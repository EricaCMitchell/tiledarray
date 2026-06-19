# Plan: A NumPy-Convention Compatibility Layer for PyTA

## Motivation

Most users come to PyTA already fluent in NumPy. Today the PyTA API diverges
from NumPy in several user-visible ways, the most jarring of which is
`einsum`:

```python
# PyTA today — result is an OUT-parameter, returns None
c = ta.TArray([4, 4], block=2)
ta.einsum("ik,kj->ij", a, b, c)        # writes into pre-allocated c

# NumPy — result is RETURNED
c = np.einsum("ik,kj->ij", a, b)
```

This plan catalogs every place PyTA's Python API differs from NumPy, then
proposes an **additive, pure-Python compatibility layer** that gives users
NumPy-shaped entry points (returning new arrays, free-function reductions,
`@`/`.T`, `zeros`/`ones`/`eye`, `dtype=`) without removing or breaking any of
the current annotation-string / out-parameter API that downstream code
(notably PyMPQC) already relies on.

The work is deliberately scoped to a wrapper. No C++ binding changes are
required for Phase 1 — the existing pybind11 surface
(`python/bindings/*.h`, see `pyta_architecture_plan.md`) already exposes every
primitive the wrapper needs.

## Where PyTA differs from NumPy

The current Python API lives entirely in the compiled `_tiledarray` extension
module (bound in `python/bindings/`, per `pyta_architecture_plan.md`);
`python/tiledarray/__init__.py` re-exports it with `from ._tiledarray import *`.
Relevant surface:

- `einsum.h` — `ta.einsum(expr, a0, *args)` with the **result array as the last
  positional**, in-place, returns `None`.
- `expression.h` — `a["i,j"]` returns an `Expression`; reductions (`sum`,
  `norm`, `dot`, `trace`, `min`, `max`, …) are **methods on `Expression`**, so
  they require an annotation string. `*` between two expressions builds a
  `ContractionExpression`.
- `array.h` — per-dtype classes (`TArray`, `TArrayF`, `TArrayZ`, `TArrayC` and
  the sparse `TSpArray*` variants); construction via `Array(shape, block,
  world, op)` / `Array(trange, …)`; `fill()` is a method; `from_array()` is a
  static method requiring an explicit trange; `a[i, j]` indexing returns a
  **tile block**, not an element; `.shape` exists but `.dtype`/`.ndim`/`.size`/
  `.T` do not.
- `trange.h` — `TiledRange` / `TiledRange1` are exposed with `.data`, `.tiles`,
  `make_uniform`, etc. (This is the key enabler — the wrapper can compute output
  tilings in pure Python.)

| # | Topic | NumPy | PyTA today | Divergence |
|---|-------|-------|------------|------------|
| 1 | `einsum` | `c = np.einsum("ik,kj->ij", a, b)` returns result | `ta.einsum("ik,kj->ij", a, b, c)` writes into trailing `c`, returns `None` | Result is an out-param; caller must pre-allocate `c` with the correct trange |
| 2 | matmul | `c = a @ b`, `np.matmul`, `np.dot`, `np.tensordot` | `c["i,j"] = a["i,k"] * b["k,j"]` into pre-allocated `c` | No `@`, no `dot`/`matmul`/`tensordot`; needs annotation strings + output array |
| 3 | transpose | `a.T`, `a.transpose()` | `b["i,j"] = a["j,i"]` into pre-allocated `b` | No `.T` / `.transpose()` |
| 4 | elementwise | `a + b`, `a - b`, `a * b`, `2 * a` (array±array returns new array) | `c["i,j"] = a["i,j"] + b["i,j"]`; operators live on `Expression`, not array | Array±array has no operator; must annotate and pre-allocate |
| 5 | reductions | `a.sum()`, `np.sum(a)`, `a.sum(axis=0)`, `np.linalg.norm(a)` | `a["i,j"].sum()`, `a["i,j"].norm()` | Methods on `Expression` (need annotation string); no free functions; no `axis=` |
| 6 | construction | `np.zeros`, `ones`, `full`, `empty`, `eye`, `arange`, `asarray`, `array` | `Array(shape, block, …)` + `.fill()`; `from_array(data, trange, world)` | No factory functions; `fill` is a method; `from_array` needs explicit trange |
| 7 | dtype | one `ndarray`, `dtype=np.float32` | distinct classes `TArray`/`TArrayF`/`TArrayZ`/`TArrayC` (+ `TSp*`) | No `dtype=` selector; dtype encoded in the class name |
| 8 | attributes | `a.dtype`, `a.ndim`, `a.size`, `a.shape`, `a.T` | only `a.shape` (and `a.trange`, `a.world`) | Missing `dtype`/`ndim`/`size`/`T` |
| 9 | indexing | `a[0, 0]` → scalar element; `a[0:2, :]` → subarray | `a[0, 0]` → **tile block** (np.ndarray); `a["i,j"]` → `Expression` | `a[ints]` returns a tile, not an element; element/slice access is not NumPy-like (left out of scope, see below) |

## Design principles

1. **Additive and non-breaking.** Every existing call site keeps working. The
   wrapper adds new names and new accepted signatures; it never removes the
   out-parameter `einsum`, the annotation-string `Expression` API, or the
   per-dtype classes.
2. **Pure Python, Phase 1.** Implement the layer in `python/tiledarray/` on top
   of the already-bound primitives. No recompilation needed to iterate.
3. **Explicit about the tiling.** NumPy has no concept of tiles; PyTA must pick
   one for every freshly-created array. The wrapper derives output tilings from
   the operands wherever possible (einsum/matmul) and exposes a `block=`/
   `trange=` knob for the factory functions, with a documented default.
4. **Fail loudly on the genuinely-distributed mismatches** (e.g. operands on
   different worlds), rather than silently picking a world.

## Proposed compatibility layer

Create `python/tiledarray/numpy_compat.py` and re-export its public names from
`python/tiledarray/__init__.py` (after the `from ._tiledarray import *`). Keep the
compiled `einsum` reachable internally (e.g. bind it as `_einsum_into` or call it
through `tiledarray._tiledarray`) so the Python `einsum` wrapper can delegate.

### A. `einsum` that returns its result (headline fix)

```python
def einsum(subscripts, *operands, out=None, world=None):
    """NumPy-style einsum. Returns a new array unless `out=` is given."""
```

Dispatch rule that preserves back-compat with the current out-parameter form:

- Parse `subscripts` to count input terms `n` (left of `->`).
- If `len(operands) == n` → **NumPy mode**: compute the output trange from the
  operands (see *Output-trange helper*), allocate the result array, call the
  compiled in-place primitive, return the result.
- If `len(operands) == n + 1` → **legacy mode**: the trailing operand is the
  pre-allocated out array; call the compiled primitive and return `None` (exact
  current behavior). Emit no warning initially; optionally `DeprecationWarning`
  later.
- `out=` keyword, when supplied, always takes the in-place path and is returned.

Result dtype/sparsity: pick the result class from the operands (all operands
must share element type in the current binding — keep that constraint; choose
dense vs sparse by "sparse if any operand is sparse", matching how TA propagates
shape). Validate the operands share a `world` and reuse it.

### B. `matmul` / `@` / `dot` / `tensordot`

Thin wrappers over `einsum`:

```python
def matmul(a, b, out=None):      # 2-D: "ik,kj->ij"; batched/stacked later
def dot(a, b, out=None):          # 1-D·1-D scalar, 2-D·2-D matmul, etc.
def tensordot(a, b, axes=2):      # build subscripts from `axes`, delegate
```

Add `__matmul__` (and `__rmatmul__`) to each array class so `a @ b` works:

```python
for _cls in (ta.TArray, ta.TSpArray, ta.TArrayF, ...):
    _cls.__matmul__ = lambda self, other: matmul(self, other)
```

(pybind11 classes accept Python-side attribute injection; if that proves
brittle, fall back to a small pure-Python `ndarray`-like facade — see *Open
questions*.)

### C. Factory functions with `dtype=`

```python
def zeros(shape, *, block=None, trange=None, dtype=float, sparse=False, world=None)
def ones (shape, *, block=None, trange=None, dtype=float, sparse=False, world=None)
def full (shape, fill_value, *, block=..., dtype=..., sparse=..., world=...)
def empty(shape, *, ...)          # alias of zeros for dense; empty array for sparse
def eye  (n, m=None, *, block=..., dtype=..., world=...)
def asarray(data, *, block=None, trange=None, dtype=None, sparse=False, world=None)
def array  (data, ...)            # alias of asarray (copy semantics)
```

Implementation notes:

- `dtype` → class via a mapping table:
  `{float/float64: TArray|TSpArray, float32: TArrayF|TSpArrayF,
    complex128: TArrayZ|TSpArrayZ, complex64: TArrayC|TSpArrayC}`,
  honoring `sparse=`. Accept NumPy dtypes and Python `float`/`complex`.
- Tiling default: if neither `block` nor `trange` is given, use a documented
  default block size (proposal: a single tile per dimension, i.e. `block=max
  extent`, which is the least-surprising "it just works" choice; revisit for
  large arrays). When `block` is an int, build a uniform `TiledRange` via
  `TiledRange1.make_uniform`.
- `zeros`/`empty` on a **sparse** type must respect the documented footgun
  (`fill(0)` yields an empty sparse array). Provide `zeros(..., sparse=True)`
  that returns a structurally-empty array and document it, matching TA
  semantics rather than fabricating zero tiles.
- `asarray` infers `shape` and `dtype` from the NumPy input and delegates to the
  existing `from_array` (supplying a default trange when none is given), so it is
  the NumPy-inference counterpart to the current explicit-trange static method.

### D. NumPy-like array attributes and methods

Inject onto the array classes (read-only properties / methods):

- `dtype` → the NumPy dtype matching the class.
- `ndim` → `len(self.shape)`.
- `size` → product of `self.shape`.
- `T` and `transpose(*axes)` → return a **new array** computed via the existing
  permutation path (`out["...perm..."] = self["...ident..."]`), defaulting to a
  full reversal of axes like NumPy.

### E. Free-function reductions

```python
def sum(a, axis=None)             # axis=None: full reduction (scalar)
def trace(a)
def norm(a)                       # Frobenius, == np.linalg.norm default
def min(a); def max(a)
def dot(a, b)                     # see B
```

Each builds the canonical annotation string (`"i,j,k,..."` for the operand's
rank) internally and calls the corresponding `Expression` method, so the user
never types annotation indices. Also add NumPy-named array methods
(`a.sum()`, `a.trace()`, `a.norm()`) as thin wrappers.

`axis=` support: Phase 1 implements only `axis=None` (full reduction) and raises
`NotImplementedError` for partial axes, with a TODO. Partial reductions are an
einsum with a dropped output index (`"ij->i"`); they can be added in Phase 2 by
routing through the einsum wrapper once that path is solid.

### F. Elementwise array operators (optional, Phase 2)

Add `__add__`/`__sub__`/`__mul__`/`__rmul__`/`__neg__` on the array classes that
return new arrays, by allocating an output with the operands' shared trange and
evaluating `out[idx] = a[idx] (op) b[idx]` with a generated annotation string.
Gate behind Phase 2 because the shape/trange-compatibility and broadcasting
rules need care, and because `a * b` is genuinely ambiguous between elementwise
(NumPy `*`) and contraction (TA `*` on expressions) — we will bind array `*` to
**elementwise** (NumPy semantics) and keep contraction on the expression API and
`@`.

## The output-trange helper (core enabler)

NumPy-style `einsum`/`matmul`/factories all need to construct a result
`TiledRange` that PyTA can allocate before the in-place primitive runs. The
mapping is mechanical and already supported by the exposed `TiledRange` API:

1. Parse `subscripts` into per-operand index-label lists and the output label
   list.
2. For each label, record the `TiledRange1` (boundary list) of the operand mode
   where it appears. Operand tilings are reachable from `array.trange` (the
   list-of-boundary-lists property bound in `array.h`) and/or `TiledRange.data`.
3. Assert consistency: a label appearing on multiple operands must have the same
   `TiledRange1` on each (this mirrors how TA's expression engine requires
   matching tilings across a contracted index).
4. Assemble the output `TiledRange` from the output labels' `TiledRange1`s and
   construct the result array on the shared world.

This helper lives in `numpy_compat.py`, is unit-tested directly, and is shared
by sections A, B, and D.

## Files to touch

- **New:** `python/tiledarray/numpy_compat.py` — the entire wrapper.
- **Edit:** `python/tiledarray/__init__.py` — after `from ._tiledarray import *`,
  import the wrapper, inject the array-class attributes/operators, and re-export
  the public names (`einsum`, `matmul`, `dot`, `tensordot`, `zeros`, `ones`,
  `full`, `empty`, `eye`, `asarray`, `array`, `sum`, `trace`, `norm`, `min`,
  `max`). Keep the compiled in-place `einsum` reachable for delegation.
- **No C++ changes in Phase 1.** (Phase 3 may move hot loops or the
  output-trange computation into `einsum.h` for performance, but only after the
  Python surface is validated.)

## Testing

Add `tests/python/test_numpy_compat.py`, parametrized over `[ta.TArray,
ta.TSpArray]` like the existing suites, and cross-checked against NumPy:

- `einsum` returns a correct new array; legacy out-parameter form still returns
  `None` and writes in place (back-compat regression test).
- `matmul`/`@`/`dot`/`tensordot` match `np.matmul`/`np.dot`/`np.tensordot`.
- `zeros`/`ones`/`full`/`eye`/`asarray` produce arrays whose `np.array(...)`
  matches the NumPy equivalent, across all four dtypes and both densities.
- `dtype`/`ndim`/`size`/`T`/`transpose` match NumPy.
- Free-function reductions match `np.sum`/`np.trace`/`np.linalg.norm` (and the
  `axis=None` path of `a.sum()`).
- Sparse-specific: `zeros(sparse=True)` yields a structurally-empty array
  (documented footgun), and reductions over it behave.
- Run under the existing np=1 and np=2 pytest-marker harness (see
  `python_pytest_markers_plan.md`) so the distributed path is exercised.

## Phasing

- **Phase 1 (headline):** output-trange helper, `einsum` (returning), `matmul`/
  `@`/`dot`/`tensordot`, factory functions with `dtype=`, array attributes
  (`dtype`/`ndim`/`size`/`T`/`transpose`), free-function reductions with
  `axis=None`. All pure-Python; full test coverage.
- **Phase 2:** elementwise array operators (`+`/`-`/`*`), partial-axis
  reductions via einsum, `tensordot(axes=...)` general form, `arange`/`linspace`
  where they make sense for a tiled array.
- **Phase 3 (optional):** push the output-trange computation and/or hot paths
  into the C++ binding if profiling shows the Python layer is a bottleneck.

## Out of scope / open questions

- **Element/slice indexing (`a[0, 0]` → scalar, `a[0:2, :]` → subarray).** This
  is the deepest semantic mismatch (#9): today `a[ints]` returns a *tile block*.
  Changing it would break the existing tile-access API and conflicts with the
  string-indexing overload that returns an `Expression`. Left explicitly out of
  scope; revisit separately with a deprecation path if there's demand.
- **Attribute injection onto pybind11 classes.** Injecting `__matmul__`/`T`/
  `dtype` onto the compiled classes from Python is convenient but can be brittle
  (some pybind11 builds restrict setting attributes on the type object). If it
  fails, the fallback is a thin pure-Python `ndarray`-like facade that wraps a
  PyTA array and forwards to it — decide during Phase 1 spike.
- **`a * b` semantics.** Binding array `*` to elementwise (NumPy) while the
  `Expression` `*` remains contraction is a deliberate, documented split; confirm
  with users before shipping Phase 2.
- **Default tiling.** "One tile per dimension" is the least-surprising default
  but is wrong for large arrays. Decide whether to expose a module-level
  `set_default_block()` knob instead of hard-coding.
- **Mixed-dtype operands.** The current bindings require operands of identical
  element type; NumPy promotes. Phase 1 keeps the strict rule and raises a clear
  error; type promotion is a possible later addition.
```