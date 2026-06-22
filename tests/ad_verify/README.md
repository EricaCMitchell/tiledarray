# AD dual-verification harness (TiledArray AD ⇄ JAX + PyTorch)

An independent, re-runnable cross-check of the native C++ autodiff layer
(`src/TiledArray/ad/`) against two independent oracle frameworks: JAX and
PyTorch. See `ad_torch_verify_plan.md` (repo root) for the full design.

## Why this exists

The C++ AD layer already ships a Boost suite (`tests/ad_*.cpp`), but those tests
verify the implementation against **internal** oracles — central finite
differences of the same `ad::` primitives, and the adjoint-consistency identity
`Re⟨C̄, JVP(dX)⟩ == Re⟨VJP(C̄), dX⟩`. Both are computed *with the code under
test*, so a self-consistent convention error (a sign flip, a wrong factor, a
shared missing conjugation in both the forward and reverse paths) can satisfy
them and still be wrong relative to the rest of the world.

JAX and PyTorch are genuinely independent oracles. Agreement between the tape,
JAX, and PyTorch to roundoff is evidence the AD math is correct in an absolute
sense, not merely internally consistent. Two oracles approaching the tape from
*opposite* sides of the complex-conjugation seam (see below) provides stronger
evidence than either alone.

## Two oracles, one seam

For non-holomorphic functions (anything using `conj`, `abs`, an inner product,
or a norm), the complex VJP has two standard, mutually incompatible sign
conventions — a framework must pick one (`AUTODIFF_BACKGROUND.md` §6.3, after
Krämer 2024):

| Framework | Complex-grad convention | Complex VJP comparison | `½z²` grad at `z=1+i` |
|---|---|---|---|
| **TiledArray tape** | **plus / conjugating** | — (subject under test) | `1−i` |
| **PyTorch** | **plus / conjugating** | **direct, no adapter** (positive control) | `1−i` |
| JAX | minus / non-conjugating | `conj(s̄)`→`conj(out)` adapter | `1+i` raw |

- **"plus"/conjugating** (PyTorch, TensorFlow, TiledArray): gradient is
  `conj(∂f/∂x)`. Makes the gradient a steepest-ascent direction for optimization.
- **"minus"/non-conjugating** (JAX): `jax.vjp` is a complex-linear cotangent
  map `ct ↦ Σ ct·(∂f/∂x)` (no conjugation), prioritizing a literal
  vector–Jacobian product.

Because PyTorch uses the same convention as the tape, its complex VJP
comparison is **direct** (no adapter) — it is the positive control. JAX sits on
the opposite side and requires an adapter — it is the negative control that
proves the seam is real and load-bearing.

Under TA's plus convention the complex VJPs conjugate:

- `dot` (bilinear, `s = Σ A·B`): `Ā = s̄·conj(B)`, `B̄ = s̄·conj(A)`;
- `inner_product` (sesquilinear, `s = Σ conj(A)·B`): `Ā = conj(s̄)·B`,
  `B̄ = s̄·A`;
- `elementwise` `f`: `Ā = conj(f'(A))·C̄` (the `½z²` litmus above).

### The JAX adapter (minus → plus)

Because JAX is a minus-convention oracle, every complex VJP comparison routes
JAX's result through a documented adapter:

```
x̄_tape  ==  conj( jax.vjp(f)(conj(s̄)) )
```

For **real** `s̄`, `conj` is the identity, so all real scenarios need no adapter.

Two assertions make the seam itself a tested property:

1. With the adapter, JAX and the tape agree to roundoff (`Abar(+adapter)`,
   `Bbar(+adapter)` — ~1e-16).
2. Without the adapter the cotangents differ by ~O(1) (`seam(no-adapter differs)`)
   — the negative control proves the adapter is load-bearing.

### PyTorch direct match (plus convention)

Because PyTorch is a plus-convention oracle, complex VJP comparisons are direct:

```python
_, vjp_fn = torch.func.vjp(f, A, B)
ja, jb = vjp_fn(sbar)          # matches tape directly
```

The reconstructed minus value (negative control):

```python
ma, _ = vjp_fn(torch.conj(sbar))
minus_ref = torch.conj(ma)     # JAX-convention answer; must DIFFER from tape
```

### Forward mode needs no adapter

The JVP (pushforward) is convention-independent: both `jax.jvp` and
`torch.func.jvp` compare directly against the tape. The complex JVP scenarios
pin the `AUTODIFF_BACKGROUND.md` §6.2 structural feature: a non-holomorphic op
carries a **conjugated tangent** (`conj(dA)`) not present in the real rules.

## How to run

```sh
tests/ad_verify/run.sh [jax|torch|both]
```

Default is `both`. `run.sh` builds the producer once, runs it to emit a golden
file, runs the selected oracle(s), and propagates a nonzero exit code if any
verifier fails.

```sh
# JAX only (original gate)
tests/ad_verify/run.sh jax

# PyTorch only
tests/ad_verify/run.sh torch

# Both (default)
tests/ad_verify/run.sh both
# or simply:
tests/ad_verify/run.sh
```

Config:

- `TA_BUILD_DIR` — build directory holding the `ad_produce` target
  (default: `build/`).
- `GOLDEN` — path for the golden file (default: a temp file).
- `BACKEND` — fallback backend selector if no positional arg given.

Requirements:

- A configured TA build directory.
- `jax` + `numpy` for the JAX oracle (tested with JAX 0.10.1 / NumPy 2.4.6).
- `torch` >= 2.0 with `torch.func` for the PyTorch oracle; complex128 is
  enabled via `torch.set_default_dtype(torch.float64)`.

## Pieces

- **`produce.cpp`** -> `ad_produce` (CMake target, `EXCLUDE_FROM_ALL`). A
  standalone `main` (not a Boost module) that runs the `ad::` layer on
  deterministic inputs and writes `golden.json`. Because it is `EXCLUDE_FROM_ALL`
  it never participates in the `ta_test` link. Build it with `ninja ad_produce`.
- **`golden.json`** — the only interface: a frozen, human-inspectable,
  framework-agnostic record of exactly what was compared. Regenerated each run
  (not a checked-in fixture).
- **`ad_verify_common.py`** — framework-neutral utilities: golden decoding
  (`tensor_np`, `scalar`), the `Check` comparison class, tolerances, and the
  table-printing scenario runner. Imported by both verifiers.
- **`verify_jax.py`** — loads the golden file, recomputes each reference in JAX,
  and prints a per-scenario table (max abs diff, max rel diff, PASS/FAIL). Uses
  the JAX-specific adapter for complex VJP scenarios.
- **`verify_torch.py`** — same at full scenario parity, using `torch.func.*` for
  the AD transforms. Direct match for complex VJP (plus convention).

## np = 1 rationale

The producer runs at **np = 1**. A single process owns every tile, so the dump
is complete and canonical without any cross-rank gather; both JAX and PyTorch
are single-process, so there is nothing on the oracle side to distribute to; and
multi-rank SPMD invariance of the AD layer is already covered by the Boost suite
at np = 2 (`run-np-2`). This harness verifies the *math*, which is
rank-count-invariant by construction.

Inputs are generated by a fixed-seed `std::mt19937_64` + `uniform_real(-1, 1)`
in a fixed traversal order, so the producer reproduces **byte-identical inputs**
run-to-run. The RNG is **not** shared with Python — the producer emits the
actual input values it used, and the verifiers consume those exact values.

Reduction-derived *outputs* (e.g. the `squared_norm_jvp` tangent and the `hvp`
gradient) can differ in their last bit between runs because TA's parallel
reductions sum in task order, which is not bit-stable under `MAD_NUM_THREADS>1`.
This is ~1e-16, well inside the verifier tolerances.

## Tolerances

Scenario-appropriate:

- **tight** (`rtol=1e-11, atol=1e-12`) for value / JVP / VJP checks exact up to
  roundoff;
- **loose** (`rtol=1e-6, atol=1e-9`) for norm-type ops (`norm2`,
  `squared_norm`) and the HVP.

## Notes

- `norm2` is non-differentiable at `A = 0`; inputs are random in `[-1, 1]` so
  norms are comfortably nonzero.
- `torch.func.jvp` with `torch.conj` uses PyTorch's forward-mode AD rule for
  conjugation (tangent is conjugated). If a future PyTorch version drops this
  rule, the fallback is `torch.autograd.forward_ad` dual tensors
  (`make_dual`/`unpack_dual`) for the `complex_conj_jvp` scenario.
- The producer re-implements the scenario *setup* (it calls the real `ad::`
  ops, not a re-implementation of them). Scenarios are named identically to the
  Boost cases they mirror so drift is easy to spot.
