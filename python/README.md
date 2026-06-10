# PyTiledArray

Python bindings for [TiledArray](https://github.com/ValeevGroup/tiledarray) — a distributed,
tiled, order-N tensor library built on MADWorld (MADNESS parallel runtime).

PyTiledArray exposes TiledArray's core functionality to Python: distributed array construction,
a lazy expression DSL for tensor arithmetic, Einstein summation, NumPy interoperability, and
sparse array support.

## Building

PyTiledArray is built as part of TiledArray when `-DTA_PYTHON=ON` is passed to CMake.
`pytest` and `numpy` must be importable by the Python interpreter CMake selects.

```sh
cmake -B build -DTA_PYTHON=ON -DBUILD_TESTING=ON ..
cmake --build build --target python-tiledarray
```

The compiled extension (`tiledarray.so` / `tiledarray.dylib`) is placed in the build
directory. To import it from Python, add that directory to `PYTHONPATH`:

```sh
export PYTHONPATH=/path/to/build
python -c "import tiledarray"
```

### Running the tests

```sh
cd build
ctest -R tiledarray/unit/python -V
```

The multi-rank test (`test_mpi`) runs with `mpiexec -n 2` and is separate from the
single-rank suite.

## Quick start

```python
import tiledarray as ta
import numpy as np

# The runtime is initialized on import and finalized via atexit.
world = ta.get_default_world()
print(f"rank {world.rank} of {world.size}")

# Create a 6x8 dense float64 array, tiled 2x4
a = ta.TArray([6, 8], block=2)
a.fill(1.0)

# Create another array and perform a contraction (matrix multiply)
b = ta.TArray([8, 4], block=2)
b.fill(2.0)

c = ta.TArray([6, 4], block=2)
c["i,j"] = a["i,k"] * b["k,j"]

print(c["i,j"].norm())
```

## Array types

PyTiledArray exposes four element dtypes, each in dense (`TArray*`) and sparse (`TSpArray*`)
variants:

| Class       | C++ type           | NumPy dtype  |
|-------------|--------------------|--------------|
| `TArray`    | `double`           | `float64`    |
| `TSpArray`  | `double`           | `float64`    |
| `TArrayF`   | `float`            | `float32`    |
| `TSpArrayF` | `float`            | `float32`    |
| `TArrayZ`   | `complex<double>`  | `complex128` |
| `TSpArrayZ` | `complex<double>`  | `complex128` |
| `TArrayC`   | `complex<float>`   | `complex64`  |
| `TSpArrayC` | `complex<float>`   | `complex64`  |

## Constructing arrays

**From a shape and uniform block size:**

```python
# 10x10 array, tiles of size 5
a = ta.TArray([10, 10], block=5)
```

**From explicit tile boundaries (a `TiledRange`):**

```python
# Mode 0: tiles [0,3), [3,6); mode 1: tiles [0,4), [4,8), [8,10)
a = ta.TArray([[0, 3, 6], [0, 4, 8, 10]])
```

**From a `TiledRange` object:**

```python
tr1_rows = ta.TiledRange1([0, 3, 6])
tr1_cols = ta.TiledRange1([0, 4, 8, 10])
tr = ta.TiledRange([tr1_rows, tr1_cols])
a = ta.TArray(tr)
```

**With a tile initializer:**

```python
# op receives each tile's Range; return a NumPy array of matching shape
a = ta.TArray([6, 8], block=2, op=lambda r: np.ones(r.shape))
```

**From a NumPy array:**

```python
data = np.arange(24.0).reshape(6, 4)
a = ta.TArray.from_array(data, [[0, 3, 6], [0, 2, 4]])
```

**Element-wise initializer:**

```python
a = ta.TArray([4, 4], block=2)
a.init_elements(lambda idx: float(idx[0] + idx[1]))
```

## Tiled ranges

`TiledRange1` represents a 1-D tiling:

```python
tr1 = ta.TiledRange1([0, 10, 20, 30])   # 3 tiles of width 10
tr1 = ta.TiledRange1.make_uniform(30, 10)  # same result

print(tr1.tile_extent)  # 3
print(tr1.extent)       # 30
print(tr1.tiles)        # [[0,10], [10,20], [20,30]]
```

`TiledRange` is the N-dimensional Cartesian product:

```python
tr = ta.TiledRange([[0, 4, 8], [0, 3, 6, 9]])
print(tr.rank)              # 2
print(tr.elements_range)    # Range over [0,8) x [0,9)
print(tr.tiles_range)       # Range over 2x3 tile grid
```

## Filling and initializing

```python
a.fill(0.0)        # fill all tiles with a scalar
a.fill(1.0)
```

**Sparse note:** `fill(0)` on a `TSpArray*` yields an *empty* array — below-threshold
tiles are not materialized. Use a small nonzero value, or use a dense array, if you
need the tiles to exist.

## Expression DSL

Annotate arrays with index strings to enter the lazy expression DSL. Evaluation happens
when the result is assigned.

```python
# Elementwise sum
c["i,j"] = a["i,j"] + b["i,j"]

# Scale
c["i,j"] = 2.0 * a["i,j"]

# Contraction (matrix multiply equivalent)
c["i,j"] = a["i,k"] * b["k,j"]

# Transpose: any permutation of indices is supported
c["j,i"] = a["i,j"]

# Chain operations
d["i,j"] = a["i,k"] * b["k,j"] + c["i,j"]
```

### Reductions

Reductions drive evaluation of the expression and return a Python scalar:

```python
expr = a["i,j"]

expr.norm()          # Frobenius norm
expr.squared_norm()  # squared Frobenius norm
expr.sum()           # sum of all elements
expr.dot(b["i,j"])   # inner product

# Real-valued dtypes (TArray, TArrayF) only:
expr.min()
expr.max()
expr.abs_min()
expr.abs_max()
expr.trace()
```

## Einstein summation

`einsum` handles general multi-operand contractions. The last argument is a
pre-constructed output array; all arguments must share the same dtype and policy:

```python
# Matrix multiply: ik,kj->ij
c = ta.TArray([6, 4], block=2)
ta.einsum("ik,kj->ij", a, b, c)

# Three-tensor contraction: ijk,jl,km->ilm
result = ta.TArray([6, 4, 5], block=2)
ta.einsum("ijk,jl,km->ilm", a3, b3, c3, result)
```

## NumPy interoperability

**Read a single tile:**

```python
tile = a[[0, 0]]   # returns a NumPy array for the tile at tile index (0,0)
```

**Write a single tile:**

```python
a[[0, 0]] = np.zeros((3, 4))
```

**Iterate over local tiles:**

```python
for ref in a:
    print(ref.index)  # tile coordinate, e.g. [0, 1]
    print(ref.range)  # element Range of this tile
    print(ref.data)   # NumPy view into the tile; assignable
    ref.data = np.ones(ref.data.shape)
```

**Build from NumPy:**

```python
np_data = np.random.rand(6, 8)
a = ta.TArray.from_array(np_data, [[0, 3, 6], [0, 4, 8]])
```

## Sparse arrays

`TSpArray*` types only store tiles whose Frobenius norm exceeds a global threshold.

```python
import numpy as np

# Control the threshold (default is a small positive float)
ta.set_sparse_threshold(np.finfo(np.float32).tiny)  # "binary" sparsity
print(ta.get_sparse_threshold())

# Create a sparse array
s = ta.TSpArray([10, 10], block=5)
s.fill(0.0)           # yields an empty array (tiles below threshold are dropped)

# Check and skip zero tiles
for ref in s:
    if not s.is_zero(ref.index):
        print(ref.data)

# Truncate: drop tiles that have fallen below threshold after in-place ops
s.truncate()
```

## Cloning (deep copy)

`b = a` shares storage (shallow copy). Use `clone()` for an independent copy:

```python
b = a.clone()
b.fill(99.0)   # does not affect a
```

## MPI / distributed usage

PyTiledArray initializes the MADWorld runtime and MPI on import, using
`MPI_COMM_WORLD`. If MPI is already initialized before importing PyTA, it
must have been started with `MPI_THREAD_MULTIPLE`.

All collective operations (array construction, fences) must be executed by all
ranks in the same order.

```python
# Run with: mpiexec -n 4 python script.py
import tiledarray as ta

world = ta.get_default_world()

# Each rank constructs the same array collectively
a = ta.TArray([100, 100], block=10)
a.fill(1.0)

# Expressions chain automatically without explicit fences
b = ta.TArray([100, 100], block=10)
b.fill(2.0)
c = ta.TArray([100, 100], block=10)
c["i,j"] = a["i,j"] + b["i,j"]

# Force a global barrier (rarely needed — prefer observing ops)
world.fence()

if world.rank == 0:
    print("norm:", c["i,j"].norm())
```

**Synchronization notes:**
- Expressions chain automatically via futures — no fence is needed between
  `D = A * B` and `E = D + C`.
- Observing operations (`norm`, `dot`, `sum`, …) drive only the work they
  depend on; they do not require a preceding fence.
- `world.fence()` is a global barrier across all ranks. Use it only when
  you need a true rendezvous.

## Reference

Full type stubs (with docstrings for every class and method) are in
[`stubs/tiledarray.pyi`](stubs/tiledarray.pyi). The underlying C++ source is in
[`src/TiledArray/python/`](src/TiledArray/python/).

API documentation for the C++ library is at
<https://valeevgroup.github.io/tiledarray/dox-master/>.
