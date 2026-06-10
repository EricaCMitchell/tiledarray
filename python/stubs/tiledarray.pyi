"""Type stubs for the TiledArray Python interface (PyTA).

These stubs describe the API of the compiled ``tiledarray`` extension module
(built from ``python/src/tiledarray.cc``) for editor/IDE autocompletion and
static type checking. They are hand-maintained; keep them in sync with the
pybind11 bindings under ``python/src/TiledArray/python/``.

PyTA exposes four element dtypes, each as a dense (``TArray*``) and sparse
(``TSpArray*``) array type:

==============  =====================  ===============
suffix          C++ element type       NumPy dtype
==============  =====================  ===============
(none)          ``double``             ``float64``
``F``           ``float``              ``float32``
``Z``           ``complex<double>``    ``complex128``
``C``           ``complex<float>``     ``complex64``
==============  =====================  ===============
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence, TypeAlias, overload

import numpy as np

__all__ = [
    "World",
    "get_default_world",
    "finalize",
    "Range",
    "TiledRange",
    "TiledRange1",
    "einsum",
    "get_sparse_threshold",
    "set_sparse_threshold",
    "TArray",
    "TSpArray",
    "TArrayF",
    "TSpArrayF",
    "TArrayZ",
    "TSpArrayZ",
    "TArrayC",
    "TSpArrayC",
    "Expression",
    "SparseExpression",
    "ExpressionF",
    "SparseExpressionF",
    "ExpressionZ",
    "SparseExpressionZ",
    "ExpressionC",
    "SparseExpressionC",
    "ContractionExpression",
    "SparseContractionExpression",
    "ContractionExpressionF",
    "SparseContractionExpressionF",
    "ContractionExpressionZ",
    "SparseContractionExpressionZ",
    "ContractionExpressionC",
    "SparseContractionExpressionC",
]

# A Python scalar acceptable as a tile/array element value.
Scalar: TypeAlias = complex | float | int

# Anything accepted where a tiling is expected: a first-class TiledRange, or a
# nested list giving the tile boundaries per mode (e.g. [[0, 2, 4], [0, 3]]).
TRangeLike: TypeAlias = "TiledRange | Sequence[Sequence[int]]"

# =============================================================================
# Runtime / World
# =============================================================================

class World:
    """A MADWorld parallel context (a wrapper over an MPI sub-communicator).

    Obtain the process-global default world with :func:`get_default_world`;
    PyTA does not currently support constructing sub-worlds from Python.
    """

    @property
    def rank(self) -> int:
        """This process's rank within the world."""

    @property
    def size(self) -> int:
        """Number of processes in the world."""

    def fence(self) -> None:
        """Global barrier: block until all enqueued tasks on all ranks finish.

        This is the brute-force synchronization option. Prefer observing ops
        (``norm``, ``dot``, ...) which drive only the work they depend on.
        """

def get_default_world() -> World:
    """Return the process-global default :class:`World` initialized on import."""

def finalize() -> None:
    """Tear down TiledArray (and MADNESS, if PyTA initialized it).

    Registered with ``atexit`` on import, so explicit calls are rarely needed.
    A process that let PyTA initialize MADNESS gets exactly one init/finalize
    cycle; MADWorld cannot be re-initialized afterward.
    """

# =============================================================================
# Index spaces
# =============================================================================

class Range:
    """A rectilinear N-dimensional integer index range (not necessarily 0-based)."""

    def __init__(self, ranges: Sequence[tuple[int, int]]) -> None:
        """Construct from a ``[lo, hi)`` half-open bound pair per dimension."""

    @property
    def ndim(self) -> int:
        """Number of dimensions (rank)."""

    @property
    def size(self) -> int:
        """Total number of elements (volume)."""

    @property
    def shape(self) -> list[int]:
        """Per-dimension extent (``hi - lo``)."""

    @property
    def start(self) -> list[int]:
        """Per-dimension lower bound (lobound)."""

    @property
    def stop(self) -> list[int]:
        """Per-dimension upper bound (upbound)."""

    def __str__(self) -> str: ...

class TiledRange1:
    """A 1-D tiling: a contiguous sequence of tiles partitioning an element range."""

    def __init__(self, boundaries: Sequence[int]) -> None:
        """Construct from tile boundaries, e.g. ``[0, 2, 5, 6]`` gives 3 tiles."""

    @staticmethod
    def make_uniform(extent: int, block: int) -> TiledRange1:
        """Tile ``[0, extent)`` into uniform tiles of size ``block`` (last may be short)."""

    @property
    def tile_extent(self) -> int:
        """Number of tiles."""

    @property
    def extent(self) -> int:
        """Number of elements spanned."""

    @property
    def tiles(self) -> list[list[int]]:
        """``[lo, hi]`` element bounds for each tile."""

    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class TiledRange:
    """An N-dimensional tiling: the Cartesian product of per-mode :class:`TiledRange1`."""

    @overload
    def __init__(self, trange1_list: Sequence[TiledRange1]) -> None:
        """Construct from one :class:`TiledRange1` per dimension."""
    @overload
    def __init__(self, trange_list: Sequence[Sequence[int]]) -> None:
        """Construct from nested tile-boundary lists, one per dimension."""

    @property
    def rank(self) -> int:
        """Number of dimensions."""

    @property
    def elements_range(self) -> Range:
        """The element index space spanned by the tiling."""

    @property
    def tiles_range(self) -> Range:
        """The index space of tile coordinates."""

    @property
    def data(self) -> list[TiledRange1]:
        """The per-mode :class:`TiledRange1` tilings."""

    def make_tile_range(self, idx: Sequence[int]) -> Range:
        """Return the element :class:`Range` footprint of the tile at ``idx``."""

    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# =============================================================================
# Distributed arrays
# =============================================================================

class _Array:
    """Common interface shared by every concrete ``TArray*`` / ``TSpArray*`` type.

    A :class:`_Array` is a distributed, tiled, order-N tensor and a *shallow-copy
    handle*: ``b = a`` shares the underlying storage (mutations through ``b`` are
    seen through ``a``); use :meth:`clone` for a deep copy.

    This base type is not instantiable from Python and exists only to give the
    stubs a single place to declare the shared API.
    """

    class Reference:
        """A handle to one local tile, yielded when iterating an array."""

        @property
        def index(self) -> list[int]:
            """The tile's coordinate within the array."""

        @property
        def range(self) -> Range:
            """The tile's element :class:`Range` (its footprint in the array)."""

        @property
        def data(self) -> np.ndarray:
            """The tile's elements as a NumPy view; assignable to overwrite the tile."""
        @data.setter
        def data(self, value: np.ndarray) -> None: ...

    @overload
    def __init__(self) -> None:
        """Construct a null array."""
    @overload
    def __init__(
        self,
        shape: Sequence[int],
        block: int,
        world: World | None = ...,
        op: Callable[[Range], np.ndarray] | None = ...,
    ) -> None:
        """Construct from an element ``shape`` uniformly tiled with the given ``block``.

        If ``op`` is given it is called per tile with the tile's :class:`Range`
        and must return that tile's data as a NumPy array.
        """
    @overload
    def __init__(
        self,
        trange: TRangeLike,
        world: World | None = ...,
        op: Callable[[Range], np.ndarray] | None = ...,
    ) -> None:
        """Construct from an explicit tiling (a :class:`TiledRange` or nested lists)."""

    @property
    def world(self) -> World:
        """The :class:`World` this array lives on."""

    @property
    def trange(self) -> list[list[int]]:
        """The tiling as nested tile-boundary lists, one per dimension."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The element extent per dimension."""

    @property
    def is_dense(self) -> bool:
        """``True`` for ``TArray*`` (dense) types, ``False`` for ``TSpArray*``."""

    def fill(self, value: Scalar, skip_set: bool = ...) -> None:
        """Fill every tile with ``value``.

        Note the sparse footgun: ``fill(0)`` on a ``TSpArray*`` yields an *empty*
        array (below-threshold tiles are not materialized).
        """

    def init(self, op: Callable[[Range], np.ndarray]) -> None:
        """Populate each tile by calling ``op`` with the tile's :class:`Range`."""

    def init_elements(self, op: Callable[[list[int]], Scalar]) -> None:
        """Populate each element by calling ``op`` with the element's index."""

    def clone(self) -> Any:
        """Return a deep copy (distinct from the shallow-copy ``b = a`` aliasing)."""

    def truncate(self) -> None:
        """Drop below-threshold tiles (sparse arrays); no-op semantics for dense."""

    def is_zero(self, idx: Sequence[int]) -> bool:
        """Whether the tile at ``idx`` is structurally zero (absent from storage)."""

    @staticmethod
    def from_array(
        data: np.ndarray, trange: TRangeLike, world: World | None = ...
    ) -> Any:
        """Build an array from a dense NumPy array, partitioned per ``trange``."""

    def __iter__(self) -> Iterator[_Array.Reference]:
        """Iterate over local tiles as :class:`Reference` handles."""

    @overload
    def __getitem__(self, idx: str) -> Any:
        """Annotate the array for the expression DSL, e.g. ``a["i,j"]``.

        Returns the matching ``Expression*`` type for this array's dtype/policy.
        """
    @overload
    def __getitem__(self, idx: Sequence[int]) -> np.ndarray:
        """Fetch the tile at coordinate ``idx`` as a NumPy array."""

    @overload
    def __setitem__(self, idx: str, value: Any) -> None:
        """Assign an :class:`Expression` or ContractionExpression result, e.g.
        ``c["i,j"] = a["i,k"] * b["k,j"]``."""
    @overload
    def __setitem__(self, idx: Sequence[int], value: np.ndarray) -> None:
        """Set the tile at coordinate ``idx`` from a NumPy array."""

class TArray(_Array):
    """Dense array, ``double`` (float64) elements."""

class TSpArray(_Array):
    """Sparse array, ``double`` (float64) elements."""

class TArrayF(_Array):
    """Dense array, ``float`` (float32) elements."""

class TSpArrayF(_Array):
    """Sparse array, ``float`` (float32) elements."""

class TArrayZ(_Array):
    """Dense array, ``complex<double>`` (complex128) elements."""

class TSpArrayZ(_Array):
    """Sparse array, ``complex<double>`` (complex128) elements."""

class TArrayC(_Array):
    """Dense array, ``complex<float>`` (complex64) elements."""

class TSpArrayC(_Array):
    """Sparse array, ``complex<float>`` (complex64) elements."""

# =============================================================================
# Lazy expression DSL
# =============================================================================

class _Expression:
    """An annotated array term (built by ``array["i,j"]``).

    Supports elementwise ``+``/``-`` between expressions, scalar ``*``/``/``,
    unary ``-``, and reductions. Multiplying two expressions builds a
    :class:`_ContractionExpression` (a contraction, not an elementwise product).
    """

    def __add__(self, other: _Expression) -> _Expression: ...
    def __sub__(self, other: _Expression) -> _Expression: ...
    @overload
    def __mul__(self, other: _Expression) -> _ContractionExpression:
        """Contraction over shared indices, e.g. ``a["i,k"] * b["k,j"]``."""
    @overload
    def __mul__(self, other: float) -> _Expression:
        """Scale by a scalar."""
    def __rmul__(self, other: float) -> _Expression: ...
    def __truediv__(self, other: float) -> _Expression: ...
    def __neg__(self) -> _Expression: ...

    def norm(self) -> float:
        """2-norm (Frobenius norm) of the expression."""
    def squared_norm(self) -> float:
        """Squared 2-norm."""
    def dot(self, other: _Expression) -> Scalar:
        """Inner product with another expression."""
    def sum(self) -> Scalar:
        """Sum of all elements."""

    # Real-valued (float32/float64) arrays only; absent on complex types.
    def min(self) -> float:
        """Minimum element (real dtypes only)."""
    def max(self) -> float:
        """Maximum element (real dtypes only)."""
    def abs_min(self) -> float:
        """Minimum absolute value (real dtypes only)."""
    def abs_max(self) -> float:
        """Maximum absolute value (real dtypes only)."""
    def trace(self) -> float:
        """Trace (real dtypes only)."""

class _ContractionExpression:
    """An unevaluated product of two expressions; assign it into an array to evaluate."""

class Expression(_Expression):
    """Expression over a :class:`TArray` (dense, float64)."""

class SparseExpression(_Expression):
    """Expression over a :class:`TSpArray` (sparse, float64)."""

class ExpressionF(_Expression):
    """Expression over a :class:`TArrayF` (dense, float32)."""

class SparseExpressionF(_Expression):
    """Expression over a :class:`TSpArrayF` (sparse, float32)."""

class ExpressionZ(_Expression):
    """Expression over a :class:`TArrayZ` (dense, complex128)."""

class SparseExpressionZ(_Expression):
    """Expression over a :class:`TSpArrayZ` (sparse, complex128)."""

class ExpressionC(_Expression):
    """Expression over a :class:`TArrayC` (dense, complex64)."""

class SparseExpressionC(_Expression):
    """Expression over a :class:`TSpArrayC` (sparse, complex64)."""

class ContractionExpression(_ContractionExpression): ...
class SparseContractionExpression(_ContractionExpression): ...
class ContractionExpressionF(_ContractionExpression): ...
class SparseContractionExpressionF(_ContractionExpression): ...
class ContractionExpressionZ(_ContractionExpression): ...
class SparseContractionExpressionZ(_ContractionExpression): ...
class ContractionExpressionC(_ContractionExpression): ...
class SparseContractionExpressionC(_ContractionExpression): ...

# =============================================================================
# Einstein summation
# =============================================================================

def einsum(expr: str, a0: _Array, *args: _Array) -> None:
    """Evaluate an Einstein-summation, e.g. ``einsum("ik,kj->ij", a, b, c)``.

    The trailing argument is the (pre-constructed) output array; all arguments
    must share the same dtype/policy. This is the supported entry point for
    general tensor-of-tensor products that the ``*`` DSL cannot express.
    """

# =============================================================================
# Sparse-shape control
# =============================================================================

def get_sparse_threshold() -> float:
    """Return the global ``SparseShape`` Frobenius-norm threshold."""

def set_sparse_threshold(threshold: float) -> None:
    """Set the global ``SparseShape`` threshold.

    For "binary" sparsity (zero iff the tile norm is exactly zero) use
    ``numpy.finfo(numpy.float32).tiny`` so any nonzero norm clears the bar.
    """
