# Plan: Adopt pytest markers so the Python suite runs at np=1 and np=2

## Context

TiledArray's core value is distributed computation, but the Python test suite
does not exercise it the way the C++ suite does. Today (`python/CMakeLists.txt`
lines 61–99):

- The eleven "regular" modules in `TA_PYTHON_TEST_MODULES` are each registered
  **only at `-np 1`** (hardcoded `${MPIEXEC_NUMPROC_FLAG} 1` on line 78).
- A single hand-written `test_mpi.py` is registered **only at `-np 2`** and is
  *not* in the module list, so it never runs at np=1 either.

Net result: **no Python test runs at both world sizes.** All of the
construction / indexing / expression / contraction / reduction / numpy-interop
coverage is single-rank only. Multi-rank regressions in those paths (e.g. the
remote-tile `getitem` bug just fixed) are invisible to CI unless someone
duplicated the case by hand into `test_mpi.py`.

The C++ suite already solved this. In `tests/CMakeLists.txt:171-189` the *same*
`ta_test` binary runs at np=1 with `--run_test=!@distributed` and np=2 with
`--run_test=!@serial`, and sets `TA_UT_DISTRIBUTED=1` for np≥2. Untagged tests
therefore run at **both** sizes; a test opts out of one size only by being
tagged. We will reproduce this model on the Python side with pytest markers:

| C++ Boost.Test tag        | pytest marker            | meaning                              |
|---------------------------|--------------------------|--------------------------------------|
| `@distributed`            | `@pytest.mark.distributed` | only meaningful multi-rank; skip np=1 |
| `@serial`                 | `@pytest.mark.serial`      | assumes world size 1; skip np≥2       |
| (untagged)                | (untagged)               | runs at every world size             |

**Intended outcome:** every Python test runs at np=1 and np=2 by default, giving
the Python bindings the same distributed coverage the C++ layer has, with markers
as the explicit, reviewable escape hatch for the rare size-specific case.

## Changes

### 1. Register the markers — `tests/python/pytest.ini` (new file)

pytest emits warnings (and, under `--strict-markers`, errors) for unknown
markers, so they must be declared. Add a small ini next to `conftest.py`:

```ini
[pytest]
addopts = --strict-markers -ra
markers =
    distributed: test is only meaningful with world size > 1; skipped at np=1 (mirrors C++ @distributed)
    serial: test assumes world size == 1; skipped at np>=2 (mirrors C++ @serial)
```

`--strict-markers` makes a typo'd marker a hard error instead of a silent
no-op, matching the rigor of the C++ tag scheme.

### 2. Drive selection from CTest — `python/CMakeLists.txt`

Replace the two separate registration blocks (the np=1-only module loop, lines
75–86, and the standalone np=2 `test_mpi`, lines 88–99) with a single loop that
registers **every** module at **both** world sizes, passing the marker filter
that mirrors the C++ `!@distributed` / `!@serial` args:

- Fold `test_mpi` into `TA_PYTHON_TEST_MODULES` (it is no longer special).
- For each module, `foreach(p RANGE 1 2)`:
  - np=1 → `pytest ... -m "not distributed"`
  - np=2 → `pytest ... -m "not serial"`
  - Test name: `tiledarray/unit/python/${_pymod}-np${p}`.
  - Properties: `FIXTURES_REQUIRED TA_UNIT_TESTS_PYTHON_EXEC`; for np=2 add
    `PROCESSORS 2` and `TA_UT_DISTRIBUTED=1` to the `ENVIRONMENT` (parity with
    `tests/CMakeLists.txt:183`). Keep `MAD_NUM_THREADS=2;PYTHONPATH=...`.

Pattern to follow verbatim is the C++ `foreach(p RANGE 1 2)` block at
`tests/CMakeLists.txt:171-189`. Keep one CTest entry per (module × world size)
so failures stay isolated to a file/size pair, matching today's per-module
granularity.

### 3. Mark the genuinely size-specific tests

Default is **no marker** — that is what makes a test run at both sizes, which is
the whole point. Only two categories need a marker:

- **`tests/python/test_mpi.py`**: mark the truly multi-rank cases
  `@pytest.mark.distributed` so np=1 skips them, matching C++ `@distributed`
  semantics. Candidates: `test_distributed_fill_gather`,
  `test_remote_tile_index_access`, `test_numpy_diagonal_is_replicated`,
  `test_distributed_contraction`, `test_from_array_distributed`. Leave the
  trivially-portable `test_world_size_at_least_one`, `test_rank_range`, and
  `test_sparse_zero_tile_index_raises` untagged so they run at both sizes.
  (Markers can be applied at class scope and overridden per-method, or per
  method — match whichever the file already reads cleanly with.)
- **`@pytest.mark.serial`**: apply **only** to a test that actually breaks at
  np≥2. The audit below found none today, so initially nothing gets `serial`.
  The marker exists for the validation step: if a module hangs or asserts at
  np=2, the fix is either to make it rank-count-agnostic (add the missing
  `world.fence()` / replicated-read pattern) or, if it is inherently
  single-rank, tag it `serial` with a one-line comment saying why.

No marker is needed on the pure-logic tests (`test_trange.py` non-array cases,
`test_world.py`, `test_sparse.py` threshold getters): they are world-size
agnostic and harmless at np=2. Leaving them untagged keeps marking minimal and
avoids over-tagging — the same way the C++ suite leaves most cases untagged.

## Audit of existing tests for np=2 readiness

All array-bearing tests already follow the fence-fill-fence idiom and use
collective construction, so they are expected to pass at np=2 as-is. One pattern
to watch during validation:

- `test_array_indexing.py::test_tile_setitem_getitem` sets a tile under
  `if world.rank == 0:` then reads it on all ranks after a fence — this is
  exactly the remote-tile read path the recent `getitem` fix repaired, so it is
  a *good* thing to run at np=2, not a problem.
- Any `np.array(dist_array)` round-trip relies on every rank receiving the full
  replicated buffer (the diagonal regression in `test_mpi.py`). If a reduction
  or interop test returns rank-local data at np=2, that surfaces here and is a
  real bug to fix (or, if intended, a `serial` tag).

## Verification

1. Configure/build with MPI and the throw assert policy (a faithful test build):
   `cmake --build <build> --target python-tiledarray`.
2. Run the full Python suite through CTest and confirm **both** np variants
   appear and pass:
   `ctest --test-dir <build> -R tiledarray/unit/python -V`
   Expect `…-np1` and `…-np2` entries for every module.
3. Spot-check the marker filtering directly:
   - `mpirun -np 1 python -m pytest tests/python/test_mpi.py -m "not distributed" -v`
     → distributed cases reported as deselected.
   - `mpirun -np 2 python -m pytest tests/python/test_array_indexing.py -m "not serial" -v`
     → all cases run on 2 ranks and pass.
4. Confirm `--strict-markers` works: temporarily mistype a marker and verify
   pytest errors instead of silently ignoring it; revert.
5. Triage any np=2 failures per the audit section: fix the test to be
   rank-agnostic where possible; tag `serial` only when the test is inherently
   single-rank, with a comment explaining why.

## Files touched

- `tests/python/pytest.ini` — **new**; marker registration + `--strict-markers`.
- `python/CMakeLists.txt` — replace lines ~61–99 with a single module×{1,2}
  registration loop carrying the `-m "not distributed"` / `-m "not serial"`
  filters and the np=2 `TA_UT_DISTRIBUTED`/`PROCESSORS` properties.
- `tests/python/test_mpi.py` — add `@pytest.mark.distributed` to the multi-rank
  cases.
- (As needed from validation) `@pytest.mark.serial` on any test proven to
  require world size 1.
