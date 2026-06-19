# Plan: PyTA Package Architecture

## Context

PyTA (the `python/` subtree) currently has a confusing and partly broken layout:
two directories that read like "the package" — `python/src/TiledArray/python/`
(C++ binding headers) and `python/tiledarray/` (a Python package) — plus a
compiled extension whose name collides with the intended package. This plan
**only** addresses the package/build architecture so PyTA has one coherent,
importable `tiledarray` package. It deliberately says nothing about any
NumPy-compatibility API; that is a separate effort that this architecture
enables but does not require.

This plan applies the conventions in `common_python_cpp_architecture.md`
(the general pattern for Python bindings that are **superbuild-only** — built
via `-DTA_PYTHON=ON` / `add_subdirectory(python)`, never pip-installed
standalone). PyTA is exactly that case, so the concrete choices below
(glue-dir naming, build-tree-as-complete-package, version source, install
destination, stub generation) are the TA-specific instantiation of that
document.

### Current state (verified against the code + runtime)

- The compiled extension is bound as `PYBIND11_MODULE(tiledarray, …)` in
  `python/src/tiledarray.cc`, with CMake `OUTPUT_NAME tiledarray`. It builds to
  `build/python/tiledarray.cpython-*.so`.
- The binding sources live under `python/src/TiledArray/python/` (`array.h`,
  `einsum.h`, `expression.h`, `module.h`, `python.h`, `range.h`, `trange.h`).
  These are **private to the PyTA extension** — nothing outside this subtree
  consumes them. `tiledarray.cc` pulls them in via
  `#include "TiledArray/python/module.h"`; `module.h` then uses flat
  same-directory includes (`"array.h"`, etc.).
- The test harness (`python/CMakeLists.txt`) and `python/README.md` both import
  the extension via `PYTHONPATH=build/python` → `import tiledarray`, i.e. they
  load the **bare `.so` directly**.
- `python/tiledarray/__init__.py` is **dead code**: it inserts its own directory
  onto `sys.path` and does `from tiledarray import *`, which re-imports the
  package itself (circular) and exposes none of the compiled symbols. It is never
  exercised by the tests or README, which use the bare `.so`.
- The hand-written stub is `python/stubs/tiledarray.pyi`.
- Importing the extension initializes MADNESS/MPI, so it must be run under
  `mpiexec` with the OpenMPI root env vars (`OMPI_ALLOW_RUN_AS_ROOT=1`,
  `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`,
  `OMPI_MCA_btl_vader_single_copy_mechanism=none`), matching the ctest harness.

### Problems

1. **Name collision.** The extension and the intended wrapper package are both
   named `tiledarray`, so only one can win on `import tiledarray`. Today the bare
   `.so` wins and the package is shadowed/dead.
2. **No place for pure-Python code.** Because the package is non-functional,
   there is nowhere to put Python-level code that wraps or extends the compiled
   core. Any future pure-Python surface has no home.
3. **Visual/structural confusion.** `python/src/TiledArray/python/` (C++) and
   `python/tiledarray/` (Python) look parallel but are different layers, and the
   dead package compounds the confusion. The `TiledArray/python/` nesting exists
   only to present a public include path — but no downstream consumes it, so the
   nesting is pure confusion with no remaining purpose.

## Decision

Adopt the standard scientific-Python layout: a **private compiled core**
(`_tiledarray`) wrapped by a **real `tiledarray` package**.

- Rename the compiled extension `tiledarray` → **`_tiledarray`** (the underscore
  marks it private). This renames the `.so` PyTA builds.
- Make `python/tiledarray/` a **functional package** whose `__init__.py` does
  `from ._tiledarray import *`, re-exporting every compiled symbol. `import
  tiledarray` then yields a package that behaves exactly like the old bare
  extension (so existing tests keep working) and now also has a place to host
  pure-Python code in the future.
- **Move the private binding source into `python/bindings/`** — both
  `tiledarray.cc` and the seven headers, flat in one directory — and **delete
  the now-empty `python/src/` tree** (including its `TiledArray/python/`
  nesting). Two reasons: (1) the headers are private implementation detail of
  the single extension, so the `TiledArray/python/` include namespace served
  no remaining purpose; (2) per `common_python_cpp_architecture.md`, the
  glue-source directory must **not** be named `src/` when the project already
  has a top-level `src/` (TA's C++ library) — that name reads as a Python
  *src-layout* package directory, which this is not. `bindings/` is
  unambiguous. Update `tiledarray.cc`'s include to `#include "module.h"`;
  `module.h`'s own includes are already flat (`"array.h"`, …) and need no
  change.
- **Generate the type stubs** with `pybind11-stubgen` instead of hand-writing
  them. The current `python/stubs/tiledarray.pyi` is hand-maintained and drifts
  from the compiled surface; replace it with generated output
  (`tiledarray/__init__.pyi` + `tiledarray/_tiledarray.pyi`) plus a `py.typed`
  marker. The common doc's *default* is to not commit `.pyi` and regenerate them
  on every build via a custom command that `DEPENDS` on the extension. That
  default does not fit PyTA: importing the module initializes MADNESS/MPI and
  must run under `mpiexec` with the OpenMPI root env vars, so wiring stubgen into
  the ordinary build would force every build to launch MPI. PyTA therefore takes
  the doc's explicit escape hatch — *"treat regenerating them as a separate
  manual maintainer step, not a build guarantee"* — via a dedicated, opt-in
  CMake target (see §5), and commits the generated stubs so IDEs/type-checkers
  work without a build.
- **Single version source via CMake.** Per the common doc, generate
  `python/tiledarray/_version.py` from the CMake project version with
  `configure_file()` at configure time, and have `__init__.py` read it
  (`__version__`). There is no `pyproject.toml` / `setuptools_scm` here — the
  CMake project version is authoritative.

### Resulting layout

```
python/
  CMakeLists.txt
  README.md
  bindings/                         # private C++ binding sources (was python/src/)
    tiledarray.cc                   # PYBIND11_MODULE(_tiledarray, …)   <- renamed macro
    module.h                        # #include "array.h" … (already flat)
    array.h einsum.h expression.h
    python.h range.h trange.h
  tiledarray/                       # the one real Python package (source tree)
    __init__.py                     # from ._tiledarray import *  (re-export) + version
    __init__.pyi                    # generated by pybind11-stubgen (committed)
    _tiledarray.pyi                 # generated by pybind11-stubgen (committed)
    py.typed                        # PEP 561 marker
```

Build tree (assembled by `python/CMakeLists.txt`, so `PYTHONPATH=<build>/python`
imports a complete package without `cmake --install`):

```
<build>/python/tiledarray/
  __init__.py                       # staged (configure_file COPYONLY)
  __init__.pyi  _tiledarray.pyi     # staged
  py.typed                          # staged
  _version.py                       # generated by configure_file() from CMake version
  _tiledarray.cpython-*.so          # the build output (LIBRARY_OUTPUT_DIRECTORY)
```

After the change there is exactly **one** functional `tiledarray` import target
(the package), and the `TiledArray`/`tiledarray` confusion is **fully
eliminated**: the `python/src/` tree (and its `TiledArray/python/` nesting) no
longer exists, so there is no longer a `TiledArray` directory — nor a
`src/`-named directory — anywhere in the Python subtree.

## Scope of changes

### 1. Extension module name
- `tiledarray.cc`: `PYBIND11_MODULE(tiledarray, m)` →
  `PYBIND11_MODULE(_tiledarray, m)`. (One-line change; recompiles this single TU
  and relinks the `.so`.)

### 2. Rename the glue directory and flatten the headers
- `git mv python/src python/bindings` to rename the binding-source directory
  (avoids the top-level-`src/` collision flagged in
  `common_python_cpp_architecture.md`).
- `git mv python/bindings/TiledArray/python/{array,einsum,expression,module,python,range,trange}.h
  python/bindings/`, then remove the emptied `python/bindings/TiledArray/` tree.
- `python/bindings/tiledarray.cc`: `#include "TiledArray/python/module.h"` →
  `#include "module.h"`. `module.h` is unchanged — its flat same-directory
  includes already resolve against the new location.

### 3. Build/package wiring — `python/CMakeLists.txt`
- `pybind11_add_module(... bindings/tiledarray.cc)` — point at the renamed glue
  directory; include resolution still works because the headers sit beside the
  `.cc`.
- `OUTPUT_NAME _tiledarray` and
  `LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tiledarray`, so the core `.so`
  lands inside the package directory in the build tree.
- Stage the static package files (`__init__.py`, `__init__.pyi`,
  `_tiledarray.pyi`, `py.typed`) into `${PROJECT_BINARY_DIR}/tiledarray/` via
  `configure_file(... COPYONLY)` (records a configure dependency so edits restage
  on rebuild). Together with the `.so` and generated `_version.py` below, this
  makes `${PROJECT_BINARY_DIR}/tiledarray/` a complete, importable package in the
  build tree.
- **Generate `_version.py`** into `${PROJECT_BINARY_DIR}/tiledarray/` with
  `configure_file()` from the CMake project version (e.g. a
  `_version.py.in` template containing `__version__ = "@TILEDARRAY_VERSION@"`).
  Not committed — it is a build artifact.
- **Install destination as a cache variable**, not a hardcoded libdir:
  ```cmake
  set(TA_PYTHON_INSTALL_DIR "lib/python3/site-packages" CACHE PATH
      "Where to install the PyTA bindings, relative to CMAKE_INSTALL_PREFIX
       unless an absolute path is given")
  ```
  `install(...)` then places the compiled core **plus** the package files
  (`__init__.py`, stubs, `py.typed`, generated `_version.py`) into a
  `tiledarray/` package directory under `${TA_PYTHON_INSTALL_DIR}`. This leaves
  the choice (self-contained path the user adds to `PYTHONPATH`, vs. the active
  env's `Python3_SITEARCH`) to the builder — HPC/toolchain builds are often not
  inside a venv.
- The test `PYTHONPATH=${PROJECT_BINARY_DIR}` is unchanged: `import tiledarray`
  now resolves the staged package, which loads `_tiledarray`. No per-test edits
  needed.

### 4. The package — `python/tiledarray/__init__.py`
- Replace the dead circular stub with a real one:
  ```python
  from ._tiledarray import *      # re-export all compiled classes/functions
  from . import _tiledarray       # keep the core module reachable as tiledarray._tiledarray
  from ._version import __version__  # CMake-generated single version source
  ```
- `from ._tiledarray import *` pulls every public (non-underscore) name the
  pybind module defines (all `TArray*`/`TSpArray*`, the `Expression*` types,
  `einsum`, `World`, `get_default_world`, `TiledRange*`, `Range`,
  `get/set_sparse_threshold`, …), so `tiledarray.X` keeps resolving exactly as
  before. The extension's `atexit` finalize registration still runs on import of
  `_tiledarray`.

### 5. Stubs (auto-generated)
- **Delete the hand-written `python/stubs/tiledarray.pyi`** and the `python/stubs/`
  directory. Stubs are no longer maintained by hand.
- Generate stubs with `pybind11-stubgen` pointed at the built package, e.g.
  `pybind11-stubgen tiledarray -o python/` with `PYTHONPATH=${PROJECT_BINARY_DIR}`.
  Run once at world size 1, it produces `python/tiledarray/__init__.pyi` (the
  public package surface) and `python/tiledarray/_tiledarray.pyi` (the compiled
  core) in one pass, both reflecting the actual compiled symbols.
- **Import-environment wrinkle.** `pybind11-stubgen` imports the module, which
  initializes MADNESS/MPI, so generation must run under `mpiexec -n 1` with the
  OpenMPI root env vars (same as the ctest harness), not a bare `python`.
- Add a dedicated CMake target — `python-tiledarray-stubs` — that builds
  `python-tiledarray` first, then runs the stubgen command above under that
  environment. It is **not** wired into the default build (so an ordinary build
  needs neither `pybind11-stubgen` nor an MPI launch); developers run it to
  refresh stubs after changing the bindings.
- The generated `__init__.pyi` / `_tiledarray.pyi` are **committed** to
  `python/tiledarray/` so IDEs and type-checkers work without a build; the target
  regenerates them on demand. This is a deliberate deviation from the common
  doc's "don't commit, regenerate every build" default — see the Decision
  section: the MPI-init-on-import wrinkle rules out a build-time custom command,
  so PyTA uses the doc's "separate manual maintainer step" path and accepts the
  staleness trade-off in exchange for build-free IDE support. Add
  `python/tiledarray/py.typed` (PEP 561 marker; hand-written, one empty file).
- `pybind11-stubgen` becomes a dev/tooling dependency (documented in
  `python/README.md`, checked for only when the stubs target is invoked).

### 6. Docs
- `python/README.md`: note that `import tiledarray` now loads the package (which
  re-exports the compiled core); the build/`PYTHONPATH` instructions are otherwise
  unchanged.

## Files to touch

- **Edit:** `python/bindings/tiledarray.cc` (module-name macro + include path),
  `python/CMakeLists.txt` (glue path + output dir + staging + `_version.py`
  generation + `TA_PYTHON_INSTALL_DIR` install + `python-tiledarray-stubs`
  target), `python/tiledarray/__init__.py` (real package + version),
  `python/README.md` (note stub-regeneration target + `pybind11-stubgen` dev
  dependency).
- **New:** `python/tiledarray/py.typed` (hand-written marker);
  `python/tiledarray/_version.py.in` (configure_file template for the
  CMake-sourced version).
- **Generated (committed):** `python/tiledarray/__init__.pyi`,
  `python/tiledarray/_tiledarray.pyi` (produced by the `python-tiledarray-stubs`
  target, not hand-written).
- **Generated (not committed):** `${PROJECT_BINARY_DIR}/tiledarray/_version.py`
  (build artifact from `configure_file()`).
- **Move/rename:** `python/src` → `python/bindings`; then
  `python/bindings/TiledArray/python/*.h` (7 headers) → `python/bindings/*.h`.
- **Delete:** the hand-written `python/stubs/tiledarray.pyi` and the
  `python/stubs/` directory; the now-empty `python/bindings/TiledArray/` tree.

## Non-goals

- No NumPy-compatibility API, no new Python-level functions or array methods — the
  package `__init__.py` is a transparent re-export only. (Hosting such code later
  is *why* this architecture is worth doing, but it is out of scope here.)
- No change to the runtime/init behavior or the test markers.

## Verification

1. Reconfigure + build: `cmake --build build --target python-tiledarray`
   (confirms the relocated headers still resolve, recompiles `tiledarray.cc`,
   relinks as `_tiledarray`, and stages the package files into
   `build/python/tiledarray/`).
2. Import resolves to the package and re-exports the core (run under mpiexec with
   the OMPI root env vars):
   `PYTHONPATH=build/python mpiexec -n 1 python -c "import tiledarray as ta; print(ta.__file__); print(ta.__version__); print(ta.TArray, ta.get_default_world(), hasattr(ta,'einsum'))"`
   — `ta.__file__` should be `build/python/tiledarray/__init__.py`,
   `ta.__version__` should match the CMake project version (from the generated
   `_version.py`), and the core symbols should be present.
3. Existing Python suite still passes at np=1 and np=2 (confirms the bare-import
   consumers were not regressed by the rename + package):
   `cd build && ctest -R 'tiledarray/unit/python' -V`.
4. Stub generation works and is reproducible: `cmake --build build --target
   python-tiledarray-stubs` regenerates `python/tiledarray/{__init__,_tiledarray}.pyi`
   under `mpiexec -n 1` with the OMPI root env vars, and re-running it on an
   unchanged build produces no diff (stubs match the compiled surface).
