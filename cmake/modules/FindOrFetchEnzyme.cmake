# Locate or fetch Enzyme (https://github.com/EnzymeAD/Enzyme), the LLVM-level
# automatic-differentiation plugin. Only included when TA_ENABLE_ENZYME=ON.
#
# Enzyme is NOT an ordinary link library: the artifact that matters is the Clang
# pass plugin "ClangEnzyme-<LLVM_VERSION_MAJOR>.so", applied to *specific*
# translation units at compile time via -fpass-plugin=. This module's job is to
#   (1) verify the toolchain can host Enzyme (a Clang whose LLVM matches),
#   (2) make the plugin target available (find_package, else FetchContent),
#   (3) expose the plugin path + a helper that attaches it per-target,
#   (4) set TILEDARRAY_HAS_ENZYME.
# See enzyme_dependency_plan.md and tiledarray_autodiff_plan.md (B5).

# --- (1) toolchain preconditions ------------------------------------------------
# Enzyme is built against an installed LLVM; the Clang compiling TA's
# differentiated TUs must be that same LLVM version. A mismatch yields a plugin
# that crashes Clang or miscompiles, so fail loudly at configure time.
find_package(LLVM REQUIRED CONFIG)

if (LLVM_VERSION_MAJOR LESS 15)
  message(FATAL_ERROR "TA_ENABLE_ENZYME=ON requires LLVM >= 15 (found ${LLVM_VERSION}); "
                      "Enzyme does not support older LLVM.")
endif()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "TA_ENABLE_ENZYME=ON requires a Clang C++ compiler so the Enzyme "
                      "pass plugin can be applied to differentiated translation units "
                      "(found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}). "
                      "Set CMAKE_CXX_COMPILER to a clang++ matching LLVM ${LLVM_VERSION_MAJOR}.")
endif()

# Compare the Clang major version to the LLVM the plugin builds against.
string(REGEX MATCH "^[0-9]+" _ta_clang_major "${CMAKE_CXX_COMPILER_VERSION}")
if (NOT _ta_clang_major STREQUAL LLVM_VERSION_MAJOR)
  message(FATAL_ERROR "TA_ENABLE_ENZYME=ON requires the Clang version to match the LLVM "
                      "Enzyme is built against: Clang is ${CMAKE_CXX_COMPILER_VERSION} "
                      "(major ${_ta_clang_major}) but LLVM is ${LLVM_VERSION} "
                      "(major ${LLVM_VERSION_MAJOR}). Point LLVM_DIR and CMAKE_CXX_COMPILER "
                      "at the same LLVM/Clang toolchain.")
endif()

set(_ta_enzyme_plugin_target "ClangEnzyme-${LLVM_VERSION_MAJOR}")

# Enzyme's ClangEnzyme plugin is only built when it locates Clang's CMake config
# (Clang_FOUND). Enzyme tries to derive Clang_DIR from LLVM_DIR, but its probe
# looks for a *static* libclangBasic.a that distro LLVM packages (e.g. Ubuntu)
# do not ship, so it silently skips the Clang plugin. Find Clang here and pass
# Clang_DIR through; Enzyme honors a pre-set Clang_DIR.
if (NOT TARGET ${_ta_enzyme_plugin_target} AND NOT DEFINED Clang_DIR)
  find_package(Clang CONFIG QUIET HINTS "${LLVM_DIR}/../clang")
  if (NOT Clang_FOUND)
    message(FATAL_ERROR "TA_ENABLE_ENZYME=ON could not find Clang's CMake config (needed to "
                        "build the ${_ta_enzyme_plugin_target} plugin). Set Clang_DIR to the "
                        "directory containing ClangConfig.cmake for LLVM ${LLVM_VERSION_MAJOR}.")
  endif()
endif()

# --- (2) find, else fetch -------------------------------------------------------
# Try an installed Enzyme first (best-effort: upstream's exported config is
# immature, EnzymeAD/Enzyme#981). Skip the FetchContent fallback in expert mode.
if (NOT TARGET ${_ta_enzyme_plugin_target})
  find_package(Enzyme CONFIG QUIET)
endif()

if (NOT TARGET ${_ta_enzyme_plugin_target})
  if (TA_EXPERT)
    message(FATAL_ERROR "TA_EXPERT=ON and Enzyme not found; install Enzyme (providing target "
                        "${_ta_enzyme_plugin_target}) or unset TA_EXPERT to allow fetching it.")
  endif()

  include(FetchContent)
  FetchContent_Declare(
      enzyme
      GIT_REPOSITORY ${TA_TRACKED_ENZYME_URL}
      GIT_TAG        ${TA_TRACKED_ENZYME_TAG}
      # Enzyme's CMake root is the enzyme/ subdirectory of the repo.
      SOURCE_SUBDIR  enzyme
  )
  FetchContent_MakeAvailable(enzyme)
endif()

# --- (3) postcondition + handle -------------------------------------------------
if (NOT TARGET ${_ta_enzyme_plugin_target})
  message(FATAL_ERROR "FindOrFetchEnzyme could not make the ${_ta_enzyme_plugin_target} "
                      "plugin target available.")
endif()

# Path to the plugin .so, resolved at generate time from the target, plus the
# target name itself so dependents can order their build against it.
set(TA_ENZYME_PLUGIN "$<TARGET_FILE:${_ta_enzyme_plugin_target}>"
    CACHE INTERNAL "Path to the ClangEnzyme pass plugin for TiledArray AD")
set(TA_ENZYME_PLUGIN_TARGET "${_ta_enzyme_plugin_target}"
    CACHE INTERNAL "Name of the ClangEnzyme pass-plugin target for TiledArray AD")
set(TILEDARRAY_HAS_ENZYME 1)
message(STATUS "Enzyme: using plugin target ${_ta_enzyme_plugin_target} (LLVM ${LLVM_VERSION})")

# --- (4) configure-time plugin smoke compile (enzyme_integration_suggestions.md P3.1)
# A mismatched clang / LLVM / Enzyme-plugin ABI otherwise fails *deep* in the
# build with a cryptic symptom (the plugin fails to load, the transformation
# silently does not run, or `__enzyme_autodiff` stays an undefined symbol — the
# FAQ's top three failure modes). Compile a tiny `__enzyme_autodiff` of x*x under
# the *actual* plugin now, so a broken toolchain fails fast with a clear message.
# Compile-and-link is the whole test: if the Enzyme pass does not run, the
# `__enzyme_autodiff` call is never lowered and the link fails on the undefined
# symbol; if it runs, the call is replaced and the probe links.
#
# This needs the plugin .so to exist on disk at configure time. That holds for an
# installed Enzyme (an IMPORTED target) and for a *warm* reconfigure of a
# FetchContent build, but NOT for a *cold* FetchContent build — there the .so is
# produced during the build step, after configure. In that one case the smoke is
# deferred with a note and reattempted on the next configure; either way the
# ad_enzyme_* test TUs exercise the plugin at build time.
set(_ta_enzyme_plugin_file "")
get_target_property(_ta_enzyme_imported ${_ta_enzyme_plugin_target} IMPORTED)
if (_ta_enzyme_imported)
  foreach (_cfg "" _RELEASE _RELWITHDEBINFO _DEBUG _MINSIZEREL)
    get_target_property(_loc ${_ta_enzyme_plugin_target} IMPORTED_LOCATION${_cfg})
    if (_loc AND EXISTS "${_loc}")
      set(_ta_enzyme_plugin_file "${_loc}")
      break()
    endif()
  endforeach()
elseif (DEFINED enzyme_BINARY_DIR)
  # FetchContent layout: <bindir>/Enzyme/ClangEnzyme-<major>.so (present once built).
  set(_loc "${enzyme_BINARY_DIR}/Enzyme/${_ta_enzyme_plugin_target}.so")
  if (EXISTS "${_loc}")
    set(_ta_enzyme_plugin_file "${_loc}")
  endif()
endif()

if (_ta_enzyme_plugin_file)
  include(CheckCXXSourceCompiles)
  set(_ta_enzyme_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
  string(APPEND CMAKE_REQUIRED_FLAGS " -fpass-plugin=${_ta_enzyme_plugin_file}")
  check_cxx_source_compiles("
    extern \"C\" double __enzyme_autodiff(void *, ...);
    static double ta_enzyme_smoke_sq(double x) { return x * x; }
    int main() {
      return __enzyme_autodiff((void *)ta_enzyme_smoke_sq, 3.0) > 5.0 ? 0 : 1;
    }
  " TA_ENZYME_PLUGIN_SMOKE_OK)
  set(CMAKE_REQUIRED_FLAGS "${_ta_enzyme_saved_required_flags}")
  if (NOT TA_ENZYME_PLUGIN_SMOKE_OK)
    message(FATAL_ERROR
        "TA_ENABLE_ENZYME=ON: the Enzyme plugin smoke compile failed — the plugin "
        "(${_ta_enzyme_plugin_file}) did not transform a trivial __enzyme_autodiff "
        "of x*x with ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION}). This is "
        "almost always a clang / LLVM / Enzyme-plugin version mismatch: confirm clang, "
        "LLVM ${LLVM_VERSION}, and the Enzyme tag ${TA_TRACKED_ENZYME_TAG} agree. See "
        "the configure log (CMakeError.log / CMakeConfigureLog.yaml) for the compiler "
        "output.")
  endif()
  message(STATUS "Enzyme: plugin smoke compile (x*x) passed")
else()
  message(STATUS "Enzyme: plugin is built from source and not yet on disk; deferring "
                 "the configure-time smoke compile (reattempted on the next configure). "
                 "The ad_enzyme_* test TUs exercise the plugin at build time.")
endif()

# Attach the Enzyme pass plugin to one target's compilation. Deliberately
# per-target, never global: only differentiated TUs opt in (autodiff plan B5 —
# MPI/MADNESS/BLAS translation units compile unchanged).
#
# Opt-level stance (enzyme_integration_suggestions.md P0.2): this attaches only
# -fpass-plugin and inherits the target's optimization level. The plugin runs
# *after* the regular optimizer, so a differentiated TU must be built at the same
# -O the rules were validated at; the primal rule shims are marked `noinline`
# (enzyme_rules.h, TA_AD_PRIMAL) so the inliner cannot erase, at -O2+, the call
# site Enzyme matches against a registered rule. The integration is validated at
# Release (-O2). If a differentiated TU hits an Enzyme type-analysis or
# pre-optimization crash, the documented first workaround is to add
# `-mllvm -enzyme-preopt=0` to its compile options; keep all differentiated TUs
# on one optimization level.
function(ta_enable_enzyme_on _target)
  target_compile_options(${_target} PRIVATE "-fpass-plugin=${TA_ENZYME_PLUGIN}")
endfunction()
