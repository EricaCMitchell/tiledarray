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

# Attach the Enzyme pass plugin to one target's compilation. Deliberately
# per-target, never global: only differentiated TUs opt in (autodiff plan B5 —
# MPI/MADNESS/BLAS translation units compile unchanged).
function(ta_enable_enzyme_on _target)
  target_compile_options(${_target} PRIVATE "-fpass-plugin=${TA_ENZYME_PLUGIN}")
endfunction()
