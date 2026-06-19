cmake_minimum_required(VERSION 3.12)

project(python-tiledarray)

if (NOT TARGET Python::Module)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
endif()

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY      https://github.com/pybind/pybind11.git
  GIT_TAG             v3.0.3
  )
FetchContent_MakeAvailable(pybind11)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall)

pybind11_add_module(python-tiledarray MODULE src/tiledarray.cc)

# mpi c libs need come from tiledarray
# Eigen and Boost also transitively come from tiledarray
target_link_libraries(python-tiledarray PRIVATE tiledarray)

# pybind uses sized deallocations when using C++17, clang requires -fsized-deallocation compiler flag to enable them
# see https://reviews.llvm.org/D8467 and https://github.com/pybind/pybind11/issues/1818
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(python-tiledarray PRIVATE -fsized-deallocation)
endif()

set_target_properties(
  python-tiledarray
  PROPERTIES
  #PREFIX ""
  OUTPUT_NAME tiledarray
  )

# how to install
install(TARGETS python-tiledarray EXPORT tiledarray
        COMPONENT python
        LIBRARY DESTINATION "${TILEDARRAY_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${TILEDARRAY_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${TILEDARRAY_INSTALL_INCLUDEDIR}")

# tests
if (BUILD_TESTING)
  # build step
  add_test(tiledarray/unit/python/build "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target python-tiledarray)
  set_tests_properties(tiledarray/unit/python/build PROPERTIES FIXTURES_SETUP TA_UNIT_TESTS_PYTHON_EXEC)

  # check for presence of prerequisite modules
  foreach(_mod pytest numpy)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import ${_mod}"
        OUTPUT_QUIET ERROR_QUIET
        RESULTS_VARIABLE check_for_${_mod})
    if (check_for_${_mod})
      message(FATAL_ERROR "Python module \"${_mod}\" is not installed; install via \"${Python_EXECUTABLE} -m pip install ${_mod}\" and rerun cmake")
    endif(check_for_${_mod})
  endforeach(_mod)

  set(TA_PYTHON_TEST_MODULES
    test_world
    test_array_construction
    test_array_indexing
    test_trange
    test_expression
    test_contraction
    test_reductions
    test_numpy_interop
    test_sparse
    test_lifetime
    test_dtypes
    test_mpi
  )

  # Register every module at both world sizes, mirroring the C++ suite
  # (tests/CMakeLists.txt): np=1 deselects @distributed cases, np>=2 deselects
  # @serial cases.  Untagged tests therefore run at every world size.
  set(_pytest_np_1_args -m "not distributed")
  set(_pytest_np_2_args -m "not serial")
  foreach(_pymod ${TA_PYTHON_TEST_MODULES})
    foreach(p RANGE 1 2)
      add_test(
        NAME tiledarray/unit/python/${_pymod}-np${p}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${p}
                ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests/python/${_pymod}.py -v
                ${_pytest_np_${p}_args}
                --import-mode=importlib
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      )
      set(_pymod_env "MAD_NUM_THREADS=2;PYTHONPATH=${PROJECT_BINARY_DIR}")
      if (p GREATER 1)
        set(_pymod_env "${_pymod_env};TA_UT_DISTRIBUTED=1")
      endif()
      set_tests_properties(tiledarray/unit/python/${_pymod}-np${p}
              PROPERTIES FIXTURES_REQUIRED TA_UNIT_TESTS_PYTHON_EXEC
              ENVIRONMENT "${_pymod_env}")
      if (p GREATER 1)
        set_tests_properties(tiledarray/unit/python/${_pymod}-np${p}
                PROPERTIES PROCESSORS ${p})
      endif()
    endforeach()
  endforeach()

endif(BUILD_TESTING)
