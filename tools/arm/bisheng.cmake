# Applied before the top-level project by CMAKE_PROJECT_TOP_LEVEL_INCLUDES.
if(NOT DEFINED ENV{HPCKIT_ROOT})
  message(FATAL_ERROR "Source tools/arm/environment.sh first")
endif()
set(_hpckit "$ENV{HPCKIT_ROOT}")
set(_compiler "${_hpckit}/compiler/bisheng")
set(CMAKE_DISABLE_FIND_PACKAGE_OpenSSL TRUE CACHE BOOL "Gloo TLS is disabled" FORCE)
set(OpenMP_C_FLAGS "-fopenmp=libomp" CACHE STRING "" FORCE)
set(OpenMP_CXX_FLAGS "-fopenmp=libomp" CACHE STRING "" FORCE)
set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
set(OpenMP_omp_LIBRARY "${_compiler}/lib/libomp.so" CACHE FILEPATH "" FORCE)
set(OpenMP_libomp_LIBRARY "${_compiler}/lib/libomp.so" CACHE FILEPATH "" FORCE)
set(MPI_C_COMPILER "$ENV{HMPI_ROOT}/bin/mpicc" CACHE FILEPATH "" FORCE)
set(MPI_CXX_COMPILER "$ENV{HMPI_ROOT}/bin/mpicxx" CACHE FILEPATH "" FORCE)
set(MPI_CXX_SKIP_MPICXX ON CACHE BOOL "" FORCE)
set(BLAS_LIBRARIES "$ENV{KML_BLAS_DIR}/libkblas.so" CACHE FILEPATH "" FORCE)
set(CMAKE_JOB_POOLS "compile=$ENV{MAX_JOBS};link=1" CACHE STRING "" FORCE)
set(CMAKE_JOB_POOL_COMPILE compile CACHE STRING "" FORCE)
set(CMAKE_JOB_POOL_LINK link CACHE STRING "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

# BiSheng 5.2 / LLVM 19 spends tens of minutes in OpenMPOpt's Attributor on
# these generated registration wrappers, which contain no OpenMP regions.
# Apply the workaround in the target's directory after it has been added;
# leave numerical kernels and their OpenMP optimizations unchanged.
function(_bisheng_configure_generated_wrappers)
  set(_wrappers)
  foreach(_shard RANGE 0 4)
    list(APPEND _wrappers
      "${CMAKE_SOURCE_DIR}/torch/csrc/autograd/generated/TraceType_${_shard}.cpp")
  endforeach()
  foreach(_shard RANGE 0 1)
    list(APPEND _wrappers
      "${CMAKE_SOURCE_DIR}/torch/csrc/autograd/generated/ADInplaceOrViewType_${_shard}.cpp")
  endforeach()
  set_property(SOURCE ${_wrappers}
    DIRECTORY "${CMAKE_SOURCE_DIR}/caffe2"
    APPEND PROPERTY COMPILE_OPTIONS
      -mllvm -openmp-opt-disable -mllvm -attributor-enable=none)
  # HMPI/KML can load libc++abi before PyTorch's libstdc++, interposing C++
  # ABI symbols and breaking e.g. locale facets. Establish the selected
  # PyTorch C++ runtime in the global dependency scope before MPI plugins.
  if("stdc++" IN_LIST CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
    get_target_property(_global_dependencies torch_global_deps LINK_LIBRARIES)
    set_property(TARGET torch_global_deps PROPERTY LINK_LIBRARIES
      "-Wl,--no-as-needed;stdc++;${_global_dependencies}")
  endif()
  target_sources(torch_global_deps PRIVATE
    "${CMAKE_SOURCE_DIR}/tools/arm/runtime_init.c")
  target_compile_options(torch_global_deps PRIVATE -fopenmp=libomp)
  target_link_libraries(torch_global_deps
    "$ENV{HPCKIT_ROOT}/compiler/bisheng/lib/libomp.so")
endfunction()
cmake_language(DEFER CALL _bisheng_configure_generated_wrappers)
