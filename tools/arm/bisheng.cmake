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
  # oneDNN 3.7.1 selects gemm:jit:bf16 on this CPU, but a 512^3 GEMM takes
  # 1.14 s versus 2.8 ms through KML sbgemm. Prefer the standard BLAS path
  # for BF16 matmul only; retain oneDNN for convolution and other operators.
  set_property(SOURCE "${CMAKE_SOURCE_DIR}/aten/src/ATen/native/mkldnn/Matmul.cpp"
    DIRECTORY "${CMAKE_SOURCE_DIR}/caffe2"
    APPEND PROPERTY COMPILE_DEFINITIONS ATEN_ARM_PREFER_BLAS_BF16)
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
  # Load this independently before global MPI/KML dependencies. Loading the
  # C++ runtime in the same dlopen group as libc++abi is not sufficient to
  # avoid locale/RTTI symbol interposition on this toolkit.
  add_library(torch_arm_runtime SHARED
    "${CMAKE_SOURCE_DIR}/tools/arm/runtime_init.c")
  set_target_properties(torch_arm_runtime PROPERTIES
    LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  target_compile_options(torch_arm_runtime PRIVATE -fopenmp=libomp)
  target_link_options(torch_arm_runtime PRIVATE -Wl,--no-as-needed)
  target_link_libraries(torch_arm_runtime PRIVATE
    "$ENV{HPCKIT_ROOT}/compiler/bisheng/lib/libomp.so")
  install(TARGETS torch_arm_runtime LIBRARY DESTINATION lib)
endfunction()
cmake_language(DEFER CALL _bisheng_configure_generated_wrappers)
