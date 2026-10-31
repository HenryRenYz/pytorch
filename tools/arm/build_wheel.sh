#!/usr/bin/env bash
set -euo pipefail
arm_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$arm_source_root/tools/arm/environment.sh"
# Host tools such as protoc are built and executed during this build too.
# The Kylin system runtime is GCC 10, older than the selected C++20 headers.
export LD_LIBRARY_PATH="$HPCKIT_ROOT/compiler/gcc/lib64:$LD_LIBRARY_PATH"
source "$HOME/venvs/pytorch-arm-build/bin/activate"
cd "$arm_source_root"
mkdir -p "$HOME/pytorch-arm-artifacts-2.14"
hostname
"$CC" --version
free -h
ps -eo user,pid,pcpu,comm --sort=-pcpu | head -12 || true
taskset -pc $$
# Budget 4 GiB per compiler plus 32 GiB for linking and the build driver.
arm_memory_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
arm_allocation_kb="$(( ${BUILD_MEMORY_GB:-256} * 1048576 ))"
if (( arm_memory_kb > arm_allocation_kb )); then arm_memory_kb="$arm_allocation_kb"; fi
arm_budget_jobs="$(( (arm_memory_kb / 1048576 - 32) / 4 ))"
arm_affinity_jobs="$(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
export MAX_JOBS="${MAX_JOBS:-64}"
if (( MAX_JOBS > arm_budget_jobs )); then MAX_JOBS="$arm_budget_jobs"; fi
if (( MAX_JOBS > arm_affinity_jobs )); then MAX_JOBS="$arm_affinity_jobs"; fi
if (( MAX_JOBS < 1 )); then echo 'Insufficient build memory' >&2; exit 1; fi
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export CMAKE_C_COMPILER="$CC" CMAKE_CXX_COMPILER="$CXX"
export CMAKE_PROJECT_TOP_LEVEL_INCLUDES="$arm_source_root/tools/arm/bisheng.cmake"
# Gloo TLS is disabled. Avoid parsing the vendor OpenSSL 1.1.1wa patch number,
# which makes CMake 3.31 emit a non-UTF-8 byte into CMakeCache.txt on Kylin.
export CMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON
export CMAKE_C_FLAGS='-mcpu=native'
export CMAKE_CXX_FLAGS='-mcpu=native'
# CMake adds the C compiler's system GCC directory when linking C dependencies.
# Search the C++20 toolchain first, before that explicit system GCC 10 -L path.
export CMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -L$HPCKIT_ROOT/compiler/gcc/lib64"
export CMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -L$HPCKIT_ROOT/compiler/gcc/lib64"
export CMAKE_MODULE_LINKER_FLAGS="$CMAKE_SHARED_LINKER_FLAGS"
export CMAKE_BUILD_TYPE=Release
export USE_CUDA=0 USE_CUDNN=0 USE_NCCL=0 USE_ROCM=0 USE_XPU=0
export USE_DISTRIBUTED=1 USE_MPI=1 USE_GLOO=1 USE_TENSORPIPE=0
export USE_OPENMP=1 USE_MKLDNN=1 USE_MKLDNN_ACL=0 USE_MKL=0
export USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_XNNPACK=0
export USE_KLEIDIAI=0 USE_KINETO=0 BUILD_TEST=0
export BLAS=Generic GENERIC_BLAS_LIBRARIES=kblas GENERIC_LAPACK_LIBRARIES=klapack_full
export PYTORCH_BUILD_VERSION=2.14.0+cpu.sve512.bisheng PYTORCH_BUILD_NUMBER=1
export OMP_NUM_THREADS=1 BLAS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_NO_BUILD_ISOLATION=1
printf 'MAX_JOBS=%s\n' "$MAX_JOBS"
python -m pip wheel --config-settings="cmake.define.CMAKE_PROJECT_TOP_LEVEL_INCLUDES=$CMAKE_PROJECT_TOP_LEVEL_INCLUDES" --verbose --no-build-isolation --no-deps --wheel-dir "$arm_source_root/dist" . 2>&1 | tee "$HOME/pytorch-arm-artifacts-2.14/build-wheel.log"
