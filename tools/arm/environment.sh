#!/usr/bin/env bash
# Source after the HPCKit setvars script and Python environment activation.
if [[ -z "${HPCKIT_ROOT:-}" ]]; then
  if [[ -x /work_ssd/software/HPCKit/25.2.1/compiler/bisheng/bin/clang ]]; then
    export HPCKIT_ROOT=/work_ssd/software/HPCKit/25.2.1
  else
    export HPCKIT_ROOT="$HOME/HPCKit/latest"
  fi
fi
export HPCKIT_ROOT="$(readlink -f "$HPCKIT_ROOT")"
export CC="$HPCKIT_ROOT/compiler/bisheng/bin/clang"
export CXX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bisheng-cxx"
export FC="$HPCKIT_ROOT/compiler/bisheng/bin/flang"
export KML_ROOT="$HPCKIT_ROOT/kml/bisheng"
export HMPI_ROOT="$HPCKIT_ROOT/hmpi/bisheng/release/hmpi"
if [[ -z "${KML_BLAS_DIR:-}" ]]; then
  if [[ -f "$KML_ROOT/lib/matrix/kblas/multi/libkblas.so" ]]; then
    export KML_BLAS_DIR="$KML_ROOT/lib/matrix/kblas/multi"
  else
    export KML_BLAS_DIR="$KML_ROOT/lib/sme/kblas/multi"
  fi
fi
for arm_required_file in "$CC" "$CXX" "$HMPI_ROOT/bin/mpicc" \
  "$KML_BLAS_DIR/libkblas.so" "$KML_ROOT/lib/sve512/libklapack_full.so" \
  "$HPCKIT_ROOT/compiler/bisheng/lib/libomp.so"; do
  if [[ ! -f "$arm_required_file" ]]; then
    echo "Missing HPCKit component: $arm_required_file" >&2
    return 1
  fi
done
# Override activation scripts which put NEON first. Include MPI dependencies
# explicitly so the library search path is reproducible in worker processes.
export LD_LIBRARY_PATH="$HPCKIT_ROOT/compiler/bisheng/lib:$KML_BLAS_DIR:$KML_ROOT/lib/sve512:$KML_ROOT/lib/noarch:$HMPI_ROOT/lib:$HPCKIT_ROOT/hmpi/bisheng/release/hucx/lib:$HPCKIT_ROOT/hmpi/bisheng/release/xucg/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$HPCKIT_ROOT/compiler/bisheng/lib:$KML_BLAS_DIR:$KML_ROOT/lib/sve512:${LIBRARY_PATH:-}"
export PATH="$HPCKIT_ROOT/compiler/bisheng/bin:$HMPI_ROOT/bin:$PATH"
export OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
unset arm_required_file
# Set OMP_NUM_THREADS and BLAS_NUM_THREADS per MPI rank/workload, not globally.
