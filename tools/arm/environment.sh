#!/usr/bin/env bash
# Source after `source ~/.bashrc; conda activate af3`.
export HPCKIT_ROOT="${HPCKIT_ROOT:-/work_ssd/software/HPCKit/25.2.1}"
export CC="$HPCKIT_ROOT/compiler/bisheng/bin/clang"
export CXX="$HPCKIT_ROOT/compiler/bisheng/bin/clang++"
export FC="$HPCKIT_ROOT/compiler/bisheng/bin/flang"
export KML_ROOT="$HPCKIT_ROOT/kml/bisheng"
export HMPI_ROOT="$HPCKIT_ROOT/hmpi/bisheng/release/hmpi"
# The HPCKit activation currently puts NEON first. Override that deliberately.
export LD_LIBRARY_PATH="$HPCKIT_ROOT/compiler/bisheng/lib:$KML_ROOT/lib/sme/kblas/multi:$KML_ROOT/lib/sve512:$KML_ROOT/lib/noarch:$HMPI_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$HPCKIT_ROOT/compiler/bisheng/lib:$KML_ROOT/lib/sme/kblas/multi:$KML_ROOT/lib/sve512:${LIBRARY_PATH:-}"
export PATH="$HPCKIT_ROOT/compiler/bisheng/bin:$HMPI_ROOT/bin:$PATH"
export OMP_DYNAMIC=FALSE
export OMP_MAX_ACTIVE_LEVELS=1
# Set OMP_NUM_THREADS and BLAS_NUM_THREADS per MPI rank/workload, not globally.
