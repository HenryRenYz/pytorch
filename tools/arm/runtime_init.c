/* HPCKit matrix BLAS requires a 512-bit SME streaming vector length. */
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#define PR_SME_GET_VL 64
#endif
#define ARM_SME_VL_MASK 0xffff
#define ARM_SME_VL_INHERIT (1 << 17)

static int set_sme512(void) {
  int result = prctl(PR_SME_SET_VL, 64 | ARM_SME_VL_INHERIT, 0, 0, 0);
  return result < 0 || (result & ARM_SME_VL_MASK) != 64;
}

__attribute__((constructor)) static void initialize_kml_sme512(void) {
  int previous = prctl(PR_SME_GET_VL, 0, 0, 0, 0);
  int failed = set_sme512();
  if (previous >= 0 && (previous & ARM_SME_VL_MASK) != 64 && !failed) {
    /* Also initialize workers in a pre-existing libomp pool. New workers
       inherit the caller's length. OMP_NUM_THREADS sets the team budget. */
#pragma omp parallel reduction(+ : failed)
    { failed += set_sme512(); }
  }
  if (failed) {
    fputs("PyTorch HPCKit build requires 512-bit SME for KML matrix BLAS.\n", stderr);
    abort();
  }
}
