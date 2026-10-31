"""Run from outside the source tree; emit JSON lines for comparable CPU runs."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--min-run-time", type=float, default=0.3)
    parser.add_argument("--allow-accuracy-mismatch", action="store_true", help="diagnostic baseline runs only; report mismatches without aborting")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        for threads in args.threads:
            env = dict(os.environ)
            for variable in ("OMP_NUM_THREADS", "BLAS_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[variable] = str(threads)
            command = [
                sys.executable, __file__, "--worker", "--threads", str(threads),
                "--min-run-time", str(args.min_run_time),
            ]
            if args.allow_accuracy_mismatch:
                command.append("--allow-accuracy-mismatch")
            subprocess.run(command, env=env, check=True)
        return

    import torch
    import torch.nn.functional as F
    from torch.utils.benchmark import Timer

    torch.manual_seed(2026)
    torch.set_num_interop_threads(1)
    vector_lengths = {}
    if platform.machine() == "aarch64":
        libc = ctypes.CDLL(None)
        for name, request in (("sve_bytes", 51), ("sme_bytes", 64)):
            value = libc.prctl(request, 0, 0, 0, 0)
            vector_lengths[name] = value & 0xffff if value >= 0 else None
    libraries = sorted({
        line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in ("libkblas", "libklapack", "libomp", "libgomp", "libmpi.so", "libopenblas"))
    })
    print(json.dumps({
        "type": "environment", "host": platform.node(),
        "torch": torch.__version__,
        "capability": torch.backends.cpu.get_cpu_capability(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "vector_lengths": vector_lengths,
        "loaded_libraries": libraries,
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "BLAS_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        )},
        "config": torch.__config__.show(),
    }), flush=True)
    for threads in args.threads:
        torch.set_num_threads(threads)
        for dtype in (torch.float32, torch.bfloat16):
            a = torch.randn(1024, 1024, dtype=dtype)
            b = torch.randn(1024, 1024, dtype=dtype)
            batch_a = torch.randn(8, 128, 256, dtype=dtype)
            batch_b = torch.randn(8, 256, 128, dtype=dtype)
            x = torch.randn(256, 1024, dtype=dtype)
            y = torch.randn_like(x)
            # Validate the exact GEMM shapes before reporting their speed.
            # Vendor BLAS kernels can select different paths for larger tiles.
            tol = 0.04 if dtype == torch.bfloat16 else 1e-4
            correctness = {}
            for name, actual, reference in (
                ("mm_1024", a @ b, a.double() @ b.double()),
                ("bmm_8x128x256x128", torch.bmm(batch_a, batch_b), torch.bmm(batch_a.double(), batch_b.double())),
            ):
                error = actual.double() - reference
                failure = None
                try:
                    torch.testing.assert_close(actual, reference.to(dtype), atol=tol, rtol=tol)
                except AssertionError as exc:
                    failure = exc
                correctness[name] = failure is None
                print(json.dumps({
                    "type": "correctness", "name": name, "dtype": str(dtype), "threads": threads,
                    "passed": failure is None, "allow_mismatch": args.allow_accuracy_mismatch,
                    "max_absolute_error_fp64": error.abs().max().item(),
                    "relative_l2_error_fp64": (error.norm() / reference.norm()).item(),
                }), flush=True)
                if failure is not None and not args.allow_accuracy_mismatch:
                    raise failure
            cases = [
                ("mm_1024", "torch.mm(a, b)", 2 * 1024**3),
                ("bmm_8x128x256x128", "torch.bmm(batch_a, batch_b)", 2 * 8 * 128 * 256 * 128),
                ("softmax_256x1024", "F.softmax(x, dim=-1)", None),
                ("layer_norm_256x1024", "F.layer_norm(x, (1024,))", None),
                ("pointwise_256x1024", "(x + y).relu() * 0.5", None),
                ("sum_256x1024", "x.sum(dim=-1)", None),
            ]
            scope = dict(torch=torch, F=F, a=a, b=b, batch_a=batch_a, batch_b=batch_b, x=x, y=y)
            for name, statement, flops in cases:
                result = Timer(statement, globals=scope, num_threads=threads).blocked_autorange(
                    min_run_time=args.min_run_time
                )
                print(json.dumps({
                    "type": "measurement", "name": name, "dtype": str(dtype),
                    "threads": threads, "median_seconds": result.median,
                    "iqr_seconds": result.iqr,
                    "gflops": flops / result.median / 1e9 if flops else None,
                    "gemm_correctness_checked": flops is not None,
                    "gemm_correctness_passed": correctness.get(name),
                }), flush=True)


if __name__ == "__main__":
    main()
