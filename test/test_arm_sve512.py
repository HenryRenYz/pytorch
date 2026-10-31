"""Focused installation checks; run outside the source root on the target CPU."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F


@unittest.skipUnless(os.uname().machine == "aarch64", "requires AArch64")
class TestArmSVE512(unittest.TestCase):
    def test_capability(self):
        self.assertEqual(torch.backends.cpu.get_cpu_capability(), "SVE512")
        self.assertTrue(torch.distributed.is_mpi_available())

    def test_openmp_runtime(self):
        torch.ones(1 << 18).add_(1)
        maps = Path("/proc/self/maps").read_text()
        self.assertIn("libomp.so", maps)
        self.assertNotIn("libgomp", maps)

    def test_active_vector_length_dispatch(self):
        bootstrap = """
import ctypes, os, sys
bits = int(sys.argv[1])
result = ctypes.CDLL(None).prctl(50, bits // 8 | (1 << 17), 0, 0, 0)
assert result >= 0, result
os.execv(sys.executable, [sys.executable, '-c', sys.argv[2]])
"""
        for bits, expected in ((128, "DEFAULT"), (256, "SVE256"), (512, "SVE512")):
            code = (
                "import torch; "
                f"assert torch.backends.cpu.get_cpu_capability() == {expected!r}; "
                "assert torch.arange(1, 32, dtype=torch.float32).sum().item() == 496"
            )
            env = dict(os.environ)
            env.pop("ATEN_CPU_CAPABILITY", None)
            subprocess.run([sys.executable, "-c", bootstrap, str(bits), code], env=env, check=True)

    def test_reductions_and_tails(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for n in (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 513):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.arange(1, n + 1, dtype=torch.float32).to(dtype)
                    ref = x.double()
                    torch.testing.assert_close(x.sum(), ref.sum().to(dtype))
                    torch.testing.assert_close(x.amin(), ref.amin().to(dtype))
                    torch.testing.assert_close(x.amax(), ref.amax().to(dtype))
                    torch.testing.assert_close(x + x, (ref + ref).to(dtype))

    def test_softmax_layer_norm(self):
        torch.manual_seed(24)
        for dtype in (torch.float32, torch.bfloat16):
            for n in (15, 16, 17, 31, 32, 33, 127, 128, 129, 1024):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.randn(7, n).to(dtype)
                    tol = 0.02 if dtype == torch.bfloat16 else 2e-5
                    torch.testing.assert_close(
                        F.softmax(x, -1), F.softmax(x.double(), -1).to(dtype),
                        atol=tol, rtol=tol,
                    )
                    torch.testing.assert_close(
                        F.layer_norm(x, (n,)), F.layer_norm(x.double(), (n,)).to(dtype),
                        atol=tol, rtol=tol,
                    )

    def test_noncontiguous_and_conversions(self):
        x = torch.arange(33 * 17, dtype=torch.float32).reshape(33, 17).t()
        torch.testing.assert_close((x * 0.25 + 1).sum(-1), (x.double() * 0.25 + 1).sum(-1).float())
        for dtype in (torch.bfloat16, torch.float16, torch.int32, torch.int64):
            y = x.to(dtype).float()
            expected = x.contiguous().to(dtype).float()
            torch.testing.assert_close(y, expected)

    def test_matmul_and_lapack(self):
        torch.manual_seed(25)
        a = torch.randn(65, 129)
        b = torch.randn(129, 33)
        torch.testing.assert_close(a @ b, (a.double() @ b.double()).float(), atol=3e-5, rtol=3e-5)
        x = torch.randn(17, 17, dtype=torch.float64)
        x = x @ x.t() + torch.eye(17, dtype=torch.float64)
        rhs = torch.randn(17, 3, dtype=torch.float64)
        result = torch.linalg.solve(x, rhs)
        torch.testing.assert_close(x @ result, rhs)

    def test_inductor(self):
        def fused(x, y):
            return ((x + y).relu() * 0.5).sum(-1)

        torch.manual_seed(26)
        for dtype in (torch.float32, torch.bfloat16):
            x, y = (torch.randn(9, 129).to(dtype) for _ in range(2))
            result = torch.compile(fused, fullgraph=True)(x, y)
            # Inductor can retain FP32 intermediates for BF16 fusion.
            tol = 0.03 if dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(result, fused(x, y), atol=tol, rtol=tol)


if __name__ == "__main__":
    unittest.main()
