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

    def test_sme_vector_length_initialization(self):
        bootstrap = """
import ctypes, os, sys
result = ctypes.CDLL(None).prctl(63, 32 | (1 << 17), 0, 0, 0)
assert result >= 0, result
os.execv(sys.executable, [sys.executable, '-c', sys.argv[1]])
"""
        code = """
import ctypes, torch
assert ctypes.CDLL(None).prctl(64, 0, 0, 0, 0) & 0xffff == 64
for threads in (1, 4, 2):
    torch.set_num_threads(threads)
    a = torch.ones(128, 128)
    torch.testing.assert_close(a @ a, torch.full_like(a, 128))
assert 'C++ Version:' in torch.__config__.show()
"""
        subprocess.run([sys.executable, "-c", bootstrap, code], check=True)

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
            for n in (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 513):
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
        lower = torch.linalg.cholesky(x)
        torch.testing.assert_close(lower @ lower.t(), x)

    def test_explicit_dispatch_registrations(self):
        values = torch.arange(1, 8, dtype=torch.float32)
        reduced = torch.segment_reduce(values, "sum", lengths=torch.tensor([2, 3, 2]))
        torch.testing.assert_close(reduced, torch.tensor([3.0, 12.0, 13.0]))
        indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
        sparse = torch.sparse_coo_tensor(indices, torch.tensor([2.0, 3.0, 4.0]), (3, 3))
        torch.testing.assert_close((sparse * sparse).to_dense(), sparse.to_dense().square())
        x = torch.arange(33, dtype=torch.float32)
        torch.testing.assert_close(torch.fft.irfft(torch.fft.rfft(x), n=33), x, atol=1e-5, rtol=1e-5)
        torch.manual_seed(29)
        for dtype in (torch.float32, torch.bfloat16):
            q, k, v = (torch.randn(2, 4, 17, 32).to(dtype) for _ in range(3))
            expected = F.scaled_dot_product_attention(q.double(), k.double(), v.double()).to(dtype)
            tol = 0.02 if dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(
                F.scaled_dot_product_attention(q, k, v), expected, atol=tol, rtol=tol,
            )

    def test_batched_and_low_precision_matmul(self):
        torch.manual_seed(27)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for m, k, n in ((16, 32, 17), (65, 129, 33), (128, 128, 128)):
                with self.subTest(dtype=dtype, shape=(m, k, n)):
                    a = torch.randn(3, m, k).to(dtype)
                    b = torch.randn(3, n, k).to(dtype).transpose(-1, -2)
                    expected = (a.double() @ b.double()).to(dtype)
                    tol = 0.04 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 5e-5
                    torch.testing.assert_close(torch.bmm(a, b), expected, atol=tol, rtol=tol)
                    torch.testing.assert_close(a[0] @ b[0], expected[0], atol=tol, rtol=tol)

    def test_bf16_addmm(self):
        torch.manual_seed(30)
        a = torch.randn(65, 129).bfloat16()
        b = torch.randn(33, 129).bfloat16().t()
        c = torch.randn(65, 33).bfloat16()
        expected = (0.75 * (a.double() @ b.double()) + 0.5 * c.double()).bfloat16()
        torch.testing.assert_close(torch.addmm(c, a, b, alpha=0.75, beta=0.5), expected, atol=0.04, rtol=0.04)
        c.fill_(float("nan"))
        torch.testing.assert_close(torch.addmm(c, a, b, beta=0), (a.double() @ b.double()).bfloat16(), atol=0.04, rtol=0.04)

    def test_inductor(self):
        from torch._inductor.cpu_vec_isa import pick_vec_isa

        self.assertEqual(pick_vec_isa().bit_width(), 512)

        def fused(x, y):
            return ((x + y).relu() * 0.5).sum(-1)

        torch.manual_seed(26)
        for dtype in (torch.float32, torch.bfloat16):
            x, y = (torch.randn(9, 129).to(dtype) for _ in range(2))
            result = torch.compile(fused, fullgraph=True)(x, y)
            # Inductor can retain FP32 intermediates for BF16 fusion.
            tol = 0.03 if dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(result, fused(x, y), atol=tol, rtol=tol)

    def test_inductor_cpp_gemm(self):
        from torch._inductor import config
        from torch._inductor.utils import run_and_get_code

        class FrozenMM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(129, 33))

            def forward(self, a):
                return (a @ self.weight).relu()

        torch.manual_seed(28)
        model, a = FrozenMM().eval(), torch.randn(65, 129)
        # Upstream CPP GEMM templates require a constant/prepacked RHS.
        with torch.no_grad(), config.patch(
            freezing=True, max_autotune=True, max_autotune_gemm_backends="CPP",
        ):
            result, code = run_and_get_code(torch.compile(model, fullgraph=True), a)
        torch.testing.assert_close(result, model(a), atol=5e-5, rtol=5e-5)
        self.assertIn("micro_gemm", "\n".join(code))


if __name__ == "__main__":
    unittest.main()
