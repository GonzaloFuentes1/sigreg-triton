"""Fused Triton kernel vs the float64 reference (skipped without CUDA + Triton).

Tolerance rationale. The kernel reads x in its storage dtype, works in float32, evaluates one
cos/sin pair of theta = dt * x and obtains cos/sin(k * theta) for k < K by angle addition;
each rotation adds O(1) float32 ulp, so after K - 1 = 16 steps the ECF error is bounded by
~K * eps32 ~ 2e-6 on top of the float32 summation error over N. The official float32 path
has an error of the same order (rounding of t_k * x and of the mean), so tolerances are
stated as "no worse than a small multiple of the official float32 error, with a floor".
"""

import pytest
import torch

from sigreg_triton import SIGReg, epps_pulley, sigreg
from sigreg_triton.ecf import ecf_triton, triton_available
from sigreg_triton.reference import ecf_reference, epps_pulley_grid, epps_pulley_reference

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and triton_available()), reason="needs CUDA + Triton"
    ),
]

EPS32 = torch.finfo(torch.float32).eps


def _ecf64(x, t_max, n_points):
    t, _, _ = epps_pulley_grid(t_max, n_points, torch.float64, x.device)
    return ecf_reference(x.double(), t)


def _max_err(a, b):
    return (a.double() - b.double()).abs().max().item()


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (7, 3), (1000, 77), (4097, 130), (3, 300, 50), (2, 2, 65, 33), (20000, 5)],
)
def test_ecf_forward_matches_float64(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda") * 1.5
    dt = 3.0 / 16
    cos_t, sin_t = ecf_triton(x, dt, 17)
    cos_r, sin_r = _ecf64(x, 3.0, 17)
    assert cos_t.shape == cos_r.shape == (*shape[:-2], shape[-1], 17)
    assert cos_t.dtype == torch.float32
    cos_32, sin_32 = ecf_reference(x, epps_pulley_grid(3.0, 17, device="cuda")[0])
    tol = max(16 * EPS32, 4 * max(_max_err(cos_32, cos_r), _max_err(sin_32, sin_r)))
    assert _max_err(cos_t, cos_r) <= tol
    assert _max_err(sin_t, sin_r) <= tol


def test_ecf_handles_non_contiguous_input():
    torch.manual_seed(0)
    base = torch.randn(40, 300, device="cuda")
    x = base.t()  # (300, 40), stride (1, 300)
    got = ecf_triton(x, 0.25, 9)
    ref = ecf_triton(x.contiguous(), 0.25, 9)
    torch.testing.assert_close(got[0], ref[0], rtol=0, atol=4 * EPS32)
    torch.testing.assert_close(got[1], ref[1], rtol=0, atol=4 * EPS32)


def test_ecf_is_deterministic():
    torch.manual_seed(0)
    x = torch.randn(10000, 256, device="cuda")
    a = ecf_triton(x, 3.0 / 16, 17)
    b = ecf_triton(x, 3.0 / 16, 17)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("t_max,n_points", [(3.0, 17), (5.0, 9), (3.0, 2), (4.0, 33)])
@pytest.mark.parametrize("n,m", [(1, 4), (3, 1), (513, 37), (4096, 256)])
def test_statistic_and_grad_match_float64(t_max, n_points, n, m):
    torch.manual_seed(n + m)
    x = torch.randn(n, m, device="cuda", requires_grad=True)
    stat = epps_pulley(x, t_max, n_points, backend="triton")
    stat.sum().backward()

    x64 = x.detach().double().requires_grad_(True)
    ref = epps_pulley_reference(x64, t_max, n_points)
    ref.sum().backward()

    x32 = x.detach().clone().requires_grad_(True)
    off = epps_pulley_reference(x32, t_max, n_points)  # official float32 path
    off.sum().backward()

    # The statistic is N * sum_k w_k |ecf_k - phi_k|^2: an absolute ECF error delta becomes
    # ~ 2 * N * sum(w) * |ecf - phi| * delta, so compare relative to the float32 path's error.
    stat_tol = 4 * _max_err(off, ref) + 1e-5 * ref.abs().max().item() + 1e-6
    assert _max_err(stat, ref) <= stat_tol
    grad_tol = 4 * _max_err(x32.grad, x64.grad) + 1e-5 * x64.grad.abs().max().item() + 1e-9
    assert _max_err(x.grad, x64.grad) <= grad_tol


@pytest.mark.parametrize("n_points", [18, 40, 65])
def test_many_knots_use_several_knot_groups(n_points):
    # More knots than one register-resident group: knots are split into groups whose
    # rotations start from an exact cos/sin(k0 * theta); the backward accumulates groups.
    torch.manual_seed(n_points)
    x = torch.randn(3000, 70, device="cuda", requires_grad=True)
    stat = epps_pulley(x, 4.0, n_points, backend="triton")
    stat.sum().backward()
    x64 = x.detach().double().requires_grad_(True)
    ref = epps_pulley_reference(x64, 4.0, n_points)
    ref.sum().backward()
    torch.testing.assert_close(stat.double(), ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        x.grad.double(), x64.grad, rtol=1e-4, atol=1e-5 * x64.grad.abs().max().item()
    )


def test_arbitrary_upstream_gradient_on_batched_input():
    # Per-statistic weights in the upstream gradient (e.g. clipping or a weighted sum).
    torch.manual_seed(0)
    x = torch.randn(2, 3, 500, 40, device="cuda", requires_grad=True)
    upstream = torch.randn(2, 3, 40, device="cuda", dtype=torch.float64)
    stat = epps_pulley(x, backend="triton")
    assert stat.shape == (2, 3, 40)
    (stat.double() * upstream).sum().backward()
    x64 = x.detach().double().requires_grad_(True)
    ref = epps_pulley_reference(x64)
    (ref * upstream).sum().backward()
    torch.testing.assert_close(stat.double(), ref, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        x.grad.double(), x64.grad, rtol=1e-4, atol=1e-5 * x64.grad.abs().max().item()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_inputs(dtype):
    torch.manual_seed(0)
    x = torch.randn(2048, 96, device="cuda").to(dtype).requires_grad_(True)
    stat = epps_pulley(x, backend="triton")
    assert stat.dtype == torch.float32
    stat.sum().backward()
    assert x.grad.dtype == dtype

    # Same rounded inputs in float64: the kernel computes in float32 from the stored values,
    # so the statistic keeps float32-level accuracy; the gradient is rounded to `dtype`.
    x64 = x.detach().double().requires_grad_(True)
    ref = epps_pulley_reference(x64)
    ref.sum().backward()
    torch.testing.assert_close(stat.double(), ref, rtol=1e-4, atol=1e-5)
    rel = torch.finfo(dtype).eps
    torch.testing.assert_close(
        x.grad.double(), x64.grad, rtol=2 * rel, atol=2 * rel * x64.grad.abs().max().item()
    )


def test_large_phases():
    # Unnormalized embeddings give large t * x; one accurate cos/sin per element plus
    # norm-preserving rotations keeps the error at the level of the official float32 path.
    torch.manual_seed(0)
    x = torch.randn(3000, 64, device="cuda") * 40
    cos_t, sin_t = ecf_triton(x, 3.0 / 16, 17)
    cos_r, sin_r = _ecf64(x, 3.0, 17)
    cos_32, sin_32 = ecf_reference(x, epps_pulley_grid(3.0, 17, device="cuda")[0])
    tol = max(16 * EPS32, 4 * max(_max_err(cos_32, cos_r), _max_err(sin_32, sin_r)))
    assert _max_err(cos_t, cos_r) <= tol
    assert _max_err(sin_t, sin_r) <= tol


@pytest.mark.parametrize("d", [37, 512])
def test_sigreg_triton_matches_torch_backend(d):
    torch.manual_seed(d)
    z = torch.randn(1024, d, device="cuda")
    out = {}
    for backend in ("triton", "torch"):
        zz = z.clone().requires_grad_(True)
        val = sigreg(zz, num_slices=256, global_step=4, backend=backend)
        val.backward()
        out[backend] = (val.detach(), zz.grad)
    torch.testing.assert_close(out["triton"][0], out["torch"][0], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(out["triton"][1], out["torch"][1], rtol=1e-3, atol=1e-7)


def test_bf16_autocast_end_to_end():
    torch.manual_seed(0)
    z = torch.randn(512, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = SIGReg(num_slices=128, backend="triton").cuda()(z)
        ref = SIGReg(num_slices=128, backend="torch").cuda()(z)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-6)
    (g_t,) = torch.autograd.grad(got, z)
    (g_r,) = torch.autograd.grad(ref, z)
    torch.testing.assert_close(g_t, g_r, rtol=2e-2, atol=2e-2 * g_r.abs().max().item())


def test_auto_backend_falls_back_for_float64():
    x = torch.randn(64, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    stat = epps_pulley(x, backend="auto")
    assert stat.dtype == torch.float64
    with pytest.raises(RuntimeError):
        epps_pulley(x, backend="triton")
