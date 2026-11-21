"""Tests for the pure-PyTorch reference implementation (CPU, float64)."""

import math

import pytest
import torch

from sigreg_triton import EppsPulley, SIGReg, sample_directions, sigreg
from sigreg_triton.reference import ecf_reference, epps_pulley_grid, epps_pulley_reference


def _paper_listing_statistic(x, t_max, n_knots):
    """Independent formulation following the listing in arXiv:2511.08544 (Sec. 4).

    Complex ECF on a symmetric grid [-t_max, t_max] and torch.trapz; no symmetry folding.
    """
    t = torch.linspace(-t_max, t_max, n_knots, dtype=x.dtype)
    exp_f = torch.exp(-0.5 * t**2)
    ecf = (1j * x.unsqueeze(-1) * t).exp().mean(-3)
    err = (ecf - exp_f).abs().square().mul(exp_f)
    return torch.trapz(err, t, dim=-1) * x.size(-2)


def test_grid_matches_official_construction():
    t, phi, w = epps_pulley_grid(3.0, 17, torch.float64, "cpu")
    dt = 3.0 / 16
    assert torch.allclose(t, torch.arange(17, dtype=torch.float64) * dt)
    assert torch.allclose(phi, torch.exp(-0.5 * t**2))
    expected = torch.full((17,), 2 * dt, dtype=torch.float64)
    expected[[0, -1]] = dt
    assert torch.allclose(w, expected * phi)


@pytest.mark.parametrize("t_max,n_points", [(3.0, 17), (5.0, 9), (2.0, 5)])
def test_folded_grid_equals_symmetric_trapezoid(t_max, n_points):
    # Folding [0, t_max] with doubled weights is exactly the trapezoid rule on the
    # symmetric grid with 2 * n_points - 1 knots, because |ecf(t) - phi(t)|^2 is even.
    torch.manual_seed(0)
    x = torch.randn(64, 7, dtype=torch.float64) * 1.3 + 0.2
    ours = epps_pulley_reference(x, t_max=t_max, n_points=n_points)
    listing = _paper_listing_statistic(x, t_max, 2 * n_points - 1)
    assert torch.allclose(ours, listing, rtol=1e-12, atol=1e-12)


def test_output_shapes():
    x = torch.randn(32, 5, dtype=torch.float64)
    assert epps_pulley_reference(x).shape == (5,)
    xb = torch.randn(3, 32, 5, dtype=torch.float64)
    assert epps_pulley_reference(xb).shape == (3, 5)
    assert torch.allclose(epps_pulley_reference(xb)[1], epps_pulley_reference(xb[1]))


def test_ecf_reference_shapes_and_values():
    x = torch.randn(10, 4, dtype=torch.float64)
    t, _, _ = epps_pulley_grid(3.0, 17, x.dtype, x.device)
    c, s = ecf_reference(x, t)
    assert c.shape == s.shape == (4, 17)
    assert torch.allclose(c[:, 0], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(s[:, 0], torch.zeros(4, dtype=torch.float64))
    assert torch.allclose(c[2, 5], torch.cos(t[5] * x[:, 2]).mean())


def test_statistic_detects_non_gaussian():
    torch.manual_seed(0)
    gauss = epps_pulley_reference(torch.randn(4096, 8, dtype=torch.float64))
    unif = epps_pulley_reference(torch.rand(4096, 8, dtype=torch.float64))
    assert (gauss >= 0).all()
    assert gauss.max() < unif.min()


def test_gradcheck_epps_pulley_reference():
    torch.manual_seed(0)
    x = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: epps_pulley_reference(v, 3.0, 17), (x,))


def test_gradcheck_sigreg_reference():
    torch.manual_seed(0)
    z = torch.randn(11, 6, dtype=torch.float64, requires_grad=True)
    a = sample_directions(6, 5, seed=3, device="cpu", dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda v: sigreg(v, directions=a, backend="torch"), (z,))


def test_sample_directions_follow_official_recipe():
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    a = torch.randn((10, 4), generator=g)
    a /= a.norm(p=2, dim=0)
    assert torch.equal(sample_directions(10, 4, seed=7, device="cpu"), a)
    assert torch.allclose(a.norm(dim=0), torch.ones(4))


def test_sigreg_reductions_and_clip():
    torch.manual_seed(0)
    z = torch.randn(50, 8, dtype=torch.float64)
    a = sample_directions(8, 6, seed=0, device="cpu", dtype=torch.float64)
    per_slice = epps_pulley_reference(z @ a)
    none = sigreg(z, directions=a, reduction=None, backend="torch")
    assert torch.allclose(none, per_slice)
    assert torch.allclose(sigreg(z, directions=a, backend="torch"), per_slice.mean())
    assert torch.allclose(
        sigreg(z, directions=a, reduction="sum", backend="torch"), per_slice.sum()
    )
    clip = float(per_slice.median())
    clipped = sigreg(z, directions=a, reduction=None, clip_value=clip, backend="torch")
    assert torch.equal(clipped == 0, per_slice < clip)


def test_sigreg_samples_directions_from_global_step():
    z = torch.randn(40, 12, dtype=torch.float64)
    a = sample_directions(12, 16, seed=5, device="cpu").to(torch.float64)
    expected = sigreg(z, directions=a, backend="torch")
    got = sigreg(z, num_slices=16, global_step=5, backend="torch")
    assert torch.allclose(got, expected)


def test_modules_match_functional_and_advance_step():
    torch.manual_seed(0)
    z = torch.randn(40, 12)
    mod = SIGReg(num_slices=16, backend="torch")
    first = mod(z)
    assert int(mod.global_step) == 1
    second = mod(z)
    assert torch.allclose(first, sigreg(z, num_slices=16, global_step=0, backend="torch"))
    assert torch.allclose(second, sigreg(z, num_slices=16, global_step=1, backend="torch"))
    x = torch.randn(30, 4)
    assert torch.allclose(EppsPulley(backend="torch")(x), epps_pulley_reference(x))


def test_single_sample_is_finite():
    x = torch.randn(1, 3, dtype=torch.float64)
    assert torch.isfinite(epps_pulley_reference(x)).all()


def test_empty_batch_raises():
    with pytest.raises(ValueError):
        sigreg(torch.randn(0, 4), num_slices=3, backend="torch")


def test_invalid_arguments_raise():
    z = torch.randn(8, 4)
    with pytest.raises(ValueError):
        sigreg(z, num_slices=3, n_points=1, backend="torch")
    with pytest.raises(ValueError):
        sigreg(z, num_slices=3, reduction="max", backend="torch")
    with pytest.raises(ValueError):
        sigreg(z, num_slices=3, backend="cuda-magic")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU-only error path")
def test_triton_backend_requires_cuda():
    with pytest.raises(RuntimeError):
        sigreg(torch.randn(8, 4), num_slices=3, backend="triton")


def test_standard_normal_statistic_is_small():
    # E[T] ~ integral (1 - phi^2) w dt for Gaussian data; just check it is O(1), not O(N).
    torch.manual_seed(1)
    stat = epps_pulley_reference(torch.randn(20000, 4, dtype=torch.float64))
    assert stat.max() < 5.0
    assert math.isfinite(float(stat.sum()))
