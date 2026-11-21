"""Parity against the official LeJEPA package (skipped when it is not installed).

Install it with:
    pip install --no-deps "lejepa @ git+https://github.com/galilai-group/lejepa@c293d291ca87cd4fddee9d3fffe4e914c7272052"
"""

import pytest
import torch

import sigreg_triton

lejepa = pytest.importorskip("lejepa")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _official(num_slices, t_max=3.0, n_points=17, reduction="mean", step=0):
    uni = lejepa.univariate.EppsPulley(t_max=t_max, n_points=n_points)
    mod = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=uni, num_slices=num_slices, reduction=reduction
    )
    mod.global_step.fill_(step)
    return mod


def _value_and_grad(fn, z):
    z = z.detach().clone().requires_grad_(True)
    out = fn(z)
    out.sum().backward()
    return out.detach(), z.grad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["torch", "auto"])
def test_sigreg_module_matches_official(device, backend):
    torch.manual_seed(0)
    z = torch.randn(256, 48, device=device)
    official = _official(64, step=11).to(device)
    ours = sigreg_triton.SIGReg(num_slices=64, backend=backend).to(device)
    ours.global_step.fill_(11)
    ref, ref_grad = _value_and_grad(official, z)
    got, got_grad = _value_and_grad(ours, z)
    assert int(ours.global_step) == int(official.global_step) == 12
    # Same directions; float32 on both sides. The fused kernel reorders the sum over N and
    # builds cos/sin(t_k x) by angle addition, so allow a few float32 ulps of the O(1) terms.
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got_grad, ref_grad, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_univariate_drop_in_inside_official_slicing(device):
    torch.manual_seed(1)
    z = torch.randn(128, 20, device=device)
    official = _official(32, reduction=None, step=3).to(device)
    mixed = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=sigreg_triton.EppsPulley(), num_slices=32, reduction=None
    ).to(device)
    mixed.global_step.fill_(3)
    torch.testing.assert_close(mixed(z), official(z), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("t_max,n_points", [(3.0, 17), (5.0, 9), (2.0, 33)])
def test_univariate_matches_official_grids(t_max, n_points):
    torch.manual_seed(2)
    x = torch.randn(500, 6, dtype=torch.float64) * 1.2
    # The official module only accepts float64 input once its buffers are cast.
    ref = lejepa.univariate.EppsPulley(t_max=t_max, n_points=n_points).double()(x)
    got = sigreg_triton.epps_pulley(x, t_max=t_max, n_points=n_points, backend="torch")
    # Official knots are float32 values cast to float64; ours are built in float64.
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-7)
