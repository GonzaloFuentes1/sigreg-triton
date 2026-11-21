"""Functional API: ``sigreg`` (sliced statistic) and ``epps_pulley`` (univariate)."""

import torch

from .reference import ecf_reference, epps_pulley_grid, grid_dtype, statistic_from_ecf

_BACKENDS = ("auto", "triton", "torch")
_REDUCTIONS = ("mean", "sum", None)
_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _use_triton(x: torch.Tensor, backend: str) -> bool:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend == "torch":
        return False
    from .ecf import triton_available

    supported = x.is_cuda and triton_available() and x.dtype in _TRITON_DTYPES
    if backend == "triton" and not supported:
        raise RuntimeError(
            "backend='triton' needs a CUDA tensor of dtype float32/float16/bfloat16 and an "
            f"importable triton (got device={x.device}, dtype={x.dtype})"
        )
    return supported


def _epps_pulley(
    x: torch.Tensor,
    t: torch.Tensor,
    phi: torch.Tensor,
    weights: torch.Tensor,
    t_max: float,
    backend: str,
) -> torch.Tensor:
    n = x.size(-2)
    if n == 0:
        raise ValueError("the sample axis (dim -2) is empty")
    if _use_triton(x, backend):
        from .ecf import epps_pulley_triton

        return epps_pulley_triton(x, t_max / (t.numel() - 1), t.numel(), phi, weights)
    cos_mean, sin_mean = ecf_reference(x, t)
    return statistic_from_ecf(cos_mean, sin_mean, phi, weights, n)


def epps_pulley(
    x: torch.Tensor, t_max: float = 3.0, n_points: int = 17, backend: str = "auto"
) -> torch.Tensor:
    """Epps-Pulley statistic of each column of ``x`` (shape ``(*, N, M)``) -> ``(*, M)``.

    Same definition as ``lejepa.univariate.EppsPulley(t_max, n_points)``.
    """
    t, phi, weights = epps_pulley_grid(t_max, n_points, grid_dtype(x), x.device)
    return _epps_pulley(x, t, phi, weights, t_max, backend)


def sample_directions(
    dim: int,
    num_slices: int,
    seed: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Unit-norm Gaussian directions ``(dim, num_slices)`` seeded like the official code."""
    if generator is None:
        generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    a = torch.randn((dim, num_slices), device=device, generator=generator)
    a /= a.norm(p=2, dim=0)
    return a.to(dtype)


def _reduce(stats: torch.Tensor, reduction: str | None, clip_value: float | None):
    if clip_value is not None:
        stats = torch.where(stats < clip_value, torch.zeros_like(stats), stats)
    if reduction == "mean":
        return stats.mean()
    if reduction == "sum":
        return stats.sum()
    return stats


def _check_reduction(reduction):
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")


def sigreg(
    z: torch.Tensor,
    num_slices: int = 1024,
    *,
    global_step: int = 0,
    directions: torch.Tensor | None = None,
    t_max: float = 3.0,
    n_points: int = 17,
    reduction: str | None = "mean",
    clip_value: float | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """SIGReg: sliced Epps-Pulley statistic of embeddings ``z`` with shape ``(*, N, D)``.

    Equivalent to ``lejepa.multivariate.SlicingUnivariateTest(EppsPulley(t_max, n_points),
    num_slices, reduction, clip_value)`` at ``global_step``. Pass ``directions`` (``(D, M)``)
    to use fixed projections instead of sampling them from ``global_step``.

    ``backend``: ``"triton"`` (fused CUDA kernel), ``"torch"`` (reference) or ``"auto"``
    (Triton for CUDA float32/float16/bfloat16 inputs, reference otherwise).
    """
    _check_reduction(reduction)
    if directions is None:
        with torch.no_grad():
            directions = sample_directions(z.size(-1), num_slices, global_step, z.device)
    x = z @ directions.to(z.dtype)
    stats = epps_pulley(x, t_max, n_points, backend)
    return _reduce(stats, reduction, clip_value)
