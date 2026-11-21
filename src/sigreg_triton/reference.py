"""Pure-PyTorch reference of the Epps-Pulley statistic used by SIGReg.

This path materializes the ``(*, N, M, K)`` tensor of phases, exactly like the official
implementation. It runs on any device and dtype (including float64) and is used for CPU
execution and as ground truth in the tests.
"""

import torch

from ._dist import all_reduce_mean, world_size


def epps_pulley_grid(
    t_max: float = 3.0,
    n_points: int = 17,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integration knots, standard-normal CF and quadrature weights.

    Knots are ``t_k = k * t_max / (n_points - 1)`` on ``[0, t_max]``. The weights are the
    trapezoid weights doubled to account for ``t < 0`` (the integrand is even) and multiplied
    by the Gaussian window ``exp(-t^2 / 2)``. This is the trapezoid rule on the symmetric grid
    ``[-t_max, t_max]`` with ``2 * n_points - 1`` knots.

    Returns ``(t, phi, weights)``, each of shape ``(n_points,)``.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if not t_max > 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    t = torch.linspace(0, t_max, n_points, dtype=dtype, device=device)
    dt = t_max / (n_points - 1)
    trapezoid = torch.full((n_points,), 2 * dt, dtype=dtype, device=device)
    trapezoid[0] = dt
    trapezoid[-1] = dt
    phi = torch.exp(-0.5 * t.square())
    return t, phi, trapezoid * phi


def ecf_reference(x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real and imaginary parts of the empirical CF, averaged over the sample axis ``-2``.

    ``x`` has shape ``(*, N, M)``; returns two tensors of shape ``(*, M, K)``.
    """
    x_t = x.unsqueeze(-1) * t
    return x_t.cos().mean(-3), x_t.sin().mean(-3)


def statistic_from_ecf(
    cos_mean: torch.Tensor,
    sin_mean: torch.Tensor,
    phi: torch.Tensor,
    weights: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Weighted L2 distance between the (globally averaged) ECF and the N(0, 1) CF."""
    cos_mean = all_reduce_mean(cos_mean)
    sin_mean = all_reduce_mean(sin_mean)
    err = (cos_mean - phi).square() + sin_mean.square()
    return (err @ weights) * (n * world_size())


def grid_dtype(x: torch.Tensor) -> torch.dtype:
    """float32 grid (as in the official buffers) unless the input is float64."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def epps_pulley_reference(x: torch.Tensor, t_max: float = 3.0, n_points: int = 17) -> torch.Tensor:
    """Epps-Pulley statistic of each column of ``x`` (shape ``(*, N, M)``) -> ``(*, M)``."""
    t, phi, weights = epps_pulley_grid(t_max, n_points, grid_dtype(x), x.device)
    cos_mean, sin_mean = ecf_reference(x, t)
    return statistic_from_ecf(cos_mean, sin_mean, phi, weights, x.size(-2))
