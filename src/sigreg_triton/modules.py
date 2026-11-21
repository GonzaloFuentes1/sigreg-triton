"""``nn.Module`` wrappers mirroring the official LeJEPA classes."""

import torch

from ._dist import sync_max
from .functional import _check_reduction, _epps_pulley, _reduce, sample_directions
from .reference import epps_pulley_grid


class EppsPulley(torch.nn.Module):
    """Drop-in for ``lejepa.univariate.EppsPulley`` (trapezoid integration).

    Maps ``(*, N, M)`` samples to ``(*, M)`` statistics. It can be used as the
    ``univariate_test`` of the official ``SlicingUnivariateTest``.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17, backend: str = "auto"):
        super().__init__()
        self.t_max = float(t_max)
        self.n_points = int(n_points)
        self.backend = backend
        t, phi, weights = epps_pulley_grid(self.t_max, self.n_points, torch.float32)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, phi, weights = self.t, self.phi, self.weights
        if x.dtype == torch.float64:
            t, phi, weights = t.double(), phi.double(), weights.double()
        return _epps_pulley(x, t, phi, weights, self.t_max, self.backend)


class SIGReg(torch.nn.Module):
    """Drop-in for ``SlicingUnivariateTest(EppsPulley(t_max, n_points), num_slices, ...)``.

    Directions are re-sampled at every call from a generator seeded with ``global_step``
    (synchronized across ranks with an all-reduce MAX), then ``global_step`` is incremented.
    For the same ``global_step`` the directions are bit-identical to the official ones.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
        reduction: str | None = "mean",
        clip_value: float | None = None,
        backend: str = "auto",
    ):
        super().__init__()
        _check_reduction(reduction)
        self.num_slices = num_slices
        self.reduction = reduction
        self.clip_value = clip_value
        self.univariate_test = EppsPulley(t_max, n_points, backend)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        return self._generator

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            seed = int(sync_max(self.global_step.clone()).item())
            generator = self._get_generator(z.device)
            a = sample_directions(z.size(-1), self.num_slices, seed, z.device, generator=generator)
            self.global_step.add_(1)
        stats = self.univariate_test(z @ a.to(z.dtype))
        return _reduce(stats, self.reduction, self.clip_value)
