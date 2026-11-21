"""Fused Triton kernels for the empirical characteristic function (forward + backward).

For samples ``x`` of shape ``(*, N, M)`` and knots ``t_k = k * dt`` (``k < K``) the forward
computes, without materializing the ``(*, N, M, K)`` phase tensor,

    cos_mean[..., m, k] = mean_n cos(t_k x[..., n, m])
    sin_mean[..., m, k] = mean_n sin(t_k x[..., n, m])

Design (choices measured with ``benchmarks/tune.py`` on an H100):

* One accurate cos/sin pair of ``theta = dt * x`` per element; ``cos/sin(k * theta)`` for the
  following knots come from angle addition (a complex rotation, 4 FMAs per knot).
* Knots are processed in groups of at most ``_KNOTS_PER_GROUP``; a group starting at knot
  ``k0 > 0`` begins from an exact cos/sin(``k0 * theta``). Larger groups spill registers.
* Forward: each program walks a chunk of rows and keeps one tile-shaped accumulator per knot
  (a loop-carried tuple), so the loop body has no cross-thread reductions; the tile is
  reduced once at the end and written as a per-chunk partial sum. Partials are summed in a
  fixed order: the result is deterministic (no atomics).
* Backward: only ``x`` is saved. Phases are recomputed and

      dL/dx[..., n, m] = sum_k t_k / N * (g_sin[..., m, k] cos(t_k x) - g_cos[..., m, k] sin(t_k x))

  is evaluated with the per-knot gradient vectors cached in registers. Knot groups are
  separate launches accumulated in float32 (in launch order, hence deterministic).
"""

import functools

import torch
from torch.autograd.function import once_differentiable

from ._dist import all_reduce_mean_, is_initialized, world_size

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _ecf_fwd_kernel(
        x_ptr,
        part_ptr,
        N,
        M,
        rows_per_chunk,
        stride_b,
        stride_n,
        stride_m,
        dt,
        inv_n,
        K: tl.constexpr,
        KG: tl.constexpr,
        N_GROUPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_bg = tl.program_id(2)
        n_chunks = tl.num_programs(0)
        pid_b = pid_bg // N_GROUPS
        k0 = (pid_bg % N_GROUPS) * KG

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        acc_c = ()
        acc_s = ()
        for _ in tl.static_range(KG):
            acc_c = acc_c + (tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32),)
            acc_s = acc_s + (tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32),)

        x_base = x_ptr + pid_b.to(tl.int64) * stride_b
        n_start = pid_c * rows_per_chunk
        n_end = tl.minimum(n_start + rows_per_chunk, N)
        for n0 in range(n_start, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask = (offs_n < n_end)[:, None] & mask_m[None, :]
            ptrs = x_base + offs_n[:, None].to(tl.int64) * stride_n + offs_m[None, :] * stride_m
            theta = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) * dt
            c1 = libdevice.cos(theta)
            s1 = libdevice.sin(theta)
            # Angle of the first knot of the group; masked entries contribute 0.
            if N_GROUPS > 1:
                phase0 = theta * k0.to(tl.float32)
                c = tl.where(mask, libdevice.cos(phase0), 0.0)
                s = tl.where(mask, libdevice.sin(phase0), 0.0)
            else:
                c = tl.where(mask, 1.0, 0.0)
                s = tl.zeros_like(c)
            new_c = ()
            new_s = ()
            for j in tl.static_range(KG):
                if j > 0:
                    c_next = c * c1 - s * s1
                    s = s * c1 + c * s1
                    c = c_next
                new_c = new_c + (acc_c[j] + c,)
                new_s = new_s + (acc_s[j] + s,)
            acc_c = new_c
            acc_s = new_s

        # partials layout: (B, n_chunks, 2, K, M), already divided by N.
        out = part_ptr + (pid_b * n_chunks + pid_c).to(tl.int64) * (2 * K * M)
        for j in tl.static_range(KG):
            smask = mask_m & (k0 + j < K)
            tl.store(out + (k0 + j) * M + offs_m, tl.sum(acc_c[j], axis=0) * inv_n, mask=smask)
            tl.store(out + (K + k0 + j) * M + offs_m, tl.sum(acc_s[j], axis=0) * inv_n, mask=smask)

    @triton.jit
    def _ecf_bwd_kernel(
        x_ptr,
        gc_ptr,
        gs_ptr,
        dx_ptr,
        N,
        M,
        rows_per_chunk,
        stride_xb,
        stride_xn,
        stride_xm,
        stride_gb,
        stride_gm,
        stride_gk,
        dt,
        dt_over_n,
        k0,
        K: tl.constexpr,
        KG: tl.constexpr,
        FIRST_GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        # Incoming gradients (B, M, K) scaled by d/dx of the phase, t_k / N = k * dt / N.
        g_offs = pid_b.to(tl.int64) * stride_gb + offs_m * stride_gm
        gcs = ()
        gss = ()
        for j in tl.static_range(KG):
            gmask = mask_m & (k0 + j < K)
            scale = (k0 + j) * dt_over_n
            gc = tl.load(gc_ptr + g_offs + (k0 + j) * stride_gk, mask=gmask, other=0.0)
            gs = tl.load(gs_ptr + g_offs + (k0 + j) * stride_gk, mask=gmask, other=0.0)
            gcs = gcs + (gc.to(tl.float32) * scale,)
            gss = gss + (gs.to(tl.float32) * scale,)

        x_base = x_ptr + pid_b.to(tl.int64) * stride_xb
        dx_base = dx_ptr + pid_b.to(tl.int64) * N * M
        n_start = pid_c * rows_per_chunk
        n_end = tl.minimum(n_start + rows_per_chunk, N)
        for n0 in range(n_start, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask = (offs_n < n_end)[:, None] & mask_m[None, :]
            ptrs = x_base + offs_n[:, None].to(tl.int64) * stride_xn + offs_m[None, :] * stride_xm
            theta = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) * dt
            c1 = libdevice.cos(theta)
            s1 = libdevice.sin(theta)
            if FIRST_GROUP:
                c = tl.full(theta.shape, 1.0, tl.float32)
                s = tl.zeros_like(theta)
            else:
                phase0 = theta * k0
                c = libdevice.cos(phase0)
                s = libdevice.sin(phase0)
            g = c * gss[0][None, :] - s * gcs[0][None, :]
            for j in tl.static_range(1, KG):
                c_next = c * c1 - s * s1
                s = s * c1 + c * s1
                c = c_next
                g += c * gss[j][None, :] - s * gcs[j][None, :]

            dx_ptrs = dx_base + offs_n[:, None].to(tl.int64) * M + offs_m[None, :]
            if not FIRST_GROUP:
                g += tl.load(dx_ptrs, mask=mask, other=0.0)
            tl.store(dx_ptrs, g.to(dx_ptr.dtype.element_ty), mask=mask)


# Tile configurations (BLOCK_N, BLOCK_M, num_warps, programs per SM); see benchmarks/tune.py.
_FWD_CONFIG = (4, 128, 4, 2)
_BWD_CONFIG = (8, 64, 4, 4)
_KNOTS_PER_GROUP = 17


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _knot_groups(k: int) -> tuple[int, int]:
    """(number of groups, knots per group) with groups as even as possible."""
    n_groups = _cdiv(k, _KNOTS_PER_GROUP)
    return n_groups, _cdiv(k, n_groups)


def _row_chunks(n: int, other_programs: int, block_n: int, per_sm: int, device) -> tuple:
    """Split N into chunks (multiples of BLOCK_N) so that the grid fills the GPU."""
    target = per_sm * _num_sms(device.index if device.index is not None else 0)
    n_chunks = max(1, min(_cdiv(n, block_n), _cdiv(target, other_programs)))
    rows_per_chunk = _cdiv(_cdiv(n, n_chunks), block_n) * block_n
    return _cdiv(n, rows_per_chunk), rows_per_chunk


def _ecf_forward(x3: torch.Tensor, dt: float, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, m = x3.shape
    block_n, block_m, num_warps, per_sm = _FWD_CONFIG
    n_groups, kg = _knot_groups(k)
    m_blocks = _cdiv(m, block_m)
    n_chunks, rows_per_chunk = _row_chunks(n, m_blocks * b * n_groups, block_n, per_sm, x3.device)

    partials = torch.empty((b, n_chunks, 2, k, m), device=x3.device, dtype=torch.float32)
    _ecf_fwd_kernel[(n_chunks, m_blocks, b * n_groups)](
        x3,
        partials,
        n,
        m,
        rows_per_chunk,
        x3.stride(0),
        x3.stride(1),
        x3.stride(2),
        dt,
        1.0 / n,
        K=k,
        KG=kg,
        N_GROUPS=n_groups,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        num_warps=num_warps,
        enable_fp_fusion=True,
    )
    means = partials[:, 0] if n_chunks == 1 else partials.sum(dim=1)  # (B, 2, K, M)
    return means[:, 0].transpose(1, 2), means[:, 1].transpose(1, 2)


def _ecf_backward(
    x3: torch.Tensor, grad_cos: torch.Tensor, grad_sin: torch.Tensor, dt: float, k: int
) -> torch.Tensor:
    b, n, m = x3.shape
    if grad_cos.stride() != grad_sin.stride():  # the kernel takes one set of strides
        grad_cos, grad_sin = grad_cos.contiguous(), grad_sin.contiguous()
    block_n, block_m, num_warps, per_sm = _BWD_CONFIG
    n_groups, kg = _knot_groups(k)
    m_blocks = _cdiv(m, block_m)
    n_chunks, rows_per_chunk = _row_chunks(n, m_blocks * b, block_n, per_sm, x3.device)
    # Several groups accumulate into a float32 buffer; a single group writes x.dtype directly.
    dx_dtype = x3.dtype if n_groups == 1 else torch.float32
    dx = torch.empty((b, n, m), device=x3.device, dtype=dx_dtype)
    for group in range(n_groups):
        _ecf_bwd_kernel[(n_chunks, m_blocks, b)](
            x3,
            grad_cos,
            grad_sin,
            dx,
            n,
            m,
            rows_per_chunk,
            x3.stride(0),
            x3.stride(1),
            x3.stride(2),
            grad_cos.stride(0),
            grad_cos.stride(1),
            grad_cos.stride(2),
            dt,
            dt / n,
            group * kg,
            K=k,
            KG=kg,
            FIRST_GROUP=group == 0,
            BLOCK_N=block_n,
            BLOCK_M=block_m,
            num_warps=num_warps,
            enable_fp_fusion=True,
        )
    return dx.to(x3.dtype)


class _ECF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, k):
        n, m = x.shape[-2:]
        x3 = x.reshape(-1, n, m)
        cos_mean, sin_mean = _ecf_forward(x3, dt, k)
        ctx.save_for_backward(x3)
        ctx.dt, ctx.k, ctx.x_shape = dt, k, x.shape
        out_shape = (*x.shape[:-2], m, k)
        return cos_mean.reshape(out_shape), sin_mean.reshape(out_shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_cos, grad_sin):
        (x3,) = ctx.saved_tensors
        b, _, m = x3.shape
        dx = _ecf_backward(
            x3, grad_cos.reshape(b, m, ctx.k), grad_sin.reshape(b, m, ctx.k), ctx.dt, ctx.k
        )
        return dx.reshape(ctx.x_shape), None, None


class _EppsPulleyStat(torch.autograd.Function):
    """ECF kernels and quadrature in a single autograd node.

    Same computation as ``reference.statistic_from_ecf(*ecf, ...)`` but with the backward of
    the (tiny, M x K) quadrature written by hand: composing it from autograd ops costs more
    host time than the kernels themselves for small N. Distributed semantics are those of
    the differentiable all-reduce: average the ECF across ranks in the forward and the ECF
    gradients in the backward.
    """

    @staticmethod
    def forward(ctx, x, dt, k, phi, weights):
        n, m = x.shape[-2:]
        x3 = x.reshape(-1, n, m)
        cos_mean, sin_mean = _ecf_forward(x3, dt, k)  # (B, M, K) float32
        if is_initialized():
            cos_mean = all_reduce_mean_(cos_mean.contiguous())
            sin_mean = all_reduce_mean_(sin_mean.contiguous())
        diff = cos_mean - phi
        scale = n * world_size()
        stats = ((diff.square() + sin_mean.square()) @ weights) * scale
        ctx.save_for_backward(x3, diff, sin_mean, weights)
        ctx.dt, ctx.k, ctx.scale, ctx.x_shape = dt, k, scale, x.shape
        return stats.reshape(x.shape[:-2] + (m,))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_stats):
        x3, diff, sin_mean, weights = ctx.saved_tensors
        b, _, m = x3.shape
        # d stat / d cos_mean_k = 2 * scale * w_k * (cos_mean_k - phi_k), same for sin_mean_k.
        g = grad_stats.reshape(b, m, 1).float() * (weights.float() * (2.0 * ctx.scale))
        grad_cos = g * diff
        grad_sin = g * sin_mean
        if is_initialized():
            all_reduce_mean_(grad_cos)
            all_reduce_mean_(grad_sin)
        dx = _ecf_backward(x3, grad_cos, grad_sin, ctx.dt, ctx.k)
        return dx.reshape(ctx.x_shape), None, None, None, None


def _check_input(x: torch.Tensor) -> None:
    if not triton_available():
        raise RuntimeError("Triton with a CUDA device is required")
    if x.dim() < 2 or x.size(-2) == 0:
        raise ValueError(f"expected a (*, N, M) tensor with N > 0, got shape {tuple(x.shape)}")


def epps_pulley_triton(
    x: torch.Tensor, dt: float, n_points: int, phi: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Epps-Pulley statistic ``(*, M)`` of ``x`` ``(*, N, M)`` with the fused kernels.

    ``phi`` and ``weights`` are the float32 grid tensors of ``reference.epps_pulley_grid``
    for knots ``t_k = k * dt``.
    """
    _check_input(x)
    return _EppsPulleyStat.apply(x, float(dt), int(n_points), phi, weights)


def ecf_triton(x: torch.Tensor, dt: float, n_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical CF on knots ``t_k = k * dt`` (``k < n_points``) with a fused Triton kernel.

    ``x``: CUDA tensor ``(*, N, M)`` in float32/float16/bfloat16 (any strides).
    Returns ``(cos_mean, sin_mean)``, float32 tensors of shape ``(*, M, n_points)``.
    """
    _check_input(x)
    return _ECF.apply(x, float(dt), int(n_points))
