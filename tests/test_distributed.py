"""Distributed semantics with the gloo backend (2 processes).

With equal shards, the all-reduced statistic must equal the single-process statistic on the
concatenated batch, and the per-rank gradient must follow the official convention (the
backward all-reduces the gradient, so each rank sees ``world_size`` times the partial
derivative of the global statistic; DDP's gradient averaging then divides it back).

The Triton variant runs both ranks on the same GPU (gloo handles CUDA tensors).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sigreg_triton
from sigreg_triton.ecf import triton_available

WORLD = 2

CASES = [
    pytest.param("torch", "cpu", torch.float64, 1e-12, id="torch-cpu-float64"),
    pytest.param(
        "triton",
        "cuda",
        torch.float32,
        1e-5,
        id="triton-cuda-float32",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not triton_available(), reason="needs CUDA + Triton"),
        ],
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, port, z_full, a, backend, device, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        shard = z_full.chunk(WORLD)[rank].to(device).requires_grad_(True)
        stat = sigreg_triton.sigreg(shard, directions=a.to(device), backend=backend)
        stat.backward()
        step = torch.tensor(rank * 5)
        synced = sigreg_triton._dist.sync_max(step)
        out[rank] = (stat.detach().cpu(), shard.grad.cpu(), int(synced))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend,device,dtype,rtol", CASES)
def test_two_process_gloo_matches_single_process(backend, device, dtype, rtol):
    torch.manual_seed(0)
    z = torch.randn(256, 10, dtype=dtype)
    a = sigreg_triton.sample_directions(10, 8, seed=0, dtype=dtype)

    full = z.clone().to(device).requires_grad_(True)
    expected = sigreg_triton.sigreg(full, directions=a.to(device), backend=backend)
    expected.backward()
    expected, full_grad = expected.detach().cpu(), full.grad.cpu()

    manager = mp.get_context("spawn").Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(_free_port(), z, a, backend, device, out), nprocs=WORLD, join=True)

    for rank in range(WORLD):
        stat, grad, synced = out[rank]
        torch.testing.assert_close(stat, expected, rtol=rtol, atol=rtol)
        scale = full_grad.abs().max().item()
        torch.testing.assert_close(
            grad, WORLD * full_grad.chunk(WORLD)[rank], rtol=rtol, atol=rtol * scale
        )
        assert synced == 5
