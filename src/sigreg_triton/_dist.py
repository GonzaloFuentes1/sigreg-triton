"""Distributed helpers reproducing the semantics of the official LeJEPA code.

* ECF averages are combined with a differentiable all-reduce (average over ranks), whose
  backward all-reduces the incoming gradient with the same operation.
* The statistic is scaled by ``N * world_size`` (global sample count).
* The slicing seed (``global_step``) is synchronized with an all-reduce MAX.
"""

import torch
from torch import distributed as dist


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """Differentiable average across ranks; identity when not distributed."""
    if not is_initialized():
        return x
    from torch.distributed.nn import all_reduce

    if dist.get_backend() == "nccl":
        return all_reduce(x, dist.ReduceOp.AVG)
    # Backends without AVG (e.g. gloo): SUM / W has the same value and the same gradient
    # (the backward of the SUM all-reduce sums the gradients scaled by 1 / W: their average).
    return all_reduce(x, dist.ReduceOp.SUM) / dist.get_world_size()


def all_reduce_mean_(x: torch.Tensor) -> torch.Tensor:
    """In-place, non-differentiable average across ranks (``x`` must be contiguous).

    Used by the fused Triton path, which applies it to the ECF in the forward and to the
    ECF gradients in the backward: exactly what the differentiable all-reduce does.
    """
    if not is_initialized():
        return x
    if dist.get_backend() == "nccl":
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
    else:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(dist.get_world_size())
    return x


def sync_max(x: torch.Tensor) -> torch.Tensor:
    """Non-differentiable MAX all-reduce used to synchronize ``global_step``."""
    if not is_initialized():
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x
