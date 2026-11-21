"""Sweep tile configurations of the ECF kernels and print median kernel times.

Usage: python benchmarks/tune.py [--knots 17]

Configurations are (BLOCK_N, BLOCK_M, num_warps, programs per SM). The forward keeps one
tile-shaped accumulator per knot, so tiles with more than ~4 elements per thread spill.
"""

import argparse

import torch

from sigreg_triton import ecf

SHAPES = [(65536, 1024), (16384, 1024), (8192, 256)]
FWD_CONFIGS = [
    (4, 128, 4, 2),
    (4, 128, 4, 4),
    (8, 64, 4, 2),
    (8, 64, 4, 4),
    (4, 64, 2, 4),
    (2, 128, 2, 4),
    (8, 128, 8, 2),
    (16, 64, 8, 2),
    (16, 64, 4, 2),
]
BWD_CONFIGS = [
    (8, 64, 4, 4),
    (8, 64, 4, 8),
    (16, 64, 4, 4),
    (4, 128, 4, 4),
    (16, 128, 8, 4),
    (32, 64, 8, 4),
    (32, 32, 4, 4),
    (8, 128, 4, 4),
]


def median_ms(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knots", type=int, default=17)
    args = parser.parse_args()
    k, dt = args.knots, 3.0 / (args.knots - 1)
    print(torch.cuda.get_device_name(), "| triton", ecf.triton.__version__, "| K =", k)
    default_fwd, default_bwd = ecf._FWD_CONFIG, ecf._BWD_CONFIG
    for n, m in SHAPES:
        x = torch.randn(1, n, m, device="cuda")
        gc = torch.randn(1, m, k, device="cuda")
        gs = torch.randn(1, m, k, device="cuda")
        print(f"N={n} M={m}")
        for cfg in FWD_CONFIGS:
            ecf._FWD_CONFIG = cfg
            mark = " (default)" if cfg == default_fwd else ""
            print(f"  fwd {cfg}: {median_ms(lambda x=x: ecf._ecf_forward(x, dt, k)):.4f} ms{mark}")
        ecf._FWD_CONFIG = default_fwd
        for cfg in BWD_CONFIGS:
            ecf._BWD_CONFIG = cfg
            mark = " (default)" if cfg == default_bwd else ""
            ms = median_ms(lambda x=x, gc=gc, gs=gs: ecf._ecf_backward(x, gc, gs, dt, k))
            print(f"  bwd {cfg}: {ms:.4f} ms{mark}")
        ecf._BWD_CONFIG = default_bwd


if __name__ == "__main__":
    main()
