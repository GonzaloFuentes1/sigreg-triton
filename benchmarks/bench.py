"""Benchmark SIGReg forward+backward: official LeJEPA vs torch.compile vs sigreg-triton.

Requires a CUDA GPU and the official package (not vendored, see NOTICE):
    pip install --no-deps "lejepa @ git+https://github.com/galilai-group/lejepa@c293d291ca87cd4fddee9d3fffe4e914c7272052"

Usage:
    python benchmarks/bench.py                # full grid, writes results/bench.csv + PNG
    python benchmarks/bench.py --quick        # small grid for a smoke test

Timing: CUDA events around ``loss = module(z); loss.backward()``, after warmup. The
implementations are interleaved in ``--rounds`` rounds of ``--iters`` steps each, so that
drift in host load affects all of them alike; the median over all steps is reported, with
the 20th/80th percentiles. Every implementation re-samples its directions per call exactly
like the official module (seeded torch.Generator + host sync on the step counter).
GPU time: sum of the device time of all kernels and copies in one step (torch.profiler),
i.e. the work the GPU does regardless of host-side launch overhead.
Memory: peak allocated bytes during one forward+backward minus the bytes allocated before
the call (inputs, parameters and cuBLAS workspace are excluded; z.grad is included).
"""

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import torch

import sigreg_triton

try:
    import lejepa
except ImportError as exc:  # pragma: no cover
    raise SystemExit("benchmarks need the official `lejepa` package, see the docstring") from exc

LEJEPA_COMMIT = "c293d291ca87cd4fddee9d3fffe4e914c7272052"
T_MAX = 3.0
DTYPES = {"fp32": None, "bf16": torch.bfloat16}


def grid(quick: bool):
    """(N, D, M, K) configurations: an N x M sweep plus D and K sweeps."""
    if quick:
        return [(2048, 256, 256, 17), (8192, 512, 1024, 17)]
    configs = [(n, 512, m, 17) for m in (256, 1024) for n in (1024, 4096, 16384, 65536)]
    configs += [(16384, d, 1024, 17) for d in (128, 2048)]
    configs += [(16384, 512, 1024, k) for k in (9, 33, 65)]
    return configs


def make_impl(name: str, m: int, k: int):
    if name == "official":
        uni = lejepa.univariate.EppsPulley(t_max=T_MAX, n_points=k)
        return lejepa.multivariate.SlicingUnivariateTest(univariate_test=uni, num_slices=m)
    if name == "official+compile":
        uni = torch.compile(lejepa.univariate.EppsPulley(t_max=T_MAX, n_points=k))
        return lejepa.multivariate.SlicingUnivariateTest(univariate_test=uni, num_slices=m)
    if name == "sigreg-triton":
        return sigreg_triton.SIGReg(num_slices=m, t_max=T_MAX, n_points=k, backend="triton")
    raise ValueError(name)


def step(module, z, autocast_dtype):
    z.grad = None
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        loss = module(z)
    loss.backward()
    return loss


def time_samples_ms(module, z, autocast_dtype, iters):
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step(module, z, autocast_dtype)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def percentiles(times):
    times = sorted(times)
    return times[len(times) // 2], times[len(times) // 5], times[(4 * len(times)) // 5]


def gpu_time_ms(module, z, autocast_dtype, steps=5):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step(module, z, autocast_dtype)
        torch.cuda.synchronize()
    total_us = sum(event.self_device_time_total for event in prof.key_averages())
    return total_us / 1000 / steps


def peak_memory_mb(module, z, autocast_dtype):
    z.grad = None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step(module, z, autocast_dtype)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def matmul_time_ms(z, m, autocast_dtype, iters):
    """Projection alone (z @ A forward + backward), the part left to cuBLAS."""
    a = sigreg_triton.sample_directions(z.size(-1), m, seed=0, device=z.device)

    class Proj(torch.nn.Module):
        def forward(self, v):
            return (v @ a).float().sum()

    proj = Proj()
    for _ in range(3):
        step(proj, z, autocast_dtype)
    return percentiles(time_samples_ms(proj, z, autocast_dtype, iters))[0]


def loss_at_step(module, z, autocast_dtype, global_step=123):
    module.global_step.fill_(global_step)
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None),
    ):
        return float(module(z))


def bench_config(n, d, m, k, dtype_name, autocast_dtype, impls, args):
    torch.manual_seed(0)
    z = torch.randn(n, d, device="cuda", requires_grad=True)
    base = dict(dtype=dtype_name, N=n, D=d, M=m, K=k)
    base["matmul_ms"] = matmul_time_ms(z, m, autocast_dtype, args.iters)
    torch._dynamo.reset()
    modules, rows = {}, {}
    for name in impls:
        rows[name] = dict(base, impl=name, status="ok")
        try:
            module = make_impl(name, m, k).cuda()
            rows[name]["loss"] = loss_at_step(module, z, autocast_dtype)
            for _ in range(args.warmup):
                step(module, z, autocast_dtype)
            torch.cuda.synchronize()
            modules[name] = module
        except torch.OutOfMemoryError:
            rows[name]["status"] = "oom"
        z.grad = None
        torch.cuda.empty_cache()

    samples = {name: [] for name in modules}
    for _ in range(args.rounds):
        for name, module in modules.items():
            samples[name] += time_samples_ms(module, z, autocast_dtype, args.iters)

    reference_loss = rows["official"].get("loss")
    for name, module in modules.items():
        med, p20, p80 = percentiles(samples[name])
        row = rows[name]
        row.update(time_ms=med, time_p20_ms=p20, time_p80_ms=p80)
        row["gpu_ms"] = gpu_time_ms(module, z, autocast_dtype)
        try:
            row["peak_mem_mb"] = peak_memory_mb(module, z, autocast_dtype)
        except torch.OutOfMemoryError:
            row["status"] = "oom"
        if reference_loss is not None:
            row["rel_diff_vs_official"] = abs(row["loss"] - reference_loss) / abs(reference_loss)
        z.grad = None
    for name in impls:
        print(json.dumps(rows[name]), flush=True)
    return [rows[name] for name in impls]


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.cache_size_limit = 256
    rows = []
    impls = ["official", "official+compile", "sigreg-triton"]
    for dtype_name in args.dtypes:
        autocast_dtype = DTYPES[dtype_name]
        for n, d, m, k in grid(args.quick):
            rows += bench_config(n, d, m, k, dtype_name, autocast_dtype, impls, args)
            torch.cuda.empty_cache()

    fields = [
        "dtype",
        "N",
        "D",
        "M",
        "K",
        "impl",
        "status",
        "time_ms",
        "time_p20_ms",
        "time_p80_ms",
        "gpu_ms",
        "peak_mem_mb",
        "matmul_ms",
        "loss",
        "rel_diff_vs_official",
    ]
    csv_path = out_dir / ("bench_quick.csv" if args.quick else "bench.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    (out_dir / "env.json").write_text(json.dumps(environment(), indent=2) + "\n")
    print("wrote", csv_path)
    if not args.quick:
        plot(rows, out_dir / "bench.png")


def environment():
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
                "-i",
                str(torch.cuda.current_device()),
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        smi = ""
    import triton

    return {
        "gpu": torch.cuda.get_device_name(),
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "lejepa_commit": LEJEPA_COMMIT,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"official": "o-", "official+compile": "s--", "sigreg-triton": "^-"}
    panels = [
        ("time_ms", "fwd+bwd wall time [ms] (median)"),
        ("gpu_ms", "GPU kernel time per step [ms]"),
        ("peak_mem_mb", "peak extra memory [MiB]"),
    ]
    dtypes = sorted({r["dtype"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(len(panels), len(dtypes), figsize=(6 * len(dtypes), 11), squeeze=False)
    for col, dtype_name in enumerate(dtypes):
        for m in (256, 1024):
            for impl, style in styles.items():
                sel = sorted(
                    (
                        r
                        for r in rows
                        if r["dtype"] == dtype_name
                        and r["impl"] == impl
                        and r["M"] == m
                        and r["D"] == 512
                        and r["K"] == 17
                        and r["status"] == "ok"
                    ),
                    key=lambda r: r["N"],
                )
                if not sel:
                    continue
                for row, (key, _) in enumerate(panels):
                    axes[row][col].plot(
                        [r["N"] for r in sel],
                        [r[key] for r in sel],
                        style,
                        alpha=1.0 if m == 1024 else 0.5,
                        label=f"{impl}, M={m}",
                    )
        for row, (_, ylabel) in enumerate(panels):
            ax = axes[row][col]
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("N (samples)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
        axes[0][col].set_title(f"SIGReg, D=512, K=17, {dtype_name}")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"], choices=list(DTYPES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10, help="steps per round")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", default="results")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
