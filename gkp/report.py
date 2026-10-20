"""Markdown report and roofline plot."""

from __future__ import annotations

from pathlib import Path

from .runtime import DeviceInfo
from .suites import Result


def measured_bandwidth(results: list[Result]) -> float:
    return max((r.gbps for r in results if r.suite == "stream"), default=0.0)


def write_report(info: DeviceInfo, results: list[Result], out: Path) -> str:
    bw = measured_bandwidth(results)
    lines = [f"# Results on {info.name}", "", info.summary() + ".", ""]
    if bw:
        lines.append(f"Measured memory bandwidth (best STREAM kernel): **{bw:.1f} GB/s**.")
    if info.peak_gflops and bw:
        lines.append(f"Ridge point of the roofline: {info.peak_gflops / bw:.1f} FLOP/byte. Kernels below it are "
                     "limited by memory bandwidth, kernels above it by compute.")
    lines += ["", "Speedup is against the first (v1) GPU version of the same suite. The NumPy rows run on the laptop's CPU.",
              "", "| Suite | Variant | Size | Time | GB/s | GFLOP/s | % of roof | FLOP/byte | Speedup | Check |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    first: dict[str, float] = {}
    for r in results:
        first.setdefault(r.suite, r.ms)
        cpu = r.variant.startswith("NumPy")
        speed = "" if r.suite == "stream" or cpu else f"{first[r.suite] / r.ms:.1f}x"
        pct = ""
        if cpu:
            pct = "CPU"
        elif r.flops and info.peak_gflops:
            roof = min(info.peak_gflops, r.intensity * bw) if bw else info.peak_gflops
            pct = f"{100 * r.gflops / roof:.0f}%"
        elif r.suite == "stream" and bw:
            pct = f"{100 * r.gbps / bw:.0f}% of best"
        lines.append(f"| {r.suite} | {r.variant} | {r.size} | {r.ms:.3f} ms | {r.gbps:.1f} | {r.gflops:.1f} | {pct} | "
                     f"{r.intensity:.2f} | {speed} | {'ok' if r.correct else 'WRONG'} |")
    lines += ["", "![roofline](roofline.png)", ""]
    text = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(text)
    plot_roofline(info, results, bw, out / "roofline.png")
    return text


def plot_roofline(info: DeviceInfo, results: list[Result], bw: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    xs = np.logspace(-2, 3, 200)
    if bw:
        roof = bw * xs if not info.peak_gflops else np.minimum(bw * xs, info.peak_gflops)
        ax.plot(xs, roof, color="#444", lw=1.5, label=f"roofline ({bw:.0f} GB/s"
                + (f", {info.peak_gflops:.0f} GFLOP/s)" if info.peak_gflops else ")"))
    markers = {"stream": "o", "reduce": "s", "matmul": "^", "fir": "D"}
    for r in results:
        if r.flops and not r.variant.startswith("NumPy"):
            ax.scatter(r.intensity, r.gflops, marker=markers[r.suite], s=42, zorder=3)
            ax.annotate(f"{r.suite}: {r.variant}", (r.intensity, r.gflops), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte of minimum traffic)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Roofline, {info.name}", fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
