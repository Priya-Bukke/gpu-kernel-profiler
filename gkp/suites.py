"""Benchmark suites. Each variant is timed, checked against NumPy, and reported with its bytes moved and
FLOPs, so throughput and arithmetic intensity can be compared with the device's limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyopencl as cl

from .runtime import Runtime


@dataclass
class Result:
    suite: str
    variant: str
    size: str
    ms: float
    bytes: float
    flops: float
    max_rel_error: float
    correct: bool

    @property
    def gbps(self) -> float:
        return self.bytes / self.ms / 1e6

    @property
    def gflops(self) -> float:
        return self.flops / self.ms / 1e6

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else 0.0

    def to_json(self) -> dict:
        return {**asdict(self), "gbps": self.gbps, "gflops": self.gflops, "intensity": self.intensity}


def cpu_baseline(suite: str, size: str, fn, flops: float, nbytes: float, repeats: int = 5) -> Result:
    """Same work done by NumPy on the CPU, timed the same way (median), for comparison."""
    import statistics
    import time

    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return Result(suite, "NumPy on CPU (reference)", size, statistics.median(times), nbytes, flops, 0.0, True)


def rel_error(got: np.ndarray, want: np.ndarray) -> float:
    scale = np.maximum(np.abs(want), 1.0)
    return float(np.max(np.abs(got - want) / scale))


def stream_suite(rt: Runtime, n: int = 1 << 24, repeats: int = 20) -> list[Result]:
    prg = rt.build("stream.cl")
    rng = np.random.default_rng(0)
    a = rng.random(n, dtype=np.float32)
    b = rng.random(n, dtype=np.float32)
    s = np.float32(3.0)
    A, B, C = rt.buffer(a), rt.buffer(b), rt.buffer(nbytes=a.nbytes)
    out = []
    ms = rt.time(prg.copy, (n,), None, (A, C), repeats)
    out.append(Result("stream", "copy", f"{n:,} floats", ms, 2 * a.nbytes, 0, rel_error(rt.read(C, n), a), True))
    ms = rt.time(prg.triad, (n,), None, (A, B, C, s), repeats)
    err = rel_error(rt.read(C, n), a + s * b)
    out.append(Result("stream", "triad", f"{n:,} floats", ms, 3 * a.nbytes, 2 * n, err, err < 1e-5))
    ms = rt.time(prg.triad_vec4, (n // 4,), None, (A, B, C, s), repeats)
    err = rel_error(rt.read(C, n), a + s * b)
    out.append(Result("stream", "triad float4", f"{n:,} floats", ms, 3 * a.nbytes, 2 * n, err, err < 1e-5))
    for r in out[:1]:
        r.correct = r.max_rel_error == 0.0
    return out


def reduce_suite(rt: Runtime, n: int = 1 << 24, wg: int = 256, repeats: int = 20) -> list[Result]:
    prg = rt.build("reduce.cl")
    x = np.random.default_rng(1).random(n, dtype=np.float32)
    want = float(np.sum(x, dtype=np.float64))
    X = rt.buffer(x)
    scratch = cl.LocalMemory(4 * wg)
    out = []

    def check(variant: str, kernel, groups: int, args, ms_global) -> None:
        P = rt.buffer(nbytes=4 * groups)
        ms = rt.time(kernel, (ms_global,), (wg,), (X, P, scratch, *args), repeats)
        got = float(np.sum(rt.read(P, groups), dtype=np.float64))
        err = abs(got - want) / want
        out.append(Result("reduce", variant, f"{n:,} floats", ms, x.nbytes, n, err, err < 1e-4))

    groups_full = n // wg
    check("v1 interleaved", prg.reduce_interleaved, groups_full, (), n)
    check("v2 sequential", prg.reduce_sequential, groups_full, (), n)
    groups_few = rt.info.compute_units * 8
    check("v3 grid stride", prg.reduce_grid_stride, groups_few, (np.uint32(n),), groups_few * wg)
    if rt.info.subgroups:
        check("v4 subgroups", prg.reduce_subgroup, groups_few, (np.uint32(n),), groups_few * wg)
    out.append(cpu_baseline("reduce", f"{n:,} floats", lambda: np.sum(x), n, x.nbytes))
    return out


def matmul_suite(rt: Runtime, n: int = 1024, tile: int = 16, wpt: int = 4, repeats: int = 10) -> list[Result]:
    prg = rt.build("matmul.cl", TILE=tile, WPT=wpt)
    rng = np.random.default_rng(2)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    want = a @ b
    A, B, C = rt.buffer(a), rt.buffer(b), rt.buffer(nbytes=a.nbytes)
    flops = 2.0 * n**3
    min_bytes = 3.0 * a.nbytes  # each matrix moved once: the ideal, used for arithmetic intensity
    out = []
    for variant, kernel, local in (
        ("v1 naive", prg.matmul_naive, (tile, tile)),
        ("v2 local memory tiles", prg.matmul_tiled, (tile, tile)),
        ("v3 register blocking", prg.matmul_regblock, (tile, tile // wpt)),
    ):
        glob = (n, n) if variant != "v3 register blocking" else (n, n // wpt)
        ms = rt.time(kernel, glob, local, (A, B, C, np.int32(n)), repeats)
        err = rel_error(rt.read(C, (n, n)), want)
        out.append(Result("matmul", variant, f"{n}x{n}", ms, min_bytes, flops, err, err < 1e-3))
    out.append(cpu_baseline("matmul", f"{n}x{n}", lambda: a @ b, flops, min_bytes))
    return out


def fir_suite(rt: Runtime, n: int = 1 << 22, taps: int = 64, wg: int = 256, repeats: int = 20) -> list[Result]:
    prg = rt.build("fir.cl", TAPS=taps)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n + taps - 1).astype(np.float32)
    h = rng.standard_normal(taps).astype(np.float32)
    want = np.correlate(x.astype(np.float64), h.astype(np.float64), mode="valid")
    X, H, Y = rt.buffer(x), rt.buffer(h), rt.buffer(nbytes=4 * n)
    flops = 2.0 * n * taps
    min_bytes = 4.0 * (len(x) + n)
    out = []
    ms = rt.time(prg.fir_naive, (n,), (wg,), (X, H, Y), repeats)
    err = rel_error(rt.read(Y, n), want)
    out.append(Result("fir", f"v1 naive ({taps} taps)", f"{n:,} samples", ms, min_bytes, flops, err, err < 1e-3))
    ms = rt.time(prg.fir_local, (n,), (wg,), (X, H, Y, cl.LocalMemory(4 * (wg + taps - 1))), repeats)
    err = rel_error(rt.read(Y, n), want)
    out.append(Result("fir", f"v2 local window ({taps} taps)", f"{n:,} samples", ms, min_bytes, flops, err, err < 1e-3))
    out.append(cpu_baseline("fir", f"{n:,} samples", lambda: np.correlate(x, h, mode="valid"), flops, min_bytes))
    return out


SUITES = {"stream": stream_suite, "reduce": reduce_suite, "matmul": matmul_suite, "fir": fir_suite}
QUICK = {"stream": {"n": 1 << 16}, "reduce": {"n": 1 << 16}, "matmul": {"n": 64}, "fir": {"n": 1 << 14}}
