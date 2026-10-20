"""OpenCL device selection, program building and kernel timing with event profiling."""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyopencl as cl

KERNELS = Path(__file__).resolve().parent / "kernels"


@dataclass
class DeviceInfo:
    name: str
    vendor: str
    kind: str
    compute_units: int
    clock_mhz: int
    local_mem_kb: int
    max_work_group: int
    subgroups: bool
    peak_gflops: float | None  # FP32, from the hardware description; None when unknown

    def summary(self) -> str:
        peak = f"{self.peak_gflops:.0f} GFLOP/s FP32 peak" if self.peak_gflops else "peak unknown"
        return (f"{self.name} ({self.kind}, {self.compute_units} compute units at {self.clock_mhz} MHz, "
                f"{self.local_mem_kb} KB local memory, {peak})")


def peak_gflops(device: cl.Device) -> float | None:
    """Intel Xe GPUs: each execution unit (compute unit) has 8 FP32 lanes doing one FMA (2 FLOPs) per clock."""
    if device.type & cl.device_type.GPU and "Intel" in device.vendor:
        return device.max_compute_units * 8 * 2 * device.max_clock_frequency / 1000
    return None


class Runtime:
    def __init__(self, device_hint: str | None = None):
        hint = (device_hint or os.environ.get("GKP_DEVICE", "")).lower()
        devices = [d for p in cl.get_platforms() for d in p.get_devices()]
        if not devices:
            raise RuntimeError("no OpenCL device found")
        gpus = [d for d in devices if d.type & cl.device_type.GPU]
        matches = [d for d in devices if hint and hint in d.name.lower()]
        self.device = (matches or gpus or devices)[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        exts = self.device.extensions
        self.info = DeviceInfo(
            name=self.device.name.strip(), vendor=self.device.vendor.strip(),
            kind=cl.device_type.to_string(self.device.type), compute_units=self.device.max_compute_units,
            clock_mhz=self.device.max_clock_frequency, local_mem_kb=self.device.local_mem_size // 1024,
            max_work_group=self.device.max_work_group_size,
            subgroups="cl_khr_subgroups" in exts or "cl_intel_subgroups" in exts,
            peak_gflops=peak_gflops(self.device),
        )

    def build(self, filename: str, **defines: int) -> cl.Program:
        opts = [f"-D{k}={v}" for k, v in defines.items()]
        if self.info.subgroups:
            opts.append("-DHAS_SUBGROUPS=1")
        src = (KERNELS / filename).read_text()
        return cl.Program(self.ctx, src).build(options=opts + ["-cl-std=CL2.0"] if self.info.subgroups else opts)

    def buffer(self, array: np.ndarray | None = None, nbytes: int | None = None) -> cl.Buffer:
        mf = cl.mem_flags
        if array is not None:
            return cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=array)
        return cl.Buffer(self.ctx, mf.READ_WRITE, nbytes)

    def read(self, buf: cl.Buffer, shape, dtype=np.float32) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.queue, out, buf).wait()
        return out

    def time(self, kernel: cl.Kernel, global_size, local_size, args, repeats: int = 20, warmup: int = 3) -> float:
        """Median kernel time in milliseconds, measured on the device with profiling events."""
        kernel.set_args(*args)
        for _ in range(warmup):
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size).wait()
        times = []
        for _ in range(repeats):
            ev = cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            ev.wait()
            times.append((ev.profile.end - ev.profile.start) * 1e-6)
        return statistics.median(times)
