"""Every kernel variant must match NumPy. Runs on whatever OpenCL device exists: the Iris Xe GPU locally,
PoCL on the CPU in CI."""

from types import SimpleNamespace

import pytest

from gkp.report import write_report
from gkp.runtime import Runtime, peak_gflops
from gkp.suites import QUICK, SUITES, Result


@pytest.fixture(scope="module")
def rt():
    return Runtime()


@pytest.mark.parametrize("suite", list(SUITES))
def test_every_variant_is_correct(rt, suite):
    results = SUITES[suite](rt, **QUICK[suite], repeats=2)
    assert results, suite
    for r in results:
        assert r.correct, f"{r.suite} {r.variant}: relative error {r.max_rel_error}"
        assert r.ms > 0


def test_reduce_has_all_gpu_variants(rt):
    variants = [r.variant for r in SUITES["reduce"](rt, **QUICK["reduce"], repeats=1)]
    assert variants[:3] == ["v1 interleaved", "v2 sequential", "v3 grid stride"]
    assert variants[-1].startswith("NumPy")


def test_peak_formula_for_intel_xe():
    import pyopencl as cl

    dev = SimpleNamespace(type=cl.device_type.GPU, vendor="Intel(R) Corporation", max_compute_units=80,
                          max_clock_frequency=1200)
    assert peak_gflops(dev) == 1536.0
    dev.vendor = "Other"
    assert peak_gflops(dev) is None


def test_derived_metrics():
    r = Result("matmul", "x", "n", ms=2.0, bytes=4e6, flops=8e9, max_rel_error=0.0, correct=True)
    assert r.gflops == 4000.0 and r.gbps == 2.0 and r.intensity == 2000.0


def test_report_files(rt, tmp_path):
    results = SUITES["stream"](rt, **QUICK["stream"], repeats=2) + SUITES["fir"](rt, **QUICK["fir"], repeats=2)
    text = write_report(rt.info, results, tmp_path)
    assert "Measured memory bandwidth" in text
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "roofline.png").stat().st_size > 1000
