"""Command line tool.

    python -m gkp info                       show the selected OpenCL device and its peak numbers
    python -m gkp run                        run every suite and write results/ (JSON, markdown, roofline)
    python -m gkp run --suite matmul --quick small sizes, for a fast check
    GKP_DEVICE=cpu python -m gkp run         pick a device by name
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import write_report
from .runtime import Runtime
from .suites import QUICK, SUITES

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(prog="gkp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    run = sub.add_parser("run")
    run.add_argument("--suite", choices=list(SUITES), action="append")
    run.add_argument("--quick", action="store_true")
    run.add_argument("--device")
    run.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    rt = Runtime(getattr(args, "device", None))
    print(rt.info.summary())
    if args.cmd == "info":
        return
    results = []
    for name in args.suite or list(SUITES):
        kwargs = QUICK[name] if args.quick else {}
        for r in SUITES[name](rt, **kwargs):
            results.append(r)
            print(f"{r.suite:7s} {r.variant:32s} {r.ms:9.3f} ms  {r.gbps:7.1f} GB/s  {r.gflops:8.1f} GFLOP/s  "
                  f"{'ok' if r.correct else 'WRONG'} (rel err {r.max_rel_error:.1e})")
    if args.quick:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(
        {"device": rt.info.__dict__, "results": [r.to_json() for r in results]}, indent=2))
    print(write_report(rt.info, results, args.out))
    if not all(r.correct for r in results):
        raise SystemExit("some kernels produced wrong results")


if __name__ == "__main__":
    main()
