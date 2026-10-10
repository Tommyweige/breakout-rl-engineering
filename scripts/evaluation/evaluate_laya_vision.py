"""Run Issue 69's optional frozen Laya feasibility protocol."""

import argparse
from pathlib import Path

from breakout_rl.laya_evaluation import run_laya_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "seed101", "pilot", "full"), default="pilot")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--contract", type=Path, default=Path("configs/eval/breakout_contract_v2.json"))
    parser.add_argument("--baseline-onnx", type=Path, default=Path("web/public/models/final_model/model.onnx"))
    parser.add_argument("--max-pilot-hours", type=float, default=1.0)
    args = parser.parse_args()
    evidence = run_laya_evaluation(output_dir=args.output_dir, mode=args.mode, device=args.device,
                                   contract_path=args.contract, baseline_onnx=args.baseline_onnx,
                                   max_pilot_hours=args.max_pilot_hours)
    print(f"{evidence['status']}: {args.output_dir / 'report.md'}")
    return 1 if evidence["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
