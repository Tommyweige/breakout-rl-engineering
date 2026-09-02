"""Inspect Dueling Value/Advantage/Q components on a real Contract v2 state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from breakout_rl.analysis.dueling import (
    DEFAULT_CONTRACT,
    DuelingInspection,
    collect_dueling_inspection,
    inspection_payload,
)


def print_inspection(inspection: DuelingInspection, *, seed: int) -> None:
    payload = inspection_payload(inspection, seed=seed)
    print(f"Device                       : {payload['device']}")
    print(
        "Input shape                  : "
        f"{tuple(payload['model_input_shape'])}"
    )
    print(f"Feature shape                : {tuple(payload['feature_shape'])}")
    print(
        "V(s) shape / value           : "
        f"{tuple(payload['value_shape'])} / {payload['value']:+.6f}"
    )
    print(
        "A(s,a) shape / values       : "
        f"{tuple(payload['advantage_shape'])} / "
        + str([round(value, 6) for value in payload["raw_advantage"]])
    )
    print(f"mean(A)                      : {payload['mean_advantage']:+.6f}")
    print(
        "mean-centered A(s,a)         : "
        + str([round(value, 6) for value in payload["centered_advantage"]])
    )
    print(
        "Q-values                     : "
        + str([round(value, 6) for value in payload["q_values"]])
    )
    print(
        "Reconstruction max abs error : "
        f"{payload['reconstruction_max_abs_error']:.10f}"
    )
    print(
        "Argmax action                : "
        f"{payload['argmax_action_index']} "
        f"({payload['argmax_action_meaning']})"
    )
    print(f"Parameter count              : {payload['model_config']['parameter_count']:,}")
    print(f"Interpretation               : {payload['trained_policy_claim']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inspection = collect_dueling_inspection(
        seed=args.seed,
        device_name=args.device,
        contract_path=args.contract,
        checkpoint=args.checkpoint,
    )
    print_inspection(inspection, seed=args.seed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = inspection_payload(inspection, seed=args.seed)
        payload.update(
            {
                "generation_command": (
                    "python -m scripts.analysis.inspect_dueling_network "
                    f"--device {args.device} --seed {args.seed} "
                    f"--contract {args.contract.as_posix()}"
                    + (
                        f" --checkpoint {args.checkpoint.as_posix()}"
                        if args.checkpoint is not None
                        else ""
                    )
                ),
                "source_script": "scripts/analysis/inspect_dueling_network.py",
            }
        )
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved inspection metadata: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
