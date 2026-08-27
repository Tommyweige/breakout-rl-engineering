"""Generate a Day 15 milestone report from raw evaluation JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from breakout_rl.evaluation_artifacts import (
    read_evaluation_results,
    summary_from_episode_rows,
    validate_embedded_summary,
    validate_episode_rows,
)


DAY15_FIGURE_BLOB_URL = (
    "https://github.com/Tommyweige/breakout-rl-engineering-private/blob/"
    "3f599ba/assets/day15/random-vs-dqn-returns.png"
)


def _group_by_seed(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["evaluation_seed"]), []).append(row)
    return groups


def classify_comparison(random_summary: Mapping[str, Any], dqn_summary: Mapping[str, Any]) -> str:
    """Describe observed samples; this is not a statistical significance test."""

    random_mean = float(random_summary["mean_return"])
    dqn_mean = float(dqn_summary["mean_return"])
    random_median = float(random_summary["median_return"])
    dqn_median = float(dqn_summary["median_return"])
    if dqn_mean > random_mean and dqn_median > random_median:
        if float(dqn_summary["min_return"]) > float(random_summary["max_return"]):
            return "A"
        return "B"
    if dqn_mean < random_mean and dqn_median <= random_median:
        return "D"
    if abs(dqn_mean - random_mean) <= max(
        1.0,
        0.25 * max(float(random_summary["std_return"]), float(dqn_summary["std_return"])),
    ):
        return "C"
    return "B" if dqn_mean > random_mean else "D"


def _number(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "—"
    parsed = float(value)
    return "—" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def _integer(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rationale(value: Any) -> str:
    if isinstance(value, Mapping):
        return "；".join(str(item) for item in value.values())
    return "—" if value is None else str(value).replace("|", "/")


def _path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(
    random_results_path: str | Path,
    dqn_results_path: str | Path,
    *,
    require_cuda: bool = True,
) -> str:
    random_source = Path(random_results_path)
    dqn_source = Path(dqn_results_path)
    random_payload = read_evaluation_results(random_source)
    dqn_payload = read_evaluation_results(dqn_source)
    if random_payload.get("policy_type") != "random":
        raise ValueError(f"{random_source}: expected policy_type=random")
    if dqn_payload.get("policy_type") != "dqn":
        raise ValueError(f"{dqn_source}: expected policy_type=dqn")
    if random_payload.get("evaluation_seeds") != dqn_payload.get("evaluation_seeds"):
        raise ValueError("Random and DQN results must use the same evaluation seeds")
    if random_payload.get("episodes_per_seed") != dqn_payload.get("episodes_per_seed"):
        raise ValueError("Random and DQN results must use the same episodes_per_seed")
    raw_seeds = dqn_payload.get("evaluation_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("evaluation_seeds must be a non-empty array")
    try:
        seeds = tuple(int(seed) for seed in raw_seeds)
        episodes_per_seed = int(dqn_payload["episodes_per_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation protocol contains invalid seeds or episode count") from error
    expected_episodes = len(seeds) * episodes_per_seed
    random_episodes = validate_episode_rows(
        random_payload,
        source=random_source,
        expected_seeds=seeds,
        expected_episodes_per_seed=episodes_per_seed,
        require_complete=True,
    )
    dqn_episodes = validate_episode_rows(
        dqn_payload,
        source=dqn_source,
        expected_seeds=seeds,
        expected_episodes_per_seed=episodes_per_seed,
        require_complete=True,
    )
    random_summary = summary_from_episode_rows(random_episodes)
    dqn_summary = summary_from_episode_rows(dqn_episodes)
    validate_embedded_summary(random_payload, random_summary, source=random_source)
    validate_embedded_summary(dqn_payload, dqn_summary, source=dqn_source)

    dqn_training = _mapping(dqn_payload.get("training"))
    dqn_gate = _mapping(dqn_training.get("day14_gate"))
    if require_cuda and dqn_gate.get("status") != "passed":
        raise ValueError(
            "formal Day 15 report requires Day 14 Gate A to be passed; "
            f"got {dqn_gate.get('status', 'missing')}"
        )
    dqn_resolved_device = str(dqn_payload.get("resolved_device", ""))
    if require_cuda and not dqn_resolved_device.startswith("cuda:"):
        raise ValueError(
            "formal Day 15 report requires DQN results resolved on CUDA; "
            f"got {dqn_resolved_device or 'missing device metadata'}"
        )
    if float(dqn_payload.get("evaluation_epsilon", 1.0)) != 0.0:
        raise ValueError("formal Day 15 DQN report requires evaluation epsilon=0")

    classification = classify_comparison(random_summary, dqn_summary)
    classification_text = {
        "A": "DQN 明顯高於 Random",
        "B": "DQN 中心值較高，但分布仍有重疊",
        "C": "DQN 與 Random 的差異不明顯",
        "D": "DQN 低於 Random",
    }[classification]
    dqn_checkpoint = _mapping(dqn_payload.get("checkpoint"))
    dqn_environment = _mapping(dqn_payload.get("environment"))
    trainer_runtime = _mapping(dqn_training.get("trainer_runtime"))
    gpu_profile = _mapping(dqn_training.get("gpu_profiling_summary"))
    profile_details = _mapping(gpu_profile.get("profiling"))
    gate_criteria = _mapping(dqn_gate.get("criteria"))
    gate_returns = _mapping(dqn_gate.get("return_signal"))
    gate_diagnostics = _mapping(dqn_gate.get("diagnostics"))
    delta = float(dqn_summary["mean_return"]) - float(random_summary["mean_return"])
    random_groups = _group_by_seed(random_episodes)
    dqn_groups = _group_by_seed(dqn_episodes)
    seeds = list(seeds)

    lines = [
        "# Day 15｜DQN milestone evaluation",
        "",
        "## 先看結論",
        "",
        f"在固定的 evaluation protocol 下，本次結果分類為 **{classification}：{classification_text}**。"
        "這個分類只描述本次已收集的完整 episode samples，不是多個 training seeds 的統計顯著性檢定。",
        "",
        f"Random 的平均 raw Atari return 是 **{_number(random_summary['mean_return'])}**，"
        f"DQN 是 **{_number(dqn_summary['mean_return'])}**，平均差值為 **{_number(delta)}**；"
        f"中位數則是 {_number(random_summary['median_return'])} 對 {_number(dqn_summary['median_return'])}。",
        "這裡的 raw reward 是環境回傳的原始遊戲分數，不是訓練時可能使用的 clipped reward。",
        "",
        "## Day 14 的學習訊號，為什麼還不算正式評估",
        "",
        "Day 14 的 100K 曲線和單局 GIF 是 development evidence：它們顯示模型值得再驗證。"
        "Day 15 則把 final 100K checkpoint 凍結，用獨立 seeds 和多局完整 rollout 檢查這個訊號能否重現。"
        "checkpoint selection rule 在 evaluation 前固定為完成 100,000 個環境步數的 final checkpoint，"
        "沒有從 50K、75K、100K 中挑最好的一局。",
        "",
        "### Day 14 candidate provenance",
        "",
        "| 欄位 | 實際值 |",
        "|---|---|",
        f"| source manifest | `{dqn_training.get('source_day14_manifest', '—')}` |",
        f"| source run | `{dqn_training.get('source_day14_run_id', '—')}` |",
        f"| checkpoint | `{dqn_checkpoint.get('path', '—')}` |",
        f"| checkpoint SHA-256 | `{dqn_checkpoint.get('sha256', '—')}` |",
        f"| checkpoint step | {_integer(dqn_checkpoint.get('step'))} |",
        f"| training seed | {dqn_training.get('training_seed', '—')} |",
        f"| training budget | {_integer(dqn_training.get('training_budget'))} environment steps |",
        f"| learning rate | `{dqn_training.get('learning_rate', '—')}` |",
        f"| batch size | {dqn_training.get('batch_size', '—')} |",
        f"| train frequency | {dqn_training.get('train_frequency', '—')} |",
        f"| replay backend | `{dqn_training.get('replay_backend', '—')}` |",
        f"| selection rule | {dqn_training.get('selection_rule', '—')} |",
        f"| Day 14 trainer PyTorch / CUDA | `{trainer_runtime.get('pytorch_version', '—')}` / `{trainer_runtime.get('torch_cuda_version', '—')}` |",
        f"| Day 14 trainer commit | `{trainer_runtime.get('git_commit_sha', '—')}` |",
        f"| GPU profiling source | `{gpu_profile.get('source', '—')}` |",
        f"| selected batch end-to-end SPS | {_number(gpu_profile.get('end_to_end_sps'))} |",
        f"| selected GPU utilization mean | {_number(profile_details.get('gpu_utilization_percent', {}).get('mean') if isinstance(profile_details.get('gpu_utilization_percent'), Mapping) else None)}% |",
        f"| selection rationale | {_rationale(dqn_training.get('selection_rationale'))} |",
        "",
        "### Day 14 Gate A evidence",
        "",
        "正式 milestone 不是因為 checkpoint 檔案存在就自動成立；這裡把 Day 14 的長跑結果與選擇依據明確列出。",
        "",
        "| Gate A 條件 | 實際證據 |",
        "|---|---|",
        f"| status | `{dqn_gate.get('status', '—')}` |",
        f"| 10K reference recent mean | {_number(gate_returns.get('reference_10k_recent_mean'))} ({gate_returns.get('reference_recent_episode_count', '—')} episodes) |",
        f"| 100K final recent mean | {_number(gate_returns.get('final_100k_recent_mean'))} |",
        f"| return signal | `{gate_criteria.get('return_signal', '—')}` |",
        f"| diagnostics healthy | `{gate_criteria.get('diagnostics_healthy', '—')}` |",
        f"| selection not single best episode | `{gate_criteria.get('selection_not_single_best_episode', '—')}` |",
        f"| checkpoint provenance complete | `{gate_criteria.get('checkpoint_provenance_complete', '—')}` |",
        f"| metrics diagnostics finite | `{gate_diagnostics.get('metrics_values_finite', '—')}` |",
        "",
        "## 固定的評估規則",
        "",
        "評估只讓 policy 讀取 observation、選 action，再把 raw reward 累積到該局結束。"
        "Random 與 DQN 共用 environment construction、seed handling、episode loop、"
        "terminated/truncated 判斷、統計與輸出 schema；差別只有 action 如何產生。",
        "",
        "| 規則 | 值 |",
        "|---|---|",
        f"| environment | `{dqn_payload.get('environment_id', dqn_environment.get('id', '—'))}` |",
        f"| observation shape | `{dqn_environment.get('observation_shape', dqn_payload.get('observation_shape', '—'))}` |",
        f"| action count | {dqn_payload.get('action_count', '—')} |",
        f"| evaluation seed groups | `{seeds}` |",
        f"| episodes per seed group | {dqn_payload.get('episodes_per_seed', '—')} |",
        f"| total episodes per policy | {expected_episodes} |",
        f"| DQN epsilon | {dqn_payload.get('evaluation_epsilon', '—')} (greedy) |",
        "| score | raw Atari reward; no training reward clipping |",
        f"| DQN requested / resolved device | `{dqn_payload.get('requested_device', '—')}` / `{dqn_resolved_device}` |",
        f"| GPU | `{_mapping(dqn_payload.get('runtime')).get('gpu_model', '—')}` |",
        f"| PyTorch / CUDA | `{_mapping(dqn_payload.get('runtime')).get('pytorch_version', '—')}` / `{_mapping(dqn_payload.get('runtime')).get('torch_cuda_version', '—')}` |",
        "",
        "DQN 的模型推論確實在 NVIDIA CUDA 上執行；Random 沒有 neural-network inference，"
        "所以它留在 CPU，不把兩者的 runtime 當成效能比較。本報告只比較遊戲回報。",
        "",
        "## 不只看平均：每局結果和 spread",
        "",
        "平均值描述整批樣本的中心；中位數比較不容易被極端局拉動；std（標準差）則描述回報的 spread。"
        "每個 policy 的 15 局都由環境自然 terminated 或 truncated，沒有把 evaluator cap 混進正式結果。",
        "",
        "| Policy | N | complete | mean | median | std | min | max | mean episode length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Random | {random_summary['count']} | {random_summary['complete_episodes']} | {_number(random_summary['mean_return'])} | {_number(random_summary['median_return'])} | {_number(random_summary['std_return'])} | {_number(random_summary['min_return'])} | {_number(random_summary['max_return'])} | {_number(random_summary['mean_episode_length'])} |",
        f"| DQN | {dqn_summary['count']} | {dqn_summary['complete_episodes']} | {_number(dqn_summary['mean_return'])} | {_number(dqn_summary['median_return'])} | {_number(dqn_summary['std_return'])} | {_number(dqn_summary['min_return'])} | {_number(dqn_summary['max_return'])} | {_number(dqn_summary['mean_episode_length'])} |",
        "",
        "下圖的每個點都是 raw evaluation artifact 裡的一局；箱型圖顯示中間分布，"
        "菱形是 mean，短線是 median。右側則按 evaluation seed group 顯示平均與 spread，"
        "避免只看一個總平均。",
        "",
        f"[![Random 與凍結 DQN 的每局回報分布，以及各 evaluation seed group 的平均與 spread]({DAY15_FIGURE_BLOB_URL}?raw=1)]({DAY15_FIGURE_BLOB_URL})",
        "",
        "每個 seed group 的完整 raw return 如下：",
        "",
        "| Evaluation seed group | Episode | Concrete reset seed | Random return | DQN return |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        random_by_index = {
            int(row["episode_index"]): row for row in random_groups[int(seed)]
        }
        dqn_by_index = {
            int(row["episode_index"]): row for row in dqn_groups[int(seed)]
        }
        for episode_index in sorted(dqn_by_index):
            random_row = random_by_index[episode_index]
            dqn_row = dqn_by_index[episode_index]
            lines.append(
                f"| {seed} | {episode_index} | {dqn_row['episode_seed']} | "
                f"{_number(random_row['episode_return'])} | {_number(dqn_row['episode_return'])} |"
            )

    lines.extend(
        [
            "",
            "## 這次結果能說到哪裡",
            "",
            f"在這組固定條件下，DQN 平均回報{'高於' if delta > 0 else '低於' if delta < 0 else '等於'} Random "
            f"{_number(abs(delta))} 分，"
            f"中位數{'也較高' if dqn_summary['median_return'] > random_summary['median_return'] else '較低' if dqn_summary['median_return'] < random_summary['median_return'] else '相同'}。"
            f"這支持「Day 14 checkpoint 在這批獨立 evaluation episodes 中展現較高回報」；"
            "它不支持「所有未來起始狀態都會更好」或「已完成 multi-training-seed robustness」。",
            "",
            "目前正式驗證的仍是一個 training seed（42）訓練出的 checkpoint；evaluation seed 101、202、303"
            "只改變凍結 policy 面對的環境隨機性。後續若要談訓練穩定性，還需要多個 training seeds。",
            "",
            "## Day 16 的品質基準",
            "",
            "Day 16 會把 single-environment training 改成多環境、批次 action inference 和批次 GPU Replay insertion。"
            "它必須重用本日的 seeds、每組 episode 數、greedy epsilon、raw reward、environment contract、"
            "done semantics 和 result schema，才能分辨速度最佳化是否造成 policy quality regression。"
            "之後的 Double DQN 與 Dueling Network 也應沿用同一套評估尺。",
            "",
            "### 可重建的 artifacts",
            "",
            f"- Random JSON：`{_path(random_source)}`",
            f"- DQN JSON：`{_path(dqn_source)}`",
            f"- Random CSV：`{_path(random_source.parent / 'episodes.csv')}`",
            f"- DQN CSV：`{_path(dqn_source.parent / 'episodes.csv')}`",
            "- 圖表由 `visualize_day15_evaluation.py` 從兩份 JSON 重新產生。",
            "- 結果由 `evaluate_dqn.py` 使用 `configs/eval/breakout_eval.json` 產生；正式 DQN 命令指定 `--device cuda`。",
            "",
            f"Report generated at `{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}`。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Day 15 milestone report from raw results."
    )
    parser.add_argument("--random-results", type=Path, required=True)
    parser.add_argument("--dqn-results", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/day15-dqn-milestone.md"),
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="allow a CPU DQN artifact for a portability/reference report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(
            args.random_results,
            args.dqn_results,
            require_cuda=not args.allow_non_cuda,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"Unable to generate Day 15 report: {error}", file=sys.stderr)
        return 2
    print(f"Wrote Day 15 report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
