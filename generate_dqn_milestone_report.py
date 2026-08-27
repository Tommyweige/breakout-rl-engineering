"""Generate the Day 15 milestone report from evaluation JSON artifacts."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Mapping, Sequence

from breakout_rl.evaluation import read_evaluation_results


def _display_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _episodes(payload: Mapping[str, Any], *, source: Path) -> list[dict[str, Any]]:
    raw_episodes = payload.get("per_episode")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{source}: per_episode must be a non-empty array")
    result: list[dict[str, Any]] = []
    for raw_episode in raw_episodes:
        if not isinstance(raw_episode, Mapping):
            raise ValueError(f"{source}: every episode must be an object")
        raw_return = raw_episode.get("episode_return")
        if raw_return is None:
            raw_return = raw_episode.get("return")
        if raw_return is None:
            raise ValueError(f"{source}: episode is missing episode_return")
        result.append(
            {
                "evaluation_seed": int(raw_episode["evaluation_seed"]),
                "episode_index": int(raw_episode["episode_index"]),
                "episode_seed": int(
                    raw_episode.get("episode_seed", raw_episode.get("seed"))
                ),
                "episode_return": float(raw_return),
                "episode_length": int(raw_episode["episode_length"]),
                "evaluator_truncated": bool(raw_episode.get("evaluator_truncated", False)),
                "complete": bool(raw_episode.get("complete", not raw_episode.get("evaluator_truncated", False))),
                "stop_reason": str(raw_episode.get("stop_reason", "unknown")),
            }
        )
    return result


def _summary(payload: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    values = [float(episode["episode_return"]) for episode in episodes]
    lengths = [int(episode["episode_length"]) for episode in episodes]
    raw_summary = payload.get("summary")
    if isinstance(raw_summary, Mapping):
        required = (
            "mean_return",
            "median_return",
            "std_return",
            "min_return",
            "max_return",
        )
        if all(key in raw_summary for key in required):
            return {
                "count": int(raw_summary.get("count", len(values))),
                "mean_return": float(raw_summary["mean_return"]),
                "median_return": float(raw_summary["median_return"]),
                "std_return": float(raw_summary["std_return"]),
                "min_return": float(raw_summary["min_return"]),
                "max_return": float(raw_summary["max_return"]),
                "mean_episode_length": float(
                    raw_summary.get("mean_episode_length", fmean(lengths))
                ),
                "complete_episodes": int(
                    raw_summary.get(
                        "complete_episodes",
                        sum(bool(episode.get("complete")) for episode in episodes),
                    )
                ),
                "evaluator_truncated_episodes": int(
                    raw_summary.get(
                        "evaluator_truncated_episodes",
                        sum(
                            bool(episode.get("evaluator_truncated"))
                            for episode in episodes
                        ),
                    )
                ),
            }
    complete_values = [
        float(episode["episode_return"])
        for episode in episodes
        if bool(episode.get("complete"))
    ]
    return {
        "count": len(values),
        "mean_return": float(fmean(values)),
        "median_return": float(median(values)),
        "std_return": float(pstdev(values)),
        "min_return": float(min(values)),
        "max_return": float(max(values)),
        "mean_episode_length": float(fmean(lengths)),
        "complete_episodes": len(complete_values),
        "evaluator_truncated_episodes": sum(
            bool(episode.get("evaluator_truncated")) for episode in episodes
        ),
    }


def _group_by_seed(episodes: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        groups.setdefault(int(episode["evaluation_seed"]), []).append(episode)
    return groups


def classify_comparison(
    random_summary: Mapping[str, Any],
    dqn_summary: Mapping[str, Any],
    *,
    evaluator_truncated_episodes: int = 0,
) -> str:
    """Classify the observed sample without pretending to run a significance test."""

    random_mean = float(random_summary["mean_return"])
    dqn_mean = float(dqn_summary["mean_return"])
    random_median = float(random_summary["median_return"])
    dqn_median = float(dqn_summary["median_return"])
    delta = dqn_mean - random_mean
    spread = max(float(random_summary["std_return"]), float(dqn_summary["std_return"]))
    if evaluator_truncated_episodes > 0:
        return "B" if delta > 0 else "C"
    if dqn_mean > random_mean and dqn_median > random_median:
        return "A" if delta > max(1.0, spread) else "B"
    if dqn_mean < random_mean and dqn_median <= random_median:
        return "D"
    if abs(delta) <= max(1.0, spread * 0.25):
        return "C"
    return "B" if delta > 0 else "D"


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    parsed = float(value)
    if not math.isfinite(parsed):
        return "—"
    return f"{parsed:.{digits}f}"


def _format_int(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def _format_rationale(value: Any) -> str:
    if isinstance(value, Mapping):
        return "；".join(str(item) for item in value.values())
    if value is None:
        return "—"
    return str(value).replace("|", "/")


def _runtime_value(payload: Mapping[str, Any], key: str) -> Any:
    runtime = payload.get("runtime")
    return runtime.get(key) if isinstance(runtime, Mapping) else None


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
    dqn_resolved_device = str(dqn_payload.get("resolved_device", ""))
    if require_cuda and not dqn_resolved_device.startswith("cuda:"):
        raise ValueError(
            "formal Day 15 report requires DQN results resolved on CUDA; "
            f"got {dqn_resolved_device or 'missing device metadata'}"
        )
    if float(dqn_payload.get("evaluation_epsilon", 1.0)) != 0.0:
        raise ValueError("formal Day 15 DQN report requires evaluation epsilon=0")

    random_episodes = _episodes(random_payload, source=random_source)
    dqn_episodes = _episodes(dqn_payload, source=dqn_source)
    random_summary = _summary(random_payload, random_episodes)
    dqn_summary = _summary(dqn_payload, dqn_episodes)
    dqn_training = dqn_payload.get("training", {})
    if not isinstance(dqn_training, Mapping):
        dqn_training = {}
    dqn_checkpoint = dqn_payload.get("checkpoint", {})
    if not isinstance(dqn_checkpoint, Mapping):
        dqn_checkpoint = {}
    dqn_trainer_runtime = dqn_training.get("trainer_runtime", {})
    if not isinstance(dqn_trainer_runtime, Mapping):
        dqn_trainer_runtime = {}
    dqn_gpu_profile = dqn_training.get("gpu_profiling_summary", {})
    if not isinstance(dqn_gpu_profile, Mapping):
        dqn_gpu_profile = {}
    selected_profiling = dqn_gpu_profile.get("profiling", {})
    if not isinstance(selected_profiling, Mapping):
        selected_profiling = {}
    dqn_environment = dqn_payload.get("environment", {})
    if not isinstance(dqn_environment, Mapping):
        dqn_environment = {}
    evaluation_seeds = dqn_payload.get("evaluation_seeds", [])
    total_episodes = len(dqn_episodes)
    delta = float(dqn_summary["mean_return"]) - float(random_summary["mean_return"])
    random_groups = _group_by_seed(random_episodes)
    dqn_groups = _group_by_seed(dqn_episodes)
    dqn_evaluator_truncated = sum(
        bool(episode.get("evaluator_truncated")) for episode in dqn_episodes
    )
    classification = classify_comparison(
        random_summary,
        dqn_summary,
        evaluator_truncated_episodes=dqn_evaluator_truncated,
    )
    if dqn_evaluator_truncated:
        classification_text = (
            "DQN 觀察平均較高，但有 evaluator cap 的局，尚不能當作完整終局的穩定優勢"
        )
    else:
        classification_text = {
            "A": "DQN 明顯高於 Random",
            "B": "DQN 稍高，但 spread 仍有重疊",
            "C": "DQN 與 Random 的差異不明顯",
            "D": "DQN 低於 Random",
        }[classification]

    lines = [
        "# Day 15｜DQN milestone evaluation",
        "",
        "## 先看結論",
        "",
        f"在固定的 evaluation protocol 下，本次結果分類為 **{classification}：{classification_text}**。"
        "這個分類只描述本次已收集的 episode samples，不是多個 training seeds 的統計顯著性檢定。",
        "",
        f"Random 的平均 raw Atari return 是 **{_format_number(random_summary['mean_return'])}**，"
        f"DQN 是 **{_format_number(dqn_summary['mean_return'])}**，平均差值為 **{_format_number(delta)}**；"
        f"中位數則是 {_format_number(random_summary['median_return'])} 對 {_format_number(dqn_summary['median_return'])}。",
        "這裡的 raw reward 是環境回傳的原始遊戲分數，沒有沿用訓練時的 reward clipping。",
        "",
        "## Day 14 的學習訊號，為什麼還不算 milestone",
        "",
        "Day 14 的 100K 曲線與單局 GIF 是 development evidence：它們讓我們知道訓練流程出現值得追蹤的訊號。"
        "正式 milestone 要回答的是同一個凍結 checkpoint 在沒有更新權重、沒有寫 Replay Buffer、"
        "沒有繼續套用 training epsilon schedule 的情況下，能否在一組事先固定的環境隨機性中穩定行動。",
        "因此本次使用 Day 14 final 100K checkpoint；checkpoint rule 在看 evaluation 結果前就已固定，"
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
        f"| checkpoint step | {_format_int(dqn_checkpoint.get('step'))} |",
        f"| training seed | {dqn_training.get('training_seed', '—')} |",
        f"| training budget | {_format_int(dqn_training.get('training_budget'))} environment steps |",
        f"| learning rate | `{dqn_training.get('learning_rate', '—')}` |",
        f"| batch size | {dqn_training.get('batch_size', '—')} |",
        f"| replay backend | `{dqn_training.get('replay_backend', '—')}` |",
        f"| selection rule | {dqn_training.get('selection_rule', '—')} |",
        f"| Day 14 trainer PyTorch / CUDA | `{dqn_trainer_runtime.get('pytorch_version', '—')}` / `{dqn_trainer_runtime.get('torch_cuda_version', '—')}` |",
        f"| Day 14 trainer commit | `{dqn_trainer_runtime.get('git_commit_sha', '—')}` |",
        f"| GPU profiling source | `{dqn_gpu_profile.get('source', '—')}` |",
        f"| selected batch end-to-end SPS | {_format_number(dqn_gpu_profile.get('end_to_end_sps'))} |",
        f"| selected GPU utilization mean | {_format_number((selected_profiling.get('gpu_utilization_percent') or {}).get('mean') if isinstance(selected_profiling.get('gpu_utilization_percent'), Mapping) else None)}% |",
        f"| selection rationale | {_format_rationale(dqn_training.get('selection_rationale'))} |",
        "",
        "## 固定的評估規則",
        "",
        "評估時的 policy 是已凍結的行動規則；它只接收 observation，選一個合法 action，"
        "再把環境回傳的 raw reward 累積到該局結束。Random 與 DQN 共用 environment construction、"
        "seed handling、episode loop、terminated/truncated 判斷與輸出 schema。",
        "",
        "| 規則 | 值 |",
        "|---|---|",
        f"| environment | `{dqn_payload.get('environment_id', dqn_environment.get('id', '—'))}` |",
        f"| observation shape | `{dqn_environment.get('observation_shape', dqn_payload.get('observation_shape', '—'))}` |",
        f"| action count | {dqn_payload.get('action_count', '—')} |",
        f"| evaluation seed groups | `{evaluation_seeds}` |",
        f"| episodes per seed group | {dqn_payload.get('episodes_per_seed', '—')} |",
        f"| total episodes per policy | {total_episodes} |",
        f"| DQN epsilon | {dqn_payload.get('evaluation_epsilon', '—')} (greedy) |",
        "| score | raw Atari reward; no training reward clipping |",
        f"| DQN requested device | `{dqn_payload.get('requested_device', '—')}` |",
        f"| DQN resolved device | `{dqn_resolved_device}` |",
        f"| GPU | `{_runtime_value(dqn_payload, 'gpu_model') or '—'}` |",
        f"| PyTorch / CUDA | `{_runtime_value(dqn_payload, 'pytorch_version') or '—'}` / `{_runtime_value(dqn_payload, 'torch_cuda_version') or '—'}` |",
        "",
        "DQN 的模型推論確實在 NVIDIA CUDA 上執行；Random 沒有神經網路推論，因此使用 CPU 不構成效能比較。"
        "本報告比較的是遊戲回報，不是兩者的 latency 或 steps per second。",
        "",
        "## 不只看平均：每局結果與 spread",
        "",
        "平均值適合描述整批樣本的中心，但它可能被少數高分或低分拉動。中位數是排序後位於中間的回報，"
        "比較不容易被極端局影響；std 是每局回報離平均值的典型距離，這裡使用 population standard deviation"
        "（把本次完整收集的 episode 當成要描述的整體）來呈現 spread。",
        "",
        "| Policy | N | complete | evaluator cap | mean | median | std | min | max | mean episode length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Random | {random_summary['count']} | {random_summary['complete_episodes']} | {random_summary['evaluator_truncated_episodes']} | {_format_number(random_summary['mean_return'])} | {_format_number(random_summary['median_return'])} | {_format_number(random_summary['std_return'])} | {_format_number(random_summary['min_return'])} | {_format_number(random_summary['max_return'])} | {_format_number(random_summary['mean_episode_length'])} |",
        f"| DQN | {dqn_summary['count']} | {dqn_summary['complete_episodes']} | {dqn_summary['evaluator_truncated_episodes']} | {_format_number(dqn_summary['mean_return'])} | {_format_number(dqn_summary['median_return'])} | {_format_number(dqn_summary['std_return'])} | {_format_number(dqn_summary['min_return'])} | {_format_number(dqn_summary['max_return'])} | {_format_number(dqn_summary['mean_episode_length'])} |",
        "",
        "下圖的每一個點都是 evaluation JSON 裡的一局，不是由平均值反推的資料；箱型圖保留中間分布，"
        "菱形是 mean，短線是 median。右圖再把同一個 evaluation seed group 的平均與 spread 分開，"
        "方便觀察結果是否只由某一組 seed 推高。",
        "",
        "[![Random 與凍結 DQN 的每局回報分布，以及各 evaluation seed group 的平均與 spread](../assets/day15/random-vs-dqn-returns.png)](../assets/day15/random-vs-dqn-returns.png)",
        "",
        "這張圖支持的結論是「本次 samples 中兩個 policy 的回報中心與分布如何重疊」，"
        "不能單獨證明 DQN 對所有未來 Breakout 起始狀態都更好，也不能把 Random CPU 與 DQN CUDA 的執行時間拿來比較。",
        "",
        "### Per-seed / per-episode raw returns",
        "",
        "| Evaluation seed group | Episode | Concrete reset seed | Random return | DQN return | DQN stop |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for seed in evaluation_seeds:
        seed_int = int(seed)
        random_by_index = {
            int(row["episode_index"]): row for row in random_groups.get(seed_int, [])
        }
        dqn_by_index = {
            int(row["episode_index"]): row for row in dqn_groups.get(seed_int, [])
        }
        episode_indexes = sorted(set(random_by_index) | set(dqn_by_index))
        for episode_index in episode_indexes:
            random_row = random_by_index.get(episode_index, {})
            dqn_row = dqn_by_index.get(episode_index, {})
            concrete_seed = dqn_row.get("episode_seed", random_row.get("episode_seed", "—"))
            lines.append(
                f"| {seed_int} | {episode_index} | {concrete_seed} | "
                f"{_format_number(random_row.get('episode_return'))} | "
                f"{_format_number(dqn_row.get('episode_return'))} | "
                f"{dqn_row.get('stop_reason', '—')} |"
            )

    lines.extend(
        [
            "",
            "## 這次結果能說到哪裡",
            "",
            f"本次固定協議得到的觀察是：DQN 平均回報比 Random {'高' if delta > 0 else '低' if delta < 0 else '相同'} "
            f"{_format_number(abs(delta))} 分，中位數則{'較高' if float(dqn_summary['median_return']) > float(random_summary['median_return']) else '較低' if float(dqn_summary['median_return']) < float(random_summary['median_return']) else '相同'}。"
            f"分類 **{classification}** 的門檻只是透明的描述性規則，不能取代更多 training seeds 或正式統計檢定。",
            "",
            f"DQN 有 **{dqn_evaluator_truncated} / {len(dqn_episodes)}** 局在 evaluator 的 "
            f"{_format_int(dqn_payload.get('max_steps_per_episode'))} 步安全上限停止；這些回報是到上限為止的累積值，"
            "不是環境自然終局的完整 return。因此本次雖然觀察到 DQN 的中心值較高，仍不能把它寫成「所有完整 episode 都穩定超過 Random」。",
            "",
            "目前仍只有一個 training seed（Day 14 seed 42）。training seed 影響模型初始化、探索與 Replay 抽樣；"
            "evaluation seed 則控制凍結 policy 在環境中遇到的隨機性。這次使用 101、202、303 三個獨立 seed group，"
            "所以它能檢查多局表現，卻還不能宣稱 multi-training-seed robustness。",
            "",
            "如果 DQN 沒有超過 Random，正確解讀是這個 100K candidate 尚未通過正式 milestone，而不是立刻斷言 CNN 容量不足；"
            "下一步應保留 evaluation evidence，再用更長的 training horizon、多個 training seeds 或後續演算法比較釐清原因。",
            "",
            "## Day 16 的品質基準",
            "",
            "Day 16 會把單一環境訓練改成多環境、批次 action inference 與批次 GPU Replay insertion。"
            "這套 evaluation contract 不隨 systems optimization 改變：仍使用相同 seeds、每個 seed 的 episodes、"
            "epsilon=0、raw reward、preprocessing、terminated/truncated semantics 與 result schema。"
            "Day 16 要重新回答的不是「速度變快了嗎」而已，而是「速度改造有沒有讓 policy quality 明顯退步」。",
            "",
            "### 可重建的 artifacts",
            "",
            f"- Random JSON：`{_display_path(random_source)}`",
            f"- DQN JSON：`{_display_path(dqn_source)}`",
            f"- Random CSV：`{_display_path(random_source.parent / 'episodes.csv')}`",
            f"- DQN CSV：`{_display_path(dqn_source.parent / 'episodes.csv')}`",
            "- 圖表由 `visualize_day15_evaluation.py` 從兩份 JSON 重新產生。",
            "- 結果由 `evaluate_dqn.py` 使用 `configs/eval/breakout_eval.json` 產生；正式 DQN 命令指定 `--device cuda`。",
            "",
            f"Report generated at `{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}`。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Day 15 DQN milestone report from real results."
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
        help="allow a CPU DQN result for a portability/reference report",
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
