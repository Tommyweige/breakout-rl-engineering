"""Record a fair real-ALE Breakout gameplay progression as MP4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from breakout_env import ENVIRONMENT_ID, make_breakout_env
from breakout_rl.evaluation_contract import (
    breakout_environment_kwargs,
    load_evaluation_contract,
    validate_breakout_runtime_contract,
)
from breakout_rl.models.factory import build_q_network, checkpoint_architecture
from breakout_rl.tensors import observation_to_tensor
from breakout_rl.training.diagnostics import ATARI_ACTION_NAMES
from breakout_rl.training.dqn_trainer import resolve_device


EXPECTED_EXPERIMENT_ID = "day21-gameplay-progression"
EXPECTED_ENVIRONMENT_ID = "ALE/Breakout-v5"
EXPECTED_LEARNED_SEED = 2022
EXPECTED_ALGORITHM = "double_dqn"
EXPECTED_ARCHITECTURE = "dueling"
EXPECTED_FINAL_MODEL_SHA256 = "6002029dcdbcbb7c93fca0c589880611aed2e2e7924db0f6b0c1f5160824389a"
EXPECTED_ACTION_MAPPING = {str(index): name for index, name in ATARI_ACTION_NAMES.items()}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _repository_root(config_path: Path) -> Path:
    source = config_path.resolve()
    for parent in source.parents:
        if (parent / ".git").exists():
            return parent
    return source.parents[2]


def _resolve_path(repository_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repository_root / candidate).resolve()


def _git_commit(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_status(repository_root: Path, *, exclude_path: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    lines = result.stdout.splitlines()
    if exclude_path is not None:
        try:
            excluded = exclude_path.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            excluded = None
        if excluded:
            lines = [
                line
                for line in lines
                if not line[3:].replace("\\", "/").startswith(excluded.rstrip("/") + "/")
                and line[3:].replace("\\", "/") != excluded.rstrip("/")
            ]
    return "\n".join(lines)


@dataclass(frozen=True)
class StageSpec:
    ordinal: int
    stage_id: str
    label: str
    policy: str
    training_seed: int | None
    algorithm: str | None
    architecture: str | None
    requested_transitions: int
    actual_checkpoint_step: int
    checkpoint: str | None
    checkpoint_sha256: str | None
    output: str
    source_run_id: str | None
    source_stage: str | None
    source_checkpoint: str | None
    source_checkpoint_sha256: str | None
    substitution_reason: str | None


@dataclass(frozen=True)
class GameplayProgressionConfig:
    source_path: Path
    repository_root: Path
    experiment_id: str
    canonical_output_dir: Path
    contract_path: Path
    showcase_evaluation_seed: int
    episodes: int
    max_steps_per_episode: int
    evaluation_epsilon: float
    video_fps: int
    capture_every_agent_step: int
    frame_repeat: int
    native_atari_fps: int
    max_width: int
    requested_device: str
    codec_preference: tuple[str, ...]
    pixel_format: str
    fairness: Mapping[str, Any]
    stages: tuple[StageSpec, ...]
    raw: Mapping[str, Any]

    @property
    def frame_size_seconds_per_agent_step(self) -> float:
        return self.frame_repeat / self.video_fps


def _parse_stage(payload: Mapping[str, Any], *, expected_ordinal: int) -> StageSpec:
    ordinal = int(payload.get("ordinal", -1))
    if ordinal != expected_ordinal:
        raise ValueError(f"stage ordinal {ordinal} is not {expected_ordinal}")
    stage_id = str(payload.get("id", "")).strip()
    label = str(payload.get("label", "")).strip()
    policy = str(payload.get("policy", "")).strip()
    output = str(payload.get("output", "")).strip()
    if not stage_id or not label or policy not in {"random", "checkpoint"}:
        raise ValueError("each stage needs a non-empty id/label and a known policy")
    if not output.endswith(".mp4") or Path(output).name != output:
        raise ValueError("stage outputs must be simple .mp4 filenames")
    checkpoint = payload.get("checkpoint")
    checkpoint_value = None if checkpoint is None else str(checkpoint)
    if policy == "random" and checkpoint_value is not None:
        raise ValueError("random baseline cannot have a checkpoint")
    if policy == "checkpoint" and not checkpoint_value:
        raise ValueError("checkpoint stages require a checkpoint path")
    checkpoint_hash = payload.get("checkpoint_sha256")
    checkpoint_hash_value = None if checkpoint_hash is None else str(checkpoint_hash).lower()
    if policy == "checkpoint" and (
        checkpoint_hash_value is None or len(checkpoint_hash_value) != 64
    ):
        raise ValueError("checkpoint stages require a 64-character SHA256")
    training_seed = payload.get("training_seed")
    training_seed_value = None if training_seed is None else int(training_seed)
    algorithm = payload.get("algorithm")
    architecture = payload.get("architecture")
    if policy == "checkpoint" and (
        training_seed_value != EXPECTED_LEARNED_SEED
        or algorithm != EXPECTED_ALGORITHM
        or architecture != EXPECTED_ARCHITECTURE
    ):
        raise ValueError("learned stages must use the frozen Day 21 seed-2022 model")
    source_checkpoint = payload.get("source_checkpoint")
    source_checkpoint_value = None if source_checkpoint is None else str(source_checkpoint)
    source_checkpoint_hash = payload.get("source_checkpoint_sha256")
    source_checkpoint_hash_value = (
        None if source_checkpoint_hash is None else str(source_checkpoint_hash).lower()
    )
    if source_checkpoint_value and (
        source_checkpoint_hash_value is None or len(source_checkpoint_hash_value) != 64
    ):
        raise ValueError("source_checkpoint requires a 64-character SHA256")
    return StageSpec(
        ordinal=ordinal,
        stage_id=stage_id,
        label=label,
        policy=policy,
        training_seed=training_seed_value,
        algorithm=None if algorithm is None else str(algorithm),
        architecture=None if architecture is None else str(architecture),
        requested_transitions=int(payload.get("requested_transitions", -1)),
        actual_checkpoint_step=int(payload.get("actual_checkpoint_step", -1)),
        checkpoint=checkpoint_value,
        checkpoint_sha256=checkpoint_hash_value,
        output=output,
        source_run_id=(
            None if payload.get("source_run_id") is None else str(payload["source_run_id"])
        ),
        source_stage=(
            None if payload.get("source_stage") is None else str(payload["source_stage"])
        ),
        source_checkpoint=source_checkpoint_value,
        source_checkpoint_sha256=source_checkpoint_hash_value,
        substitution_reason=(
            None
            if payload.get("substitution_reason") is None
            else str(payload["substitution_reason"])
        ),
    )


def load_gameplay_progression_config(
    path: str | Path = "configs/eval/gameplay-progression.json",
    *,
    repository_root: str | Path | None = None,
) -> GameplayProgressionConfig:
    source_path = Path(path).resolve()
    payload = _read_json(source_path)
    root = Path(repository_root).resolve() if repository_root else _repository_root(source_path)
    if payload.get("schema_version") != 1 or payload.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("unsupported gameplay progression config")
    output_dir_value = payload.get("canonical_output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        raise ValueError("canonical_output_dir must be non-empty")
    canonical_output_dir = _resolve_path(root, output_dir_value)
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("environment must be an object")
    contract_value = environment.get("contract_path")
    if not isinstance(contract_value, str) or not contract_value.strip():
        raise ValueError("environment.contract_path must be non-empty")
    contract_path = _resolve_path(root, contract_value)
    contract = load_evaluation_contract(contract_path)
    validate_breakout_runtime_contract(contract)
    if contract.environment_id != EXPECTED_ENVIRONMENT_ID:
        raise ValueError("showcase must use ALE/Breakout-v5")
    if environment.get("environment_id") != EXPECTED_ENVIRONMENT_ID:
        raise ValueError("environment.environment_id must match Contract v2")
    showcase_seed = int(environment.get("showcase_evaluation_seed", -1))
    episodes = int(environment.get("episodes", -1))
    max_steps = int(environment.get("max_steps_per_episode", -1))
    epsilon = float(environment.get("evaluation_epsilon", -1.0))
    fps = int(environment.get("video_fps", -1))
    capture_every = int(environment.get("capture_every_agent_step", -1))
    frame_repeat = int(environment.get("frame_repeat", -1))
    native_fps = int(environment.get("native_atari_fps", -1))
    max_width = int(environment.get("max_width", -1))
    requested_device = str(environment.get("requested_device", "")).strip().lower()
    codec_preference_value = environment.get("codec_preference")
    if isinstance(codec_preference_value, (str, bytes)) or not isinstance(codec_preference_value, Sequence):
        raise ValueError("environment.codec_preference must be a sequence")
    codec_preference = tuple(str(codec).strip() for codec in codec_preference_value)
    if (
        showcase_seed < 0
        or showcase_seed not in contract.concrete_episode_seeds
        or episodes != 1
        or max_steps != int(contract.time_limit_semantics["agent_step_limit"])
        or not math.isclose(epsilon, 0.0, abs_tol=0.0)
        or fps != 30
        or capture_every != 1
        or frame_repeat != 2
        or native_fps != 60
        or max_width < 1
        or requested_device != "cuda"
        or not codec_preference
        or any(codec not in {"avc1", "H264", "X264"} for codec in codec_preference)
        or environment.get("pixel_format") != "yuv420p"
    ):
        raise ValueError("gameplay progression recording settings are not fixed")
    if not math.isclose(
        frame_repeat / fps,
        contract.frame_skip / native_fps,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("frame_repeat/video_fps must preserve Contract v2 playback speed")
    fairness = payload.get("fairness")
    if not isinstance(fairness, Mapping):
        raise ValueError("fairness must be an object")
    inventory = payload.get("checkpoint_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("checkpoint_inventory must be an object")
    inventory_directory = inventory.get("checkpoints_directory")
    available_steps = inventory.get("available_checkpoint_steps")
    candidates = inventory.get("target_250k_candidates")
    if (
        not isinstance(inventory_directory, str)
        or not inventory_directory.strip()
        or isinstance(available_steps, (str, bytes))
        or not isinstance(available_steps, Sequence)
        or isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
    ):
        raise ValueError("checkpoint_inventory fields are incomplete")
    parsed_available_steps = [int(step) for step in available_steps]
    if parsed_available_steps != sorted(set(parsed_available_steps)):
        raise ValueError("checkpoint_inventory steps must be sorted and unique")
    if [int(item.get("step", -1)) for item in candidates if isinstance(item, Mapping)] != [200000, 300000]:
        raise ValueError("checkpoint_inventory must register both 200K and 300K candidates")
    if len(candidates) != 2 or inventory.get("selected_step") != 200000:
        raise ValueError("checkpoint_inventory must select the earlier 200K tie-break")
    if not str(inventory.get("tie_break_rule", "")).strip():
        raise ValueError("checkpoint_inventory.tie_break_rule must be recorded")
    for item in candidates:
        if not isinstance(item, Mapping) or len(str(item.get("sha256", ""))) != 64:
            raise ValueError("checkpoint inventory candidates need SHA256 metadata")
    stages_value = payload.get("stages")
    if isinstance(stages_value, (str, bytes)) or not isinstance(stages_value, Sequence):
        raise ValueError("stages must be a sequence")
    stages = tuple(
        _parse_stage(stage, expected_ordinal=index)
        for index, stage in enumerate(stages_value)
        if isinstance(stage, Mapping)
    )
    if len(stages) != len(stages_value) or len(stages) != 7:
        raise ValueError("gameplay progression must contain exactly seven ordered stages")
    if stages[0].policy != "random" or any(stage.policy != "checkpoint" for stage in stages[1:]):
        raise ValueError("only the first stage may be the random baseline")
    outputs = [stage.output for stage in stages]
    if len(set(outputs)) != len(outputs):
        raise ValueError("stage output filenames must be unique")
    learned_steps = [stage.actual_checkpoint_step for stage in stages[1:]]
    if learned_steps != sorted(learned_steps) or learned_steps != [100000, 200000, 500000, 1000000, 2500000, 5000000]:
        raise ValueError("learned stages must follow the verified checkpoint progression")
    if stages[2].requested_transitions != 250000 or not stages[2].substitution_reason:
        raise ValueError("the nearest-available 250K substitution must be recorded")
    return GameplayProgressionConfig(
        source_path=source_path,
        repository_root=root,
        experiment_id=EXPECTED_EXPERIMENT_ID,
        canonical_output_dir=canonical_output_dir,
        contract_path=contract_path,
        showcase_evaluation_seed=showcase_seed,
        episodes=episodes,
        max_steps_per_episode=max_steps,
        evaluation_epsilon=epsilon,
        video_fps=fps,
        capture_every_agent_step=capture_every,
        frame_repeat=frame_repeat,
        native_atari_fps=native_fps,
        max_width=max_width,
        requested_device=requested_device,
        codec_preference=codec_preference,
        pixel_format="yuv420p",
        fairness=fairness,
        stages=stages,
        raw=payload,
    )


def validate_checkpoint_inventory(config: GameplayProgressionConfig) -> dict[str, Any]:
    inventory = config.raw["checkpoint_inventory"]
    directory = _resolve_path(config.repository_root, inventory["checkpoints_directory"])
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    discovered: dict[int, Path] = {}
    for path in directory.glob("step-*.pt"):
        match = re.fullmatch(r"step-(\d+)\.pt", path.name)
        if match is not None:
            discovered[int(match.group(1))] = path
    configured_steps = [int(step) for step in inventory["available_checkpoint_steps"]]
    if sorted(discovered) != configured_steps:
        raise ValueError(
            "checkpoint inventory does not match the verified external run: "
            f"discovered={sorted(discovered)} configured={configured_steps}"
        )
    candidate_records: list[dict[str, Any]] = []
    for candidate in inventory["target_250k_candidates"]:
        step = int(candidate["step"])
        path = _resolve_path(config.repository_root, candidate["path"])
        if discovered.get(step) != path:
            raise ValueError(f"checkpoint inventory path mismatch for {step}")
        actual_hash = _sha256(path)
        if actual_hash != str(candidate["sha256"]).lower():
            raise ValueError(f"checkpoint inventory SHA256 mismatch for {step}")
        candidate_records.append(
            {"step": step, "path": path.as_posix(), "sha256": actual_hash}
        )
    return {
        "source_run_id": inventory["source_run_id"],
        "checkpoints_directory": directory.as_posix(),
        "available_checkpoint_steps": configured_steps,
        "target_250k_candidates": candidate_records,
        "selected_step": int(inventory["selected_step"]),
        "tie_break_rule": inventory["tie_break_rule"],
    }


def validate_checkpoint_sources(
    config: GameplayProgressionConfig,
    *,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if inventory is None:
        validate_checkpoint_inventory(config)
    validated: dict[str, dict[str, Any]] = {}
    for stage in config.stages:
        if stage.policy != "checkpoint" or stage.checkpoint is None:
            continue
        checkpoint_path = _resolve_path(config.repository_root, stage.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_hash = _sha256(checkpoint_path)
        if actual_hash != stage.checkpoint_sha256:
            raise ValueError(
                f"{stage.stage_id} checkpoint SHA256 mismatch: {actual_hash} != {stage.checkpoint_sha256}"
            )
        source_hash = None
        if stage.source_checkpoint:
            source_path = _resolve_path(config.repository_root, stage.source_checkpoint)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_hash = _sha256(source_path)
            if source_hash != stage.source_checkpoint_sha256:
                raise ValueError(
                    f"{stage.stage_id} source checkpoint SHA256 mismatch: {source_hash} != {stage.source_checkpoint_sha256}"
                )
        validated[stage.stage_id] = {
            "path": checkpoint_path.as_posix(),
            "sha256": actual_hash,
            "source_checkpoint_sha256": source_hash,
        }
    return validated


def _even_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("rendered frame must be uint8 RGB")
    pad_bottom = array.shape[0] % 2
    pad_right = array.shape[1] % 2
    if pad_bottom or pad_right:
        array = np.pad(array, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    return np.ascontiguousarray(array)


def _prepare_frame(frame: Any, *, max_width: int) -> np.ndarray:
    import cv2

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise TypeError("environment render must return uint8 pixels")
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("environment render must be grayscale, RGB, or RGBA")
    scale = max_width / array.shape[1]
    resized = cv2.resize(
        np.ascontiguousarray(array),
        (max_width, max(1, round(array.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    return _even_frame(resized)


@dataclass(frozen=True)
class CodecInfo:
    tag: str
    name: str
    pixel_format_tag: str


def _probe_codec(
    output_path: Path,
    *,
    frame: np.ndarray,
    fps: int,
    codec_preference: Sequence[str],
) -> CodecInfo:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    height, width = frame.shape[:2]
    for tag in codec_preference:
        if len(tag) != 4:
            reasons.append(f"{tag}: invalid FourCC")
            continue
        if tag not in {"avc1", "H264", "X264"}:
            reasons.append(f"{tag}: non-H264 codecs are not permitted")
            continue
        probe_path = output_path.with_name(f".{output_path.stem}.{tag}.probe.mp4")
        probe_path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(
            str(probe_path),
            cv2.VideoWriter_fourcc(*tag),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            reasons.append(f"{tag}: VideoWriter did not open")
            probe_path.unlink(missing_ok=True)
            continue
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        data = probe_path.read_bytes() if probe_path.is_file() else b""
        readable = False
        pixel_format_tag = "unknown"
        container_codec_tag = "unknown"
        capture = cv2.VideoCapture(str(probe_path))
        try:
            readable, _ = capture.read()
            container_codec_tag = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
            pixel_format_tag = _fourcc_text(
                capture.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)
            )
        finally:
            capture.release()
        probe_path.unlink(missing_ok=True)
        if not readable or not data:
            reasons.append(f"{tag}: probe output was not readable")
            continue
        if container_codec_tag.lower() not in {"h264", "avc1"}:
            reasons.append(
                f"{tag}: probe codec tag {container_codec_tag} is not H.264/AVC"
            )
            continue
        if pixel_format_tag not in {"I420", "IYUV", "YV12"}:
            reasons.append(
                f"{tag}: output pixel format {pixel_format_tag} is not YUV420 planar"
            )
            continue
        return CodecInfo(
            tag=tag,
            name="H.264/AVC" if tag in {"avc1", "H264", "X264"} else "MPEG-4 Part 2",
            pixel_format_tag=pixel_format_tag,
        )
    raise RuntimeError("no usable MP4 codec: " + "; ".join(reasons))


def _write_video(
    frames: Iterable[np.ndarray],
    *,
    output_path: Path,
    fps: int,
    codec_preference: Sequence[str],
) -> dict[str, Any]:
    import cv2

    iterator = iter(frames)
    try:
        first = _even_frame(next(iterator))
    except StopIteration as error:
        raise RuntimeError("no frames available for MP4") from error
    codec = _probe_codec(
        output_path,
        frame=first,
        fps=fps,
        codec_preference=codec_preference,
    )
    temp_path = output_path.with_name(f".{output_path.stem}.{codec.tag}.tmp.mp4")
    temp_path.unlink(missing_ok=True)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*codec.tag),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"selected codec {codec.tag} failed to open for final output")
    frame_count = 0
    try:
        for frame in _prepend(first, iterator):
            prepared = _even_frame(frame)
            if prepared.shape != first.shape:
                raise ValueError("all video frames must have the same even dimensions")
            bgr = cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frame_count += 1
    finally:
        writer.release()
    if frame_count < 1 or not temp_path.is_file() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("MP4 encoder produced no usable output")
    os.replace(temp_path, output_path)
    capture = cv2.VideoCapture(str(output_path))
    pixel_format_tag = "unknown"
    measured_codec_tag = "unknown"
    try:
        readable, _ = capture.read()
        measured_fps = float(capture.get(cv2.CAP_PROP_FPS))
        measured_codec_tag = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
        pixel_format_tag = _fourcc_text(
            capture.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)
        )
    finally:
        capture.release()
    if not readable:
        raise RuntimeError(f"encoded MP4 is not readable: {output_path}")
    if measured_codec_tag.lower() not in {"h264", "avc1"}:
        raise RuntimeError(
            f"encoded MP4 codec is {measured_codec_tag}, expected H.264/AVC"
        )
    if pixel_format_tag not in {"I420", "IYUV", "YV12"}:
        raise RuntimeError(
            f"encoded MP4 pixel format is {pixel_format_tag}, expected YUV420 planar"
        )
    return {
        "path": output_path.as_posix(),
        "sha256": _sha256(output_path),
        "frame_count": frame_count,
        "fps": fps,
        "measured_fps": measured_fps,
        "frame_size": [width, height],
        "codec": codec.name,
        "codec_tag": measured_codec_tag,
        "requested_codec_tag": codec.tag,
        "pixel_format": "yuv420p",
        "codec_pixel_format_tag": pixel_format_tag,
        "file_size_bytes": output_path.stat().st_size,
    }


def _fourcc_text(value: float) -> str:
    integer = int(value)
    if integer < 0:
        return "unknown"
    return "".join(chr(integer >> (8 * index) & 255) for index in range(4)).strip("\x00")


def _prepend(first: np.ndarray, rest: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    yield first
    yield from rest


def _duplicate_frames(frames: Iterable[np.ndarray], repeat: int) -> Iterator[np.ndarray]:
    for frame in frames:
        for _ in range(repeat):
            yield frame


def _render_review_summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# Gameplay progression review summary",
        "",
        "Every row uses the same Contract v2 environment, showcase seed, episode cap, and real-time playback settings. The 250K row records the verified 200K nearest checkpoint because no 250K checkpoint exists in the source run.",
        "",
        "| Stage | Actual checkpoint | Training seed | Showcase return | MP4 |",
        "|---|---:|---:|---:|---|",
    ]
    for stage in result.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        returns = stage.get("episode_returns")
        showcase_return = returns[0] if isinstance(returns, list) and returns else None
        video = stage.get("video")
        video_path = Path(str(video.get("path"))).name if isinstance(video, Mapping) else "unavailable"
        checkpoint = stage.get("actual_checkpoint_step")
        actual_checkpoint = "—" if stage.get("policy") == "random" else f"{int(checkpoint):,}"
        seed = "—" if stage.get("training_seed") is None else str(stage.get("training_seed"))
        lines.append(
            f"| {stage.get('label')} | {actual_checkpoint} | {seed} | {showcase_return} | `{video_path}` |"
        )
    lines.extend(
        [
            "",
            f"- Showcase evaluation seed: `{result.get('protocol', {}).get('showcase_evaluation_seed')}`",
            f"- Video: `{result.get('protocol', {}).get('video_fps')}` FPS, codec `{result.get('stages', [{}])[0].get('video', {}).get('codec_tag') if result.get('stages') else None}`, pixel format `{result.get('protocol', {}).get('pixel_format')}`, playback speed `{result.get('protocol', {}).get('playback_speed')}`",
            f"- Montage: `breakout-learning-progression.mp4`",
            f"- YouTube uploaded: `{result.get('youtube_uploaded')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _load_checkpoint_model(
    stage: StageSpec,
    *,
    config: GameplayProgressionConfig,
    resolved_device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module]:
    if stage.checkpoint is None:
        raise ValueError(f"{stage.stage_id} has no checkpoint")
    checkpoint_path = _resolve_path(config.repository_root, stage.checkpoint)
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{stage.stage_id} checkpoint must contain a mapping")
    saved_config = payload.get("config", {})
    if not isinstance(saved_config, dict):
        saved_config = {}
    model_config = payload.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}
    environment = make_breakout_env(
        render_mode="rgb_array",
        **breakout_environment_kwargs(
            load_evaluation_contract(config.contract_path)
        ),
    )
    try:
        action_count = int(environment.action_space.n)
        input_shape = tuple(int(value) for value in environment.observation_space.shape)
    finally:
        environment.close()
    architecture = checkpoint_architecture(payload)
    hidden_dim = int(model_config.get("hidden_dim", 512))
    model = build_q_network(
        architecture,
        num_actions=action_count,
        input_shape=input_shape,
        hidden_dim=hidden_dim,
    ).to(resolved_device)
    state_dict = payload.get("online_network")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{stage.stage_id} checkpoint has no online_network")
    model.load_state_dict(state_dict)
    model.eval()
    if payload.get("algorithm", saved_config.get("algorithm")) != EXPECTED_ALGORITHM:
        raise ValueError(f"{stage.stage_id} algorithm is not Double DQN")
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(f"{stage.stage_id} architecture is not dueling")
    return payload, model


def _action_name(action: int) -> str:
    return ATARI_ACTION_NAMES.get(action, f"ACTION_{action}")


def _render_stage(
    stage: StageSpec,
    *,
    config: GameplayProgressionConfig,
    output_path: Path,
    resolved_device: torch.device,
    source_hashes: Mapping[str, Any],
    generation_command: str,
) -> dict[str, Any]:
    contract = load_evaluation_contract(config.contract_path)
    env = make_breakout_env(
        render_mode="rgb_array",
        **breakout_environment_kwargs(contract),
    )
    payload: dict[str, Any] | None = None
    model: torch.nn.Module | None = None
    if stage.policy == "checkpoint":
        payload, model = _load_checkpoint_model(
            stage,
            config=config,
            resolved_device=resolved_device,
        )
    rng = np.random.default_rng(config.showcase_evaluation_seed)
    requested_counts = {name: 0 for name in EXPECTED_ACTION_MAPPING.values()}
    executed_counts = {name: 0 for name in EXPECTED_ACTION_MAPPING.values()}
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    terminated_count = 0
    truncated_count = 0
    cap_reached_count = 0
    captured_frames = 0
    agent_steps = 0
    frames: list[np.ndarray] = []
    try:
        for episode_index in range(config.episodes):
            observation, _ = env.reset(seed=config.showcase_evaluation_seed + episode_index)
            episode_return = 0.0
            episode_steps = 0
            frame = env.render()
            if frame is None:
                raise RuntimeError("ALE environment did not return an RGB render")
            frames.append(_prepare_frame(frame, max_width=config.max_width))
            captured_frames += 1
            for _ in range(config.max_steps_per_episode):
                if stage.policy == "random":
                    requested_action = int(rng.integers(0, int(env.action_space.n)))
                else:
                    if model is None:
                        raise RuntimeError("checkpoint stage model was not loaded")
                    with torch.no_grad():
                        state = observation_to_tensor(observation, device=resolved_device)
                        requested_action = int(torch.argmax(model(state)[0]).item())
                requested_counts[_action_name(requested_action)] = requested_counts.get(
                    _action_name(requested_action), 0
                ) + 1
                observation, reward, terminated, truncated, _ = env.step(requested_action)
                episode_return += float(reward)
                episode_steps += 1
                agent_steps += 1
                executed_action = getattr(env, "last_executed_action", requested_action)
                try:
                    executed_action_int = int(executed_action)
                except (TypeError, ValueError):
                    executed_action_int = requested_action
                executed_counts[_action_name(executed_action_int)] = executed_counts.get(
                    _action_name(executed_action_int), 0
                ) + 1
                frame = env.render()
                if frame is None:
                    raise RuntimeError("ALE environment did not return an RGB render")
                if episode_steps % config.capture_every_agent_step == 0:
                    frames.append(_prepare_frame(frame, max_width=config.max_width))
                    captured_frames += 1
                if terminated or truncated:
                    if terminated:
                        terminated_count += 1
                    if truncated:
                        truncated_count += 1
                    break
            else:
                cap_reached_count += 1
            episode_returns.append(episode_return)
            episode_lengths.append(episode_steps)
    finally:
        env.close()
    if not frames:
        raise RuntimeError(f"{stage.stage_id} produced no render frames")
    video = _write_video(
        _duplicate_frames(frames, config.frame_repeat),
        output_path=output_path,
        fps=config.video_fps,
        codec_preference=config.codec_preference,
    )
    checkpoint_path = None if stage.checkpoint is None else _resolve_path(config.repository_root, stage.checkpoint)
    checkpoint_step = 0
    if payload is not None:
        raw_step = payload.get("global_step")
        if raw_step is None:
            raw_step = stage.actual_checkpoint_step
        checkpoint_step = int(raw_step)
        if checkpoint_step != stage.actual_checkpoint_step:
            raise ValueError(
                f"{stage.stage_id} payload step {checkpoint_step} does not match manifest {stage.actual_checkpoint_step}"
            )
    return {
        "stage": stage.stage_id,
        "label": stage.label,
        "policy": stage.policy,
        "training_seed": stage.training_seed,
        "requested_transitions": stage.requested_transitions,
        "actual_checkpoint_step": stage.actual_checkpoint_step,
        "checkpoint_step_from_payload": checkpoint_step,
        "checkpoint": None if checkpoint_path is None else checkpoint_path.as_posix(),
        "checkpoint_sha256": None if checkpoint_path is None else _sha256(checkpoint_path),
        "source_run_id": stage.source_run_id,
        "source_stage": stage.source_stage,
        "source_checkpoint": stage.source_checkpoint,
        "source_checkpoint_sha256": stage.source_checkpoint_sha256,
        "substitution_reason": stage.substitution_reason,
        "algorithm": stage.algorithm,
        "architecture": stage.architecture,
        "showcase_evaluation_seed": config.showcase_evaluation_seed,
        "evaluation_epsilon": config.evaluation_epsilon,
        "environment_id": ENVIRONMENT_ID,
        "contract_id": contract.contract_id,
        "contract_sha256": _sha256(config.contract_path),
        "contract": contract.to_dict(),
        "action_mapping": EXPECTED_ACTION_MAPPING,
        "requested_action_counts": requested_counts,
        "executed_action_counts": executed_counts,
        "device": str(resolved_device),
        "requested_device": config.requested_device,
        "episodes": config.episodes,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "recorder_cap_reached_count": cap_reached_count,
        "agent_steps": agent_steps,
        "captured_render_frames": captured_frames,
        "frame_repeat": config.frame_repeat,
        "capture_every_agent_step": config.capture_every_agent_step,
        "playback_speed": 1.0,
        "video": video,
        "source_hashes": dict(source_hashes),
        "generation": {
            "source_script": "scripts/visualization/record_gameplay_progression.py",
            "render_source": "make_breakout_env(render_mode='rgb_array', **breakout_environment_kwargs(contract)).render()",
            "generation_command": generation_command,
        },
    }


def _montage_frames(
    stage_video: Path,
    *,
    label: str,
) -> Iterator[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(stage_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stage MP4 for montage: {stage_video}")
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            title_height = 34
            canvas = np.zeros((rgb.shape[0] + title_height, rgb.shape[1], 3), dtype=np.uint8)
            canvas[title_height:] = rgb
            cv2.putText(
                canvas,
                label,
                (6, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            yield _even_frame(canvas)
    finally:
        capture.release()


def _resolve_device(config: GameplayProgressionConfig) -> torch.device:
    device = resolve_device(config.requested_device)
    if device.type != "cuda":
        raise RuntimeError(
            f"gameplay progression requested {config.requested_device} but resolved {device}; refusing silent fallback"
        )
    return device


def record_progression(
    config: GameplayProgressionConfig,
    *,
    output_dir: Path,
    overwrite: bool = False,
    generation_command: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_status = _git_status(config.repository_root, exclude_path=output_dir)
    source_hashes = {
        "config_sha256": _sha256(config.source_path),
        "contract_sha256": _sha256(config.contract_path),
        "recorder_script_sha256": _sha256(Path(__file__).resolve()),
        "source_commit": _git_commit(config.repository_root),
        "source_dirty": bool(source_status),
        "source_status_snapshot": source_status,
    }
    checkpoint_inventory = validate_checkpoint_inventory(config)
    checkpoint_sources = validate_checkpoint_sources(
        config,
        inventory=checkpoint_inventory,
    )
    resolved_device = _resolve_device(config)
    resolved_generation_command = generation_command or (
        "python -m scripts.visualization.record_gameplay_progression "
        "--manifest configs/eval/gameplay-progression.json "
        f"--output {config.canonical_output_dir.as_posix()}"
    )
    stage_metadata: list[dict[str, Any]] = []
    for stage in config.stages:
        output_path = output_dir / stage.output
        if output_path.exists() and not overwrite:
            raise FileExistsError(output_path)
        metadata = _render_stage(
            stage,
            config=config,
            output_path=output_path,
            resolved_device=resolved_device,
            source_hashes=source_hashes,
            generation_command=resolved_generation_command,
        )
        metadata["verified_checkpoint_source"] = checkpoint_sources.get(stage.stage_id)
        stage_metadata.append(metadata)
        stage_json = output_path.with_suffix(".json")
        stage_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    montage_path = output_dir / "breakout-learning-progression.mp4"
    montage_streams = (
        frame
        for metadata in stage_metadata
        for frame in _montage_frames(
            output_dir / Path(str(metadata["video"]["path"])).name,
            label=str(metadata["label"]),
        )
    )
    montage = _write_video(
        montage_streams,
        output_path=montage_path,
        fps=config.video_fps,
        codec_preference=config.codec_preference,
    )
    result = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "generated_at_utc": _utc_timestamp(),
        "config": config.source_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "protocol": {
            "environment_id": EXPECTED_ENVIRONMENT_ID,
            "contract_path": config.contract_path.as_posix(),
            "contract_sha256": source_hashes["contract_sha256"],
            "showcase_evaluation_seed": config.showcase_evaluation_seed,
            "episodes": config.episodes,
            "max_steps_per_episode": config.max_steps_per_episode,
            "evaluation_epsilon": config.evaluation_epsilon,
            "video_fps": config.video_fps,
            "capture_every_agent_step": config.capture_every_agent_step,
            "frame_repeat": config.frame_repeat,
            "playback_speed": 1.0,
            "requested_device": config.requested_device,
            "resolved_device": str(resolved_device),
            "pixel_format": config.pixel_format,
            "action_mapping": EXPECTED_ACTION_MAPPING,
        },
        "fairness": dict(config.fairness),
        "checkpoint_inventory": checkpoint_inventory,
        "source_hashes": source_hashes,
        "stages": stage_metadata,
        "montage": {
            **montage,
            "source_stage_ids": [stage.stage_id for stage in config.stages],
            "stage_order_is_manifest_order": True,
        },
        "youtube_uploaded": False,
        "review_summary": "review-summary.md",
        "generation_command": resolved_generation_command,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "review-summary.md").write_text(
        _render_review_summary(result),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/eval/gameplay-progression.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_gameplay_progression_config(args.manifest)
        output_dir = (
            config.canonical_output_dir
            if args.output is None
            else args.output.resolve()
        )
        result = record_progression(
            config,
            output_dir=output_dir,
            overwrite=args.overwrite,
            generation_command=(
                "python -m scripts.visualization.record_gameplay_progression "
                f"--manifest {args.manifest.as_posix()} --output {output_dir.as_posix()}"
            ),
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Gameplay progression recording failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
