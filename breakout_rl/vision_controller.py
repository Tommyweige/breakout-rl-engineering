"""Explainable pixel perception and predictive control for Atari Breakout.

The controller accepts only the current screen observation. It does not hold an
environment reference and cannot read ALE RAM or emulator object state.
"""

from __future__ import annotations

import math
import operator
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # OpenCV is already a project runtime dependency.
    import cv2
except ImportError:  # pragma: no cover - exercised only in incomplete installs.
    cv2 = None  # type: ignore[assignment]


NOOP = "NOOP"
LEFT = "LEFT"
RIGHT = "RIGHT"
CONTROLLER_ACTIONS = (NOOP, LEFT, RIGHT)


@dataclass(frozen=True)
class PlayfieldBounds:
    """Inclusive pixel bounds for the playable court."""

    left: float
    right: float
    top: float
    bottom: float
    confidence: float


@dataclass(frozen=True)
class PaddleDetection:
    left: float
    right: float
    center_x: float
    top: float
    bottom: float
    confidence: float


@dataclass(frozen=True)
class BallEstimate:
    x: float | None
    y: float | None
    vx: float | None
    vy: float | None
    confidence: float
    last_seen_frame: int | None
    age_frames: int | None
    directly_detected: bool


@dataclass(frozen=True)
class VisionObservation:
    frame_index: int
    bounds: PlayfieldBounds
    paddle: PaddleDetection | None
    ball: BallEstimate
    ball_candidate_count: int


@dataclass(frozen=True)
class ControlDecision:
    action: str
    observation: VisionObservation
    predicted_intercept_x: float | None
    time_to_intercept_frames: float | None
    paddle_error: float | None
    decision_latency_ms: float


@dataclass(frozen=True)
class _Blob:
    x: float
    y: float
    left: int
    top: int
    width: int
    height: int
    area: int
    confidence: float


def _as_image(frame: Any) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim not in (2, 3):
        raise ValueError("screen frame must have shape (height, width) or (height, width, channels)")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("screen frame must have 1, 3, or 4 channels")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("screen frame is too small for Breakout perception")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("screen frame must contain numeric pixel values")
    if image.dtype != np.uint8:
        if np.any(image < 0) or np.any(image > 255):
            raise ValueError("screen pixel values must be between 0 and 255")
        image = image.astype(np.uint8)
    return image


def _gray_max(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return image[:, :, :3].max(axis=2)


def detect_playfield_bounds(frame: Any) -> PlayfieldBounds:
    """Find the court from the persistent side rails and the HUD transition."""

    image = _as_image(frame)
    height, width = image.shape[:2]
    non_black = _gray_max(image) > 24
    y0 = max(0, int(height * 0.12))
    y1 = max(y0 + 1, min(height, int(height * 0.94)))
    column_density = non_black[y0:y1].mean(axis=0)
    rail_threshold = 0.78

    left = 0
    while left < width // 4 and column_density[left] >= rail_threshold:
        left += 1
    right = width - 1
    while right >= width * 3 // 4 and column_density[right] >= rail_threshold:
        right -= 1

    horizontal_found = left > 0 and right < width - 1 and right > left + 20
    if not horizontal_found:
        left = max(1, int(round(width * 0.05)))
        right = min(width - 2, int(round(width * 0.95)) - 1)

    inner_left = min(right - 1, left + 2)
    inner_right = max(inner_left + 1, right - 2)
    row_density = non_black[:, inner_left : inner_right + 1].mean(axis=1)
    top_scan_start = int(height * 0.10)
    top_scan_end = min(height, int(height * 0.50))
    quiet_rows = np.flatnonzero(row_density[top_scan_start:top_scan_end] < 0.12)
    if quiet_rows.size:
        top = top_scan_start + int(quiet_rows[0])
        vertical_top_found = True
    else:
        top = int(round(height * 0.15))
        vertical_top_found = False

    side_pixels = np.concatenate(
        (non_black[:, :left], non_black[:, right + 1 :]), axis=1
    )
    if side_pixels.shape[1] == 0:
        rail_rows = np.zeros(height, dtype=bool)
    else:
        rail_rows = side_pixels.mean(axis=1) >= rail_threshold
    bottom_scan_start = int(height * 0.55)
    visible_rail_rows = np.flatnonzero(rail_rows[bottom_scan_start:])
    if visible_rail_rows.size:
        bottom = bottom_scan_start + int(visible_rail_rows[-1])
        vertical_bottom_found = True
    else:
        bottom = int(round(height * 0.92))
        vertical_bottom_found = False

    bottom = min(max(bottom, top + 20), height - 1)
    return PlayfieldBounds(
        left=float(left),
        right=float(right),
        top=float(top),
        bottom=float(bottom),
        confidence=(0.4 + 0.2 * horizontal_found + 0.2 * vertical_top_found + 0.2 * vertical_bottom_found),
    )


def _target_mask(image: np.ndarray) -> np.ndarray:
    """Mask the audited Breakout ball/paddle palette plus bright monochrome sprites."""

    if image.ndim == 2 or image.shape[2] == 1:
        gray = image if image.ndim == 2 else image[:, :, 0]
        return gray >= 190
    rgb = image[:, :, :3]
    red_sprite = (
        (rgb[:, :, 0] >= 160)
        & (rgb[:, :, 1] <= 130)
        & (rgb[:, :, 2] <= 130)
        & (rgb[:, :, 0].astype(np.int16) - rgb[:, :, 1].astype(np.int16) >= 55)
    )
    white_sprite = (rgb[:, :, 0] >= 190) & (rgb[:, :, 1] >= 190) & (rgb[:, :, 2] >= 190)
    return red_sprite | white_sprite


def _connected_blobs(mask: np.ndarray, bounds: PlayfieldBounds) -> list[_Blob]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for connected-component perception")
    left = max(0, int(math.floor(bounds.left)))
    right = min(mask.shape[1] - 1, int(math.ceil(bounds.right)))
    top = max(0, int(math.floor(bounds.top)))
    bottom = min(mask.shape[0] - 1, int(math.ceil(bounds.bottom)))
    region = np.ascontiguousarray(mask[top : bottom + 1, left : right + 1], dtype=np.uint8)
    if not np.any(region):
        return []
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        region, connectivity=8
    )
    blobs: list[_Blob] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        absolute_y = y + top
        absolute_x = x + left
        area_ratio = area / max(width * height, 1)
        if (
            4 <= area <= 24
            and 2 <= width <= 5
            and 2 <= height <= 7
            and area_ratio >= 0.45
        ):
            shape_confidence = 1.0 - min(abs(width - 2) * 0.08 + abs(height - 4) * 0.05, 0.3)
            blobs.append(
                _Blob(
                    x=float(centroids[index, 0] + left),
                    y=float(centroids[index, 1] + top),
                    left=absolute_x,
                    top=absolute_y,
                    width=width,
                    height=height,
                    area=area,
                    confidence=shape_confidence,
                )
            )
    return blobs


def _detect_paddle(mask: np.ndarray, bounds: PlayfieldBounds) -> PaddleDetection | None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for connected-component perception")
    left = max(0, int(bounds.left))
    right = min(mask.shape[1] - 1, int(bounds.right))
    top = max(0, int(bounds.top))
    bottom = min(mask.shape[0] - 1, int(bounds.bottom))
    region = np.ascontiguousarray(mask[top : bottom + 1, left : right + 1], dtype=np.uint8)
    if not np.any(region):
        return None
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        region, connectivity=8
    )
    paddle_candidates: list[tuple[float, PaddleDetection]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        absolute_y = y + top
        if (
            width < 4
            or width > 28
            or height < 2
            or height > 8
            or area < 12
            or absolute_y < bounds.bottom - 18
        ):
            continue
        absolute_x = x + left
        visible_right = absolute_x + width - 1
        center_x = float(centroids[index, 0] + left)
        # The paddle can be clipped by the playfield edge. Estimate its center
        # from the visible edge and its standard 16-pixel sprite width.
        if absolute_x == bounds.left and width < 16:
            center_x = bounds.left + 7.5
        elif visible_right == bounds.right and width < 16:
            center_x = bounds.right - 7.5
        paddle = PaddleDetection(
            left=float(absolute_x),
            right=float(visible_right),
            center_x=center_x,
            top=float(absolute_y),
            bottom=float(absolute_y + height - 1),
            confidence=min(1.0, 0.65 + area / 200.0),
        )
        paddle_candidates.append((abs(paddle.bottom - bounds.bottom), paddle))
    return min(paddle_candidates, key=lambda item: item[0])[1] if paddle_candidates else None


def reflect_x(x: float, left: float, right: float) -> float:
    """Fold an unrestricted horizontal coordinate through repeated wall bounces."""

    if not all(math.isfinite(value) for value in (x, left, right)):
        raise ValueError("reflection coordinates must be finite")
    if right <= left:
        raise ValueError("right wall must be greater than left wall")
    span = right - left
    offset = (x - left) % (2.0 * span)
    return left + (offset if offset <= span else 2.0 * span - offset)


def predict_paddle_intercept(
    *,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    paddle_plane_y: float,
    bounds: PlayfieldBounds,
    ball_radius: float = 2.0,
    minimum_downward_speed: float = 0.15,
) -> tuple[float, float] | None:
    """Predict a descending ball's first paddle-plane crossing with wall bounces."""

    values = (ball_x, ball_y, ball_vx, ball_vy, paddle_plane_y, ball_radius)
    if not all(math.isfinite(float(value)) for value in values):
        return None
    if ball_vy <= minimum_downward_speed or paddle_plane_y <= ball_y:
        return None
    frames = (paddle_plane_y - ball_y) / ball_vy
    if not math.isfinite(frames) or frames <= 0.0:
        return None
    left = bounds.left + ball_radius
    right = bounds.right - ball_radius
    if right <= left:
        return None
    intercept = reflect_x(ball_x + ball_vx * frames, left, right)
    return intercept, frames


def choose_paddle_action(
    *,
    error: float,
    previous_direction: str = NOOP,
    deadband: float = 3.0,
    hysteresis: float = 1.0,
) -> str:
    """Apply a Schmitt-style deadband to observed paddle-to-target error."""

    if not math.isfinite(error):
        return NOOP
    if not math.isfinite(deadband) or deadband < 0:
        raise ValueError("deadband must be finite and non-negative")
    if not math.isfinite(hysteresis) or hysteresis < 0 or hysteresis > deadband:
        raise ValueError("hysteresis must be finite and between 0 and deadband")
    if previous_direction not in CONTROLLER_ACTIONS:
        raise ValueError("previous_direction must be NOOP, LEFT, or RIGHT")

    engage = deadband + hysteresis
    release = max(0.0, deadband - hysteresis)
    if previous_direction == RIGHT:
        if error <= -engage:
            return LEFT
        return RIGHT if error > release else NOOP
    if previous_direction == LEFT:
        if error >= engage:
            return RIGHT
        return LEFT if error < -release else NOOP
    if error > engage:
        return RIGHT
    if error < -engage:
        return LEFT
    return NOOP


class PredictiveBreakoutController:
    """Track visible sprites and steer toward the reflected paddle-plane intercept."""

    def __init__(
        self,
        *,
        deadband: float = 3.0,
        hysteresis: float = 1.0,
        max_missing_frames: int = 4,
        max_ball_speed_pixels_per_frame: float = 8.0,
    ) -> None:
        if isinstance(max_missing_frames, bool):
            raise TypeError("max_missing_frames must be a non-negative integer")
        try:
            parsed_missing_frames = operator.index(max_missing_frames)
        except TypeError as error:
            raise TypeError("max_missing_frames must be a non-negative integer") from error
        if parsed_missing_frames < 0:
            raise ValueError("max_missing_frames must be a non-negative integer")
        if not math.isfinite(max_ball_speed_pixels_per_frame) or max_ball_speed_pixels_per_frame <= 0:
            raise ValueError("max_ball_speed_pixels_per_frame must be positive and finite")
        choose_paddle_action(
            error=0.0, deadband=deadband, hysteresis=hysteresis
        )
        self.deadband = float(deadband)
        self.hysteresis = float(hysteresis)
        self.max_missing_frames = int(parsed_missing_frames)
        self.max_ball_speed_pixels_per_frame = float(max_ball_speed_pixels_per_frame)
        self.reset()

    def reset(self) -> None:
        self._frame_index = -1
        self._previous_frame: np.ndarray | None = None
        self._last_seen_frame: int | None = None
        self._last_ball_x: float | None = None
        self._last_ball_y: float | None = None
        self._vx: float | None = None
        self._vy: float | None = None
        self._last_confidence = 0.0
        self._track_live = False
        self._previous_direction = NOOP
        self._last_movement_action: str | None = None
        self._observation_count = 0
        self._ball_detected_count = 0
        self._paddle_detected_count = 0
        self._lost_ball_frames = 0
        self._trajectory_prediction_count = 0
        self._left_right_reversal_count = 0
        self._action_counts: Counter[str] = Counter({name: 0 for name in CONTROLLER_ACTIONS})
        self._decision_latencies_ms: list[float] = []

    def _candidate_blobs(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bounds: PlayfieldBounds,
        paddle: PaddleDetection | None,
    ) -> list[_Blob]:
        candidates = _connected_blobs(mask, bounds)
        if self._previous_frame is not None:
            previous = self._previous_frame
            if previous.shape == image.shape:
                if image.ndim == 2:
                    changed = image != previous
                else:
                    changed = np.any(image[:, :, :3] != previous[:, :, :3], axis=2)
                moving_mask = mask & changed
                for blob in _connected_blobs(moving_mask, bounds):
                    if not any(math.hypot(blob.x - item.x, blob.y - item.y) < 2.0 for item in candidates):
                        candidates.append(
                            _Blob(
                                x=blob.x,
                                y=blob.y,
                                left=blob.left,
                                top=blob.top,
                                width=blob.width,
                                height=blob.height,
                                area=blob.area,
                                confidence=blob.confidence * 0.82,
                            )
                        )
        if paddle is not None:
            # The moving edges of a paddle can resemble a small sprite for one frame.
            candidates = [
                candidate
                for candidate in candidates
                if candidate.y < paddle.top - 2.0
            ]
        return candidates

    def _select_ball_candidate(
        self, candidates: list[_Blob], bounds: PlayfieldBounds
    ) -> _Blob | None:
        if not candidates:
            return None
        if self._track_live and self._last_seen_frame is not None:
            gap = self._frame_index - self._last_seen_frame
            if gap <= self.max_missing_frames + 1 and self._last_ball_x is not None and self._last_ball_y is not None:
                projected_x = self._last_ball_x + (self._vx or 0.0) * gap
                projected_x = reflect_x(
                    projected_x,
                    bounds.left + 2.0,
                    bounds.right - 2.0,
                )
                projected_y = self._last_ball_y + (self._vy or 0.0) * gap
                max_displacement = self.max_ball_speed_pixels_per_frame * gap + 4.0
                plausible: list[tuple[float, float, _Blob]] = []
                for candidate in candidates:
                    distance_from_last = math.hypot(
                        candidate.x - self._last_ball_x,
                        candidate.y - self._last_ball_y,
                    )
                    if distance_from_last > max_displacement:
                        continue
                    prediction_error = math.hypot(
                        candidate.x - projected_x,
                        candidate.y - projected_y,
                    )
                    plausible.append((prediction_error, -candidate.confidence, candidate))
                if plausible:
                    return min(plausible, key=lambda item: (item[0], item[1]))[2]
                if gap <= self.max_missing_frames:
                    return None
        # The first observation and long-gap reacquisition favor the canonical 2x4 sprite.
        return max(candidates, key=lambda candidate: candidate.confidence)

    def observe(self, frame: Any) -> VisionObservation:
        image = _as_image(frame)
        if self._previous_frame is not None and self._previous_frame.shape != image.shape:
            self.reset()
        self._frame_index += 1
        bounds = detect_playfield_bounds(image)
        mask = _target_mask(image)
        paddle = _detect_paddle(mask, bounds)
        candidates = self._candidate_blobs(image, mask, bounds, paddle)
        selected = self._select_ball_candidate(candidates, bounds)

        self._observation_count += 1
        if paddle is not None:
            self._paddle_detected_count += 1
        if selected is not None:
            self._ball_detected_count += 1
            if self._last_seen_frame is not None and self._last_ball_x is not None and self._last_ball_y is not None and self._track_live:
                delta_frames = self._frame_index - self._last_seen_frame
                dx = (selected.x - self._last_ball_x) / max(delta_frames, 1)
                dy = (selected.y - self._last_ball_y) / max(delta_frames, 1)
                if (
                    abs(dx) <= self.max_ball_speed_pixels_per_frame
                    and abs(dy) <= self.max_ball_speed_pixels_per_frame
                ):
                    # Use the newest segment so wall/brick direction changes take effect immediately.
                    if abs(dx) > 0.05 or self._vx is None:
                        self._vx = float(dx)
                    if abs(dy) > 0.05 or self._vy is None:
                        self._vy = float(dy)
                else:
                    self._vx = None
                    self._vy = None
                    self._last_confidence = selected.confidence * 0.55
            else:
                self._vx = None
                self._vy = None
            self._last_seen_frame = self._frame_index
            self._last_ball_x = selected.x
            self._last_ball_y = selected.y
            self._last_confidence = selected.confidence
            self._track_live = True
            ball = BallEstimate(
                x=selected.x,
                y=selected.y,
                vx=self._vx,
                vy=self._vy,
                confidence=self._last_confidence,
                last_seen_frame=self._last_seen_frame,
                age_frames=0,
                directly_detected=True,
            )
        else:
            self._lost_ball_frames += 1
            if self._last_seen_frame is not None and self._last_ball_x is not None and self._last_ball_y is not None:
                age = self._frame_index - self._last_seen_frame
            else:
                age = None
            if (
                self._track_live
                and age is not None
                and age <= self.max_missing_frames
            ):
                projected_x = self._last_ball_x + (self._vx or 0.0) * age
                projected_x = reflect_x(
                    projected_x,
                    bounds.left + 2.0,
                    bounds.right - 2.0,
                )
                ball = BallEstimate(
                    x=projected_x,
                    y=self._last_ball_y + (self._vy or 0.0) * age,
                    vx=self._vx,
                    vy=self._vy,
                    confidence=self._last_confidence * (0.65**age),
                    last_seen_frame=self._last_seen_frame,
                    age_frames=age,
                    directly_detected=False,
                )
            else:
                if self._track_live and age is not None and age > self.max_missing_frames:
                    self._track_live = False
                    self._vx = None
                    self._vy = None
                ball = BallEstimate(
                    x=None,
                    y=None,
                    vx=None,
                    vy=None,
                    confidence=0.0,
                    last_seen_frame=self._last_seen_frame,
                    age_frames=age,
                    directly_detected=False,
                )

        self._previous_frame = np.array(image, copy=True)
        return VisionObservation(
            frame_index=self._frame_index,
            bounds=bounds,
            paddle=paddle,
            ball=ball,
            ball_candidate_count=len(candidates),
        )

    def select_action(self, frame: Any) -> ControlDecision:
        started = time.perf_counter_ns()
        observation = self.observe(frame)
        intercept: float | None = None
        time_to_intercept: float | None = None
        error: float | None = None
        action = NOOP
        ball = observation.ball
        paddle = observation.paddle
        if (
            paddle is not None
            and ball.x is not None
            and ball.y is not None
            and ball.vx is not None
            and ball.vy is not None
        ):
            paddle_plane_y = paddle.top - 2.0
            prediction = predict_paddle_intercept(
                ball_x=ball.x,
                ball_y=ball.y,
                ball_vx=ball.vx,
                ball_vy=ball.vy,
                paddle_plane_y=paddle_plane_y,
                bounds=observation.bounds,
            )
            if prediction is not None:
                intercept, time_to_intercept = prediction
                self._trajectory_prediction_count += 1
                error = intercept - paddle.center_x
                action = choose_paddle_action(
                    error=error,
                    previous_direction=self._previous_direction,
                    deadband=self.deadband,
                    hysteresis=self.hysteresis,
                )

        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        self._decision_latencies_ms.append(latency_ms)
        self._action_counts[action] += 1
        if action in (LEFT, RIGHT):
            if self._last_movement_action is not None and action != self._last_movement_action:
                self._left_right_reversal_count += 1
            self._last_movement_action = action
        self._previous_direction = action
        return ControlDecision(
            action=action,
            observation=observation,
            predicted_intercept_x=intercept,
            time_to_intercept_frames=time_to_intercept,
            paddle_error=error,
            decision_latency_ms=latency_ms,
        )

    @property
    def diagnostics(self) -> dict[str, Any]:
        observation_count = self._observation_count
        decision_count = sum(self._action_counts.values())
        latencies = np.asarray(self._decision_latencies_ms, dtype=np.float64)
        return {
            "observation_count": observation_count,
            "ball_detected_frames": self._ball_detected_count,
            "ball_detection_success_rate": (
                self._ball_detected_count / observation_count if observation_count else None
            ),
            "paddle_detected_frames": self._paddle_detected_count,
            "paddle_detection_success_rate": (
                self._paddle_detected_count / observation_count if observation_count else None
            ),
            "lost_ball_frames": self._lost_ball_frames,
            "trajectory_prediction_count": self._trajectory_prediction_count,
            "requested_action_distribution": dict(self._action_counts),
            "left_right_reversal_count": self._left_right_reversal_count,
            "left_right_reversal_rate": (
                self._left_right_reversal_count / decision_count if decision_count else 0.0
            ),
            "noop_rate": (
                self._action_counts[NOOP] / decision_count if decision_count else 0.0
            ),
            "controller_decision_latency_ms": {
                "mean": float(np.mean(latencies)) if latencies.size else None,
                "median": float(np.median(latencies)) if latencies.size else None,
                "p95": float(np.percentile(latencies, 95)) if latencies.size else None,
                "max": float(np.max(latencies)) if latencies.size else None,
            },
        }


__all__ = [
    "CONTROLLER_ACTIONS",
    "LEFT",
    "NOOP",
    "RIGHT",
    "BallEstimate",
    "ControlDecision",
    "PaddleDetection",
    "PlayfieldBounds",
    "PredictiveBreakoutController",
    "VisionObservation",
    "choose_paddle_action",
    "detect_playfield_bounds",
    "predict_paddle_intercept",
    "reflect_x",
]
