"""Independent scalar references for M4 prediction, NIS, latch, and oracle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type Label = Literal["healthy", "camera-fault", "lidar-fault", "ambiguous"]
type EvidenceStatus = Literal["update-eligible", "insufficient-support"]
type OracleAction = Literal["camera-only", "lidar-only", "fixed-fusion"]


def constant_velocity_prediction(
    *,
    first_value_xy: npt.ArrayLike,
    first_covariance_xy: npt.ArrayLike,
    first_time_s: float,
    second_value_xy: npt.ArrayLike,
    second_covariance_xy: npt.ArrayLike,
    second_time_s: float,
    reference_time_s: float,
) -> tuple[FloatArray, FloatArray]:
    """Scalar-oracle two-sample extrapolation with independent covariance."""

    first = np.asarray(first_value_xy, dtype=np.float64)
    second = np.asarray(second_value_xy, dtype=np.float64)
    first_covariance = np.asarray(first_covariance_xy, dtype=np.float64)
    second_covariance = np.asarray(second_covariance_xy, dtype=np.float64)
    if first.shape != (2,) or second.shape != (2,):
        raise ValueError("reference values must have shape (2,)")
    if first_covariance.shape != (2, 2) or second_covariance.shape != (2, 2):
        raise ValueError("reference covariances must have shape (2, 2)")
    if not first_time_s < second_time_s < reference_time_s:
        raise ValueError("reference prediction times must be strictly increasing")
    h = (reference_time_s - second_time_s) / (second_time_s - first_time_s)
    value = np.empty(2, dtype=np.float64)
    covariance = np.empty((2, 2), dtype=np.float64)
    for coordinate in range(2):
        value[coordinate] = (1.0 + h) * second[coordinate] - h * first[coordinate]
        for other in range(2):
            covariance[coordinate, other] = (1.0 + h) ** 2 * second_covariance[
                coordinate, other
            ] + h**2 * first_covariance[coordinate, other]
    return value, covariance


def nis(
    *,
    current_value_xy: npt.ArrayLike,
    current_covariance_xy: npt.ArrayLike,
    predicted_value_xy: npt.ArrayLike,
    predicted_covariance_xy: npt.ArrayLike,
) -> float:
    """Direct inverse-based 2D NIS reference."""

    current = np.asarray(current_value_xy, dtype=np.float64)
    prediction = np.asarray(predicted_value_xy, dtype=np.float64)
    covariance = np.asarray(current_covariance_xy, dtype=np.float64) + np.asarray(
        predicted_covariance_xy,
        dtype=np.float64,
    )
    if current.shape != (2,) or prediction.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("reference NIS inputs must be two dimensional")
    residual = current - prediction
    return float(residual @ np.linalg.inv(covariance) @ residual)


def ecdf_rank(clean_values: Sequence[float], value: float) -> float:
    """Loop-based strict-less-than empirical rank."""

    if not clean_values:
        raise ValueError("reference ECDF requires clean values")
    return sum(clean_value < value for clean_value in clean_values) / len(clean_values)


def latch_trace(
    labels: Sequence[Label],
    statuses: Sequence[EvidenceStatus],
) -> tuple[Label, ...]:
    """Literal scalar recurrence independent of the production state class."""

    if len(labels) != len(statuses):
        raise ValueError("reference latch inputs must align")
    state: Label = "healthy"
    activation_candidate: Label | None = None
    activation_count = 0
    recovery_count = 0
    output: list[Label] = []
    for label, status in zip(labels, statuses, strict=True):
        if status == "insufficient-support":
            output.append(state)
            continue
        if state == "healthy":
            if label == "healthy":
                activation_candidate = None
                activation_count = 0
            else:
                if activation_candidate == label:
                    activation_count += 1
                else:
                    activation_candidate = label
                    activation_count = 1
                if activation_count == 2:
                    state = label
                    activation_candidate = None
                    activation_count = 0
        elif label == "healthy":
            recovery_count += 1
            if recovery_count == 3:
                state = "healthy"
                recovery_count = 0
        else:
            recovery_count = 0
        output.append(state)
    return tuple(output)


def frame_oracle_action(
    *,
    truth_xy: npt.ArrayLike,
    camera_xy: npt.ArrayLike | None,
    lidar_xy: npt.ArrayLike | None,
    fixed_xy: npt.ArrayLike | None,
) -> OracleAction:
    """Choose one whole-frame action with the frozen exact-tie order."""

    truth = np.asarray(truth_xy, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1:] != (2,) or truth.shape[0] == 0:
        raise ValueError("oracle truth must have nonempty shape (objects, 2)")
    candidates: tuple[tuple[OracleAction, npt.ArrayLike | None], ...] = (
        ("camera-only", camera_xy),
        ("lidar-only", lidar_xy),
        ("fixed-fusion", fixed_xy),
    )
    losses: list[tuple[float, OracleAction]] = []
    for action, raw_value in candidates:
        if raw_value is None:
            continue
        value = np.asarray(raw_value, dtype=np.float64)
        if value.shape != truth.shape:
            raise ValueError("oracle action values must align with truth")
        losses.append((float(np.square(value - truth).sum(axis=1).mean()), action))
    if not losses:
        raise ValueError("oracle requires at least one defined action")
    # min is stable, so exact ties retain camera, LiDAR, fixed declaration order.
    return min(losses, key=lambda item: item[0])[1]
