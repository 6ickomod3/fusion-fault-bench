"""Two-input covariance-reported information fusion in Cartesian BEV."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type Vector2Like = Sequence[float] | FloatArray


@dataclass(frozen=True, slots=True)
class DiagonalFusionResult:
    """Fused Cartesian value and its reported diagonal variance."""

    value_xy: FloatArray
    reported_variance_xy: FloatArray


def _vector2(value: Vector2Like, *, field_name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{field_name} must have shape (2,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True)


def fuse_diagonal_information(
    *,
    first_value_xy: Vector2Like,
    first_reported_variance_xy: Vector2Like,
    second_value_xy: Vector2Like,
    second_reported_variance_xy: Vector2Like,
) -> DiagonalFusionResult:
    """Fuse two independent estimates using only their reported variances."""

    first_value = _vector2(first_value_xy, field_name="first_value_xy")
    second_value = _vector2(second_value_xy, field_name="second_value_xy")
    first_variance = _vector2(
        first_reported_variance_xy,
        field_name="first_reported_variance_xy",
    )
    second_variance = _vector2(
        second_reported_variance_xy,
        field_name="second_reported_variance_xy",
    )
    if np.any(first_variance <= 0.0) or np.any(second_variance <= 0.0):
        raise ValueError("reported variances must be strictly positive")

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        first_information = 1.0 / first_variance
        second_information = 1.0 / second_variance
        fused_variance = 1.0 / (first_information + second_information)
        fused_value = fused_variance * (
            first_information * first_value + second_information * second_value
        )

    if not np.all(np.isfinite(fused_variance)) or np.any(fused_variance <= 0.0):
        raise ValueError("information fusion produced an invalid variance")
    if not np.all(np.isfinite(fused_value)):
        raise ValueError("information fusion produced a non-finite value")

    value_result = np.asarray(fused_value, dtype=np.float64)
    variance_result = np.asarray(fused_variance, dtype=np.float64)
    value_result.setflags(write=False)
    variance_result.setflags(write=False)
    return DiagonalFusionResult(
        value_xy=value_result,
        reported_variance_xy=variance_result,
    )
