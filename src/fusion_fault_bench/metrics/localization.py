"""Matched-center localization loss in ego-frame BEV coordinates."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def _finite_array(value: npt.ArrayLike, *, field_name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def squared_center_error(
    estimate_xy: npt.ArrayLike,
    truth_xy: npt.ArrayLike,
) -> float:
    """Return squared Euclidean center error ``ex**2 + ey**2``."""

    estimate = _finite_array(estimate_xy, field_name="estimate_xy")
    truth = _finite_array(truth_xy, field_name="truth_xy")
    if estimate.shape != (2,) or truth.shape != (2,):
        raise ValueError("estimate_xy and truth_xy must each have shape (2,)")

    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.subtract(estimate, truth, dtype=np.float64)
    x_error = float(difference[0])
    y_error = float(difference[1])
    try:
        loss = math.fsum((x_error * x_error, y_error * y_error))
    except OverflowError as error:
        raise ValueError("squared center error must be finite") from error
    if not math.isfinite(loss):
        raise ValueError("squared center error must be finite")
    return loss


def matched_center_mse(
    estimates_xy: npt.ArrayLike,
    truths_xy: npt.ArrayLike,
) -> float:
    """Average squared Euclidean loss over aligned eligible object-frames."""

    estimates = _finite_array(estimates_xy, field_name="estimates_xy")
    truths = _finite_array(truths_xy, field_name="truths_xy")
    if (
        estimates.ndim != 2
        or truths.ndim != 2
        or estimates.shape[1:] != (2,)
        or truths.shape[1:] != (2,)
    ):
        raise ValueError("estimates_xy and truths_xy must each have shape (n, 2)")
    if estimates.shape != truths.shape:
        raise ValueError("estimates_xy and truths_xy must have the same shape")
    if estimates.shape[0] == 0:
        raise ValueError("matched-center MSE requires at least one object-frame")

    losses = (
        squared_center_error(estimate, truth)
        for estimate, truth in zip(estimates, truths, strict=True)
    )
    try:
        result = math.fsum(losses) / estimates.shape[0]
    except OverflowError as error:
        raise ValueError("matched-center MSE must be finite") from error
    if not math.isfinite(result):
        raise ValueError("matched-center MSE must be finite")
    return result
