from __future__ import annotations

import numpy as np
import pytest

from fusion_fault_bench.metrics import matched_center_mse, squared_center_error


def test_squared_center_error_sums_coordinates_instead_of_averaging_them() -> None:
    assert squared_center_error((3.0, 4.0), (0.0, 0.0)) == 25.0


def test_squared_center_error_uses_estimate_minus_truth() -> None:
    assert squared_center_error((-1.0, 5.0), (2.0, 1.0)) == 25.0


def test_matched_center_mse_averages_object_frame_losses() -> None:
    estimates = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    truths = np.zeros((2, 2), dtype=np.float64)

    assert matched_center_mse(estimates, truths) == 14.5
    assert matched_center_mse(estimates[:1], truths[:1]) == 25.0


@pytest.mark.parametrize(
    ("estimate", "truth", "message"),
    [
        ((1.0,), (0.0, 0.0), "shape"),
        ((1.0, 2.0), (0.0, 0.0, 0.0), "shape"),
        ((float("nan"), 0.0), (0.0, 0.0), "finite"),
        ((1.0, 2.0), (float("inf"), 0.0), "finite"),
        ((1e308, 0.0), (-1e308, 0.0), "must be finite"),
        ((1e154, 1e154), (0.0, 0.0), "must be finite"),
    ],
)
def test_squared_center_error_rejects_invalid_inputs(estimate, truth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        squared_center_error(estimate, truth)


@pytest.mark.parametrize(
    ("estimates", "truths", "message"),
    [
        ([], [], "shape"),
        ([[1.0, 2.0]], [[0.0, 0.0], [1.0, 1.0]], "same shape"),
        ([[1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0]], "shape"),
        ([[float("nan"), 0.0]], [[0.0, 0.0]], "finite"),
        (np.empty((0, 2)), np.empty((0, 2)), "at least one"),
        ([[1e154, 0.0], [1e154, 0.0]], [[0.0, 0.0], [0.0, 0.0]], "must be finite"),
    ],
)
def test_matched_center_mse_rejects_invalid_inputs(estimates, truths, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        matched_center_mse(estimates, truths)
