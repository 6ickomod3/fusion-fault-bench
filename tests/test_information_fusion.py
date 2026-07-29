from __future__ import annotations

import numpy as np
import pytest

from fusion_fault_bench.fusion import fuse_diagonal_information


def test_information_fusion_matches_independent_matrix_inverse() -> None:
    camera_value = np.asarray([2.5, -0.75], dtype=np.float64)
    lidar_value = np.asarray([0.125, 0.5], dtype=np.float64)
    camera_variance = np.asarray([1.5**2, 0.6**2], dtype=np.float64)
    lidar_variance = np.asarray([0.25**2, 0.25**2], dtype=np.float64)

    result = fuse_diagonal_information(
        first_value_xy=camera_value,
        first_reported_variance_xy=camera_variance,
        second_value_xy=lidar_value,
        second_reported_variance_xy=lidar_variance,
    )

    camera_covariance = np.diag(camera_variance)
    lidar_covariance = np.diag(lidar_variance)
    expected_covariance = np.linalg.inv(
        np.linalg.inv(camera_covariance) + np.linalg.inv(lidar_covariance)
    )
    expected_value = expected_covariance @ (
        np.linalg.inv(camera_covariance) @ camera_value
        + np.linalg.inv(lidar_covariance) @ lidar_value
    )
    assert result.reported_variance_xy == pytest.approx([9 / 148, 9 / 169])
    assert result.reported_variance_xy == pytest.approx(np.diag(expected_covariance))
    assert result.value_xy == pytest.approx(expected_value)


def test_information_fusion_uses_reported_precision_weights() -> None:
    result = fuse_diagonal_information(
        first_value_xy=(37.0, 169.0),
        first_reported_variance_xy=(9 / 4, 9 / 25),
        second_value_xy=(0.0, 0.0),
        second_reported_variance_xy=(1 / 16, 1 / 16),
    )

    assert result.value_xy == pytest.approx([1.0, 25.0])
    assert result.reported_variance_xy == pytest.approx([9 / 148, 9 / 169])
    assert result.value_xy.dtype == np.float64
    assert result.reported_variance_xy.dtype == np.float64
    assert not result.value_xy.flags.writeable
    assert not result.reported_variance_xy.flags.writeable


def test_information_fusion_is_symmetric_in_its_two_inputs() -> None:
    forward = fuse_diagonal_information(
        first_value_xy=(1.0, 2.0),
        first_reported_variance_xy=(0.5, 4.0),
        second_value_xy=(-3.0, 5.0),
        second_reported_variance_xy=(2.0, 1.0),
    )
    reverse = fuse_diagonal_information(
        first_value_xy=(-3.0, 5.0),
        first_reported_variance_xy=(2.0, 1.0),
        second_value_xy=(1.0, 2.0),
        second_reported_variance_xy=(0.5, 4.0),
    )

    assert np.array_equal(forward.value_xy, reverse.value_xy)
    assert np.array_equal(forward.reported_variance_xy, reverse.reported_variance_xy)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("first_value_xy", (1.0,), "shape"),
        ("second_value_xy", (1.0, 2.0, 3.0), "shape"),
        ("first_value_xy", (float("nan"), 0.0), "finite"),
        ("second_value_xy", (float("inf"), 0.0), "finite"),
        ("first_reported_variance_xy", (0.0, 1.0), "strictly positive"),
        ("second_reported_variance_xy", (-1.0, 1.0), "strictly positive"),
        ("first_reported_variance_xy", (float("nan"), 1.0), "finite"),
    ],
)
def test_information_fusion_rejects_invalid_inputs(field, bad_value, message: str) -> None:
    arguments = {
        "first_value_xy": (1.0, 2.0),
        "first_reported_variance_xy": (1.0, 1.0),
        "second_value_xy": (3.0, 4.0),
        "second_reported_variance_xy": (1.0, 1.0),
    }
    arguments[field] = bad_value

    with pytest.raises(ValueError, match=message):
        fuse_diagonal_information(**arguments)


def test_information_fusion_rejects_unrepresentable_precision() -> None:
    with pytest.raises(ValueError, match="invalid variance"):
        fuse_diagonal_information(
            first_value_xy=(1.0, 2.0),
            first_reported_variance_xy=(1e-320, 1.0),
            second_value_xy=(3.0, 4.0),
            second_reported_variance_xy=(1e-320, 1.0),
        )


def test_information_fusion_rejects_nonfinite_weighted_value() -> None:
    with pytest.raises(ValueError, match="non-finite value"):
        fuse_diagonal_information(
            first_value_xy=(1e308, 2.0),
            first_reported_variance_xy=(1e-308, 1.0),
            second_value_xy=(0.0, 4.0),
            second_reported_variance_xy=(1.0, 1.0),
        )
