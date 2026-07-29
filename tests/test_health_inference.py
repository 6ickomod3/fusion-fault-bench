from __future__ import annotations

import numpy as np
import pytest

from fusion_fault_bench.health_inference import (
    bootstrap_equal_weighted_regret,
    conditional_loss_interval,
    conditional_mean_interval,
    equal_target_family_condition_mean,
    recovery_fraction_interval,
    sequence_mean_interval,
)


def _identity_bootstrap(sequence_count: int, replicates: int = 40) -> np.ndarray:
    return np.tile(np.arange(sequence_count, dtype=np.int64), (replicates, 1))


def test_sequence_mean_interval_uses_complete_sequence_statistics() -> None:
    result = sequence_mean_interval([1.0, 3.0], _identity_bootstrap(2))
    assert result.estimate == 2.0
    assert result.lower == 2.0
    assert result.upper == 2.0
    assert result.defined_replicates == 40


@pytest.mark.parametrize(
    ("values", "indices"),
    [
        ([], np.zeros((1, 0), dtype=np.int64)),
        ([1.0], np.zeros((1, 2), dtype=np.int64)),
        ([1.0], np.ones((1, 1), dtype=np.int64)),
        ([float("nan")], np.zeros((1, 1), dtype=np.int64)),
    ],
)
def test_sequence_mean_rejects_invalid_vectors_and_indices(
    values: list[float],
    indices: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        sequence_mean_interval(values, indices)


def test_conditional_loss_aggregates_sums_and_counts() -> None:
    result = conditional_loss_interval(
        [2.0, 6.0],
        np.asarray([2, 3], dtype=np.int64),
        _identity_bootstrap(2),
    )
    assert result.estimate == 1.6
    assert result.lower == 1.6
    assert result.upper == 1.6


def test_conditional_loss_censors_insufficient_bootstrap_support() -> None:
    indices = np.vstack(
        (
            np.zeros((39, 2), dtype=np.int64),
            np.ones((1, 2), dtype=np.int64),
        )
    )
    result = conditional_loss_interval(
        [0.0, 2.0],
        np.asarray([0, 1], dtype=np.int64),
        indices,
    )
    assert result.estimate is None
    assert result.defined_replicates == 1

    for sums, counts in (
        ([1.0], np.asarray([0], dtype=np.int64)),
        ([-1.0], np.asarray([1], dtype=np.int64)),
        ([1.0], np.asarray([1.0])),
        ([1.0, 2.0], np.asarray([1], dtype=np.int64)),
    ):
        with pytest.raises(ValueError):
            conditional_loss_interval(sums, counts, _identity_bootstrap(len(sums)))


def test_conditional_mean_allows_signed_latency_relative_to_first_missing() -> None:
    result = conditional_mean_interval(
        [-2.0, 4.0],
        np.asarray([1, 1], dtype=np.int64),
        _identity_bootstrap(2),
    )
    assert result.estimate == 1.0


def test_recovery_fraction_is_unclipped_and_recomputed_per_replicate() -> None:
    result = recovery_fraction_interval(
        [4.0, 6.0],
        [-1.0, 1.0],
        [2.0, 2.0],
        _identity_bootstrap(2),
    )
    assert result.estimate == pytest.approx(5.0 / 3.0)
    assert result.lower == pytest.approx(5.0 / 3.0)
    assert result.upper == pytest.approx(5.0 / 3.0)


def test_recovery_fraction_censors_zero_or_insufficient_denominators() -> None:
    result = recovery_fraction_interval(
        [1.0, 1.0],
        [0.5, 0.5],
        [1.0, 1.0],
        _identity_bootstrap(2),
    )
    assert result.estimate is None

    with pytest.raises(ValueError):
        recovery_fraction_interval(
            [1.0],
            [0.5, 0.6],
            [0.0],
            _identity_bootstrap(1),
        )
    with pytest.raises(ValueError):
        recovery_fraction_interval(
            [1.0],
            [0.5],
            [0.0],
            _identity_bootstrap(1),
            denominator_tolerance_m2=0.0,
        )


def test_equal_weighting_is_target_then_family_then_condition() -> None:
    values = np.asarray([0.0, 2.0, 10.0, 20.0], dtype=np.float64)
    targets = ("camera", "camera", "camera", "lidar")
    families = ("bias", "bias", "noise", "bias")
    # Camera: mean(mean(0,2), 10)=5.5. Lidar: 20. Equal target mean=12.75.
    assert equal_target_family_condition_mean(values, targets=targets, families=families) == 12.75
    matrix = np.column_stack((values, values + 2.0))
    boot = _identity_bootstrap(2)
    result = bootstrap_equal_weighted_regret(
        matrix,
        boot,
        targets=targets,
        families=families,
    )
    assert np.array_equal(result, np.full(40, 13.75))


def test_equal_weighting_rejects_bad_labels_and_shapes() -> None:
    with pytest.raises(ValueError):
        equal_target_family_condition_mean(
            [1.0],
            targets=("unknown",),
            families=("bias",),
        )
    with pytest.raises(ValueError):
        equal_target_family_condition_mean(
            [1.0],
            targets=("camera", "lidar"),
            families=("bias",),
        )
    with pytest.raises(ValueError):
        bootstrap_equal_weighted_regret(
            np.asarray([1.0]),
            _identity_bootstrap(1),
            targets=("camera",),
            families=("bias",),
        )
