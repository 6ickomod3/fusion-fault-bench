from __future__ import annotations

import numpy as np
import pytest

from fusion_fault_bench.inference import (
    bootstrap_conditional_loss,
    bootstrap_count_ratio,
    bootstrap_crossover_roots,
    bootstrap_mean,
    first_zero_crossover,
    paired_bootstrap_indices,
    pava_non_decreasing,
    percentile_interval,
)


def test_pcg64dxsm_bootstrap_index_golden() -> None:
    indices = paired_bootstrap_indices(seed=2718, replicates=3, sequence_count=5)

    assert indices.dtype == np.int64
    assert indices.tolist() == [
        [3, 3, 2, 1, 4],
        [0, 3, 2, 2, 1],
        [3, 4, 2, 0, 2],
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([3.0, 1.0, 2.0], [2.0, 2.0, 2.0]),
        ([1.0, 3.0, 2.0, 4.0], [1.0, 2.5, 2.5, 4.0]),
        ([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]),
    ],
)
def test_pava_equal_weight_adjacent_pooling(values: list[float], expected: list[float]) -> None:
    assert pava_non_decreasing(values).tolist() == expected


@pytest.mark.parametrize("values", [[], [[1.0, 2.0]], [1.0, float("nan")]])
def test_pava_rejects_invalid_arrays(values: list) -> None:
    with pytest.raises(ValueError):
        pava_non_decreasing(values)


@pytest.mark.parametrize(
    ("grid", "fitted", "expected"),
    [
        ([0.0, 1.0], [0.1, 0.2], 0.0),
        ([0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], 1.0),
        ([0.0, 1.0, 2.0], [-1.0, -0.5, 0.5], 1.5),
        ([0.0, 1.0], [-2.0, -1.0], None),
        ([0.0, 1.0], [-1e-13, 1.0], 0.0),
    ],
)
def test_first_zero_crossover_boundaries(
    grid: list[float],
    fitted: list[float],
    expected: float | None,
) -> None:
    assert first_zero_crossover(grid, fitted, zero_tolerance=1e-12) == expected


@pytest.mark.parametrize(
    ("grid", "fitted", "tolerance"),
    [
        ([0.0], [-1.0], 1e-12),
        ([0.0, 0.0], [-1.0, 1.0], 1e-12),
        ([0.0, 1.0], [1.0, -1.0], 1e-12),
        ([0.0, 1.0], [-1.0, 1.0], 0.0),
        ([0.0, 1.0], [-1.0], 1e-12),
    ],
)
def test_first_zero_crossover_rejects_invalid_inputs(
    grid: list[float],
    fitted: list[float],
    tolerance: float,
) -> None:
    with pytest.raises(ValueError):
        first_zero_crossover(grid, fitted, zero_tolerance=tolerance)


def test_linear_percentile_interval_has_fixed_quantiles() -> None:
    lower, upper = percentile_interval(
        [0.0, 1.0, 2.0, 3.0],
        confidence_level=0.95,
    )

    assert lower == pytest.approx(0.075)
    assert upper == pytest.approx(2.925)


@pytest.mark.parametrize(
    ("values", "confidence"),
    [([], 0.95), ([1.0], 0.5), ([1.0], 1.0)],
)
def test_percentile_interval_rejects_invalid_inputs(values: list[float], confidence: float) -> None:
    with pytest.raises(ValueError):
        percentile_interval(values, confidence_level=confidence)


def test_bootstrap_scalar_and_ratio_statistics() -> None:
    indices = np.asarray([[0, 1], [1, 1]], dtype=np.int64)
    means = bootstrap_mean(np.asarray([1.0, 3.0]), indices)
    ratios = bootstrap_count_ratio(
        np.asarray([1, 3], dtype=np.int64),
        np.asarray([2, 4], dtype=np.int64),
        indices,
    )
    conditional = bootstrap_conditional_loss(
        np.asarray([2.0, 0.0]),
        np.asarray([1, 0], dtype=np.int64),
        indices,
    )

    assert means.tolist() == [2.0, 3.0]
    assert ratios.tolist() == [4 / 6, 6 / 8]
    assert conditional.tolist() == [2.0]


def test_bootstrap_crossover_refits_each_paired_replicate() -> None:
    roots = bootstrap_crossover_roots(
        magnitudes=np.asarray([0.0, 1.0, 2.0]),
        sequence_contrasts=np.asarray(
            [
                [-1.0, -1.0],
                [-0.5, 0.5],
                [1.0, 1.0],
            ]
        ),
        indices=np.asarray([[0, 0], [1, 1]], dtype=np.int64),
        zero_tolerance=1e-12,
    )

    assert roots == (1.3333333333333333, 0.6666666666666666)


@pytest.mark.parametrize(
    ("contrasts", "indices"),
    [
        (np.asarray([1.0, 2.0]), np.asarray([[0, 1]], dtype=np.int64)),
        (np.ones((2, 2)), np.asarray([[0, 1, 1]], dtype=np.int64)),
        (np.ones((3, 2)), np.asarray([[0, 1]], dtype=np.int64)),
    ],
)
def test_bootstrap_crossover_rejects_shape_mismatch(
    contrasts: np.ndarray, indices: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        bootstrap_crossover_roots(
            magnitudes=np.asarray([0.0, 1.0]),
            sequence_contrasts=contrasts,
            indices=indices,
            zero_tolerance=1e-12,
        )
