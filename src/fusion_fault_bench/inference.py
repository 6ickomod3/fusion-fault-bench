"""Deterministic point, bootstrap, isotonic, and crossover computations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


def paired_bootstrap_indices(
    *,
    seed: int,
    replicates: int,
    sequence_count: int,
) -> IntArray:
    """Draw the one paired index matrix reused by every method and severity."""

    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    return generator.integers(
        0,
        sequence_count,
        size=(replicates, sequence_count),
        dtype=np.int64,
    )


def pava_non_decreasing(values: Sequence[float] | FloatArray) -> FloatArray:
    """Equal-weight nondecreasing PAVA with deterministic adjacent pooling."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("PAVA requires a nonempty one-dimensional array")
    if not np.all(np.isfinite(source)):
        raise ValueError("PAVA values must be finite")

    means: list[float] = []
    weights: list[int] = []
    for raw_value in source:
        means.append(float(raw_value))
        weights.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            right_mean = means.pop()
            left_mean = means.pop()
            right_weight = weights.pop()
            left_weight = weights.pop()
            pooled_weight = left_weight + right_weight
            pooled_mean = (left_mean * left_weight + right_mean * right_weight) / pooled_weight
            means.append(pooled_mean)
            weights.append(pooled_weight)

    fitted = np.empty_like(source)
    cursor = 0
    for mean, weight in zip(means, weights, strict=True):
        fitted[cursor : cursor + weight] = mean
        cursor += weight
    return fitted


def first_zero_crossover(
    magnitudes: Sequence[float] | FloatArray,
    fitted_values: Sequence[float] | FloatArray,
    *,
    zero_tolerance: float,
) -> float | None:
    """Return the identity/first interpolated zero, or None when right-censored."""

    grid = np.asarray(magnitudes, dtype=np.float64)
    fitted = np.asarray(fitted_values, dtype=np.float64)
    if grid.ndim != 1 or fitted.ndim != 1 or grid.size != fitted.size:
        raise ValueError("crossover grid and fitted values must be aligned vectors")
    if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("crossover magnitudes must be strictly increasing")
    if zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be positive")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(fitted)):
        raise ValueError("crossover inputs must be finite")
    if np.any(np.diff(fitted) < 0.0):
        raise ValueError("crossover fitted values must be nondecreasing")

    adjusted = np.where(np.abs(fitted) <= zero_tolerance, 0.0, fitted)
    if adjusted[0] >= 0.0:
        return float(grid[0])
    for index in range(1, grid.size):
        if adjusted[index] < 0.0:
            continue
        if adjusted[index] == 0.0:
            return float(grid[index])
        left_x = float(grid[index - 1])
        right_x = float(grid[index])
        left_y = float(adjusted[index - 1])
        right_y = float(adjusted[index])
        fraction = -left_y / (right_y - left_y)
        return left_x + fraction * (right_x - left_x)
    return None


def percentile_interval(
    values: Sequence[float] | FloatArray,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    """Two-sided linear percentile interval with a fixed quantile convention."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("percentile interval requires a nonempty vector")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    alpha = 1.0 - confidence_level
    quantiles = np.quantile(
        samples,
        (alpha / 2.0, 1.0 - alpha / 2.0),
        method="linear",
    )
    return float(quantiles[0]), float(quantiles[1])


def bootstrap_mean(values: FloatArray, indices: IntArray) -> FloatArray:
    """Bootstrap sequence means for a complete scalar sequence statistic."""

    return np.asarray(values[indices].mean(axis=1), dtype=np.float64)


def bootstrap_count_ratio(
    numerators: IntArray,
    denominators: IntArray,
    indices: IntArray,
) -> FloatArray:
    """Bootstrap a ratio of clustered integer counts."""

    numerator = numerators[indices].sum(axis=1, dtype=np.int64)
    denominator = denominators[indices].sum(axis=1, dtype=np.int64)
    return np.asarray(numerator / denominator, dtype=np.float64)


def bootstrap_conditional_loss(
    loss_sums: FloatArray,
    valid_counts: IntArray,
    indices: IntArray,
) -> FloatArray:
    """Bootstrap valid-count-weighted loss, omitting zero-denominator replicates."""

    numerator = loss_sums[indices].sum(axis=1, dtype=np.float64)
    denominator = valid_counts[indices].sum(axis=1, dtype=np.int64)
    defined = denominator > 0
    return np.asarray(numerator[defined] / denominator[defined], dtype=np.float64)


def bootstrap_crossover_roots(
    *,
    magnitudes: FloatArray,
    sequence_contrasts: FloatArray,
    indices: IntArray,
    zero_tolerance: float,
) -> tuple[float | None, ...]:
    """Refit PAVA and the grid root for each paired bootstrap replicate."""

    if sequence_contrasts.ndim != 2:
        raise ValueError("sequence_contrasts must have severity and sequence axes")
    if sequence_contrasts.shape[0] != magnitudes.size:
        raise ValueError("severity axis must match the magnitude grid")
    if sequence_contrasts.shape[1] != indices.shape[1]:
        raise ValueError("sequence axis must match the bootstrap index width")

    bootstrapped = sequence_contrasts[:, indices].mean(axis=2)
    roots: list[float | None] = []
    for replicate in range(indices.shape[0]):
        fitted = pava_non_decreasing(bootstrapped[:, replicate])
        roots.append(
            first_zero_crossover(
                magnitudes,
                fitted,
                zero_tolerance=zero_tolerance,
            )
        )
    return tuple(roots)
