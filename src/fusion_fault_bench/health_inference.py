"""M4 sequence-first inference, weighting, and oracle-fraction utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.inference import percentile_interval

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HealthInterval:
    """A point estimate and its pointwise paired-bootstrap interval."""

    estimate: float | None
    lower: float | None
    upper: float | None
    defined_replicates: int


def _float_vector(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _indices(value: npt.ArrayLike, *, sequence_count: int) -> IntArray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] != sequence_count
        or array.shape[0] == 0
        or array.dtype.kind not in {"i", "u"}
    ):
        raise ValueError("bootstrap indices must be a nonempty aligned integer matrix")
    result = np.asarray(array, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= sequence_count):
        raise ValueError("bootstrap index lies outside the sequence population")
    return result


def sequence_mean_interval(
    values: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> HealthInterval:
    """Infer on complete sequence statistics with one paired index matrix."""

    vector = _float_vector(values, name="values")
    indices = _indices(bootstrap_indices, sequence_count=vector.size)
    replicates = vector[indices].mean(axis=1)
    lower, upper = percentile_interval(replicates, confidence_level=confidence_level)
    return HealthInterval(
        estimate=math.fsum(float(value) for value in vector) / vector.size,
        lower=lower,
        upper=upper,
        defined_replicates=int(replicates.size),
    )


def conditional_loss_interval(
    loss_sums: npt.ArrayLike,
    valid_counts: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> HealthInterval:
    """Cluster-bootstrap a valid-count-weighted conditional loss."""

    losses = _float_vector(loss_sums, name="loss_sums")
    raw_counts = np.asarray(valid_counts)
    if (
        raw_counts.shape != losses.shape
        or raw_counts.dtype.kind not in {"i", "u"}
        or np.any(raw_counts < 0)
    ):
        raise ValueError("valid_counts must be a nonnegative aligned integer vector")
    counts = np.asarray(raw_counts, dtype=np.int64)
    if np.any(losses < 0.0) or np.any((counts == 0) & (losses != 0.0)):
        raise ValueError("conditional loss sums and counts are inconsistent")
    return conditional_mean_interval(
        losses,
        counts,
        bootstrap_indices,
        confidence_level=confidence_level,
    )


def conditional_mean_interval(
    value_sums: npt.ArrayLike,
    valid_counts: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> HealthInterval:
    """Cluster-bootstrap a possibly signed conditional mean."""

    losses = _float_vector(value_sums, name="value_sums")
    raw_counts = np.asarray(valid_counts)
    if (
        raw_counts.shape != losses.shape
        or raw_counts.dtype.kind not in {"i", "u"}
        or np.any(raw_counts < 0)
    ):
        raise ValueError("valid_counts must be a nonnegative aligned integer vector")
    counts = np.asarray(raw_counts, dtype=np.int64)
    if np.any((counts == 0) & (losses != 0.0)):
        raise ValueError("conditional value sums and counts are inconsistent")
    indices = _indices(bootstrap_indices, sequence_count=losses.size)
    numerator = losses[indices].sum(axis=1, dtype=np.float64)
    denominator = counts[indices].sum(axis=1, dtype=np.int64)
    defined = denominator > 0
    replicates = numerator[defined] / denominator[defined]
    required = (1.0 - (1.0 - confidence_level) / 2.0) * indices.shape[0]
    point_count = int(counts.sum(dtype=np.int64))
    if point_count == 0 or int(replicates.size) <= required:
        return HealthInterval(None, None, None, int(replicates.size))
    lower, upper = percentile_interval(replicates, confidence_level=confidence_level)
    point = math.fsum(float(value) for value in losses) / point_count
    return HealthInterval(point, lower, upper, int(replicates.size))


def recovery_fraction_interval(
    fixed_loss: npt.ArrayLike,
    policy_loss: npt.ArrayLike,
    oracle_loss: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
    *,
    denominator_tolerance_m2: float = 1e-12,
    confidence_level: float = 0.95,
) -> HealthInterval:
    """Recompute the unclipped three-way recovery fraction per replicate."""

    fixed = _float_vector(fixed_loss, name="fixed_loss")
    policy = _float_vector(policy_loss, name="policy_loss")
    oracle = _float_vector(oracle_loss, name="oracle_loss")
    if policy.shape != fixed.shape or oracle.shape != fixed.shape:
        raise ValueError("recovery-fraction inputs must be aligned")
    if denominator_tolerance_m2 <= 0.0:
        raise ValueError("denominator tolerance must be positive")
    indices = _indices(bootstrap_indices, sequence_count=fixed.size)
    fixed_boot = fixed[indices].mean(axis=1)
    policy_boot = policy[indices].mean(axis=1)
    oracle_boot = oracle[indices].mean(axis=1)
    denominators = fixed_boot - oracle_boot
    defined = denominators > denominator_tolerance_m2
    replicates = (fixed_boot[defined] - policy_boot[defined]) / denominators[defined]
    required = (1.0 - (1.0 - confidence_level) / 2.0) * indices.shape[0]
    point_denominator = float(fixed.mean() - oracle.mean())
    if point_denominator <= denominator_tolerance_m2 or int(replicates.size) <= required:
        return HealthInterval(None, None, None, int(replicates.size))
    lower, upper = percentile_interval(replicates, confidence_level=confidence_level)
    point = float((fixed.mean() - policy.mean()) / point_denominator)
    return HealthInterval(point, lower, upper, int(replicates.size))


def equal_target_family_condition_mean(
    condition_values: npt.ArrayLike,
    *,
    targets: Sequence[str],
    families: Sequence[str],
) -> float:
    """Average conditions within family, families within target, then targets."""

    values = _float_vector(condition_values, name="condition_values")
    if len(targets) != values.size or len(families) != values.size:
        raise ValueError("target and family labels must align with condition values")
    unique_targets = tuple(dict.fromkeys(targets))
    if not unique_targets or any(target not in {"camera", "lidar"} for target in unique_targets):
        raise ValueError("utility targets must be camera and/or lidar")
    target_means: list[float] = []
    for target in unique_targets:
        target_indices = [index for index, label in enumerate(targets) if label == target]
        unique_families = tuple(dict.fromkeys(families[index] for index in target_indices))
        family_means = [
            math.fsum(float(values[index]) for index in target_indices if families[index] == family)
            / sum(families[index] == family for index in target_indices)
            for family in unique_families
        ]
        target_means.append(math.fsum(family_means) / len(family_means))
    return math.fsum(target_means) / len(target_means)


def bootstrap_equal_weighted_regret(
    condition_sequence_regrets: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
    *,
    targets: Sequence[str],
    families: Sequence[str],
) -> FloatArray:
    """Recompute equal-target/family/condition regret inside every replicate."""

    values = np.asarray(condition_sequence_regrets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("condition regrets must be a nonempty finite matrix")
    if len(targets) != values.shape[0] or len(families) != values.shape[0]:
        raise ValueError("condition labels must align with the first matrix axis")
    indices = _indices(bootstrap_indices, sequence_count=values.shape[1])
    boot_condition_means = values[:, indices].mean(axis=2)
    return np.asarray(
        [
            equal_target_family_condition_mean(
                boot_condition_means[:, replicate],
                targets=targets,
                families=families,
            )
            for replicate in range(indices.shape[0])
        ],
        dtype=np.float64,
    )
