"""Complete-scene M5-A aggregation, directional selectors, and crossovers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    GeometryCrossoverManifest,
)
from fusion_fault_bench.contracts.replay_v1 import M5_SCENE_NAMES
from fusion_fault_bench.inference import (
    bootstrap_crossover_roots,
    bootstrap_crossover_status,
    first_zero_crossover,
    pava_non_decreasing,
    percentile_interval,
)
from fusion_fault_bench.replay_inference import (
    ReplayInterval,
    equal_scene_loss_interval,
    equal_scene_ratio_interval,
    equal_scene_value_interval,
    pooled_availability_interval,
    pooled_conditional_loss_interval,
    replay_bootstrap_indices,
)
from fusion_fault_bench.replay_persistent import (
    ReplayPersistentSceneEvaluation,
)
from fusion_fault_bench.replay_plan import ReplayPersistentCase

type ReplayPersistentAggregation = Literal[
    "equal-scene-mean",
    "pooled-valid-eligible-count-ratio",
    "pooled-valid-loss",
]
type ReplayPersistentMetricId = Literal[
    "matched-center-mse",
    "fused-minus-healthy",
    "coverage",
    "conditional-matched-center-mse",
    "undefined-output-rate",
    "scene-equal-coverage",
    "camera-lidar-disagreement-mse",
]
type ReplayCrossoverDirection = Literal["negative", "positive", "increase"]
type ReplayCrossoverStatus = Literal["observed", "not-observed", "undetermined"]
type ReplayCrossoverUpper = float | Literal["positive-infinity"] | None

_REPLAY_SEQUENCE_IDS = tuple(f"nuscenes:{name}" for name in M5_SCENE_NAMES)

M5_A_DIRECTIONAL_EXPECTATIONS: dict[str, Literal["positive", "negative"]] = {
    "replay-lidar-y-bias:0": "negative",
    "replay-camera-noise-correctly-reported:1": "negative",
    "replay-camera-noise-underreported:1": "negative",
    "replay-camera-calibration-x:0": "negative",
    "replay-camera-calibration-yaw:0": "negative",
    "replay-camera-timestamp-offset:0": "negative",
    "replay-lidar-y-bias:-4": "positive",
    "replay-lidar-y-bias:+4": "positive",
    "replay-camera-calibration-x:-4": "positive",
    "replay-camera-calibration-x:+4": "positive",
    "replay-camera-calibration-yaw:-0.08": "positive",
    "replay-camera-calibration-yaw:+0.08": "positive",
    "replay-camera-timestamp-offset:-0.8": "positive",
    "replay-camera-timestamp-offset:+0.8": "positive",
    "replay-camera-noise-underreported:4": "positive",
    "replay-camera-noise-correctly-reported:4": "negative",
}


@dataclass(frozen=True, slots=True)
class ReplayPersistentPopulationMetric:
    """One aggregate plus its complete ten-scene sufficient statistics."""

    replay_experiment_identity_sha256: str
    condition_id: str
    condition_selector: str
    method_id: str
    metric_id: ReplayPersistentMetricId
    aggregation: ReplayPersistentAggregation
    interval: ReplayInterval
    scene_numerators: tuple[float, ...]
    scene_denominators: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.replay_experiment_identity_sha256) != 64:
            raise ValueError("persistent population identity digest is invalid")
        if (
            not self.condition_id
            or not self.condition_selector.startswith(f"{self.condition_id}:")
            or not self.method_id
        ):
            raise ValueError("persistent population coordinate is invalid")
        if (
            len(self.scene_numerators) != 10
            or len(self.scene_denominators) != 10
            or any(not math.isfinite(value) for value in self.scene_numerators)
            or any(type(value) is not int or value < 0 for value in self.scene_denominators)
        ):
            raise ValueError("persistent population statistics require ten finite scenes")

    @property
    def scene_values(self) -> tuple[float | None, ...]:
        """Reconstruct scene values without reducing away zero support."""

        return tuple(
            None if denominator == 0 else numerator / denominator
            for numerator, denominator in zip(
                self.scene_numerators,
                self.scene_denominators,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class ReplayPersistentCrossoverEstimate:
    """One M3-equivalent PAVA/crossover result on ten replay scenes."""

    replay_experiment_identity_sha256: str
    condition_id: str
    direction: ReplayCrossoverDirection
    severity_unit: str
    tested_maximum: float
    status: ReplayCrossoverStatus
    point_estimate: float | None
    interval_lower: float | None
    interval_upper: ReplayCrossoverUpper
    bootstrap_crossing_count: int
    bootstrap_replicates: int

    def __post_init__(self) -> None:
        if (
            len(self.replay_experiment_identity_sha256) != 64
            or not self.condition_id
            or not math.isfinite(self.tested_maximum)
            or self.tested_maximum <= 0.0
            or type(self.bootstrap_crossing_count) is not int
            or type(self.bootstrap_replicates) is not int
            or not 0 <= self.bootstrap_crossing_count <= self.bootstrap_replicates
        ):
            raise ValueError("persistent replay crossover record is invalid")

    @property
    def bootstrap_crossing_fraction(self) -> float:
        """Return the finite-root fraction across the frozen bootstrap."""

        return self.bootstrap_crossing_count / self.bootstrap_replicates


def _ordered_scene_rows(
    case: ReplayPersistentCase,
    evaluations: tuple[ReplayPersistentSceneEvaluation, ...],
) -> tuple[ReplayPersistentSceneEvaluation, ...]:
    if tuple(row.sequence_id for row in evaluations) != _REPLAY_SEQUENCE_IDS:
        raise ValueError("M5-A aggregation requires all ten scenes in frozen order")
    if any(
        row.replay_experiment_identity_sha256 != case.identity_sha256
        or row.condition_id != case.identity.experiment_id
        or row.condition_selector != case.fault_condition.selector
        for row in evaluations
    ):
        raise ValueError("M5-A scene row disagrees with its frozen case")
    return evaluations


def _metric(
    case: ReplayPersistentCase,
    *,
    method_id: str,
    metric_id: ReplayPersistentMetricId,
    aggregation: ReplayPersistentAggregation,
    interval: ReplayInterval,
    numerators: npt.ArrayLike,
    denominators: npt.ArrayLike,
) -> ReplayPersistentPopulationMetric:
    return ReplayPersistentPopulationMetric(
        replay_experiment_identity_sha256=case.identity_sha256,
        condition_id=case.identity.experiment_id,
        condition_selector=case.fault_condition.selector,
        method_id=method_id,
        metric_id=metric_id,
        aggregation=aggregation,
        interval=interval,
        scene_numerators=tuple(float(value) for value in np.asarray(numerators)),
        scene_denominators=tuple(int(value) for value in np.asarray(denominators)),
    )


def aggregate_replay_persistent_case(
    case: ReplayPersistentCase,
    evaluations: tuple[ReplayPersistentSceneEvaluation, ...],
    *,
    bootstrap_indices: npt.ArrayLike | None = None,
) -> tuple[ReplayPersistentPopulationMetric, ...]:
    """Aggregate one exact M5-A severity without changing M3 estimands."""

    rows = _ordered_scene_rows(case, evaluations)
    indices = replay_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    results: list[ReplayPersistentPopulationMetric] = []

    if isinstance(case.source_manifest, AvailabilityControlManifest):
        for method in case.source_manifest.methods:
            method_rows = tuple(row.result(method) for row in rows)
            losses = np.asarray([row.loss_sum_m2 for row in method_rows], dtype=np.float64)
            valid = np.asarray(
                [row.valid_object_frame_count for row in method_rows],
                dtype=np.int64,
            )
            eligible = np.asarray(
                [row.eligible_object_frame_count for row in method_rows],
                dtype=np.int64,
            )
            undefined = eligible - valid
            results.extend(
                (
                    _metric(
                        case,
                        method_id=method,
                        metric_id="coverage",
                        aggregation="pooled-valid-eligible-count-ratio",
                        interval=pooled_availability_interval(valid, eligible, indices),
                        numerators=valid,
                        denominators=eligible,
                    ),
                    _metric(
                        case,
                        method_id=method,
                        metric_id="conditional-matched-center-mse",
                        aggregation="pooled-valid-loss",
                        interval=pooled_conditional_loss_interval(losses, valid, indices),
                        numerators=losses,
                        denominators=valid,
                    ),
                    _metric(
                        case,
                        method_id=method,
                        metric_id="undefined-output-rate",
                        aggregation="pooled-valid-eligible-count-ratio",
                        interval=pooled_availability_interval(undefined, eligible, indices),
                        numerators=undefined,
                        denominators=eligible,
                    ),
                    _metric(
                        case,
                        method_id=method,
                        metric_id="scene-equal-coverage",
                        aggregation="equal-scene-mean",
                        interval=equal_scene_ratio_interval(valid, eligible, indices),
                        numerators=valid,
                        denominators=eligible,
                    ),
                )
            )
        return tuple(results)

    for method in case.source_manifest.methods:
        method_rows = tuple(row.result(method) for row in rows)
        losses = np.asarray([row.loss_sum_m2 for row in method_rows], dtype=np.float64)
        valid = np.asarray(
            [row.valid_object_frame_count for row in method_rows],
            dtype=np.int64,
        )
        eligible = np.asarray(
            [row.eligible_object_frame_count for row in method_rows],
            dtype=np.int64,
        )
        results.append(
            _metric(
                case,
                method_id=method,
                metric_id="matched-center-mse",
                aggregation="equal-scene-mean",
                interval=equal_scene_loss_interval(losses, valid, eligible, indices),
                numerators=losses,
                denominators=eligible,
            )
        )

    if isinstance(case.source_manifest, GeometryCrossoverManifest):
        healthy_method = "lidar-only" if case.fault_condition.target == "camera" else "camera-only"
        fixed = next(row for row in results if row.method_id == "fixed-fusion").scene_values
        healthy = next(row for row in results if row.method_id == healthy_method).scene_values
        if any(value is None for value in (*fixed, *healthy)):
            raise ValueError("M5-A fused-minus-healthy requires complete scene support")
        contrasts = np.asarray(
            [
                cast(float, fixed_value) - cast(float, healthy_value)
                for fixed_value, healthy_value in zip(fixed, healthy, strict=True)
            ],
            dtype=np.float64,
        )
        results.append(
            _metric(
                case,
                method_id="fixed-fusion",
                metric_id="fused-minus-healthy",
                aggregation="equal-scene-mean",
                interval=equal_scene_value_interval(contrasts, indices),
                numerators=contrasts,
                denominators=np.ones(10, dtype=np.int64),
            )
        )
    else:
        disagreement_sums = np.asarray(
            [row.cross_modal_disagreement_sum_m2 for row in rows],
            dtype=np.float64,
        )
        disagreement_counts = np.asarray(
            [row.cross_modal_common_count for row in rows],
            dtype=np.int64,
        )
        eligible = np.asarray(
            [row.results[0].eligible_object_frame_count for row in rows],
            dtype=np.int64,
        )
        results.append(
            _metric(
                case,
                method_id="camera-lidar-pair",
                metric_id="camera-lidar-disagreement-mse",
                aggregation="equal-scene-mean",
                interval=equal_scene_loss_interval(
                    disagreement_sums,
                    disagreement_counts,
                    eligible,
                    indices,
                ),
                numerators=disagreement_sums,
                denominators=disagreement_counts,
            )
        )
    return tuple(results)


def evaluate_replay_persistent_crossovers(
    cases: tuple[ReplayPersistentCase, ...],
    aggregates: tuple[ReplayPersistentPopulationMetric, ...],
    *,
    bootstrap_indices: npt.ArrayLike | None = None,
) -> tuple[ReplayPersistentCrossoverEstimate, ...]:
    """Apply the unchanged M3 PAVA and right-censored crossover algorithm."""

    if not cases or not isinstance(cases[0].source_manifest, GeometryCrossoverManifest):
        raise ValueError("replay crossover requires one geometry experiment")
    identity_sha256 = cases[0].identity_sha256
    condition_id = cases[0].identity.experiment_id
    if any(
        case.identity_sha256 != identity_sha256
        or case.identity.experiment_id != condition_id
        or case.source_manifest != cases[0].source_manifest
        for case in cases
    ):
        raise ValueError("replay crossover cases must belong to one frozen experiment")
    contrast_by_selector = {
        row.condition_selector: row for row in aggregates if row.metric_id == "fused-minus-healthy"
    }
    if set(contrast_by_selector) != {case.fault_condition.selector for case in cases}:
        raise ValueError("replay crossover lacks the complete severity curve")

    manifest = cases[0].source_manifest
    indices = replay_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    directions = tuple(
        dict.fromkeys(
            case.source_condition.direction
            for case in cases
            if case.source_condition.direction != "identity"
        )
    )
    output: list[ReplayPersistentCrossoverEstimate] = []
    for raw_direction in directions:
        curve_cases = tuple(
            case for case in cases if case.source_condition.direction in {"identity", raw_direction}
        )
        magnitudes = np.asarray(
            [case.source_condition.magnitude for case in curve_cases],
            dtype=np.float64,
        )
        contrasts = np.asarray(
            [
                tuple(
                    cast(float, value)
                    for value in contrast_by_selector[case.fault_condition.selector].scene_values
                )
                for case in curve_cases
            ],
            dtype=np.float64,
        )
        point_curve = np.asarray(
            [math.fsum(float(value) for value in scene_values) / 10 for scene_values in contrasts],
            dtype=np.float64,
        )
        point_root = first_zero_crossover(
            magnitudes,
            pava_non_decreasing(point_curve),
            zero_tolerance=manifest.evaluation.crossover.zero_tolerance_m2,
        )
        roots = bootstrap_crossover_roots(
            magnitudes=magnitudes,
            sequence_contrasts=contrasts,
            indices=np.asarray(indices, dtype=np.int64),
            zero_tolerance=manifest.evaluation.crossover.zero_tolerance_m2,
        )
        crossing_count = sum(root is not None for root in roots)
        status = bootstrap_crossover_status(
            point_crossed=point_root is not None,
            crossing_count=crossing_count,
            bootstrap_replicates=len(roots),
        )
        if status == "observed":
            censored = np.asarray(
                [root if root is not None else np.inf for root in roots],
                dtype=np.float64,
            )
            interval_lower, interval_upper = percentile_interval(
                censored,
                confidence_level=manifest.evaluation.bootstrap.confidence_level,
            )
        elif status == "not-observed":
            interval_lower = float(magnitudes[-1])
            interval_upper = "positive-infinity"
        else:
            interval_lower = None
            interval_upper = None
        output.append(
            ReplayPersistentCrossoverEstimate(
                replay_experiment_identity_sha256=identity_sha256,
                condition_id=condition_id,
                direction=cast(ReplayCrossoverDirection, raw_direction),
                severity_unit=manifest.fault_sweep.unit,
                tested_maximum=float(magnitudes[-1]),
                status=status,
                point_estimate=point_root,
                interval_lower=interval_lower,
                interval_upper=interval_upper,
                bootstrap_crossing_count=crossing_count,
                bootstrap_replicates=len(roots),
            )
        )
    return tuple(output)
