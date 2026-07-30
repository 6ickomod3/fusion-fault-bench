"""Exact all-ten-scene orchestration for the frozen M5 replay benchmark.

The orchestration layer consumes an already extracted metadata-only replay
population.  It never accepts a dataset path and never opens sensor payloads.
Per-scene random draws and the paired scene-bootstrap matrix are each created
once and shared across every preregistered condition in both M5 panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    GeometryCrossoverManifest,
)
from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayHealthResultV1,
    ReplayHealthSequenceEventV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_SHA256,
    M5_SCENE_NAMES,
)
from fusion_fault_bench.replay_descriptors import (
    ReplayDescriptorAggregate,
    build_m3_comparator_descriptor_aggregates,
    build_replay_descriptor_aggregates,
)
from fusion_fault_bench.replay_evaluation import (
    ReplayHealthSceneEvaluation,
    evaluate_replay_health_sequence,
)
from fusion_fault_bench.replay_experiments import (
    M5_DATA_MASTER_SEED,
    draw_replay_scene_randomness,
    generate_replay_condition_sequence,
)
from fusion_fault_bench.replay_fit import (
    M4_FROZEN_CALIBRATION_SHA256,
    FrozenReplayHealthFit,
    load_frozen_replay_health_fit,
    validate_frozen_replay_health_fit,
)
from fusion_fault_bench.replay_health import replay_health_schedule
from fusion_fault_bench.replay_health_population import (
    ReplayHealthPopulationMetric,
    aggregate_replay_health_case,
    validate_replay_health_population_grid,
)
from fusion_fault_bench.replay_inference import (
    ReplayHealthSequenceContrast,
    replay_bootstrap_indices,
)
from fusion_fault_bench.replay_persistent import (
    ReplayPersistentSceneEvaluation,
    evaluate_replay_persistent_sequence,
)
from fusion_fault_bench.replay_persistent_inference import (
    ReplayPersistentCrossoverEstimate,
    ReplayPersistentPopulationMetric,
    aggregate_replay_persistent_case,
    evaluate_replay_persistent_crossovers,
)
from fusion_fault_bench.replay_plan import LoadedReplayPlan, load_replay_plan
from fusion_fault_bench.replay_source import (
    ReplayPopulation,
    build_scene_descriptor_primitives,
    log_group_count,
)

_M5_PERSISTENT_CASE_COUNT = 71
_M5_HEALTH_CASE_COUNT = 43
_M5_SCENE_COUNT = 10
_M5_CROSSOVER_COUNT = 10
_M5_BOOTSTRAP_REPLICATES = 2_000
_M5_PERSISTENT_METRIC_COUNT = 464
_M5_HEALTH_RESULT_COUNT = 12_660
_M5_HEALTH_CONTRAST_COUNT = 6_450
_M5_HEALTH_EVENT_COUNT = 1_720
_M5_HEALTH_METRIC_COUNT = 14_988
_M5_SEQUENCE_IDS = tuple(f"nuscenes:{name}" for name in M5_SCENE_NAMES)
_M5_CROSSOVER_COORDINATES = (
    ("replay-lidar-y-bias", "negative", "m", 4.0),
    ("replay-lidar-y-bias", "positive", "m", 4.0),
    ("replay-camera-noise-correctly-reported", "increase", "std-scale", 4.0),
    ("replay-camera-noise-underreported", "increase", "std-scale", 4.0),
    ("replay-camera-calibration-x", "negative", "m", 4.0),
    ("replay-camera-calibration-x", "positive", "m", 4.0),
    ("replay-camera-calibration-yaw", "negative", "rad", 0.08),
    ("replay-camera-calibration-yaw", "positive", "rad", 0.08),
    ("replay-camera-timestamp-offset", "negative", "s", 0.8),
    ("replay-camera-timestamp-offset", "positive", "s", 0.8),
)
_QUANTILE_SUFFIXES = ("q0", "q25", "q50", "q75", "q100")
_SHARED_QUANTILE_DESCRIPTOR_BASES = (
    "reference-time-delta",
    "eligible-track-length",
    "ego-range",
    "ego-bearing",
    "finite-difference-speed",
)
_REPLAY_ONLY_QUANTILE_DESCRIPTOR_BASES = (
    "camera-minus-lidar-acquisition-offset",
    "box-width",
    "box-length",
    "box-height",
    "finite-difference-acceleration",
    "lidar-point-count",
)
_M5_REPLAY_DESCRIPTOR_IDS = frozenset(
    {
        "sample-count",
        "eligible-object-frame-count",
        "unique-eligible-track-count",
        "zero-order-hold-velocity-fraction",
        "support-all-annotations",
        "support-roi-pass",
        "support-camera-center-pass",
        "support-lidar-points-positive",
        "support-final-eligible",
        "visibility-level",
        "visibility-level-per-scene-fraction",
        "category-composition",
        "category-composition-per-scene-fraction",
        "distinct-log-group-count",
        *(
            f"{base}-{suffix}"
            for base in (
                *_SHARED_QUANTILE_DESCRIPTOR_BASES,
                *_REPLAY_ONLY_QUANTILE_DESCRIPTOR_BASES,
            )
            for suffix in _QUANTILE_SUFFIXES
        ),
    }
)
_M3_COMPARATOR_DESCRIPTOR_IDS = frozenset(
    {
        "sample-count",
        "eligible-object-frame-count",
        *(
            f"{base}-{suffix}"
            for base in _SHARED_QUANTILE_DESCRIPTOR_BASES
            for suffix in _QUANTILE_SUFFIXES
        ),
        "camera-minus-lidar-acquisition-offset",
        "box-width",
        "box-length",
        "box-height",
        "finite-difference-acceleration",
        "visibility-level",
        "lidar-point-count",
        "zero-order-hold-velocity-fraction",
        "category-composition",
        "distinct-log-group-count",
    }
)
_M3_COMPARATOR_DESCRIPTOR_COUNT = 91
_HEALTH_WINDOWS = ("score", "event", "recovery")
_HEALTH_POLICIES = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
)
_HEALTH_EVENT_POLICIES = _HEALTH_POLICIES[:-1]
_HEALTH_BASE_METHODS = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    *_HEALTH_POLICIES,
)
_HEALTH_STANDARD_METHODS = (
    *_HEALTH_BASE_METHODS,
    "fault-target-drop-policy",
    "frame-action-performance-oracle",
)


def _require_unique[RowT](
    rows: tuple[RowT, ...],
    *,
    keys: tuple[object, ...],
    label: str,
) -> None:
    if len(rows) != len(keys) or len(keys) != len(set(keys)):
        raise ValueError(f"replay benchmark {label} coordinates are incomplete or duplicated")


def _validate_descriptor_grid(
    rows: tuple[ReplayDescriptorAggregate, ...],
    *,
    scene_frame_counts: tuple[int, ...],
    distinct_log_group_count: int,
) -> None:
    keys = tuple(
        (
            row.population,
            row.descriptor_id,
            row.statistic,
            row.category_label,
            row.status,
            row.unit,
        )
        for row in rows
    )
    _require_unique(rows, keys=keys, label="descriptor")
    replay_ids = {row.descriptor_id for row in rows if row.population == "nuscenes-mini-replay"}
    comparator_rows = tuple(row for row in rows if row.population == "m3-main-test-comparator")
    comparator_ids = {row.descriptor_id for row in comparator_rows}
    if (
        frozenset(replay_ids) != _M5_REPLAY_DESCRIPTOR_IDS
        or frozenset(comparator_ids) != _M3_COMPARATOR_DESCRIPTOR_IDS
        or len(comparator_rows) != _M3_COMPARATOR_DESCRIPTOR_COUNT
    ):
        raise ValueError("replay benchmark descriptor comparison is incomplete")

    expected: set[tuple[object, ...]] = set()
    replay_scalar_units = {
        "sample-count": "count",
        "eligible-object-frame-count": "count",
        "unique-eligible-track-count": "count",
        "zero-order-hold-velocity-fraction": "fraction",
        "support-all-annotations": "count",
        "support-roi-pass": "count",
        "support-camera-center-pass": "count",
        "support-lidar-points-positive": "count",
        "support-final-eligible": "count",
    }
    for descriptor_id, unit in replay_scalar_units.items():
        expected.update(
            (
                "nuscenes-mini-replay",
                descriptor_id,
                statistic,
                None,
                "ok",
                unit,
            )
            for statistic in ("minimum", "median", "maximum")
        )
    replay_quantile_units = {
        "reference-time-delta": "s",
        "eligible-track-length": "frames",
        "ego-range": "m",
        "ego-bearing": "rad",
        "finite-difference-speed": "m/s",
        "camera-minus-lidar-acquisition-offset": "s",
        "box-width": "m",
        "box-length": "m",
        "box-height": "m",
        "finite-difference-acceleration": "m/s^2",
        "lidar-point-count": "count",
    }
    for descriptor_base, unit in replay_quantile_units.items():
        for suffix in _QUANTILE_SUFFIXES:
            expected.update(
                (
                    "nuscenes-mini-replay",
                    f"{descriptor_base}-{suffix}",
                    statistic,
                    None,
                    "ok",
                    unit,
                )
                for statistic in ("minimum", "median", "maximum")
            )
    expected.add(
        (
            "nuscenes-mini-replay",
            "distinct-log-group-count",
            "count",
            None,
            "ok",
            "count",
        )
    )
    for descriptor_id in ("visibility-level", "category-composition"):
        labels = {
            row.category_label
            for row in rows
            if row.population == "nuscenes-mini-replay"
            and row.descriptor_id == descriptor_id
            and row.statistic == "count"
        }
        if not labels or None in labels:
            raise ValueError("replay categorical descriptor labels are incomplete")
        for label in labels:
            expected.update(
                (
                    "nuscenes-mini-replay",
                    descriptor_id,
                    statistic,
                    label,
                    "ok",
                    unit,
                )
                for statistic, unit in (("count", "count"), ("fraction", "fraction"))
            )
            expected.update(
                (
                    "nuscenes-mini-replay",
                    f"{descriptor_id}-per-scene-fraction",
                    statistic,
                    label,
                    "ok",
                    "fraction",
                )
                for statistic in ("minimum", "median", "maximum")
            )

    for descriptor_id in ("sample-count", "eligible-object-frame-count"):
        expected.update(
            (
                "m3-main-test-comparator",
                descriptor_id,
                statistic,
                None,
                "ok",
                "count",
            )
            for statistic in ("minimum", "median", "maximum")
        )
    comparator_quantile_units = {
        "reference-time-delta": "s",
        "eligible-track-length": "frames",
        "ego-range": "m",
        "ego-bearing": "rad",
        "finite-difference-speed": "m/s",
    }
    for descriptor_base, unit in comparator_quantile_units.items():
        for suffix in _QUANTILE_SUFFIXES:
            expected.update(
                (
                    "m3-main-test-comparator",
                    f"{descriptor_base}-{suffix}",
                    statistic,
                    None,
                    "ok",
                    unit,
                )
                for statistic in ("minimum", "median", "maximum")
            )
    modeled_absences = _M3_COMPARATOR_DESCRIPTOR_IDS - {
        "sample-count",
        "eligible-object-frame-count",
        *(
            f"{base}-{suffix}"
            for base in _SHARED_QUANTILE_DESCRIPTOR_BASES
            for suffix in _QUANTILE_SUFFIXES
        ),
    }
    expected.update(
        (
            "m3-main-test-comparator",
            descriptor_id,
            "not-modeled",
            None,
            "not-applicable",
            "unitless",
        )
        for descriptor_id in modeled_absences
    )
    if set(keys) != expected:
        raise ValueError("replay benchmark descriptor statistic grid is incomplete")

    replay_values = {
        (row.descriptor_id, row.statistic): row.value
        for row in rows
        if row.population == "nuscenes-mini-replay" and row.category_label is None
    }
    frame_array = np.asarray(scene_frame_counts, dtype=np.float64)
    expected_frame_values = {
        "minimum": float(np.min(frame_array)),
        "median": float(np.quantile(frame_array, 0.5, method="linear")),
        "maximum": float(np.max(frame_array)),
    }
    if any(
        replay_values[("sample-count", statistic)] != value
        for statistic, value in expected_frame_values.items()
    ) or replay_values[("distinct-log-group-count", "count")] != float(distinct_log_group_count):
        raise ValueError("replay benchmark descriptor provenance is inconsistent")


def _validate_crossover_grid(
    rows: tuple[ReplayPersistentCrossoverEstimate, ...],
    *,
    plan: LoadedReplayPlan,
) -> None:
    identity_by_experiment = {
        case.identity.experiment_id: case.identity_sha256 for case in plan.persistent_cases
    }
    coordinates = tuple(
        (
            row.replay_experiment_identity_sha256,
            row.condition_id,
            row.direction,
            row.severity_unit,
            row.tested_maximum,
        )
        for row in rows
    )
    expected = tuple(
        (identity_by_experiment[condition_id], condition_id, direction, unit, maximum)
        for condition_id, direction, unit, maximum in _M5_CROSSOVER_COORDINATES
    )
    if coordinates != expected:
        raise ValueError("replay benchmark crossover grid is incomplete or out of order")


def _validate_persistent_grid(
    plan: LoadedReplayPlan,
    scene_rows: tuple[ReplayPersistentSceneEvaluation, ...],
    metrics: tuple[ReplayPersistentPopulationMetric, ...],
) -> None:
    expected_scene_coordinates = tuple(
        (
            case.identity_sha256,
            case.identity.experiment_id,
            case.fault_condition.selector,
            sequence_id,
            tuple(case.source_manifest.methods),
        )
        for case in plan.persistent_cases
        for sequence_id in _M5_SEQUENCE_IDS
    )
    actual_scene_coordinates = tuple(
        (
            row.replay_experiment_identity_sha256,
            row.condition_id,
            row.condition_selector,
            row.sequence_id,
            tuple(result.method for result in row.results),
        )
        for row in scene_rows
    )
    if actual_scene_coordinates != expected_scene_coordinates:
        raise ValueError("replay benchmark persistent scene grid is incomplete or out of order")

    expected_metric_coordinates: list[tuple[str, str, str, str, str, str]] = []
    for case in plan.persistent_cases:
        prefix = (
            case.identity_sha256,
            case.identity.experiment_id,
            case.fault_condition.selector,
        )
        if isinstance(case.source_manifest, AvailabilityControlManifest):
            for method in case.source_manifest.methods:
                expected_metric_coordinates.extend(
                    (
                        (*prefix, method, "coverage", "pooled-valid-eligible-count-ratio"),
                        (*prefix, method, "conditional-matched-center-mse", "pooled-valid-loss"),
                        (
                            *prefix,
                            method,
                            "undefined-output-rate",
                            "pooled-valid-eligible-count-ratio",
                        ),
                        (*prefix, method, "scene-equal-coverage", "equal-scene-mean"),
                    )
                )
            continue

        expected_metric_coordinates.extend(
            (
                *prefix,
                method,
                "matched-center-mse",
                "equal-scene-mean",
            )
            for method in case.source_manifest.methods
        )
        if isinstance(case.source_manifest, GeometryCrossoverManifest):
            expected_metric_coordinates.append(
                (
                    *prefix,
                    "fixed-fusion",
                    "fused-minus-healthy",
                    "equal-scene-mean",
                )
            )
        else:
            expected_metric_coordinates.append(
                (
                    *prefix,
                    "camera-lidar-pair",
                    "camera-lidar-disagreement-mse",
                    "equal-scene-mean",
                )
            )

    actual_metric_coordinates = tuple(
        (
            row.replay_experiment_identity_sha256,
            row.condition_id,
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.aggregation,
        )
        for row in metrics
    )
    expected_metrics = tuple(expected_metric_coordinates)
    if (
        len(expected_metrics) != _M5_PERSISTENT_METRIC_COUNT
        or actual_metric_coordinates != expected_metrics
    ):
        raise ValueError("replay benchmark persistent metric grid is incomplete or out of order")


def _validate_health_raw_grid(
    plan: LoadedReplayPlan,
    *,
    scene_frame_counts: tuple[int, ...],
    results: tuple[ReplayHealthResultV1, ...],
    contrasts: tuple[ReplayHealthSequenceContrast, ...],
    events: tuple[ReplayHealthSequenceEventV1, ...],
) -> None:
    expected_results = {
        (
            case.identity_sha256,
            case.identity.experiment_id,
            case.selector,
            sequence_id,
            method,
            window,
        )
        for case in plan.health_cases
        for sequence_id in _M5_SEQUENCE_IDS
        for method in (
            _HEALTH_BASE_METHODS
            if case.family == "common-mode-position-bias"
            else _HEALTH_STANDARD_METHODS
        )
        for window in _HEALTH_WINDOWS
    }
    actual_results = tuple(
        (
            row.replay_experiment_identity_sha256,
            row.condition_id,
            row.condition_selector,
            row.sequence_id,
            row.method,
            row.window,
        )
        for row in results
    )
    if (
        len(expected_results) != _M5_HEALTH_RESULT_COUNT
        or len(actual_results) != _M5_HEALTH_RESULT_COUNT
        or len(set(actual_results)) != len(actual_results)
        or set(actual_results) != expected_results
    ):
        raise ValueError("replay benchmark health result grid is incomplete")

    expected_contrasts = {
        (
            case.identity_sha256,
            case.identity.experiment_id,
            case.selector,
            sequence_id,
            policy,
            window,
        )
        for case in plan.health_cases
        for sequence_id in _M5_SEQUENCE_IDS
        for policy in _HEALTH_POLICIES
        for window in _HEALTH_WINDOWS
    }
    actual_contrasts = tuple(
        (
            row.replay_experiment_identity_sha256,
            row.condition_id,
            row.condition_selector,
            row.sequence_id,
            row.policy,
            row.window,
        )
        for row in contrasts
    )
    if (
        len(expected_contrasts) != _M5_HEALTH_CONTRAST_COUNT
        or len(actual_contrasts) != _M5_HEALTH_CONTRAST_COUNT
        or len(set(actual_contrasts)) != len(actual_contrasts)
        or set(actual_contrasts) != expected_contrasts
    ):
        raise ValueError("replay benchmark health contrast grid is incomplete")

    expected_events = {
        (
            case.identity_sha256,
            case.identity.experiment_id,
            case.selector,
            sequence_id,
            policy,
        )
        for case in plan.health_cases
        for sequence_id in _M5_SEQUENCE_IDS
        for policy in _HEALTH_EVENT_POLICIES
    }
    actual_events = tuple(
        (
            row.replay_experiment_identity_sha256,
            row.condition_id,
            row.condition_selector,
            row.sequence_id,
            row.policy,
        )
        for row in events
    )
    if (
        len(expected_events) != _M5_HEALTH_EVENT_COUNT
        or len(actual_events) != _M5_HEALTH_EVENT_COUNT
        or len(set(actual_events)) != len(actual_events)
        or set(actual_events) != expected_events
    ):
        raise ValueError("replay benchmark health event grid is incomplete")

    case_by_selector = {case.selector: case for case in plan.health_cases}
    frame_count_by_sequence = dict(zip(_M5_SEQUENCE_IDS, scene_frame_counts, strict=True))
    for row in events:
        case = case_by_selector[row.condition_selector]
        if (
            row.fault_family != case.family
            or row.fault_target != case.target
            or row.schedule != replay_health_schedule(frame_count_by_sequence[row.sequence_id])
        ):
            raise ValueError("replay benchmark health event semantics disagree with frozen intent")


@dataclass(frozen=True, slots=True, repr=False)
class ReplayBenchmarkEvidence:
    """Complete local M5 evidence plus aggregate-only public inputs.

    Sequence-level members remain local-only.  Their opaque object rows are
    intentionally absent; the retained rows contain only sufficient statistics,
    public sequence IDs, and preregistered condition coordinates.
    """

    plan: LoadedReplayPlan
    fit_calibration_sha256: str
    data_master_seed: int
    bootstrap_replicates: int
    log_group_ordinals: tuple[str, ...]
    scene_frame_counts: tuple[int, ...]
    descriptor_aggregates: tuple[ReplayDescriptorAggregate, ...]
    persistent_scene_evaluations: tuple[ReplayPersistentSceneEvaluation, ...]
    persistent_metrics: tuple[ReplayPersistentPopulationMetric, ...]
    persistent_crossovers: tuple[ReplayPersistentCrossoverEstimate, ...]
    health_results: tuple[ReplayHealthResultV1, ...]
    health_contrasts: tuple[ReplayHealthSequenceContrast, ...]
    health_events: tuple[ReplayHealthSequenceEventV1, ...]
    health_metrics: tuple[ReplayHealthPopulationMetric, ...]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __post_init__(self) -> None:
        if (
            self.plan.intent.intent_sha256 != M5_REPLAY_INTENT_SHA256
            or len(self.plan.persistent_cases) != _M5_PERSISTENT_CASE_COUNT
            or len(self.plan.health_cases) != _M5_HEALTH_CASE_COUNT
        ):
            raise ValueError("replay benchmark evidence does not bind the exact M5 plan")
        if (
            self.fit_calibration_sha256 != M4_FROZEN_CALIBRATION_SHA256
            or self.data_master_seed != M5_DATA_MASTER_SEED
            or self.bootstrap_replicates != _M5_BOOTSTRAP_REPLICATES
        ):
            raise ValueError("replay benchmark evidence does not bind frozen randomness and fit")
        if (
            len(self.log_group_ordinals) != _M5_SCENE_COUNT
            or len(self.scene_frame_counts) != _M5_SCENE_COUNT
            or any(
                type(frame_count) is not int or frame_count < 16
                for frame_count in self.scene_frame_counts
            )
            or len(self.persistent_scene_evaluations) != _M5_PERSISTENT_CASE_COUNT * _M5_SCENE_COUNT
        ):
            raise ValueError("replay benchmark evidence is incomplete")
        distinct_groups = tuple(
            sorted(set(self.log_group_ordinals), key=lambda value: value.encode("utf-8"))
        )
        if distinct_groups != tuple(
            f"log-group:{index:02d}" for index in range(len(distinct_groups))
        ):
            raise ValueError("replay benchmark log-group ordinals are not contiguous")

        _validate_persistent_grid(
            self.plan,
            self.persistent_scene_evaluations,
            self.persistent_metrics,
        )
        _validate_crossover_grid(
            self.persistent_crossovers,
            plan=self.plan,
        )
        _validate_health_raw_grid(
            self.plan,
            scene_frame_counts=self.scene_frame_counts,
            results=self.health_results,
            contrasts=self.health_contrasts,
            events=self.health_events,
        )
        validate_replay_health_population_grid(
            self.plan,
            self.health_metrics,
            contrasts=self.health_contrasts,
        )
        _validate_descriptor_grid(
            self.descriptor_aggregates,
            scene_frame_counts=self.scene_frame_counts,
            distinct_log_group_count=len(distinct_groups),
        )


def _validate_population(population: ReplayPopulation) -> tuple[str, ...]:
    """Validate all data-dependent release prerequisites before any fault run."""

    log_groups: list[str] = []
    for scene in population.scenes:
        if len(scene.frames) < 16:
            raise ValueError("every M5 scene must contain at least 16 frames")
        replay_health_schedule(len(scene.frames))
        if not any(frame.eligible_objects for frame in scene.frames):
            raise ValueError("every M5 scene must have nonempty frozen base support")
        log_groups.append(scene.log_group_id)
    distinct_groups = tuple(sorted(set(log_groups), key=lambda value: value.encode("utf-8")))
    if len(distinct_groups) < 2:
        raise ValueError("M5 persistence analysis requires at least two log groups")
    if distinct_groups != tuple(f"log-group:{index:02d}" for index in range(len(distinct_groups))):
        raise ValueError("M5 log-group ordinals must be contiguous")
    return tuple(log_groups)


def _persistent_crossovers(
    plan: LoadedReplayPlan,
    metrics: tuple[ReplayPersistentPopulationMetric, ...],
    *,
    bootstrap_indices: npt.ArrayLike,
) -> tuple[ReplayPersistentCrossoverEstimate, ...]:
    experiment_order = tuple(
        dict.fromkeys(
            case.identity.experiment_id
            for case in plan.persistent_cases
            if isinstance(case.source_manifest, GeometryCrossoverManifest)
        )
    )
    output: list[ReplayPersistentCrossoverEstimate] = []
    for experiment_id in experiment_order:
        cases = tuple(
            case for case in plan.persistent_cases if case.identity.experiment_id == experiment_id
        )
        experiment_metrics = tuple(row for row in metrics if row.condition_id == experiment_id)
        output.extend(
            evaluate_replay_persistent_crossovers(
                cases,
                experiment_metrics,
                bootstrap_indices=bootstrap_indices,
            )
        )
    return tuple(output)


def _run_replay_benchmark(
    population: ReplayPopulation,
    *,
    plan: LoadedReplayPlan,
    fit: FrozenReplayHealthFit,
) -> ReplayBenchmarkEvidence:
    """Execute the frozen M5 computation with authenticated dependencies."""

    validate_frozen_replay_health_fit(fit)
    log_groups = _validate_population(population)

    # Descriptors are intentionally frozen before fault outcomes are generated.
    primitives = tuple(build_scene_descriptor_primitives(scene) for scene in population.scenes)
    descriptor_aggregates = (
        *build_replay_descriptor_aggregates(
            primitives,
            distinct_log_group_count=log_group_count(population),
        ),
        *build_m3_comparator_descriptor_aggregates(plan),
    )

    # Exactly one draw set per scene and one bootstrap matrix are shared by both
    # panels and every severity coordinate.
    draws = tuple(draw_replay_scene_randomness(scene) for scene in population.scenes)
    bootstrap = replay_bootstrap_indices()

    persistent_scene_rows: list[ReplayPersistentSceneEvaluation] = []
    persistent_metrics: list[ReplayPersistentPopulationMetric] = []
    for case in plan.persistent_cases:
        evaluations: list[ReplayPersistentSceneEvaluation] = []
        for scene, scene_draws in zip(population.scenes, draws, strict=True):
            sequence = generate_replay_condition_sequence(
                scene,
                condition=case.fault_condition,
                draws=scene_draws,
            )
            evaluations.append(evaluate_replay_persistent_sequence(case, sequence))
        case_rows = tuple(evaluations)
        persistent_scene_rows.extend(case_rows)
        persistent_metrics.extend(
            aggregate_replay_persistent_case(
                case,
                case_rows,
                bootstrap_indices=bootstrap,
            )
        )
    persistent_metric_rows = tuple(persistent_metrics)

    health_results: list[ReplayHealthResultV1] = []
    health_contrasts: list[ReplayHealthSequenceContrast] = []
    health_events: list[ReplayHealthSequenceEventV1] = []
    health_metrics: list[ReplayHealthPopulationMetric] = []
    for case in plan.health_cases:
        health_scene_rows: list[ReplayHealthSceneEvaluation] = []
        for scene, scene_draws in zip(population.scenes, draws, strict=True):
            sequence = generate_replay_condition_sequence(
                scene,
                condition=case.for_frame_count(len(scene.frames)),
                draws=scene_draws,
            )
            health_scene_rows.append(
                evaluate_replay_health_sequence(
                    sequence,
                    case=case,
                    fit=fit,
                )
            )
        case_rows = tuple(health_scene_rows)
        health_metrics.extend(
            aggregate_replay_health_case(
                case,
                case_rows,
                bootstrap_indices=bootstrap,
            )
        )
        for evaluation in case_rows:
            health_results.extend(evaluation.results)
            health_contrasts.extend(evaluation.contrasts)
            health_events.extend(evaluation.events)

    return ReplayBenchmarkEvidence(
        plan=plan,
        fit_calibration_sha256=fit.calibration_sha256,
        data_master_seed=M5_DATA_MASTER_SEED,
        bootstrap_replicates=int(bootstrap.shape[0]),
        log_group_ordinals=log_groups,
        scene_frame_counts=tuple(len(scene.frames) for scene in population.scenes),
        descriptor_aggregates=descriptor_aggregates,
        persistent_scene_evaluations=tuple(persistent_scene_rows),
        persistent_metrics=persistent_metric_rows,
        persistent_crossovers=_persistent_crossovers(
            plan,
            persistent_metric_rows,
            bootstrap_indices=bootstrap,
        ),
        health_results=tuple(health_results),
        health_contrasts=tuple(health_contrasts),
        health_events=tuple(health_events),
        health_metrics=tuple(health_metrics),
    )


def run_replay_benchmark(
    population: ReplayPopulation,
    *,
    source_root: Path,
) -> ReplayBenchmarkEvidence:
    """Authenticate M5 intent/M4 fit and execute the complete CPU-only replay."""

    plan = load_replay_plan(source_root=source_root)
    fit = load_frozen_replay_health_fit(source_root)
    return _run_replay_benchmark(population, plan=plan, fit=fit)
