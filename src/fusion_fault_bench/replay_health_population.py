"""Complete-scene M5-B aggregation with variable support and dynamic events."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.health_result_v1 import (
    HealthMethod,
    HealthPolicyMethod,
    HealthWindow,
)
from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayHealthResultV1,
    ReplayHealthSequenceEventV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_SHA256,
    M5_SCENE_NAMES,
)
from fusion_fault_bench.health_inference import recovery_fraction_interval
from fusion_fault_bench.replay_evaluation import ReplayHealthSceneEvaluation
from fusion_fault_bench.replay_inference import (
    ReplayHealthSequenceContrast,
    ReplayInterval,
    conditional_observed_mean_interval,
    equal_scene_contrast_interval,
    equal_scene_loss_interval,
    equal_scene_ratio_interval,
    equal_scene_value_interval,
    observed_fraction_interval,
    pooled_availability_interval,
    pooled_conditional_loss_interval,
    replay_bootstrap_indices,
    replay_sequence_contrast_values,
)
from fusion_fault_bench.replay_plan import (
    LoadedReplayPlan,
    ReplayHealthCaseSpec,
)

type ReplayHealthAggregation = Literal[
    "equal-scene-mean",
    "pooled-valid-eligible-count-ratio",
    "pooled-valid-loss",
    "conditional-observed-scene-mean",
    "unclipped-recovery-ratio",
]
type ReplayHealthMetricStatus = Literal["ok", "undefined", "not-applicable"]
type ReplayHealthUnit = Literal[
    "m^2",
    "fraction",
    "observation-step",
    "s",
    "count",
    "unitless",
]

_REPLAY_SEQUENCE_IDS = tuple(f"nuscenes:{name}" for name in M5_SCENE_NAMES)
_POLICIES: tuple[HealthPolicyMethod, ...] = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
)
_EVENT_POLICIES = _POLICIES[:-1]
_WINDOWS: tuple[HealthWindow, ...] = ("score", "event", "recovery")
_BASE_RESULT_METHODS: tuple[HealthMethod, ...] = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    *_POLICIES,
)
_STANDARD_RESULT_METHODS: tuple[HealthMethod, ...] = (
    *_BASE_RESULT_METHODS,
    "fault-target-drop-policy",
    "frame-action-performance-oracle",
)
_RESULT_METRICS: tuple[
    tuple[str, ReplayHealthUnit, ReplayHealthAggregation],
    ...,
] = (
    ("matched-center-mse", "m^2", "equal-scene-mean"),
    ("conditional-matched-center-mse", "m^2", "pooled-valid-loss"),
    ("coverage", "fraction", "pooled-valid-eligible-count-ratio"),
    ("undefined-output-rate", "fraction", "pooled-valid-eligible-count-ratio"),
    ("scene-equal-coverage", "fraction", "equal-scene-mean"),
)
_STRUCTURAL_COMMON_MODE_METRICS = frozenset(
    {
        "gap-vs-fault-target-drop",
        "gap-vs-frame-oracle",
        "frame-oracle-recoverable-loss-fraction",
    }
)


@dataclass(frozen=True, slots=True)
class ReplayHealthPopulationCoordinate:
    """One authoritative M5-B aggregate coordinate derived from frozen intent."""

    replay_experiment_identity_sha256: str
    condition_id: str
    condition_selector: str
    method_id: str
    metric_id: str
    window: HealthWindow
    unit: ReplayHealthUnit
    aggregation: ReplayHealthAggregation


@dataclass(frozen=True, slots=True)
class ReplayHealthPopulationMetric:
    """One M5-B aggregate and its ten-scene reconstruction statistics."""

    replay_experiment_identity_sha256: str
    condition_id: str
    condition_selector: str
    method_id: str
    metric_id: str
    window: HealthWindow
    unit: ReplayHealthUnit
    aggregation: ReplayHealthAggregation
    status: ReplayHealthMetricStatus
    interval: ReplayInterval
    scene_numerators: tuple[float, ...]
    scene_denominators: tuple[float, ...]
    scene_defined: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.replay_experiment_identity_sha256) != 64:
            raise ValueError("health population identity digest is invalid")
        if (
            not self.condition_id
            or not self.condition_selector.startswith(f"{self.condition_id}:")
            or not self.method_id
            or not self.metric_id
        ):
            raise ValueError("health population coordinate is invalid")
        interval_defined = self.interval.estimate is not None
        if self.status not in {"ok", "undefined", "not-applicable"} or interval_defined != (
            self.status == "ok"
        ):
            raise ValueError("health population status disagrees with its interval")
        if (
            len(self.scene_numerators) != 10
            or len(self.scene_denominators) != 10
            or len(self.scene_defined) != 10
            or any(not math.isfinite(value) for value in self.scene_numerators)
            or any(not math.isfinite(value) or value < 0.0 for value in self.scene_denominators)
            or any(type(value) is not bool for value in self.scene_defined)
            or any(
                defined and denominator == 0.0
                for denominator, defined in zip(
                    self.scene_denominators,
                    self.scene_defined,
                    strict=True,
                )
            )
        ):
            raise ValueError("health population statistics require ten valid scenes")
        if self.status == "not-applicable" and (
            self.interval.defined_replicates != 0
            or any(self.scene_defined)
            or any(value != 0.0 for value in self.scene_numerators)
            or any(value != 0.0 for value in self.scene_denominators)
        ):
            raise ValueError("not-applicable health metrics cannot carry numeric support")

    @property
    def scene_values(self) -> tuple[float | None, ...]:
        """Reconstruct per-scene values without imputing missing observations."""

        return tuple(
            None if not defined else numerator / denominator
            for numerator, denominator, defined in zip(
                self.scene_numerators,
                self.scene_denominators,
                self.scene_defined,
                strict=True,
            )
        )

    @property
    def coordinate(self) -> ReplayHealthPopulationCoordinate:
        """Return the complete identity-bound aggregate coordinate."""

        return ReplayHealthPopulationCoordinate(
            replay_experiment_identity_sha256=self.replay_experiment_identity_sha256,
            condition_id=self.condition_id,
            condition_selector=self.condition_selector,
            method_id=self.method_id,
            metric_id=self.metric_id,
            window=self.window,
            unit=self.unit,
            aggregation=self.aggregation,
        )


def _metric(
    case: ReplayHealthCaseSpec,
    *,
    method_id: str,
    metric_id: str,
    window: HealthWindow,
    unit: ReplayHealthUnit,
    aggregation: ReplayHealthAggregation,
    interval: ReplayInterval,
    numerators: npt.ArrayLike,
    denominators: npt.ArrayLike,
    defined: npt.ArrayLike | None = None,
    status: ReplayHealthMetricStatus | None = None,
) -> ReplayHealthPopulationMetric:
    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    scene_defined = denominator > 0.0 if defined is None else np.asarray(defined, dtype=np.bool_)
    if numerator.shape != (10,) or denominator.shape != (10,) or scene_defined.shape != (10,):
        raise ValueError("health aggregate statistics must align with ten scenes")
    return ReplayHealthPopulationMetric(
        replay_experiment_identity_sha256=case.identity_sha256,
        condition_id=case.identity.experiment_id,
        condition_selector=case.selector,
        method_id=method_id,
        metric_id=metric_id,
        window=window,
        unit=unit,
        aggregation=aggregation,
        status=(
            ("ok" if interval.estimate is not None else "undefined") if status is None else status
        ),
        interval=interval,
        scene_numerators=tuple(float(value) for value in numerator),
        scene_denominators=tuple(float(value) for value in denominator),
        scene_defined=tuple(bool(value) for value in scene_defined),
    )


def _not_applicable_metric(
    case: ReplayHealthCaseSpec,
    *,
    method_id: str,
    metric_id: str,
    window: HealthWindow,
    unit: ReplayHealthUnit,
    aggregation: ReplayHealthAggregation,
    bootstrap_indices: npt.ArrayLike,
) -> ReplayHealthPopulationMetric:
    """Retain one frozen coordinate unavailable by structure or common support."""

    bootstrap = np.asarray(bootstrap_indices)
    return _metric(
        case,
        method_id=method_id,
        metric_id=metric_id,
        window=window,
        unit=unit,
        aggregation=aggregation,
        status="not-applicable",
        interval=ReplayInterval(
            estimate=None,
            lower=None,
            upper=None,
            defined_replicates=0,
            bootstrap_replicates=bootstrap.shape[0],
        ),
        numerators=np.zeros(10, dtype=np.float64),
        denominators=np.zeros(10, dtype=np.float64),
        defined=np.zeros(10, dtype=np.bool_),
    )


def _validate_binding(
    case: ReplayHealthCaseSpec,
    rows: Sequence[object],
) -> None:
    for row in rows:
        if (
            getattr(row, "replay_experiment_identity_sha256", None) != case.identity_sha256
            or getattr(row, "condition_id", None) != case.identity.experiment_id
            or getattr(row, "condition_selector", None) != case.selector
        ):
            raise ValueError("M5-B sequence row disagrees with its frozen case")


def expected_replay_health_population_coordinates(
    plan: LoadedReplayPlan,
) -> frozenset[ReplayHealthPopulationCoordinate]:
    """Derive the exact 14,988-row M5-B aggregate authority from frozen intent."""

    if plan.intent.intent_sha256 != M5_REPLAY_INTENT_SHA256 or len(plan.health_cases) != 43:
        raise ValueError("health coordinate authority requires the exact frozen M5 plan")

    result_coordinates: set[ReplayHealthPopulationCoordinate] = set()
    contrast_coordinates: set[ReplayHealthPopulationCoordinate] = set()
    event_coordinates: set[ReplayHealthPopulationCoordinate] = set()

    def add(
        destination: set[ReplayHealthPopulationCoordinate],
        case: ReplayHealthCaseSpec,
        *,
        method_id: str,
        metric_id: str,
        window: HealthWindow,
        unit: ReplayHealthUnit,
        aggregation: ReplayHealthAggregation,
    ) -> None:
        destination.add(
            ReplayHealthPopulationCoordinate(
                replay_experiment_identity_sha256=case.identity_sha256,
                condition_id=case.identity.experiment_id,
                condition_selector=case.selector,
                method_id=method_id,
                metric_id=metric_id,
                window=window,
                unit=unit,
                aggregation=aggregation,
            )
        )

    def equal(
        case: ReplayHealthCaseSpec,
        *,
        method_id: str,
        metric_id: str,
        window: HealthWindow,
        unit: ReplayHealthUnit,
    ) -> None:
        add(
            event_coordinates,
            case,
            method_id=method_id,
            metric_id=metric_id,
            window=window,
            unit=unit,
            aggregation="equal-scene-mean",
        )

    def pooled(
        case: ReplayHealthCaseSpec,
        *,
        method_id: str,
        metric_id: str,
        window: HealthWindow,
    ) -> None:
        add(
            event_coordinates,
            case,
            method_id=method_id,
            metric_id=metric_id,
            window=window,
            unit="fraction",
            aggregation="pooled-valid-eligible-count-ratio",
        )

    def conditional(
        case: ReplayHealthCaseSpec,
        *,
        method_id: str,
        metric_id: str,
        window: HealthWindow,
        unit: ReplayHealthUnit,
    ) -> None:
        equal(
            case,
            method_id=method_id,
            metric_id=f"{metric_id}-observed-fraction",
            window=window,
            unit="fraction",
        )
        add(
            event_coordinates,
            case,
            method_id=method_id,
            metric_id=metric_id,
            window=window,
            unit=unit,
            aggregation="conditional-observed-scene-mean",
        )

    for case in plan.health_cases:
        result_methods = (
            _BASE_RESULT_METHODS
            if case.family == "common-mode-position-bias"
            else _STANDARD_RESULT_METHODS
        )
        for method_id in result_methods:
            for window in _WINDOWS:
                for metric_id, unit, aggregation in _RESULT_METRICS:
                    add(
                        result_coordinates,
                        case,
                        method_id=method_id,
                        metric_id=metric_id,
                        window=window,
                        unit=unit,
                        aggregation=aggregation,
                    )

        for method_id in _POLICIES:
            for window in _WINDOWS:
                for metric_id in (
                    "policy-gain-vs-fixed",
                    "gap-vs-fault-target-drop",
                    "gap-vs-frame-oracle",
                ):
                    add(
                        contrast_coordinates,
                        case,
                        method_id=method_id,
                        metric_id=metric_id,
                        window=window,
                        unit="m^2",
                        aggregation="equal-scene-mean",
                    )
                add(
                    contrast_coordinates,
                    case,
                    method_id=method_id,
                    metric_id="frame-oracle-recoverable-loss-fraction",
                    window=window,
                    unit="unitless",
                    aggregation="unclipped-recovery-ratio",
                )

        if case.family == "dropout":
            equal(
                case,
                method_id="none",
                metric_id="realized-dropout-fraction",
                window="event",
                unit="fraction",
            )
            conditional(
                case,
                method_id="none",
                metric_id="first-missing-step",
                window="event",
                unit="observation-step",
            )
            conditional(
                case,
                method_id="none",
                metric_id="first-missing-elapsed-reference-time",
                window="event",
                unit="s",
            )

        for method_id in _EVENT_POLICIES:
            equal(
                case,
                method_id=method_id,
                metric_id="detection-fraction",
                window="event",
                unit="fraction",
            )
            outcomes = (
                ("ambiguous", "missed")
                if case.target in {"none", "both"}
                else ("correct", "ambiguous", "wrong-sensor", "missed")
            )
            for outcome in outcomes:
                equal(
                    case,
                    method_id=method_id,
                    metric_id=f"event-outcome-{outcome}-fraction",
                    window="event",
                    unit="fraction",
                )
            if case.family == "common-mode-position-bias":
                for label in ("camera-fault", "lidar-fault", "ambiguous"):
                    equal(
                        case,
                        method_id=method_id,
                        metric_id=f"first-latch-label-{label}-fraction",
                        window="event",
                        unit="fraction",
                    )
            if case.target in {"camera", "lidar"}:
                equal(
                    case,
                    method_id=method_id,
                    metric_id="attribution-fraction",
                    window="event",
                    unit="fraction",
                )
            for metric_id in (
                "early-clear-fraction",
                "recovery-denominator-fraction",
            ):
                equal(
                    case,
                    method_id=method_id,
                    metric_id=metric_id,
                    window="event",
                    unit="fraction",
                )
            pooled(
                case,
                method_id=method_id,
                metric_id="recovery-fraction",
                window="recovery",
            )
            for state in ("healthy", "camera-fault", "lidar-fault", "ambiguous"):
                equal(
                    case,
                    method_id=method_id,
                    metric_id=f"final-active-state-{state}-fraction",
                    window="event",
                    unit="fraction",
                )
            if case.family == "dropout":
                pooled(
                    case,
                    method_id=method_id,
                    metric_id="detection-among-realized-dropout-fraction",
                    window="event",
                )

            latency_fields: list[tuple[str, HealthWindow, ReplayHealthUnit]] = [
                ("detection-latency-steps", "event", "observation-step"),
                ("detection-elapsed-reference-time", "event", "s"),
                ("recovery-latency-steps", "recovery", "observation-step"),
                ("recovery-elapsed-reference-time", "recovery", "s"),
            ]
            if case.target in {"camera", "lidar"}:
                latency_fields.extend(
                    (
                        (
                            "attribution-latency-steps",
                            "event",
                            "observation-step",
                        ),
                        ("attribution-elapsed-reference-time", "event", "s"),
                    )
                )
            if case.family == "dropout":
                latency_fields.extend(
                    (
                        (
                            "detection-minus-first-missing-steps",
                            "event",
                            "observation-step",
                        ),
                        (
                            "detection-minus-first-missing-elapsed-reference-time",
                            "event",
                            "s",
                        ),
                    )
                )
            for metric_id, window, unit in latency_fields:
                conditional(
                    case,
                    method_id=method_id,
                    metric_id=metric_id,
                    window=window,
                    unit=unit,
                )

            episode_metrics: tuple[tuple[str, HealthWindow], ...] = (
                ("false-alert-episode-starts", "score"),
                ("latch-episode-starts", "event"),
            )
            for metric_id, window in episode_metrics:
                equal(
                    case,
                    method_id=method_id,
                    metric_id=metric_id,
                    window=window,
                    unit="count",
                )
            for metric_id in (
                "state-healthy-occupancy",
                "state-camera-fault-occupancy",
                "state-lidar-fault-occupancy",
                "state-ambiguous-occupancy",
                "action-camera-occupancy",
                "action-lidar-occupancy",
                "action-fixed-occupancy",
                "action-undefined-occupancy",
            ):
                pooled(
                    case,
                    method_id=method_id,
                    metric_id=metric_id,
                    window="event",
                )

    if (
        len(result_coordinates) != 6_330
        or len(contrast_coordinates) != 2_580
        or len(event_coordinates) != 6_078
    ):
        raise ValueError("health coordinate authority is not the frozen component matrix")
    coordinates = frozenset((*result_coordinates, *contrast_coordinates, *event_coordinates))
    if len(coordinates) != 14_988:
        raise ValueError("health coordinate authority is not the frozen 14,988-row matrix")
    return coordinates


def validate_replay_health_population_grid(
    plan: LoadedReplayPlan,
    metrics: Sequence[ReplayHealthPopulationMetric],
    *,
    contrasts: Sequence[ReplayHealthSequenceContrast],
) -> None:
    """Validate exact M5-B aggregate coordinates and support-derived status."""

    expected_metrics = expected_replay_health_population_coordinates(plan)
    checked_metrics = tuple(metrics)
    actual_metrics = tuple(metric.coordinate for metric in checked_metrics)
    if (
        len(actual_metrics) != 14_988
        or len(set(actual_metrics)) != len(actual_metrics)
        or frozenset(actual_metrics) != expected_metrics
    ):
        raise ValueError(
            "health metric population does not match the authoritative 14,988-row grid"
        )

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
        for sequence_id in _REPLAY_SEQUENCE_IDS
        for policy in _POLICIES
        for window in _WINDOWS
    }
    checked_contrasts = tuple(contrasts)
    actual_contrasts = tuple(
        (
            row.replay_experiment_identity_sha256,
            row.condition_id,
            row.condition_selector,
            row.sequence_id,
            row.policy,
            row.window,
        )
        for row in checked_contrasts
    )
    if (
        len(actual_contrasts) != 6_450
        or len(set(actual_contrasts)) != len(actual_contrasts)
        or set(actual_contrasts) != expected_contrasts
    ):
        raise ValueError("health contrasts do not match the authoritative 6,450-row support grid")

    case_by_selector = {case.selector: case for case in plan.health_cases}
    contrast_index = {
        (row.condition_selector, row.sequence_id, row.policy, row.window): row
        for row in checked_contrasts
    }
    for row in checked_contrasts:
        case = case_by_selector[row.condition_selector]
        comparator_applicable = case.family != "common-mode-position-bias"
        if (
            row.target_drop_applicable != comparator_applicable
            or row.frame_oracle_applicable != comparator_applicable
        ):
            raise ValueError(
                "health contrast comparator applicability disagrees with frozen intent"
            )

    recovery_not_applicable = {
        (case.selector, policy, window): not all(
            contrast_index[
                (case.selector, sequence_id, policy, window)
            ].identical_support_recovery_applicable
            for sequence_id in _REPLAY_SEQUENCE_IDS
        )
        for case in plan.health_cases
        if case.family != "common-mode-position-bias"
        for policy in _POLICIES
        for window in _WINDOWS
    }
    for metric in checked_metrics:
        case = case_by_selector[metric.condition_selector]
        structural_not_applicable = (
            case.family == "common-mode-position-bias"
            and metric.metric_id in _STRUCTURAL_COMMON_MODE_METRICS
        )
        support_not_applicable = (
            metric.metric_id == "frame-oracle-recoverable-loss-fraction"
            and case.family != "common-mode-position-bias"
            and recovery_not_applicable[(case.selector, metric.method_id, metric.window)]
        )
        expected_not_applicable = structural_not_applicable or support_not_applicable
        if (metric.status == "not-applicable") != expected_not_applicable:
            raise ValueError(
                "health population applicability status disagrees with frozen "
                "structure or all-scene common support"
            )


def aggregate_replay_health_results(
    case: ReplayHealthCaseSpec,
    rows: Sequence[ReplayHealthResultV1],
    *,
    bootstrap_indices: npt.ArrayLike | None = None,
) -> tuple[ReplayHealthPopulationMetric, ...]:
    """Aggregate all M5-B method/window losses and availability diagnostics."""

    checked = tuple(rows)
    _validate_binding(case, checked)
    index: dict[tuple[str, HealthMethod, HealthWindow], ReplayHealthResultV1] = {}
    for row in checked:
        key = (row.sequence_id, row.method, row.window)
        if key in index:
            raise ValueError("duplicate replay health result row")
        index[key] = row
    method_order = (
        _BASE_RESULT_METHODS
        if case.family == "common-mode-position-bias"
        else _STANDARD_RESULT_METHODS
    )
    expected = {
        (sequence_id, method, window)
        for sequence_id in _REPLAY_SEQUENCE_IDS
        for method in method_order
        for window in _WINDOWS
    }
    if set(index) != expected:
        raise ValueError("replay health result matrix is incomplete")
    indices = replay_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    output: list[ReplayHealthPopulationMetric] = []
    for method in method_order:
        for window in _WINDOWS:
            cohort = tuple(
                index[(sequence_id, method, window)] for sequence_id in _REPLAY_SEQUENCE_IDS
            )
            losses = np.asarray([row.loss_sum_m2 for row in cohort], dtype=np.float64)
            valid = np.asarray(
                [row.valid_object_frame_count for row in cohort],
                dtype=np.int64,
            )
            eligible = np.asarray(
                [row.eligible_object_frame_count for row in cohort],
                dtype=np.int64,
            )
            undefined = eligible - valid
            output.extend(
                (
                    _metric(
                        case,
                        method_id=method,
                        metric_id="matched-center-mse",
                        window=window,
                        unit="m^2",
                        aggregation="equal-scene-mean",
                        interval=equal_scene_loss_interval(
                            losses,
                            valid,
                            eligible,
                            indices,
                        ),
                        numerators=losses,
                        denominators=eligible,
                        defined=(eligible > 0) & (valid == eligible),
                    ),
                    _metric(
                        case,
                        method_id=method,
                        metric_id="conditional-matched-center-mse",
                        window=window,
                        unit="m^2",
                        aggregation="pooled-valid-loss",
                        interval=pooled_conditional_loss_interval(
                            losses,
                            valid,
                            indices,
                        ),
                        numerators=losses,
                        denominators=valid,
                    ),
                    _metric(
                        case,
                        method_id=method,
                        metric_id="coverage",
                        window=window,
                        unit="fraction",
                        aggregation="pooled-valid-eligible-count-ratio",
                        interval=pooled_availability_interval(valid, eligible, indices),
                        numerators=valid,
                        denominators=eligible,
                    ),
                    _metric(
                        case,
                        method_id=method,
                        metric_id="undefined-output-rate",
                        window=window,
                        unit="fraction",
                        aggregation="pooled-valid-eligible-count-ratio",
                        interval=pooled_availability_interval(undefined, eligible, indices),
                        numerators=undefined,
                        denominators=eligible,
                    ),
                    _metric(
                        case,
                        method_id=method,
                        metric_id="scene-equal-coverage",
                        window=window,
                        unit="fraction",
                        aggregation="equal-scene-mean",
                        interval=equal_scene_ratio_interval(valid, eligible, indices),
                        numerators=valid,
                        denominators=eligible,
                    ),
                )
            )
    return tuple(output)


def _contrast_metric(
    case: ReplayHealthCaseSpec,
    rows: tuple[ReplayHealthSequenceContrast, ...],
    *,
    metric_id: str,
    contrast: Literal[
        "fixed-policy",
        "policy-target-drop",
        "policy-frame-oracle",
    ],
    bootstrap_indices: npt.ArrayLike,
) -> ReplayHealthPopulationMetric:
    values = replay_sequence_contrast_values(rows, contrast=contrast)
    numerator = np.asarray(
        [0.0 if value is None else value for value in values],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [0.0 if value is None else 1.0 for value in values],
        dtype=np.float64,
    )
    return _metric(
        case,
        method_id=rows[0].policy,
        metric_id=metric_id,
        window=rows[0].window,
        unit="m^2",
        aggregation="equal-scene-mean",
        interval=equal_scene_contrast_interval(
            rows,
            bootstrap_indices,
            contrast=contrast,
        ),
        numerators=numerator,
        denominators=denominator,
    )


def _recovery_fraction_metric(
    case: ReplayHealthCaseSpec,
    rows: tuple[ReplayHealthSequenceContrast, ...],
    *,
    bootstrap_indices: npt.ArrayLike,
) -> ReplayHealthPopulationMetric:
    counts = np.asarray([row.fixed_policy_common_count for row in rows], dtype=np.float64)
    fixed = (
        np.asarray(
            [row.fixed_on_common_loss_sum_m2 for row in rows],
            dtype=np.float64,
        )
        / counts
    )
    policy = (
        np.asarray(
            [row.policy_on_fixed_common_loss_sum_m2 for row in rows],
            dtype=np.float64,
        )
        / counts
    )
    oracle = (
        np.asarray(
            [row.frame_oracle_on_common_loss_sum_m2 for row in rows],
            dtype=np.float64,
        )
        / counts
    )
    interval = recovery_fraction_interval(
        fixed,
        policy,
        oracle,
        bootstrap_indices,
    )
    replay_interval = ReplayInterval(
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        defined_replicates=interval.defined_replicates,
        bootstrap_replicates=np.asarray(bootstrap_indices).shape[0],
    )
    return _metric(
        case,
        method_id=rows[0].policy,
        metric_id="frame-oracle-recoverable-loss-fraction",
        window=rows[0].window,
        unit="unitless",
        aggregation="unclipped-recovery-ratio",
        interval=replay_interval,
        numerators=fixed - policy,
        denominators=fixed - oracle,
        defined=(fixed - oracle) > 1e-12,
    )


def aggregate_replay_health_contrasts(
    case: ReplayHealthCaseSpec,
    rows: Sequence[ReplayHealthSequenceContrast],
    *,
    bootstrap_indices: npt.ArrayLike | None = None,
) -> tuple[ReplayHealthPopulationMetric, ...]:
    """Aggregate exact paired-common-support policy contrasts."""

    checked = tuple(rows)
    _validate_binding(case, checked)
    index: dict[
        tuple[str, HealthPolicyMethod, HealthWindow],
        ReplayHealthSequenceContrast,
    ] = {}
    for row in checked:
        key = (row.sequence_id, row.policy, row.window)
        if key in index:
            raise ValueError("duplicate replay health contrast row")
        index[key] = row
    expected = {
        (sequence_id, policy, window)
        for sequence_id in _REPLAY_SEQUENCE_IDS
        for policy in _POLICIES
        for window in _WINDOWS
    }
    if set(index) != expected:
        raise ValueError("replay health contrast matrix is incomplete")
    indices = replay_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    output: list[ReplayHealthPopulationMetric] = []
    for policy in _POLICIES:
        for window in _WINDOWS:
            cohort = tuple(
                index[(sequence_id, policy, window)] for sequence_id in _REPLAY_SEQUENCE_IDS
            )
            output.append(
                _contrast_metric(
                    case,
                    cohort,
                    metric_id="policy-gain-vs-fixed",
                    contrast="fixed-policy",
                    bootstrap_indices=indices,
                )
            )
            target_applicable = {row.target_drop_applicable for row in cohort}
            oracle_applicable = {row.frame_oracle_applicable for row in cohort}
            if len(target_applicable) != 1 or len(oracle_applicable) != 1:
                raise ValueError("replay contrast applicability differs across scenes")
            expected_comparator_applicability = {case.family != "common-mode-position-bias"}
            if (
                target_applicable != expected_comparator_applicability
                or oracle_applicable != expected_comparator_applicability
            ):
                raise ValueError("replay contrast applicability disagrees with the frozen case")
            if target_applicable == {True}:
                output.append(
                    _contrast_metric(
                        case,
                        cohort,
                        metric_id="gap-vs-fault-target-drop",
                        contrast="policy-target-drop",
                        bootstrap_indices=indices,
                    )
                )
            else:
                output.append(
                    _not_applicable_metric(
                        case,
                        method_id=policy,
                        metric_id="gap-vs-fault-target-drop",
                        window=window,
                        unit="m^2",
                        aggregation="equal-scene-mean",
                        bootstrap_indices=indices,
                    )
                )
            if oracle_applicable == {True}:
                output.append(
                    _contrast_metric(
                        case,
                        cohort,
                        metric_id="gap-vs-frame-oracle",
                        contrast="policy-frame-oracle",
                        bootstrap_indices=indices,
                    )
                )
                if all(row.identical_support_recovery_applicable for row in cohort):
                    output.append(
                        _recovery_fraction_metric(
                            case,
                            cohort,
                            bootstrap_indices=indices,
                        )
                    )
                else:
                    output.append(
                        _not_applicable_metric(
                            case,
                            method_id=policy,
                            metric_id="frame-oracle-recoverable-loss-fraction",
                            window=window,
                            unit="unitless",
                            aggregation="unclipped-recovery-ratio",
                            bootstrap_indices=indices,
                        )
                    )
            else:
                output.extend(
                    (
                        _not_applicable_metric(
                            case,
                            method_id=policy,
                            metric_id="gap-vs-frame-oracle",
                            window=window,
                            unit="m^2",
                            aggregation="equal-scene-mean",
                            bootstrap_indices=indices,
                        ),
                        _not_applicable_metric(
                            case,
                            method_id=policy,
                            metric_id="frame-oracle-recoverable-loss-fraction",
                            window=window,
                            unit="unitless",
                            aggregation="unclipped-recovery-ratio",
                            bootstrap_indices=indices,
                        ),
                    )
                )
    return tuple(output)


def _equal_value_metric(
    case: ReplayHealthCaseSpec,
    *,
    method_id: str,
    metric_id: str,
    window: HealthWindow,
    unit: ReplayHealthUnit,
    values: Sequence[float | int | bool],
    bootstrap_indices: npt.ArrayLike,
) -> ReplayHealthPopulationMetric:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    return _metric(
        case,
        method_id=method_id,
        metric_id=metric_id,
        window=window,
        unit=unit,
        aggregation="equal-scene-mean",
        interval=equal_scene_value_interval(numeric, bootstrap_indices),
        numerators=numeric,
        denominators=np.ones(10, dtype=np.float64),
    )


def _conditional_metric_pair(
    case: ReplayHealthCaseSpec,
    *,
    method_id: str,
    metric_id: str,
    window: HealthWindow,
    unit: ReplayHealthUnit,
    values: Sequence[float | int | None],
    bootstrap_indices: npt.ArrayLike,
) -> tuple[ReplayHealthPopulationMetric, ReplayHealthPopulationMetric]:
    checked = tuple(None if value is None else float(value) for value in values)
    observed = np.asarray([value is not None for value in checked], dtype=np.float64)
    numerators = np.asarray(
        [0.0 if value is None else value for value in checked],
        dtype=np.float64,
    )
    return (
        _metric(
            case,
            method_id=method_id,
            metric_id=f"{metric_id}-observed-fraction",
            window=window,
            unit="fraction",
            aggregation="equal-scene-mean",
            interval=observed_fraction_interval(checked, bootstrap_indices),
            numerators=observed,
            denominators=np.ones(10, dtype=np.float64),
        ),
        _metric(
            case,
            method_id=method_id,
            metric_id=metric_id,
            window=window,
            unit=unit,
            aggregation="conditional-observed-scene-mean",
            interval=conditional_observed_mean_interval(checked, bootstrap_indices),
            numerators=numerators,
            denominators=observed,
        ),
    )


def _event_index(
    case: ReplayHealthCaseSpec,
    rows: Sequence[ReplayHealthSequenceEventV1],
) -> dict[tuple[str, HealthPolicyMethod], ReplayHealthSequenceEventV1]:
    checked = tuple(rows)
    _validate_binding(case, checked)
    index: dict[tuple[str, HealthPolicyMethod], ReplayHealthSequenceEventV1] = {}
    for row in checked:
        key = (row.sequence_id, row.policy)
        if key in index:
            raise ValueError("duplicate replay event row")
        index[key] = row
    expected = {
        (sequence_id, policy) for sequence_id in _REPLAY_SEQUENCE_IDS for policy in _EVENT_POLICIES
    }
    if set(index) != expected:
        raise ValueError("replay event matrix is incomplete")
    return index


def aggregate_replay_health_events(
    case: ReplayHealthCaseSpec,
    rows: Sequence[ReplayHealthSequenceEventV1],
    *,
    bootstrap_indices: npt.ArrayLike | None = None,
) -> tuple[ReplayHealthPopulationMetric, ...]:
    """Aggregate dynamic event, latency, recovery, and occupancy diagnostics."""

    index = _event_index(case, rows)
    indices = replay_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    output: list[ReplayHealthPopulationMetric] = []

    if case.family == "dropout":
        canonical = tuple(
            index[(sequence_id, _EVENT_POLICIES[0])] for sequence_id in _REPLAY_SEQUENCE_IDS
        )
        for sequence_id in _REPLAY_SEQUENCE_IDS:
            observations = {
                (
                    index[(sequence_id, policy)].realized_dropout,
                    index[(sequence_id, policy)].first_missing_step,
                    index[(sequence_id, policy)].first_missing_latency_s,
                )
                for policy in _EVENT_POLICIES
            }
            if len(observations) != 1:
                raise ValueError("dropout observation evidence differs across policies")
        output.append(
            _equal_value_metric(
                case,
                method_id="none",
                metric_id="realized-dropout-fraction",
                window="event",
                unit="fraction",
                values=tuple(bool(row.realized_dropout) for row in canonical),
                bootstrap_indices=indices,
            )
        )
        output.extend(
            _conditional_metric_pair(
                case,
                method_id="none",
                metric_id="first-missing-step",
                window="event",
                unit="observation-step",
                values=tuple(row.first_missing_step for row in canonical),
                bootstrap_indices=indices,
            )
        )
        output.extend(
            _conditional_metric_pair(
                case,
                method_id="none",
                metric_id="first-missing-elapsed-reference-time",
                window="event",
                unit="s",
                values=tuple(row.first_missing_latency_s for row in canonical),
                bootstrap_indices=indices,
            )
        )

    for policy in _EVENT_POLICIES:
        cohort = tuple(index[(sequence_id, policy)] for sequence_id in _REPLAY_SEQUENCE_IDS)
        output.append(
            _equal_value_metric(
                case,
                method_id=policy,
                metric_id="detection-fraction",
                window="event",
                unit="fraction",
                values=tuple(row.detected for row in cohort),
                bootstrap_indices=indices,
            )
        )
        outcomes = (
            ("ambiguous", "missed")
            if case.target in {"none", "both"}
            else ("correct", "ambiguous", "wrong-sensor", "missed")
        )
        for outcome in outcomes:
            output.append(
                _equal_value_metric(
                    case,
                    method_id=policy,
                    metric_id=f"event-outcome-{outcome}-fraction",
                    window="event",
                    unit="fraction",
                    values=tuple(row.outcome == outcome for row in cohort),
                    bootstrap_indices=indices,
                )
            )
        if case.family == "common-mode-position-bias":
            for label in ("camera-fault", "lidar-fault", "ambiguous"):
                output.append(
                    _equal_value_metric(
                        case,
                        method_id=policy,
                        metric_id=f"first-latch-label-{label}-fraction",
                        window="event",
                        unit="fraction",
                        values=tuple(row.first_latch_label == label for row in cohort),
                        bootstrap_indices=indices,
                    )
                )
        if case.target in {"camera", "lidar"}:
            output.append(
                _equal_value_metric(
                    case,
                    method_id=policy,
                    metric_id="attribution-fraction",
                    window="event",
                    unit="fraction",
                    values=tuple(row.correctly_attributed for row in cohort),
                    bootstrap_indices=indices,
                )
            )
        output.extend(
            (
                _equal_value_metric(
                    case,
                    method_id=policy,
                    metric_id="early-clear-fraction",
                    window="event",
                    unit="fraction",
                    values=tuple(row.early_clear for row in cohort),
                    bootstrap_indices=indices,
                ),
                _equal_value_metric(
                    case,
                    method_id=policy,
                    metric_id="recovery-denominator-fraction",
                    window="event",
                    unit="fraction",
                    values=tuple(row.recovery_eligible for row in cohort),
                    bootstrap_indices=indices,
                ),
            )
        )
        recovery_eligible = np.asarray(
            [row.recovery_eligible for row in cohort],
            dtype=np.int64,
        )
        recovered = np.asarray([row.recovered for row in cohort], dtype=np.int64)
        output.append(
            _metric(
                case,
                method_id=policy,
                metric_id="recovery-fraction",
                window="recovery",
                unit="fraction",
                aggregation="pooled-valid-eligible-count-ratio",
                interval=pooled_availability_interval(
                    recovered,
                    recovery_eligible,
                    indices,
                ),
                numerators=recovered,
                denominators=recovery_eligible,
            )
        )
        for state in ("healthy", "camera-fault", "lidar-fault", "ambiguous"):
            output.append(
                _equal_value_metric(
                    case,
                    method_id=policy,
                    metric_id=f"final-active-state-{state}-fraction",
                    window="event",
                    unit="fraction",
                    values=tuple(row.final_active_state == state for row in cohort),
                    bootstrap_indices=indices,
                )
            )
        if case.family == "dropout":
            realized = np.asarray(
                [bool(row.realized_dropout) for row in cohort],
                dtype=np.int64,
            )
            detected_realized = np.asarray(
                [bool(row.realized_dropout) and row.detected for row in cohort],
                dtype=np.int64,
            )
            output.append(
                _metric(
                    case,
                    method_id=policy,
                    metric_id="detection-among-realized-dropout-fraction",
                    window="event",
                    unit="fraction",
                    aggregation="pooled-valid-eligible-count-ratio",
                    interval=pooled_availability_interval(
                        detected_realized,
                        realized,
                        indices,
                    ),
                    numerators=detected_realized,
                    denominators=realized,
                )
            )

        latency_fields: list[
            tuple[str, HealthWindow, ReplayHealthUnit, tuple[float | int | None, ...]]
        ] = [
            (
                "detection-latency-steps",
                "event",
                "observation-step",
                tuple(row.detection_latency_steps for row in cohort),
            ),
            (
                "detection-elapsed-reference-time",
                "event",
                "s",
                tuple(row.detection_latency_s for row in cohort),
            ),
            (
                "recovery-latency-steps",
                "recovery",
                "observation-step",
                tuple(row.recovery_latency_steps for row in cohort),
            ),
            (
                "recovery-elapsed-reference-time",
                "recovery",
                "s",
                tuple(row.recovery_latency_s for row in cohort),
            ),
        ]
        if case.target in {"camera", "lidar"}:
            latency_fields.extend(
                (
                    (
                        "attribution-latency-steps",
                        "event",
                        "observation-step",
                        tuple(row.attribution_latency_steps for row in cohort),
                    ),
                    (
                        "attribution-elapsed-reference-time",
                        "event",
                        "s",
                        tuple(row.attribution_latency_s for row in cohort),
                    ),
                )
            )
        if case.family == "dropout":
            latency_fields.extend(
                (
                    (
                        "detection-minus-first-missing-steps",
                        "event",
                        "observation-step",
                        tuple(row.detection_minus_first_missing_steps for row in cohort),
                    ),
                    (
                        "detection-minus-first-missing-elapsed-reference-time",
                        "event",
                        "s",
                        tuple(row.detection_minus_first_missing_s for row in cohort),
                    ),
                )
            )
        for metric_id, window, unit, values in latency_fields:
            output.extend(
                _conditional_metric_pair(
                    case,
                    method_id=policy,
                    metric_id=metric_id,
                    window=window,
                    unit=unit,
                    values=values,
                    bootstrap_indices=indices,
                )
            )

        output.extend(
            (
                _equal_value_metric(
                    case,
                    method_id=policy,
                    metric_id="false-alert-episode-starts",
                    window="score",
                    unit="count",
                    values=tuple(row.false_alert_episode_count for row in cohort),
                    bootstrap_indices=indices,
                ),
                _equal_value_metric(
                    case,
                    method_id=policy,
                    metric_id="latch-episode-starts",
                    window="event",
                    unit="count",
                    values=tuple(row.latch_episode_count for row in cohort),
                    bootstrap_indices=indices,
                ),
            )
        )
        occupancy = (
            ("state-healthy-occupancy", "active_healthy_steps"),
            ("state-camera-fault-occupancy", "active_camera_fault_steps"),
            ("state-lidar-fault-occupancy", "active_lidar_fault_steps"),
            ("state-ambiguous-occupancy", "active_ambiguous_steps"),
            ("action-camera-occupancy", "active_camera_action_steps"),
            ("action-lidar-occupancy", "active_lidar_action_steps"),
            ("action-fixed-occupancy", "active_fixed_action_steps"),
            ("action-undefined-occupancy", "active_undefined_action_steps"),
        )
        active_counts = np.asarray(
            [row.schedule.active_frame_count for row in cohort],
            dtype=np.int64,
        )
        for metric_id, field_name in occupancy:
            counts = np.asarray(
                [getattr(row, field_name) for row in cohort],
                dtype=np.int64,
            )
            output.append(
                _metric(
                    case,
                    method_id=policy,
                    metric_id=metric_id,
                    window="event",
                    unit="fraction",
                    aggregation="pooled-valid-eligible-count-ratio",
                    interval=pooled_availability_interval(counts, active_counts, indices),
                    numerators=counts,
                    denominators=active_counts,
                )
            )
    return tuple(output)


def aggregate_replay_health_case(
    case: ReplayHealthCaseSpec,
    evaluations: Sequence[ReplayHealthSceneEvaluation],
    *,
    bootstrap_indices: npt.ArrayLike | None = None,
) -> tuple[ReplayHealthPopulationMetric, ...]:
    """Aggregate one complete ten-scene M5-B case from local scene evidence."""

    checked = tuple(evaluations)
    if tuple(row.sequence_id for row in checked) != _REPLAY_SEQUENCE_IDS:
        raise ValueError("M5-B aggregation requires all ten scenes in frozen order")
    if any(
        row.replay_experiment_identity_sha256 != case.identity_sha256
        or row.condition_selector != case.selector
        for row in checked
    ):
        raise ValueError("M5-B scene evaluation disagrees with its frozen case")
    indices = replay_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    return (
        *aggregate_replay_health_results(
            case,
            tuple(result for evaluation in checked for result in evaluation.results),
            bootstrap_indices=indices,
        ),
        *aggregate_replay_health_contrasts(
            case,
            tuple(contrast for evaluation in checked for contrast in evaluation.contrasts),
            bootstrap_indices=indices,
        ),
        *aggregate_replay_health_events(
            case,
            tuple(event for evaluation in checked for event in evaluation.events),
            bootstrap_indices=indices,
        ),
    )
