"""Aggregate-only conversion from authenticated M5 evidence to public contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from fusion_fault_bench.artifacts import derive_run_id
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    GeometryCrossoverManifest,
)
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_AGGREGATE_COORDINATE_COUNT,
    M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256,
    M5_HEALTH_CONDITION_SELECTOR_COUNT,
    M5_HEALTH_CONDITION_SELECTOR_SET_SHA256,
    M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT,
    M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256,
    M5_PERSISTENT_CONDITION_SELECTOR_COUNT,
    M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256,
    M5_PERSISTENT_HYPOTHESIS_COORDINATES,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_TRACKED_AGGREGATE_TERMS,
    REPLAY_CURATED_ARTIFACT_CONTRACT,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    ReplayProfileSummaryV1,
    ReplayRepeatVerificationV1,
    ReplaySourceMemberCommitmentV1,
    ReplayValidationV1,
)
from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayHealthResultV1,
    ReplayHealthSequenceEventV1,
)
from fusion_fault_bench.contracts.replay_release_v1 import ReplayFigureSourceBindingV1
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_PANEL_ID,
    M5_PERSISTENT_PANEL_ID,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
    M5_SCENE_NAMES,
    ReplayExperimentIdentityV1,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RunRecordV1Alpha1
from fusion_fault_bench.inference import bootstrap_crossover_status
from fusion_fault_bench.replay_artifacts import ReplayCuratedArtifactWriteRequest
from fusion_fault_bench.replay_health_population import (
    ReplayHealthPopulationMetric,
    aggregate_replay_health_contrasts,
    aggregate_replay_health_events,
    aggregate_replay_health_results,
    expected_replay_health_population_coordinates,
    validate_replay_health_population_grid,
)
from fusion_fault_bench.replay_inference import (
    H5_B_SELECTORS,
    ReplayHealthSequenceContrast,
    ReplayInterval,
    classify_persistence,
    leave_one_log_group_out,
    leave_one_scene_out,
    scene_sign_counts,
    supports_nonpositive_control,
)
from fusion_fault_bench.replay_persistent import ReplayPersistentSceneEvaluation
from fusion_fault_bench.replay_persistent_inference import (
    M5_A_DIRECTIONAL_EXPECTATIONS,
    ReplayPersistentCrossoverEstimate,
    ReplayPersistentPopulationMetric,
    aggregate_replay_persistent_case,
    evaluate_replay_persistent_crossovers,
)
from fusion_fault_bench.replay_plan import LoadedReplayPlan
from fusion_fault_bench.replay_resources import M5_PUBLIC_REPLAY_COMMAND

type PanelAggregate = ReplayPersistentAggregateV1 | ReplayHealthAggregateV1
type InferenceRole = Literal[
    "primary-directional",
    "nonpositive-control",
    "diagnostic",
    "descriptive",
]
type ExpectedDirection = Literal["positive", "negative", "nonpositive", "none"]
type HypothesisId = Literal[
    "h5-a1",
    "h5-a2",
    "h5-a3",
    "h5-a4",
    "h5-a5",
    "h5-a6",
    "h5-b1",
    "h5-b2",
    "h5-b3",
    "h5-b4",
]

_M5_A_DIRECTIONAL_COORDINATES: tuple[
    tuple[HypothesisId, str, Literal["positive", "negative"]],
    ...,
] = (
    ("h5-a1", "replay-lidar-y-bias:0", "negative"),
    ("h5-a1", "replay-camera-noise-correctly-reported:1", "negative"),
    ("h5-a1", "replay-camera-noise-underreported:1", "negative"),
    ("h5-a1", "replay-camera-calibration-x:0", "negative"),
    ("h5-a1", "replay-camera-calibration-yaw:0", "negative"),
    ("h5-a1", "replay-camera-timestamp-offset:0", "negative"),
    ("h5-a2", "replay-lidar-y-bias:-4", "positive"),
    ("h5-a2", "replay-lidar-y-bias:+4", "positive"),
    ("h5-a2", "replay-camera-calibration-x:-4", "positive"),
    ("h5-a2", "replay-camera-calibration-x:+4", "positive"),
    ("h5-a2", "replay-camera-calibration-yaw:-0.08", "positive"),
    ("h5-a2", "replay-camera-calibration-yaw:+0.08", "positive"),
    ("h5-a2", "replay-camera-timestamp-offset:-0.8", "positive"),
    ("h5-a2", "replay-camera-timestamp-offset:+0.8", "positive"),
    ("h5-a3", "replay-camera-noise-underreported:4", "positive"),
    ("h5-a4", "replay-camera-noise-correctly-reported:4", "negative"),
)
_M5_A_HYPOTHESIS_BY_SELECTOR: dict[
    str,
    tuple[HypothesisId, Literal["positive", "negative"]],
] = {
    selector: (hypothesis_id, direction)
    for hypothesis_id, selector, direction in _M5_A_DIRECTIONAL_COORDINATES
}
if {
    selector: direction for _, selector, direction in _M5_A_DIRECTIONAL_COORDINATES
} != M5_A_DIRECTIONAL_EXPECTATIONS:
    raise RuntimeError("M5-A public hypothesis map disagrees with population inference")

_M5_A_DIAGNOSTIC_COORDINATES: tuple[
    tuple[HypothesisId, str, str, str, Literal["full"], Literal["m^2", "fraction"]],
    ...,
] = (
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "fixed-fusion",
        "coverage",
        "full",
        "fraction",
    ),
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "fixed-fusion",
        "conditional-matched-center-mse",
        "full",
        "m^2",
    ),
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "fault-target-drop-policy",
        "coverage",
        "full",
        "fraction",
    ),
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "lidar-only",
        "coverage",
        "full",
        "fraction",
    ),
    *(
        (
            "h5-a6",
            selector,
            "fixed-fusion",
            "matched-center-mse",
            "full",
            "m^2",
        )
        for selector in ("replay-common-mode-x:-4", "replay-common-mode-x:+4")
    ),
    *(
        (
            "h5-a6",
            selector,
            "camera-lidar-pair",
            "camera-lidar-disagreement-mse",
            "full",
            "m^2",
        )
        for selector in (
            "replay-common-mode-x:0",
            "replay-common-mode-x:-0.25",
            "replay-common-mode-x:+0.25",
            "replay-common-mode-x:-0.5",
            "replay-common-mode-x:+0.5",
            "replay-common-mode-x:-1",
            "replay-common-mode-x:+1",
            "replay-common-mode-x:-2",
            "replay-common-mode-x:+2",
            "replay-common-mode-x:-4",
            "replay-common-mode-x:+4",
        )
    ),
)
_M5_A_DIAGNOSTIC_BY_COORDINATE: dict[
    tuple[str, str, str],
    HypothesisId,
] = {
    (selector, method_id, metric_id): hypothesis_id
    for hypothesis_id, selector, method_id, metric_id, _, _ in (_M5_A_DIAGNOSTIC_COORDINATES)
}
_LOCAL_M5_A_COORDINATES = {
    *(
        (
            hypothesis_id,
            selector,
            "fixed-fusion",
            "fused-minus-healthy",
            "full",
            "m^2",
            "equal-scene-mean",
            "primary-directional",
            direction,
        )
        for hypothesis_id, selector, direction in _M5_A_DIRECTIONAL_COORDINATES
    ),
    *(
        (
            hypothesis_id,
            selector,
            method_id,
            metric_id,
            window,
            unit,
            (
                "pooled-valid-eligible-count-ratio"
                if metric_id == "coverage"
                else "pooled-valid-loss"
                if metric_id == "conditional-matched-center-mse"
                else "equal-scene-mean"
            ),
            "diagnostic",
            "none",
        )
        for hypothesis_id, selector, method_id, metric_id, window, unit in (
            _M5_A_DIAGNOSTIC_COORDINATES
        )
    ),
}
if set(M5_PERSISTENT_HYPOTHESIS_COORDINATES) != _LOCAL_M5_A_COORDINATES:
    raise RuntimeError("M5-A curation map disagrees with the public artifact contract")

_M5_B_HYPOTHESIS_BY_SELECTOR = {row.selector: row for row in H5_B_SELECTORS}

_EXPECTED_CROSSOVERS: tuple[
    tuple[
        str,
        Literal["negative", "positive", "increase"],
        Literal["m", "rad", "s", "std-scale"],
        float,
    ],
    ...,
] = (
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
_HEALTH_WINDOWS = ("score", "event", "recovery")
_STRUCTURAL_COMMON_MODE_NA_METRICS = frozenset(
    {
        "gap-vs-fault-target-drop",
        "gap-vs-frame-oracle",
        "frame-oracle-recoverable-loss-fraction",
    }
)
_HEALTH_POLICIES = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
)
_HEALTH_EVENT_POLICIES = _HEALTH_POLICIES[:-1]
_HEALTH_BASE_RESULT_METHODS = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    *_HEALTH_POLICIES,
)
_HEALTH_STANDARD_RESULT_METHODS = (
    *_HEALTH_BASE_RESULT_METHODS,
    "fault-target-drop-policy",
    "frame-action-performance-oracle",
)


@dataclass(frozen=True, slots=True)
class ReplayCuratedAggregateEvidence:
    """Public aggregate records produced before externally evidenced release gates."""

    profile_summary: ReplayProfileSummaryV1
    descriptor_aggregates: tuple[ReplayDescriptorAggregateV1, ...]
    persistent_aggregates: tuple[ReplayPersistentAggregateV1, ...]
    persistent_crossovers: tuple[ReplayPersistentCrossoverV1, ...]
    health_aggregates: tuple[ReplayHealthAggregateV1, ...]
    cluster_sensitivity: tuple[ReplayClusterSensitivityV1, ...]
    run: RunRecordV1Alpha1


@dataclass(frozen=True, slots=True)
class ReplayLogGroupBinding:
    """One local-only log-group ordinal keyed to frozen replay scene order."""

    sequence_id: str
    log_group_ordinal: str


class _AggregateBinding(TypedDict):
    run_id: str
    replay_intent_sha256: str
    replay_identity_set_sha256: str
    identity: ReplayExperimentIdentityV1
    replay_identity_sha256: str


def _stable_id(prefix: str, coordinate: dict[str, object]) -> str:
    return f"{prefix}-{sha256_digest(coordinate)}"


def _selector_set_sha256(panel_id: str, selectors: set[str]) -> str:
    return sha256_digest(
        {
            "panel_id": panel_id,
            "condition_selectors": sorted(selectors),
        }
    )


def _validate_plan(plan: LoadedReplayPlan) -> None:
    if (
        plan.intent.intent_sha256 != M5_REPLAY_INTENT_SHA256
        or plan.intent.byte_sha256 != M5_REPLAY_INTENT_BYTE_SHA256
    ):
        raise ValueError("replay curation requires the authenticated frozen M5 intent")
    persistent_selectors = {case.fault_condition.selector for case in plan.persistent_cases}
    health_selectors = {case.selector for case in plan.health_cases}
    if (
        len(plan.persistent_cases) != M5_PERSISTENT_CONDITION_SELECTOR_COUNT
        or len(persistent_selectors) != M5_PERSISTENT_CONDITION_SELECTOR_COUNT
        or _selector_set_sha256(M5_PERSISTENT_PANEL_ID, persistent_selectors)
        != M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256
    ):
        raise ValueError("authenticated replay plan has an incomplete M5-A selector grid")
    if (
        len(plan.health_cases) != M5_HEALTH_CONDITION_SELECTOR_COUNT
        or len(health_selectors) != M5_HEALTH_CONDITION_SELECTOR_COUNT
        or _selector_set_sha256(M5_HEALTH_PANEL_ID, health_selectors)
        != M5_HEALTH_CONDITION_SELECTOR_SET_SHA256
    ):
        raise ValueError("authenticated replay plan has an incomplete M5-B selector grid")


def expected_replay_persistent_coordinates(
    plan: LoadedReplayPlan,
) -> set[tuple[str, str, str, str]]:
    coordinates: set[tuple[str, str, str, str]] = set()
    for case in plan.persistent_cases:
        selector = case.fault_condition.selector
        manifest = case.source_manifest
        if isinstance(manifest, AvailabilityControlManifest):
            metrics = (
                ("coverage", "pooled-valid-eligible-count-ratio"),
                ("conditional-matched-center-mse", "pooled-valid-loss"),
                ("undefined-output-rate", "pooled-valid-eligible-count-ratio"),
                ("scene-equal-coverage", "equal-scene-mean"),
            )
            for method in manifest.methods:
                coordinates.update(
                    (selector, method, metric_id, aggregation) for metric_id, aggregation in metrics
                )
        else:
            coordinates.update(
                (selector, method, "matched-center-mse", "equal-scene-mean")
                for method in manifest.methods
            )
            if isinstance(manifest, GeometryCrossoverManifest):
                coordinates.add(
                    (
                        selector,
                        "fixed-fusion",
                        "fused-minus-healthy",
                        "equal-scene-mean",
                    )
                )
            else:
                coordinates.add(
                    (
                        selector,
                        "camera-lidar-pair",
                        "camera-lidar-disagreement-mse",
                        "equal-scene-mean",
                    )
                )
    public_coordinates = {
        (
            selector,
            method_id,
            metric_id,
            "full",
            (
                "fraction"
                if metric_id in {"coverage", "undefined-output-rate", "scene-equal-coverage"}
                else "m^2"
            ),
            aggregation,
        )
        for selector, method_id, metric_id, aggregation in coordinates
    }
    if (
        len(coordinates) != M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT
        or sha256_digest(
            {
                "panel_id": M5_PERSISTENT_PANEL_ID,
                "aggregate_coordinates": sorted(public_coordinates),
            }
        )
        != M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256
    ):
        raise ValueError("persistent coordinate blueprint is not the frozen 464-row matrix")
    return coordinates


def _health_coordinate(
    *,
    selector: str,
    method_id: str,
    metric_id: str,
    window: str,
    unit: str,
    aggregation: str,
) -> tuple[str, str, str, str, str, str]:
    return (selector, method_id, metric_id, window, unit, aggregation)


def _conditional_health_coordinates(
    *,
    selector: str,
    method_id: str,
    metric_id: str,
    window: str,
    unit: str,
) -> set[tuple[str, str, str, str, str, str]]:
    return {
        _health_coordinate(
            selector=selector,
            method_id=method_id,
            metric_id=f"{metric_id}-observed-fraction",
            window=window,
            unit="fraction",
            aggregation="equal-scene-mean",
        ),
        _health_coordinate(
            selector=selector,
            method_id=method_id,
            metric_id=metric_id,
            window=window,
            unit=unit,
            aggregation="conditional-observed-scene-mean",
        ),
    }


def _expected_health_result_coordinates(
    plan: LoadedReplayPlan,
) -> set[tuple[str, str, str, str, str, str]]:
    coordinates: set[tuple[str, str, str, str, str, str]] = set()
    metrics = (
        ("matched-center-mse", "m^2", "equal-scene-mean"),
        ("conditional-matched-center-mse", "m^2", "pooled-valid-loss"),
        ("coverage", "fraction", "pooled-valid-eligible-count-ratio"),
        ("undefined-output-rate", "fraction", "pooled-valid-eligible-count-ratio"),
        ("scene-equal-coverage", "fraction", "equal-scene-mean"),
    )
    for case in plan.health_cases:
        methods = (
            _HEALTH_BASE_RESULT_METHODS
            if case.family == "common-mode-position-bias"
            else _HEALTH_STANDARD_RESULT_METHODS
        )
        for method_id in methods:
            for window in _HEALTH_WINDOWS:
                coordinates.update(
                    _health_coordinate(
                        selector=case.selector,
                        method_id=method_id,
                        metric_id=metric_id,
                        window=window,
                        unit=unit,
                        aggregation=aggregation,
                    )
                    for metric_id, unit, aggregation in metrics
                )
    if len(coordinates) != 6_330:
        raise ValueError("health result blueprint is not the frozen 6,330-row matrix")
    return coordinates


def _expected_health_contrast_coordinates(
    plan: LoadedReplayPlan,
) -> set[tuple[str, str, str, str, str, str]]:
    coordinates: set[tuple[str, str, str, str, str, str]] = set()
    for case in plan.health_cases:
        for method_id in _HEALTH_POLICIES:
            for window in _HEALTH_WINDOWS:
                coordinates.add(
                    _health_coordinate(
                        selector=case.selector,
                        method_id=method_id,
                        metric_id="policy-gain-vs-fixed",
                        window=window,
                        unit="m^2",
                        aggregation="equal-scene-mean",
                    )
                )
                for metric_id in (
                    "gap-vs-fault-target-drop",
                    "gap-vs-frame-oracle",
                ):
                    coordinates.add(
                        _health_coordinate(
                            selector=case.selector,
                            method_id=method_id,
                            metric_id=metric_id,
                            window=window,
                            unit="m^2",
                            aggregation="equal-scene-mean",
                        )
                    )
                coordinates.add(
                    _health_coordinate(
                        selector=case.selector,
                        method_id=method_id,
                        metric_id="frame-oracle-recoverable-loss-fraction",
                        window=window,
                        unit="unitless",
                        aggregation="unclipped-recovery-ratio",
                    )
                )
    if len(coordinates) != 2_580:
        raise ValueError("health contrast blueprint is not the frozen 2,580-row matrix")
    return coordinates


def _expected_health_event_coordinates(
    plan: LoadedReplayPlan,
) -> set[tuple[str, str, str, str, str, str]]:
    coordinates: set[tuple[str, str, str, str, str, str]] = set()

    def equal(
        *,
        selector: str,
        method_id: str,
        metric_id: str,
        window: str,
        unit: str,
    ) -> None:
        coordinates.add(
            _health_coordinate(
                selector=selector,
                method_id=method_id,
                metric_id=metric_id,
                window=window,
                unit=unit,
                aggregation="equal-scene-mean",
            )
        )

    def pooled(
        *,
        selector: str,
        method_id: str,
        metric_id: str,
        window: str,
    ) -> None:
        coordinates.add(
            _health_coordinate(
                selector=selector,
                method_id=method_id,
                metric_id=metric_id,
                window=window,
                unit="fraction",
                aggregation="pooled-valid-eligible-count-ratio",
            )
        )

    for case in plan.health_cases:
        selector = case.selector
        if case.family == "dropout":
            equal(
                selector=selector,
                method_id="none",
                metric_id="realized-dropout-fraction",
                window="event",
                unit="fraction",
            )
            coordinates.update(
                _conditional_health_coordinates(
                    selector=selector,
                    method_id="none",
                    metric_id="first-missing-step",
                    window="event",
                    unit="observation-step",
                )
            )
            coordinates.update(
                _conditional_health_coordinates(
                    selector=selector,
                    method_id="none",
                    metric_id="first-missing-elapsed-reference-time",
                    window="event",
                    unit="s",
                )
            )

        for method_id in _HEALTH_EVENT_POLICIES:
            equal(
                selector=selector,
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
                    selector=selector,
                    method_id=method_id,
                    metric_id=f"event-outcome-{outcome}-fraction",
                    window="event",
                    unit="fraction",
                )
            if case.family == "common-mode-position-bias":
                for label in ("camera-fault", "lidar-fault", "ambiguous"):
                    equal(
                        selector=selector,
                        method_id=method_id,
                        metric_id=f"first-latch-label-{label}-fraction",
                        window="event",
                        unit="fraction",
                    )
            if case.target in {"camera", "lidar"}:
                equal(
                    selector=selector,
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
                    selector=selector,
                    method_id=method_id,
                    metric_id=metric_id,
                    window="event",
                    unit="fraction",
                )
            pooled(
                selector=selector,
                method_id=method_id,
                metric_id="recovery-fraction",
                window="recovery",
            )
            for state in ("healthy", "camera-fault", "lidar-fault", "ambiguous"):
                equal(
                    selector=selector,
                    method_id=method_id,
                    metric_id=f"final-active-state-{state}-fraction",
                    window="event",
                    unit="fraction",
                )
            if case.family == "dropout":
                pooled(
                    selector=selector,
                    method_id=method_id,
                    metric_id="detection-among-realized-dropout-fraction",
                    window="event",
                )

            latency_fields = [
                ("detection-latency-steps", "event", "observation-step"),
                ("detection-elapsed-reference-time", "event", "s"),
                ("recovery-latency-steps", "recovery", "observation-step"),
                ("recovery-elapsed-reference-time", "recovery", "s"),
            ]
            if case.target in {"camera", "lidar"}:
                latency_fields.extend(
                    (
                        ("attribution-latency-steps", "event", "observation-step"),
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
                coordinates.update(
                    _conditional_health_coordinates(
                        selector=selector,
                        method_id=method_id,
                        metric_id=metric_id,
                        window=window,
                        unit=unit,
                    )
                )
            for metric_id, window in (
                ("false-alert-episode-starts", "score"),
                ("latch-episode-starts", "event"),
            ):
                equal(
                    selector=selector,
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
                    selector=selector,
                    method_id=method_id,
                    metric_id=metric_id,
                    window="event",
                )
    if len(coordinates) != 6_078:
        raise ValueError("health event blueprint is not the frozen 6,078-row matrix")
    return coordinates


def expected_replay_health_coordinates(
    plan: LoadedReplayPlan,
) -> set[tuple[str, str, str, str, str, str]]:
    coordinates = {
        *_expected_health_result_coordinates(plan),
        *_expected_health_contrast_coordinates(plan),
        *_expected_health_event_coordinates(plan),
    }
    authoritative = {
        (
            coordinate.condition_selector,
            coordinate.method_id,
            coordinate.metric_id,
            coordinate.window,
            coordinate.unit,
            coordinate.aggregation,
        )
        for coordinate in expected_replay_health_population_coordinates(plan)
    }
    if (
        len(coordinates) != M5_HEALTH_AGGREGATE_COORDINATE_COUNT
        or coordinates != authoritative
        or sha256_digest(
            {
                "panel_id": M5_HEALTH_PANEL_ID,
                "aggregate_coordinates": sorted(coordinates),
            }
        )
        != M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256
    ):
        raise ValueError("health release blueprint is not the frozen 14,988-row matrix")
    return coordinates


def _validate_context(
    *,
    profile_summary: ReplayProfileSummaryV1,
    descriptor_aggregates: Sequence[ReplayDescriptorAggregateV1],
    log_group_ordinals: Sequence[str],
    run: RunRecordV1Alpha1,
) -> tuple[tuple[ReplayDescriptorAggregateV1, ...], tuple[str, ...]]:
    if (
        run.manifest_sha256 != M5_REPLAY_INTENT_SHA256
        or run.source_dirty
        or run.status != "succeeded"
        or run.artifact_sha256 != "0" * 64
        or run.run_id
        != derive_run_id(
            manifest_sha256=M5_REPLAY_INTENT_SHA256,
            git_revision=run.git_revision,
            lockfile_sha256=run.lockfile_sha256,
            package_version=run.package_version,
            artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
        )
    ):
        raise ValueError("replay curation requires a clean successful frozen-intent run")
    if (
        profile_summary.run_id != run.run_id
        or profile_summary.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
        or profile_summary.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
    ):
        raise ValueError("replay profile summary does not bind the curation run")

    groups = tuple(log_group_ordinals)
    if len(groups) != 10:
        raise ValueError("replay curation requires one log-group ordinal per frozen scene")
    distinct_groups = tuple(sorted(set(groups), key=lambda value: value.encode("utf-8")))
    expected_groups = tuple(f"log-group:{index:02d}" for index in range(len(distinct_groups)))
    if distinct_groups != expected_groups or profile_summary.distinct_log_group_count != len(
        distinct_groups
    ):
        raise ValueError("log-group ordinals are not contiguous or disagree with provenance")

    descriptors = tuple(
        sorted(
            descriptor_aggregates,
            key=lambda row: (
                row.population,
                row.descriptor_id,
                row.statistic,
                row.category_label or "",
            ),
        )
    )
    if not descriptors:
        raise ValueError("replay curation requires source descriptor aggregates")
    if any(
        row.run_id != run.run_id
        or row.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
        or row.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
        for row in descriptors
    ):
        raise ValueError("source descriptor aggregate does not bind the curation run")
    descriptor_keys = tuple(
        (
            row.population,
            row.descriptor_id,
            row.statistic,
            row.category_label,
        )
        for row in descriptors
    )
    if len(descriptor_keys) != len(set(descriptor_keys)):
        raise ValueError("source descriptor aggregates contain a duplicate coordinate")
    return descriptors, groups


def _group_health_rows[RowT](
    rows: Sequence[RowT],
    *,
    expected_selectors: set[str],
    label: str,
) -> dict[str, tuple[RowT, ...]]:
    grouped: dict[str, list[RowT]] = {}
    for row in rows:
        selector = getattr(row, "condition_selector", None)
        if not isinstance(selector, str) or selector not in expected_selectors:
            raise ValueError(f"{label} contains an unknown replay selector")
        grouped.setdefault(selector, []).append(row)
    if set(grouped) != expected_selectors:
        raise ValueError(f"{label} does not cover the exact 43-selector M5-B grid")
    return {selector: tuple(values) for selector, values in grouped.items()}


def aggregate_replay_health_evidence(
    *,
    plan: LoadedReplayPlan,
    health_results: Sequence[ReplayHealthResultV1],
    health_contrasts: Sequence[ReplayHealthSequenceContrast],
    health_events: Sequence[ReplayHealthSequenceEventV1],
) -> tuple[ReplayHealthPopulationMetric, ...]:
    """Aggregate complete per-scene M5-B evidence without reading dataset payloads."""

    _validate_plan(plan)
    expected_selectors = {case.selector for case in plan.health_cases}
    results = _group_health_rows(
        health_results,
        expected_selectors=expected_selectors,
        label="replay health results",
    )
    contrasts = _group_health_rows(
        health_contrasts,
        expected_selectors=expected_selectors,
        label="replay health contrasts",
    )
    events = _group_health_rows(
        health_events,
        expected_selectors=expected_selectors,
        label="replay health events",
    )
    output: list[ReplayHealthPopulationMetric] = []
    for case in plan.health_cases:
        output.extend(aggregate_replay_health_results(case, results[case.selector]))
        output.extend(aggregate_replay_health_contrasts(case, contrasts[case.selector]))
        output.extend(aggregate_replay_health_events(case, events[case.selector]))
    result = tuple(output)
    validate_replay_health_population_grid(
        plan,
        result,
        contrasts=health_contrasts,
    )
    return result


def aggregate_replay_persistent_evidence(
    *,
    plan: LoadedReplayPlan,
    scene_evaluations: Sequence[ReplayPersistentSceneEvaluation],
) -> tuple[
    tuple[ReplayPersistentPopulationMetric, ...],
    tuple[ReplayPersistentCrossoverEstimate, ...],
]:
    """Recompute the complete M5-A population and crossover evidence."""

    _validate_plan(plan)
    expected_selectors = {case.fault_condition.selector for case in plan.persistent_cases}
    grouped: dict[str, list[ReplayPersistentSceneEvaluation]] = {}
    for row in scene_evaluations:
        if row.condition_selector not in expected_selectors:
            raise ValueError("persistent scene evidence contains an unknown selector")
        grouped.setdefault(row.condition_selector, []).append(row)
    if set(grouped) != expected_selectors:
        raise ValueError("persistent scene evidence does not cover the exact 71 selectors")

    metrics: list[ReplayPersistentPopulationMetric] = []
    for case in plan.persistent_cases:
        metrics.extend(
            aggregate_replay_persistent_case(
                case,
                tuple(grouped[case.fault_condition.selector]),
            )
        )
    metric_rows = tuple(metrics)

    experiment_order = tuple(
        dict.fromkeys(
            case.identity.experiment_id
            for case in plan.persistent_cases
            if isinstance(case.source_manifest, GeometryCrossoverManifest)
        )
    )
    crossovers: list[ReplayPersistentCrossoverEstimate] = []
    for experiment_id in experiment_order:
        cases = tuple(
            case for case in plan.persistent_cases if case.identity.experiment_id == experiment_id
        )
        experiment_metrics = tuple(row for row in metric_rows if row.condition_id == experiment_id)
        crossovers.extend(
            evaluate_replay_persistent_crossovers(
                cases,
                experiment_metrics,
            )
        )
    return metric_rows, tuple(crossovers)


def _metric_status(interval: ReplayInterval) -> Literal["ok", "undefined"]:
    return "ok" if interval.estimate is not None else "undefined"


def _require_release_interval(interval: ReplayInterval) -> None:
    if interval.bootstrap_replicates != 2_000:
        raise ValueError("replay curation requires the frozen 2,000-replicate bootstrap")


def _result_id(
    *,
    panel_id: str,
    identity_sha256: str,
    condition_selector: str,
    method_id: str,
    metric_id: str,
    window: str,
) -> str:
    return _stable_id(
        "replay-result",
        {
            "schema": "ffb.replay-result-coordinate/v1",
            "panel_id": panel_id,
            "replay_experiment_identity_sha256": identity_sha256,
            "condition_selector": condition_selector,
            "method_id": method_id,
            "metric_id": metric_id,
            "window": window,
        },
    )


def _aggregate_binding(
    *,
    run_id: str,
    identity: ReplayExperimentIdentityV1,
    identity_sha256: str,
) -> _AggregateBinding:
    return {
        "run_id": run_id,
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
        "identity": identity,
        "replay_identity_sha256": identity_sha256,
    }


def _sensitivity_rows(
    *,
    source: PanelAggregate,
    scene_values: Sequence[float | None],
    log_group_ordinals: Sequence[str],
) -> tuple[ReplayClusterSensitivityV1, ...]:
    loso = leave_one_scene_out(scene_values)
    lolo = leave_one_log_group_out(scene_values, log_group_ordinals)
    source_sha256 = sha256_digest(source)
    rows: list[ReplayClusterSensitivityV1] = []
    for cluster_kind, values in (
        ("leave-one-scene-out", loso),
        ("leave-one-log-group-out", lolo),
    ):
        for value in values:
            rows.append(
                ReplayClusterSensitivityV1(
                    schema="ffb.replay-cluster-sensitivity/v1",
                    **_aggregate_binding(
                        run_id=source.run_id,
                        identity=source.identity,
                        identity_sha256=source.replay_identity_sha256,
                    ),
                    sensitivity_id=_stable_id(
                        "replay-sensitivity",
                        {
                            "schema": "ffb.replay-sensitivity-coordinate/v1",
                            "source_result_id": source.result_id,
                            "cluster_kind": cluster_kind,
                            "cluster_id": value.cluster_id,
                        },
                    ),
                    source_result_id=source.result_id,
                    source_record_sha256=source_sha256,
                    cluster_kind=cast(
                        Literal["leave-one-scene-out", "leave-one-log-group-out"],
                        cluster_kind,
                    ),
                    cluster_id=value.cluster_id,
                    status="ok" if value.estimate is not None else "undefined",
                    estimate=value.estimate,
                    unit=source.unit,
                    tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
                )
            )
    return tuple(rows)


def _persistent_record(
    *,
    run_id: str,
    case_identity: ReplayExperimentIdentityV1,
    metric: ReplayPersistentPopulationMetric,
    log_group_ordinals: Sequence[str],
) -> tuple[ReplayPersistentAggregateV1, tuple[ReplayClusterSensitivityV1, ...]]:
    identity_sha256 = replay_experiment_identity_sha256(case_identity)
    if (
        metric.replay_experiment_identity_sha256 != identity_sha256
        or metric.condition_id != case_identity.experiment_id
    ):
        raise ValueError("persistent population metric disagrees with its replay identity")
    _require_release_interval(metric.interval)
    primary = (
        metric.method_id == "fixed-fusion"
        and metric.metric_id == "fused-minus-healthy"
        and metric.aggregation == "equal-scene-mean"
        and metric.condition_selector in _M5_A_HYPOTHESIS_BY_SELECTOR
    )
    diagnostic_hypothesis = _M5_A_DIAGNOSTIC_BY_COORDINATE.get(
        (metric.condition_selector, metric.method_id, metric.metric_id)
    )
    hypothesis_id: HypothesisId | None = None
    role: InferenceRole = "descriptive"
    expected_direction: ExpectedDirection = "none"
    persistence_label = "not-applicable" if metric.interval.estimate is not None else "undefined"
    nonpositive_supported = None
    signs = None
    if primary:
        hypothesis_id, raw_direction = _M5_A_HYPOTHESIS_BY_SELECTOR[metric.condition_selector]
        role = "primary-directional"
        expected_direction = raw_direction
        assessment = classify_persistence(
            metric.interval,
            metric.scene_values,
            log_group_ordinals,
            raw_direction,
        )
        persistence_label = assessment.label
        signs = assessment.scene_signs
    elif diagnostic_hypothesis is not None:
        hypothesis_id = diagnostic_hypothesis
        role = "diagnostic"

    record = ReplayPersistentAggregateV1(
        schema="ffb.replay-persistent-aggregate/v1",
        **_aggregate_binding(
            run_id=run_id,
            identity=case_identity,
            identity_sha256=identity_sha256,
        ),
        result_id=_result_id(
            panel_id=M5_PERSISTENT_PANEL_ID,
            identity_sha256=identity_sha256,
            condition_selector=metric.condition_selector,
            method_id=metric.method_id,
            metric_id=metric.metric_id,
            window="full",
        ),
        condition_id=metric.condition_id,
        condition_selector=metric.condition_selector,
        hypothesis_id=hypothesis_id,
        method_id=metric.method_id,
        metric_id=metric.metric_id,
        window="full",
        inference_role=role,
        unit=(
            "fraction"
            if metric.metric_id in {"coverage", "undefined-output-rate", "scene-equal-coverage"}
            else "m^2"
        ),
        status=_metric_status(metric.interval),
        estimate=metric.interval.estimate,
        interval_lower=metric.interval.lower,
        interval_upper=metric.interval.upper,
        bootstrap_replicates=2_000,
        defined_bootstrap_replicates=metric.interval.defined_replicates,
        confidence_level=0.95,
        interval_method="paired-scene-percentile-pointwise",
        aggregation=metric.aggregation,
        scene_count=10,
        positive_scene_count=None if signs is None else signs.positive,
        zero_scene_count=None if signs is None else signs.zero,
        negative_scene_count=None if signs is None else signs.negative,
        undefined_scene_count=None if signs is None else signs.undefined,
        expected_direction=expected_direction,
        persistence_label=persistence_label,
        nonpositive_control_supported=nonpositive_supported,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )
    sensitivity = (
        _sensitivity_rows(
            source=record,
            scene_values=metric.scene_values,
            log_group_ordinals=log_group_ordinals,
        )
        if primary
        else ()
    )
    return record, sensitivity


def _health_record(
    *,
    run_id: str,
    case_identity: ReplayExperimentIdentityV1,
    metric: ReplayHealthPopulationMetric,
    recovery_support_compatible_scene_count: int | None,
    log_group_ordinals: Sequence[str],
) -> tuple[ReplayHealthAggregateV1, tuple[ReplayClusterSensitivityV1, ...]]:
    identity_sha256 = replay_experiment_identity_sha256(case_identity)
    if (
        metric.replay_experiment_identity_sha256 != identity_sha256
        or metric.condition_id != case_identity.experiment_id
    ):
        raise ValueError("health population metric disagrees with its replay identity")
    _require_release_interval(metric.interval)

    selector = _M5_B_HYPOTHESIS_BY_SELECTOR.get(metric.condition_selector)
    selected = (
        selector is not None
        and metric.method_id == selector.method
        and metric.metric_id == selector.metric_name
        and metric.window == selector.window
        and metric.unit == selector.unit
        and metric.aggregation == "equal-scene-mean"
    )
    hypothesis_id: HypothesisId | None = None
    role: InferenceRole = "descriptive"
    expected_direction: ExpectedDirection = "none"
    persistence_label = "not-applicable" if metric.interval.estimate is not None else "undefined"
    nonpositive_supported = None
    signs = None
    requires_sensitivity = False
    if selected:
        assert selector is not None
        hypothesis_id = selector.hypothesis_id
        if selector.assessment_rule == "persistence":
            assert selector.expected_direction is not None
            role = "primary-directional"
            expected_direction = selector.expected_direction
            assessment = classify_persistence(
                metric.interval,
                metric.scene_values,
                log_group_ordinals,
                selector.expected_direction,
            )
            persistence_label = assessment.label
            signs = assessment.scene_signs
            requires_sensitivity = True
        elif selector.assessment_rule == "nonpositive-control":
            role = "nonpositive-control"
            expected_direction = "nonpositive"
            persistence_label = (
                "not-applicable" if metric.interval.estimate is not None else "undefined"
            )
            nonpositive_supported = supports_nonpositive_control(metric.interval)
            signs = scene_sign_counts(metric.scene_values)
            requires_sensitivity = True
        else:
            role = "diagnostic"

    structural_not_applicable = (
        metric.condition_id == "replay-common-mode-x"
        and metric.metric_id in _STRUCTURAL_COMMON_MODE_NA_METRICS
    )
    if structural_not_applicable:
        applicability_basis = "structural-unavailable"
    elif metric.metric_id == "frame-oracle-recoverable-loss-fraction":
        if recovery_support_compatible_scene_count is None:
            raise ValueError("recovery aggregate lacks all-scene support evidence")
        applicability_basis = (
            "applicable"
            if recovery_support_compatible_scene_count == 10
            else "support-incompatible"
        )
    else:
        applicability_basis = "applicable"

    record = ReplayHealthAggregateV1(
        schema="ffb.replay-health-aggregate/v1",
        **_aggregate_binding(
            run_id=run_id,
            identity=case_identity,
            identity_sha256=identity_sha256,
        ),
        result_id=_result_id(
            panel_id=M5_HEALTH_PANEL_ID,
            identity_sha256=identity_sha256,
            condition_selector=metric.condition_selector,
            method_id=metric.method_id,
            metric_id=metric.metric_id,
            window=metric.window,
        ),
        condition_id=metric.condition_id,
        condition_selector=metric.condition_selector,
        hypothesis_id=hypothesis_id,
        method_id=metric.method_id,
        metric_id=metric.metric_id,
        window=metric.window,
        inference_role=role,
        unit=metric.unit,
        status=metric.status,
        estimate=metric.interval.estimate,
        interval_lower=metric.interval.lower,
        interval_upper=metric.interval.upper,
        bootstrap_replicates=2_000,
        defined_bootstrap_replicates=metric.interval.defined_replicates,
        confidence_level=0.95,
        interval_method="paired-scene-percentile-pointwise",
        aggregation=metric.aggregation,
        scene_count=10,
        positive_scene_count=None if signs is None else signs.positive,
        zero_scene_count=None if signs is None else signs.zero,
        negative_scene_count=None if signs is None else signs.negative,
        undefined_scene_count=None if signs is None else signs.undefined,
        expected_direction=expected_direction,
        persistence_label=persistence_label,
        nonpositive_control_supported=nonpositive_supported,
        applicability_basis=applicability_basis,
        recovery_support_compatible_scene_count=(
            recovery_support_compatible_scene_count
            if metric.metric_id == "frame-oracle-recoverable-loss-fraction"
            and not structural_not_applicable
            else None
        ),
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )
    sensitivity = (
        _sensitivity_rows(
            source=record,
            scene_values=metric.scene_values,
            log_group_ordinals=log_group_ordinals,
        )
        if requires_sensitivity
        else ()
    )
    return record, sensitivity


def _curate_crossovers(
    *,
    plan: LoadedReplayPlan,
    crossovers: Sequence[ReplayPersistentCrossoverEstimate],
    run_id: str,
) -> tuple[ReplayPersistentCrossoverV1, ...]:
    identity_by_experiment = {
        case.identity.experiment_id: case.identity for case in plan.persistent_cases
    }
    indexed: dict[
        tuple[str, str, str],
        ReplayPersistentCrossoverEstimate,
    ] = {}
    for row in crossovers:
        key = (row.condition_id, row.direction, row.severity_unit)
        if key in indexed:
            raise ValueError("persistent crossover evidence contains a duplicate coordinate")
        indexed[key] = row
    expected_keys = {
        (condition_id, direction, severity_unit)
        for condition_id, direction, severity_unit, _ in _EXPECTED_CROSSOVERS
    }
    if set(indexed) != expected_keys:
        raise ValueError("persistent crossover evidence is not the exact ten-coordinate set")

    output: list[ReplayPersistentCrossoverV1] = []
    for condition_id, direction, severity_unit, tested_maximum in _EXPECTED_CROSSOVERS:
        row = indexed[(condition_id, direction, severity_unit)]
        identity = identity_by_experiment[condition_id]
        identity_sha256 = replay_experiment_identity_sha256(identity)
        if (
            row.replay_experiment_identity_sha256 != identity_sha256
            or row.bootstrap_replicates != 2_000
            or row.tested_maximum != tested_maximum
        ):
            raise ValueError("persistent crossover does not bind frozen replay inference")
        point_curve_crossed = row.point_estimate is not None
        expected_status = bootstrap_crossover_status(
            point_crossed=point_curve_crossed,
            crossing_count=row.bootstrap_crossing_count,
            bootstrap_replicates=row.bootstrap_replicates,
        )
        if row.status != expected_status:
            raise ValueError("persistent crossover status contradicts its bootstrap support")
        if row.status == "observed":
            if (
                row.point_estimate is None
                or row.interval_lower is None
                or not isinstance(row.interval_upper, float)
            ):
                raise ValueError("observed persistent crossover lacks a finite interval")
            status: Literal["observed", "not-observed", "undetermined"] = "observed"
            point_estimate = row.point_estimate
            interval_lower = row.interval_lower
            interval_upper = row.interval_upper
            censoring = "none"
        elif row.status == "not-observed":
            if (
                row.point_estimate is not None
                or row.interval_lower != row.tested_maximum
                or row.interval_upper != "positive-infinity"
            ):
                raise ValueError("not-observed crossover lacks exact right censoring")
            status = "not-observed"
            point_estimate = None
            interval_lower = row.tested_maximum
            interval_upper = "positive-infinity"
            censoring = "right-above-tested-maximum"
        else:
            if row.interval_lower is not None or row.interval_upper is not None:
                raise ValueError("undetermined crossover cannot carry an interval")
            status = "undetermined"
            point_estimate = row.point_estimate
            interval_lower = None
            interval_upper = None
            censoring = "mixed-bootstrap"
        output.append(
            ReplayPersistentCrossoverV1(
                schema="ffb.replay-persistent-crossover/v1",
                **_aggregate_binding(
                    run_id=run_id,
                    identity=identity,
                    identity_sha256=identity_sha256,
                ),
                crossover_id=_stable_id(
                    "replay-crossover",
                    {
                        "schema": "ffb.replay-crossover-coordinate/v1",
                        "replay_experiment_identity_sha256": identity_sha256,
                        "direction": direction,
                        "severity_unit": severity_unit,
                    },
                ),
                direction=direction,
                severity_unit=severity_unit,
                tested_maximum=row.tested_maximum,
                status=status,
                point_curve_crossed=point_curve_crossed,
                point_estimate=point_estimate,
                interval_lower=interval_lower,
                interval_upper=interval_upper,
                censoring=censoring,
                bootstrap_replicates=2_000,
                bootstrap_crossing_count=row.bootstrap_crossing_count,
                bootstrap_crossing_fraction=row.bootstrap_crossing_fraction,
                confidence_level=0.95,
                interval_method="right-censored-paired-scene-percentile",
                tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
            )
        )
    return tuple(output)


def _curate_replay_population_evidence(
    *,
    plan: LoadedReplayPlan,
    persistent_metrics: Sequence[ReplayPersistentPopulationMetric],
    persistent_crossovers: Sequence[ReplayPersistentCrossoverEstimate],
    health_metrics: Sequence[ReplayHealthPopulationMetric],
    health_contrasts: Sequence[ReplayHealthSequenceContrast],
    descriptor_aggregates: Sequence[ReplayDescriptorAggregateV1],
    log_group_ordinals: Sequence[str],
    profile_summary: ReplayProfileSummaryV1,
    run: RunRecordV1Alpha1,
) -> ReplayCuratedAggregateEvidence:
    """Convert already-authenticated population evidence into public contracts."""

    _validate_plan(plan)
    descriptors, groups = _validate_context(
        profile_summary=profile_summary,
        descriptor_aggregates=descriptor_aggregates,
        log_group_ordinals=log_group_ordinals,
        run=run,
    )
    persistent_case_by_selector = {
        case.fault_condition.selector: case for case in plan.persistent_cases
    }
    health_case_by_selector = {case.selector: case for case in plan.health_cases}

    persistent_rows = tuple(
        sorted(
            persistent_metrics,
            key=lambda row: (
                row.condition_selector,
                row.method_id,
                row.metric_id,
                row.aggregation,
            ),
        )
    )
    health_rows = tuple(
        sorted(
            health_metrics,
            key=lambda row: (
                row.condition_selector,
                row.method_id,
                row.metric_id,
                row.window,
                row.aggregation,
            ),
        )
    )
    persistent_selectors = {row.condition_selector for row in persistent_rows}
    health_selectors = {row.condition_selector for row in health_rows}
    if persistent_selectors != set(persistent_case_by_selector):
        raise ValueError("persistent population metrics do not cover the exact 71 selectors")
    if health_selectors != set(health_case_by_selector):
        raise ValueError("health population metrics do not cover the exact 43 selectors")

    persistent_keys = tuple(
        (
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.aggregation,
        )
        for row in persistent_rows
    )
    health_keys = tuple(
        (
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.aggregation,
        )
        for row in health_rows
    )
    if len(persistent_keys) != len(set(persistent_keys)):
        raise ValueError("persistent population metrics contain a duplicate coordinate")
    if len(health_keys) != len(set(health_keys)):
        raise ValueError("health population metrics contain a duplicate coordinate")
    if set(persistent_keys) != expected_replay_persistent_coordinates(plan):
        raise ValueError("persistent population metrics do not match the frozen 464-row matrix")
    if set(health_keys) != expected_replay_health_coordinates(plan):
        raise ValueError("health population metrics do not match the frozen 14,988-row matrix")
    validate_replay_health_population_grid(
        plan,
        health_rows,
        contrasts=health_contrasts,
    )
    for metric in health_rows:
        case = health_case_by_selector[metric.condition_selector]
        structural_common_mode_na = (
            case.family == "common-mode-position-bias"
            and metric.metric_id in _STRUCTURAL_COMMON_MODE_NA_METRICS
        )
        if structural_common_mode_na and metric.status != "not-applicable":
            raise ValueError(
                "common-mode target/oracle/recovery coordinates must be not-applicable"
            )
        if metric.status == "not-applicable" and not (
            structural_common_mode_na
            or metric.metric_id == "frame-oracle-recoverable-loss-fraction"
        ):
            raise ValueError(
                "not-applicable health status is only valid for structural or "
                "support-incompatible recovery coordinates"
            )

    curated_persistent: list[ReplayPersistentAggregateV1] = []
    curated_health: list[ReplayHealthAggregateV1] = []
    sensitivity: list[ReplayClusterSensitivityV1] = []
    recovery_support_counts: dict[tuple[str, str, str], int] = {}
    for row in health_contrasts:
        key = (row.condition_selector, row.policy, row.window)
        recovery_support_counts[key] = recovery_support_counts.get(key, 0) + int(
            row.identical_support_recovery_applicable
        )
    for metric in persistent_rows:
        case = persistent_case_by_selector[metric.condition_selector]
        record, related = _persistent_record(
            run_id=run.run_id,
            case_identity=case.identity,
            metric=metric,
            log_group_ordinals=groups,
        )
        curated_persistent.append(record)
        sensitivity.extend(related)
    for metric in health_rows:
        case = health_case_by_selector[metric.condition_selector]
        record, related = _health_record(
            run_id=run.run_id,
            case_identity=case.identity,
            metric=metric,
            recovery_support_compatible_scene_count=(
                recovery_support_counts[
                    (metric.condition_selector, metric.method_id, metric.window)
                ]
                if metric.metric_id == "frame-oracle-recoverable-loss-fraction"
                and case.family != "common-mode-position-bias"
                else None
            ),
            log_group_ordinals=groups,
        )
        curated_health.append(record)
        sensitivity.extend(related)

    claim_rows_a = tuple(row for row in curated_persistent if row.hypothesis_id is not None)
    actual_a = {
        (
            row.hypothesis_id,
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.aggregation,
            row.inference_role,
            row.expected_direction,
        )
        for row in claim_rows_a
    }
    expected_a = _LOCAL_M5_A_COORDINATES
    if actual_a != expected_a or len(claim_rows_a) != len(expected_a):
        raise ValueError("curated M5-A hypothesis map is incomplete")

    claim_rows_b = tuple(row for row in curated_health if row.hypothesis_id is not None)
    actual_b = {
        (
            row.hypothesis_id,
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.aggregation,
            row.inference_role,
            row.expected_direction,
        )
        for row in claim_rows_b
    }
    expected_b = {
        (
            row.hypothesis_id,
            row.selector,
            row.method,
            row.metric_name,
            row.window,
            row.unit,
            "equal-scene-mean",
            (
                "primary-directional"
                if row.assessment_rule == "persistence"
                else "nonpositive-control"
                if row.assessment_rule == "nonpositive-control"
                else "diagnostic"
            ),
            (
                row.expected_direction
                if row.expected_direction is not None
                else "nonpositive"
                if row.assessment_rule == "nonpositive-control"
                else "none"
            ),
        )
        for row in H5_B_SELECTORS
    }
    if actual_b != expected_b or len(claim_rows_b) != len(H5_B_SELECTORS):
        raise ValueError("curated M5-B hypothesis map is incomplete")

    result_ids = tuple(row.result_id for row in (*curated_persistent, *curated_health))
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("curated replay result identifiers are not globally unique")

    persistent_rank = {
        case.fault_condition.selector: index for index, case in enumerate(plan.persistent_cases)
    }
    health_rank = {case.selector: index for index, case in enumerate(plan.health_cases)}
    ordered_persistent = tuple(
        sorted(
            curated_persistent,
            key=lambda row: (
                persistent_rank[row.condition_selector],
                row.method_id,
                row.metric_id,
                row.result_id,
            ),
        )
    )
    ordered_health = tuple(
        sorted(
            curated_health,
            key=lambda row: (
                health_rank[row.condition_selector],
                row.method_id,
                row.metric_id,
                row.window,
                row.result_id,
            ),
        )
    )
    ordered_sensitivity = tuple(
        sorted(
            sensitivity,
            key=lambda row: (
                row.source_result_id,
                row.cluster_kind,
                row.cluster_id,
            ),
        )
    )
    public_run = run.model_copy(update={"command": M5_PUBLIC_REPLAY_COMMAND})
    return ReplayCuratedAggregateEvidence(
        profile_summary=profile_summary,
        descriptor_aggregates=descriptors,
        persistent_aggregates=ordered_persistent,
        persistent_crossovers=_curate_crossovers(
            plan=plan,
            crossovers=persistent_crossovers,
            run_id=run.run_id,
        ),
        health_aggregates=ordered_health,
        cluster_sensitivity=ordered_sensitivity,
        run=public_run,
    )


def curate_replay_evidence(
    *,
    plan: LoadedReplayPlan,
    persistent_scene_evaluations: Sequence[ReplayPersistentSceneEvaluation],
    persistent_metrics: Sequence[ReplayPersistentPopulationMetric],
    persistent_crossovers: Sequence[ReplayPersistentCrossoverEstimate],
    health_results: Sequence[ReplayHealthResultV1],
    health_contrasts: Sequence[ReplayHealthSequenceContrast],
    health_events: Sequence[ReplayHealthSequenceEventV1],
    descriptor_aggregates: Sequence[ReplayDescriptorAggregateV1],
    log_group_bindings: Sequence[ReplayLogGroupBinding],
    profile_summary: ReplayProfileSummaryV1,
    run: RunRecordV1Alpha1,
) -> ReplayCuratedAggregateEvidence:
    """Aggregate M5-B rows and curate both authenticated replay panels."""

    bindings = tuple(log_group_bindings)
    expected_sequence_ids = tuple(f"nuscenes:{name}" for name in M5_SCENE_NAMES)
    if tuple(binding.sequence_id for binding in bindings) != expected_sequence_ids:
        raise ValueError("log-group bindings do not match frozen replay scene order")
    recomputed_persistent, recomputed_crossovers = aggregate_replay_persistent_evidence(
        plan=plan,
        scene_evaluations=persistent_scene_evaluations,
    )

    def persistent_key(
        row: ReplayPersistentPopulationMetric,
    ) -> tuple[str, str, str, str]:
        return (
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.aggregation,
        )

    def crossover_key(
        row: ReplayPersistentCrossoverEstimate,
    ) -> tuple[str, str, str]:
        return (
            row.condition_id,
            row.direction,
            row.severity_unit,
        )

    if tuple(sorted(persistent_metrics, key=persistent_key)) != tuple(
        sorted(recomputed_persistent, key=persistent_key)
    ) or tuple(sorted(persistent_crossovers, key=crossover_key)) != tuple(
        sorted(recomputed_crossovers, key=crossover_key)
    ):
        raise ValueError("persistent population evidence differs from deterministic recomputation")
    health_metrics = aggregate_replay_health_evidence(
        plan=plan,
        health_results=health_results,
        health_contrasts=health_contrasts,
        health_events=health_events,
    )
    return _curate_replay_population_evidence(
        plan=plan,
        persistent_metrics=recomputed_persistent,
        persistent_crossovers=recomputed_crossovers,
        health_metrics=health_metrics,
        health_contrasts=health_contrasts,
        descriptor_aggregates=descriptor_aggregates,
        log_group_ordinals=tuple(binding.log_group_ordinal for binding in bindings),
        profile_summary=profile_summary,
        run=run,
    )


def assemble_replay_curated_write_request(
    evidence: ReplayCuratedAggregateEvidence,
    *,
    validation: ReplayValidationV1,
    repeat_verification: ReplayRepeatVerificationV1,
    figures: Sequence[ReplayFigureSourceBindingV1],
    source_commitments: Sequence[ReplaySourceMemberCommitmentV1],
) -> ReplayCuratedArtifactWriteRequest:
    """Attach independently evidenced release gates without inventing their hashes."""

    return ReplayCuratedArtifactWriteRequest(
        profile_summary=evidence.profile_summary,
        descriptor_aggregates=evidence.descriptor_aggregates,
        persistent_aggregates=evidence.persistent_aggregates,
        persistent_crossovers=evidence.persistent_crossovers,
        health_aggregates=evidence.health_aggregates,
        cluster_sensitivity=evidence.cluster_sensitivity,
        validation=validation,
        repeat_verification=repeat_verification,
        figures=tuple(figures),
        source_commitments=tuple(source_commitments),
        run=evidence.run,
    )
