"""Exact M4 clean fitting and validation-only threshold orchestration.

This module owns the one-way train/validation boundary.  It expands the frozen
condition groups into globally unique value-level cases, fits the eight clean
ECDF arrays on train only, evaluates all 36 combined-policy candidates on
cached validation features, and returns an in-memory fit bundle.  Artifact
publication and test-split evaluation deliberately live elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal

import numpy as np

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.health_result_v1 import (
    CROSS_THRESHOLDS,
    SELF_THRESHOLDS,
    HealthFitSummaryV1,
    HealthThresholdCandidateV1,
    HealthValidationCheckV1,
    HealthValidationV1,
)
from fusion_fault_bench.contracts.health_v1 import (
    M4_HEALTH_INTENT_SHA256,
    HealthBenchmarkIntentV1,
    load_health_benchmark_intent,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EdgeProceduralProfile,
    MainProceduralProfile,
    load_procedural_profile,
)
from fusion_fault_bench.experiments.health import (
    HealthObservationSequence,
    generate_health_observations,
)
from fusion_fault_bench.health import (
    ExecutedAction,
    HealthCalibration,
    HealthFrameInput,
    HealthThresholds,
    ModalityMeasurement,
    ObjectHealthInput,
)
from fusion_fault_bench.health_evaluation import (
    HealthPolicyTrace,
    MethodLossTrace,
    evaluate_policy_trace,
    fixed_fusion_values,
    frame_oracle_actions,
    loss_trace_for_actions,
)
from fusion_fault_bench.health_fit import (
    CANDIDATE_COUNT,
    ECDF_VALUES_PER_CHANNEL,
    FRAMES_PER_SEQUENCE,
    TRAIN_SEQUENCE_COUNT,
    VALIDATION_BOOTSTRAP_REPLICATES,
    VALIDATION_SEQUENCE_COUNT,
    HealthFeatureTrace,
    HealthThresholdSelection,
    ScoredHealthTrace,
    ValidationConditionRegret,
    compute_health_feature_trace,
    fit_clean_health_calibration,
    rescore_health_feature_trace,
    select_health_thresholds,
)
from fusion_fault_bench.reference.health import constant_velocity_prediction, nis
from fusion_fault_bench.scenarios.health import (
    HealthBaseSequence,
    HealthFaultSpec,
    generate_health_base_sequences,
    health_event_schedule,
)

type HealthBenchmarkStage = Literal["validation", "test"]
type HealthPopulationId = Literal["main-validation", "main-test", "edge-test"]

_MAIN_PROFILE_PATH = Path("examples/profiles/constant-velocity-front-roi-v1.json")
_EDGE_PROFILE_PATH = Path("examples/profiles/constant-velocity-fov-edge-v1.json")
_IDENTITY_FAULT = HealthFaultSpec(
    family="identity",
    target="none",
    axis="none",
    unit="identity",
    value=0.0,
)
_ECDF_CHANNEL_NAMES = (
    "camera_self_mean",
    "camera_self_maximum",
    "lidar_self_mean",
    "lidar_self_maximum",
    "camera_from_lidar_cross_mean",
    "camera_from_lidar_cross_maximum",
    "lidar_from_camera_cross_mean",
    "lidar_from_camera_cross_maximum",
)
_CANDIDATE_THRESHOLDS = tuple(
    HealthThresholds(self_score=self_threshold, cross_score=cross_threshold)
    for self_threshold in SELF_THRESHOLDS
    for cross_threshold in CROSS_THRESHOLDS
)
_FIXED_ACTIONS: tuple[ExecutedAction, ...] = tuple(
    "fixed-fusion" for _ in range(FRAMES_PER_SEQUENCE)
)


@dataclass(frozen=True, slots=True)
class HealthScenarioMetadata:
    """Generator-only scenario metadata that never enters ``HealthFrameInput``."""

    population: HealthPopulationId
    fault: HealthFaultSpec


@dataclass(frozen=True, slots=True)
class HealthCaseDescriptor:
    """One expanded validation or test value with a globally unique ID."""

    value_id: str
    condition_id: str
    condition_group_id: str
    value_index: int
    stage: HealthBenchmarkStage
    role: str
    scenario: HealthScenarioMetadata

    def __post_init__(self) -> None:
        if (
            not self.value_id
            or not self.condition_id
            or not self.condition_group_id
            or self.value_index < 0
            or not self.role
        ):
            raise ValueError("health case descriptors require complete value-level identity")
        expected = f"{self.condition_group_id}.value-{self.value_index:02d}"
        if self.value_id != expected or self.condition_id != expected:
            raise ValueError("health value ID does not use the canonical value-level form")
        if self.stage == "validation" and self.scenario.population != "main-validation":
            raise ValueError("validation cases must use the main validation population")

    @property
    def fault(self) -> HealthFaultSpec:
        """Return the generator-only fault coordinate."""

        return self.scenario.fault

    @property
    def population(self) -> HealthPopulationId:
        """Return the base population without exposing it to the scorer."""

        return self.scenario.population


@dataclass(frozen=True, slots=True)
class HealthPopulationProfiles:
    """The exact main and edge profiles bound to the M4 intent."""

    main_profile: MainProceduralProfile
    edge_profile: EdgeProceduralProfile
    main_profile_sha256: str
    edge_profile_sha256: str

    def __post_init__(self) -> None:
        if sha256_digest(self.main_profile) != self.main_profile_sha256:
            raise ValueError("main profile digest does not match its loaded content")
        if sha256_digest(self.edge_profile) != self.edge_profile_sha256:
            raise ValueError("edge profile digest does not match its loaded content")


@dataclass(frozen=True, slots=True, eq=False)
class HealthBenchmarkFit:
    """Complete in-memory M4 fit result before artifact publication."""

    intent: HealthBenchmarkIntentV1
    intent_sha256: str
    profiles: HealthPopulationProfiles
    validation_cases: tuple[HealthCaseDescriptor, ...]
    test_cases: tuple[HealthCaseDescriptor, ...]
    calibration: HealthCalibration
    condition_regrets: tuple[ValidationConditionRegret, ...]
    candidates: tuple[HealthThresholdCandidateV1, ...]
    selection: HealthThresholdSelection
    summary: HealthFitSummaryV1
    validation: HealthValidationV1

    def __post_init__(self) -> None:
        if self.intent_sha256 != sha256_digest(self.intent):
            raise ValueError("fit intent digest does not match its content")
        if self.candidates != self.selection.candidates:
            raise ValueError("fit candidate records disagree with threshold selection")
        if not self.validation.all_checks_passed:
            raise ValueError("an M4 fit cannot advance with a failed validation conjunction")


@dataclass(frozen=True, slots=True, eq=False)
class _CachedCleanSequence:
    observations: HealthObservationSequence
    features: HealthFeatureTrace
    scored: ScoredHealthTrace
    fused_xy_m: np.ndarray
    fixed_loss: MethodLossTrace


@dataclass(frozen=True, slots=True)
class _WindowStatistic:
    mean_loss_m2: float
    coverage: float


@dataclass(frozen=True, slots=True)
class _CausalityChecks:
    metadata_boundary: bool
    future_prefix: bool
    current_preupdate: bool
    independent_histories: bool


def _expanded_case(
    *,
    condition_id: str,
    value_index: int,
    stage: HealthBenchmarkStage,
    role: str,
    population: HealthPopulationId,
    family: str,
    target: str,
    axis: str,
    unit: str,
    value: float,
    schedule: str,
) -> HealthCaseDescriptor:
    return HealthCaseDescriptor(
        value_id=f"{condition_id}.value-{value_index:02d}",
        condition_id=f"{condition_id}.value-{value_index:02d}",
        condition_group_id=condition_id,
        value_index=value_index,
        stage=stage,
        role=role,
        scenario=HealthScenarioMetadata(
            population=population,
            fault=HealthFaultSpec(
                family=family,  # type: ignore[arg-type]
                target=target,  # type: ignore[arg-type]
                axis=axis,  # type: ignore[arg-type]
                unit=unit,  # type: ignore[arg-type]
                value=value,
                schedule=schedule,  # type: ignore[arg-type]
            ),
        ),
    )


def expand_validation_cases(
    intent: HealthBenchmarkIntentV1,
) -> tuple[HealthCaseDescriptor, ...]:
    """Expand all 32 validation fault values plus the sole clean value."""

    cases: list[HealthCaseDescriptor] = []
    for condition in (*intent.validation_controls, *intent.validation_matrix):
        for value_index, value in enumerate(condition.values):
            cases.append(
                _expanded_case(
                    condition_id=condition.condition_id,
                    value_index=value_index,
                    stage="validation",
                    role=condition.selection_role,
                    population="main-validation",
                    family=condition.family,
                    target=condition.target,
                    axis=condition.axis,
                    unit=condition.unit,
                    value=float(value),
                    schedule=condition.schedule,
                )
            )
    result = tuple(cases)
    value_ids = tuple(item.value_id for item in result)
    if len(result) != 33 or len(set(value_ids)) != len(value_ids):
        raise ValueError("M4 validation expansion must produce 33 unique value-level cases")
    return result


def expand_test_cases(
    intent: HealthBenchmarkIntentV1,
) -> tuple[HealthCaseDescriptor, ...]:
    """Expand all 38 primary test values plus nine control values."""

    cases: list[HealthCaseDescriptor] = []
    for condition in intent.test_matrix:
        for value_index, value in enumerate(condition.values):
            cases.append(
                _expanded_case(
                    condition_id=condition.condition_id,
                    value_index=value_index,
                    stage="test",
                    role=condition.test_role,
                    population="main-test",
                    family=condition.family,
                    target=condition.target,
                    axis=condition.axis,
                    unit=condition.unit,
                    value=float(value),
                    schedule=condition.schedule,
                )
            )
    for control in intent.test_controls:
        for value_index, value in enumerate(control.values):
            cases.append(
                _expanded_case(
                    condition_id=control.condition_id,
                    value_index=value_index,
                    stage="test",
                    role=control.test_role,
                    population=control.population,
                    family=control.family,
                    target=control.target,
                    axis=control.axis,
                    unit=control.unit,
                    value=float(value),
                    schedule=control.schedule,
                )
            )
    result = tuple(cases)
    value_ids = tuple(item.value_id for item in result)
    if len(result) != 47 or len(set(value_ids)) != len(value_ids):
        raise ValueError("M4 test expansion must produce 47 unique value-level cases")
    return result


def load_health_population_profiles(
    intent: HealthBenchmarkIntentV1,
    *,
    source_root: Path,
) -> HealthPopulationProfiles:
    """Load and bind the exact main and edge procedural profiles."""

    main = load_procedural_profile(source_root / _MAIN_PROFILE_PATH)
    edge = load_procedural_profile(source_root / _EDGE_PROFILE_PATH)
    if not isinstance(main, MainProceduralProfile):
        raise ValueError("M4 main profile has the wrong profile type")
    if not isinstance(edge, EdgeProceduralProfile):
        raise ValueError("M4 edge profile has the wrong profile type")
    main_digest = sha256_digest(main)
    edge_digest = sha256_digest(edge)
    if (
        main.profile_id != intent.source_population.profile_id
        or main_digest != intent.source_population.profile_sha256
    ):
        raise ValueError("M4 main profile does not match the frozen intent")
    if (
        edge.profile_id != intent.source_population.edge_profile_id
        or edge_digest != intent.source_population.edge_profile_sha256
    ):
        raise ValueError("M4 edge profile does not match the frozen intent")
    return HealthPopulationProfiles(
        main_profile=main,
        edge_profile=edge,
        main_profile_sha256=main_digest,
        edge_profile_sha256=edge_digest,
    )


def _window_statistic(
    observations: HealthObservationSequence,
    loss: MethodLossTrace,
    *,
    start: int,
    end: int,
) -> _WindowStatistic:
    eligible = observations.eligibility_mask[start:end]
    valid = np.asarray(loss.valid_mask[start:end] & eligible, dtype=np.bool_)
    eligible_count = int(np.count_nonzero(eligible))
    valid_count = int(np.count_nonzero(valid))
    if eligible_count == 0 or valid_count == 0:
        raise ValueError("M4 fit windows require nonempty eligible and method support")
    loss_values = loss.loss_m2[start:end][valid]
    mean_loss = math.fsum(float(value) for value in loss_values) / valid_count
    return _WindowStatistic(
        mean_loss_m2=mean_loss,
        coverage=valid_count / eligible_count,
    )


def _false_alert_episode_starts(
    trace: HealthPolicyTrace,
    *,
    start: int,
    end: int,
) -> int:
    previous = "healthy" if start == 0 else trace.latched_labels[start - 1]
    count = 0
    for current in trace.latched_labels[start:end]:
        if previous == "healthy" and current != "healthy":
            count += 1
        previous = current
    return count


def _fixed_loss(
    observations: HealthObservationSequence,
    *,
    fused_xy_m: np.ndarray,
) -> MethodLossTrace:
    return loss_trace_for_actions(
        observations,
        method="fixed-fusion",
        actions=_FIXED_ACTIONS,
        fused=fused_xy_m,
    )


def _cache_clean_validation(
    bases: tuple[HealthBaseSequence, ...],
    calibration: HealthCalibration,
) -> tuple[_CachedCleanSequence, ...]:
    cached: list[_CachedCleanSequence] = []
    for base in bases:
        observations = generate_health_observations(base, fault=_IDENTITY_FAULT)
        features = compute_health_feature_trace(observations.health_frame_inputs())
        scored = rescore_health_feature_trace(features, calibration)
        fused, _ = fixed_fusion_values(observations)
        cached.append(
            _CachedCleanSequence(
                observations=observations,
                features=features,
                scored=scored,
                fused_xy_m=fused,
                fixed_loss=_fixed_loss(observations, fused_xy_m=fused),
            )
        )
    return tuple(cached)


def _evaluate_clean_candidates(
    cached: tuple[_CachedCleanSequence, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence_count = len(cached)
    regression = np.empty((CANDIDATE_COUNT, sequence_count), dtype=np.float64)
    coverage = np.empty_like(regression)
    false_alerts = np.empty_like(regression)
    fixed_coverage = np.empty(sequence_count, dtype=np.float64)
    score_start, score_end = health_event_schedule("standard").score_frames
    fixed_statistics = tuple(
        _window_statistic(
            item.observations,
            item.fixed_loss,
            start=score_start,
            end=score_end,
        )
        for item in cached
    )
    for sequence_index, statistic in enumerate(fixed_statistics):
        fixed_coverage[sequence_index] = statistic.coverage

    for candidate_index, thresholds in enumerate(_CANDIDATE_THRESHOLDS):
        for sequence_index, item in enumerate(cached):
            policy = evaluate_policy_trace(
                item.scored.frames,
                method="combined-health-gate",
                thresholds=thresholds,
            )
            policy_loss = loss_trace_for_actions(
                item.observations,
                method="combined-health-gate",
                actions=policy.actions,
                fused=item.fused_xy_m,
            )
            policy_statistic = _window_statistic(
                item.observations,
                policy_loss,
                start=score_start,
                end=score_end,
            )
            regression[candidate_index, sequence_index] = (
                policy_statistic.mean_loss_m2 - fixed_statistics[sequence_index].mean_loss_m2
            )
            coverage[candidate_index, sequence_index] = policy_statistic.coverage
            false_alerts[candidate_index, sequence_index] = _false_alert_episode_starts(
                policy,
                start=score_start,
                end=score_end,
            )
    return regression, coverage, fixed_coverage, false_alerts


def _observation_identity_outside_event(
    clean: HealthObservationSequence,
    faulted: HealthObservationSequence,
    *,
    fault: HealthFaultSpec,
) -> bool:
    schedule = health_event_schedule(fault.schedule)
    active_start, active_end = schedule.fault_active_frames
    outside = np.ones(clean.frame_count, dtype=np.bool_)
    outside[active_start:active_end] = False
    if (
        clean.sequence_id != faulted.sequence_id
        or clean.object_ids != faulted.object_ids
        or clean.health_frame != faulted.health_frame
        or not np.array_equal(clean.frame_indices, faulted.frame_indices)
    ):
        return False
    frame_fields = (
        "reference_times_s",
        "truth_xy_m",
        "eligibility_mask",
        "camera_value_xy_m",
        "lidar_value_xy_m",
        "camera_actual_covariance_xy_m2",
        "lidar_actual_covariance_xy_m2",
        "camera_reported_covariance_xy_m2",
        "lidar_reported_covariance_xy_m2",
        "camera_available",
        "lidar_available",
        "camera_reported_times_s",
        "lidar_reported_times_s",
    )
    return all(
        np.array_equal(getattr(clean, name)[outside], getattr(faulted, name)[outside])
        for name in frame_fields
    )


def _candidate_regrets_for_sequence(
    observations: HealthObservationSequence,
    scored: ScoredHealthTrace,
) -> tuple[np.ndarray, bool]:
    score_start, score_end = health_event_schedule("standard").score_frames
    fused, fused_available = fixed_fusion_values(observations)
    oracle_actions = frame_oracle_actions(
        observations,
        fused=fused,
        fused_available=fused_available,
    )
    oracle_loss = loss_trace_for_actions(
        observations,
        method="frame-action-performance-oracle",
        actions=oracle_actions,
        fused=fused,
    )
    oracle_statistic = _window_statistic(
        observations,
        oracle_loss,
        start=score_start,
        end=score_end,
    )
    regrets = np.empty(CANDIDATE_COUNT, dtype=np.float64)
    oracle_dominance = True
    for candidate_index, thresholds in enumerate(_CANDIDATE_THRESHOLDS):
        policy = evaluate_policy_trace(
            scored.frames,
            method="combined-health-gate",
            thresholds=thresholds,
        )
        policy_loss = loss_trace_for_actions(
            observations,
            method="combined-health-gate",
            actions=policy.actions,
            fused=fused,
        )
        policy_statistic = _window_statistic(
            observations,
            policy_loss,
            start=score_start,
            end=score_end,
        )
        if policy_statistic.coverage != oracle_statistic.coverage:
            raise AssertionError("selection utility requires identical policy/oracle support")
        regret = policy_statistic.mean_loss_m2 - oracle_statistic.mean_loss_m2
        if regret < -1e-12:
            oracle_dominance = False
        regrets[candidate_index] = 0.0 if -1e-12 <= regret < 0.0 else regret
    return regrets, oracle_dominance


def _selection_cases(
    intent: HealthBenchmarkIntentV1,
    validation_cases: tuple[HealthCaseDescriptor, ...],
) -> tuple[HealthCaseDescriptor, ...]:
    by_condition: dict[str, list[HealthCaseDescriptor]] = {}
    for case in validation_cases:
        by_condition.setdefault(case.condition_group_id, []).append(case)
    selected = tuple(
        case
        for condition_id in intent.threshold_selection.selection_conditions
        for case in by_condition[condition_id]
    )
    if len(selected) != 20:
        raise ValueError("M4 selection utility must contain exactly 20 value-level cases")
    return selected


def _evaluate_utility_conditions(
    *,
    cases: tuple[HealthCaseDescriptor, ...],
    bases: tuple[HealthBaseSequence, ...],
    clean: tuple[_CachedCleanSequence, ...],
    calibration: HealthCalibration,
) -> tuple[tuple[ValidationConditionRegret, ...], int, int, int]:
    conditions: list[ValidationConditionRegret] = []
    identity_comparisons = 0
    identity_violations = 0
    oracle_violations = 0
    for case in cases:
        regrets = np.empty(
            (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
            dtype=np.float64,
        )
        for sequence_index, base in enumerate(bases):
            observations = generate_health_observations(base, fault=case.fault)
            identity_comparisons += 1
            identity_violations += not _observation_identity_outside_event(
                clean[sequence_index].observations,
                observations,
                fault=case.fault,
            )
            features = compute_health_feature_trace(observations.health_frame_inputs())
            scored = rescore_health_feature_trace(features, calibration)
            sequence_regrets, sequence_dominance = _candidate_regrets_for_sequence(
                observations,
                scored,
            )
            regrets[:, sequence_index] = sequence_regrets
            oracle_violations += not sequence_dominance
        target = case.fault.target
        if target not in {"camera", "lidar"}:
            raise AssertionError("selection utility must have one sensor target")
        conditions.append(
            ValidationConditionRegret(
                condition_id=case.value_id,
                target="camera" if target == "camera" else "lidar",
                family=case.fault.family,
                regret_m2_by_candidate_sequence=regrets,
            )
        )
    return (
        tuple(conditions),
        identity_comparisons,
        identity_violations,
        oracle_violations,
    )


def _mutate_camera_frame(
    frames: tuple[HealthFrameInput, ...],
    *,
    frame_index: int,
) -> tuple[HealthFrameInput, ...]:
    frame = frames[frame_index]
    mutated_objects: list[ObjectHealthInput] = []
    delta = np.asarray((17.0, -11.0), dtype=np.float64)
    for item in frame.objects:
        if item.camera is None:
            raise AssertionError("clean causality checks require camera availability")
        mutated_objects.append(
            replace(
                item,
                camera=replace(
                    item.camera,
                    value_xy_m=np.asarray(item.camera.value_xy_m) + delta,
                ),
            )
        )
    result = list(frames)
    result[frame_index] = replace(frame, objects=tuple(mutated_objects))
    return tuple(result)


def _reference_camera_self_mean(
    frames: tuple[HealthFrameInput, ...],
    *,
    frame_index: int,
) -> float:
    first_by_id = {item.object_id: item for item in frames[frame_index - 2].objects}
    second_by_id = {item.object_id: item for item in frames[frame_index - 1].objects}
    values: list[float] = []
    for current in frames[frame_index].objects:
        first = first_by_id[current.object_id]
        second = second_by_id[current.object_id]
        if first.camera is None or second.camera is None or current.camera is None:
            continue
        predicted_value, predicted_covariance = constant_velocity_prediction(
            first_value_xy=first.camera.value_xy_m,
            first_covariance_xy=first.camera.reported_covariance_xy_m2,
            first_time_s=frames[frame_index - 2].reference_time_s,
            second_value_xy=second.camera.value_xy_m,
            second_covariance_xy=second.camera.reported_covariance_xy_m2,
            second_time_s=frames[frame_index - 1].reference_time_s,
            reference_time_s=frames[frame_index].reference_time_s,
        )
        values.append(
            nis(
                current_value_xy=current.camera.value_xy_m,
                current_covariance_xy=current.camera.reported_covariance_xy_m2,
                predicted_value_xy=predicted_value,
                predicted_covariance_xy=predicted_covariance,
            )
        )
    if len(values) < 2:
        raise AssertionError("causality reference requires two mature camera objects")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _causality_checks(clean: _CachedCleanSequence) -> _CausalityChecks:
    frames = clean.observations.health_frame_inputs()
    allowed_fields = {
        HealthFrameInput: (
            "reference_time_s",
            "camera_available",
            "lidar_available",
            "objects",
        ),
        ObjectHealthInput: ("object_id", "camera", "lidar"),
        ModalityMeasurement: (
            "value_xy_m",
            "reported_covariance_xy_m2",
            "reported_time_s",
        ),
    }
    metadata_boundary = all(
        tuple(field.name for field in fields(contract)) == expected
        for contract, expected in allowed_fields.items()
    )

    current_index = 20
    current_frames = _mutate_camera_frame(frames, frame_index=current_index)
    current_mutation = compute_health_feature_trace(current_frames)
    future_prefix = clean.features.frames[:current_index] == current_mutation.frames[:current_index]
    original_channel = clean.features.frames[current_index].camera_self
    mutated_channel = current_mutation.frames[current_index].camera_self
    original_reference_mean = _reference_camera_self_mean(
        frames,
        frame_index=current_index,
    )
    mutated_reference_mean = _reference_camera_self_mean(
        current_frames,
        frame_index=current_index,
    )
    current_preupdate = (
        clean.features.frames[:current_index] == current_mutation.frames[:current_index]
        and original_channel.status == "defined"
        and mutated_channel.status == "defined"
        and original_channel.mean_nis is not None
        and mutated_channel.mean_nis is not None
        and math.isclose(
            original_channel.mean_nis,
            original_reference_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            mutated_channel.mean_nis,
            mutated_reference_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and original_channel.mean_nis != mutated_channel.mean_nis
        and clean.features.frames[current_index].lidar_from_camera_cross
        == current_mutation.frames[current_index].lidar_from_camera_cross
    )

    independent_histories = (
        all(
            original.lidar_self == mutated.lidar_self
            for original, mutated in zip(
                clean.features.frames,
                current_mutation.frames,
                strict=True,
            )
        )
        and clean.features.frames[current_index + 1].lidar_from_camera_cross
        != current_mutation.frames[current_index + 1].lidar_from_camera_cross
    )
    return _CausalityChecks(
        metadata_boundary=metadata_boundary,
        future_prefix=future_prefix,
        current_preupdate=current_preupdate,
        independent_histories=independent_histories,
    )


def _validation_check(
    check_id: str,
    *,
    observed: float | int | bool | str,
    expected: float | int | bool | str,
    passed: bool | None = None,
) -> HealthValidationCheckV1:
    return HealthValidationCheckV1(
        check_id=check_id,
        passed=observed == expected if passed is None else passed,
        observed=observed,
        expected=expected,
    )


def _build_fit_validation(
    *,
    intent: HealthBenchmarkIntentV1,
    intent_sha256: str,
    profiles: HealthPopulationProfiles,
    train_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    validation_cases: tuple[HealthCaseDescriptor, ...],
    test_cases: tuple[HealthCaseDescriptor, ...],
    selection_cases: tuple[HealthCaseDescriptor, ...],
    calibration: HealthCalibration,
    selection: HealthThresholdSelection,
    identity_comparisons: int,
    identity_violations: int,
    oracle_violations: int,
    causality: _CausalityChecks,
) -> HealthValidationV1:
    all_value_ids = tuple(case.value_id for case in (*validation_cases, *test_cases))
    ecdf_lengths = tuple(int(getattr(calibration, name).size) for name in _ECDF_CHANNEL_NAMES)
    candidate_order = tuple(
        (candidate.self_threshold, candidate.cross_threshold) for candidate in selection.candidates
    ) == tuple(
        (self_threshold, cross_threshold)
        for self_threshold in SELF_THRESHOLDS
        for cross_threshold in CROSS_THRESHOLDS
    )
    candidate_frame_evaluations = (
        (1 + len(selection_cases))
        * VALIDATION_SEQUENCE_COUNT
        * CANDIDATE_COUNT
        * FRAMES_PER_SEQUENCE
    )
    bootstrap_cells = VALIDATION_BOOTSTRAP_REPLICATES * VALIDATION_SEQUENCE_COUNT
    checks = (
        _validation_check(
            "intent-digest",
            observed=intent_sha256,
            expected=M4_HEALTH_INTENT_SHA256,
        ),
        _validation_check(
            "main-profile-digest",
            observed=profiles.main_profile_sha256,
            expected=intent.source_population.profile_sha256,
        ),
        _validation_check(
            "edge-profile-digest",
            observed=profiles.edge_profile_sha256,
            expected=intent.source_population.edge_profile_sha256,
        ),
        _validation_check(
            "train-sequence-count",
            observed=len(train_ids),
            expected=TRAIN_SEQUENCE_COUNT,
        ),
        _validation_check(
            "validation-sequence-count",
            observed=len(validation_ids),
            expected=VALIDATION_SEQUENCE_COUNT,
        ),
        _validation_check(
            "train-validation-sequence-id-overlap",
            observed=len(set(train_ids) & set(validation_ids)),
            expected=0,
        ),
        _validation_check(
            "validation-value-case-count",
            observed=len(validation_cases),
            expected=33,
        ),
        _validation_check(
            "test-value-case-count",
            observed=len(test_cases),
            expected=47,
        ),
        _validation_check(
            "value-case-id-uniqueness",
            observed=len(set(all_value_ids)),
            expected=80,
        ),
        _validation_check(
            "selection-value-case-count",
            observed=len(selection_cases),
            expected=20,
        ),
        _validation_check(
            "identity-outside-active-event",
            observed=(
                identity_comparisons
                if identity_violations == 0
                else f"{identity_comparisons - identity_violations}/{identity_comparisons}"
            ),
            expected=4_000,
        ),
        _validation_check(
            "ecdf-channel-count",
            observed=len(ecdf_lengths),
            expected=8,
        ),
        _validation_check(
            "ecdf-values-per-channel",
            observed=(
                ecdf_lengths[0]
                if len(set(ecdf_lengths)) == 1
                else ",".join(str(value) for value in ecdf_lengths)
            ),
            expected=ECDF_VALUES_PER_CHANNEL,
        ),
        _validation_check(
            "threshold-candidate-count",
            observed=len(selection.candidates),
            expected=CANDIDATE_COUNT,
        ),
        _validation_check(
            "threshold-candidate-order",
            observed=candidate_order,
            expected=True,
        ),
        _validation_check(
            "metadata-leakage-boundary",
            observed=causality.metadata_boundary,
            expected=True,
        ),
        _validation_check(
            "future-prefix-causality",
            observed=causality.future_prefix,
            expected=True,
        ),
        _validation_check(
            "current-preupdate-causality",
            observed=causality.current_preupdate,
            expected=True,
        ),
        _validation_check(
            "independent-modality-histories",
            observed=causality.independent_histories,
            expected=True,
        ),
        _validation_check(
            "frame-oracle-comparison-count",
            observed=(len(selection_cases) * VALIDATION_SEQUENCE_COUNT * CANDIDATE_COUNT),
            expected=144_000,
        ),
        _validation_check(
            "frame-oracle-dominance",
            observed=oracle_violations,
            expected=0,
        ),
        _validation_check(
            "candidate-frame-evaluation-cap",
            observed=candidate_frame_evaluations,
            expected=intent.resource_caps.candidate_frame_evaluations_max,
            passed=(
                candidate_frame_evaluations < intent.resource_caps.candidate_frame_evaluations_max
            ),
        ),
        _validation_check(
            "bootstrap-cell-cap",
            observed=bootstrap_cells,
            expected=intent.resource_caps.bootstrap_cells_max,
            passed=bootstrap_cells < intent.resource_caps.bootstrap_cells_max,
        ),
        _validation_check(
            "scientific-feature-trace-count",
            observed=(
                TRAIN_SEQUENCE_COUNT
                + VALIDATION_SEQUENCE_COUNT
                + len(selection_cases) * VALIDATION_SEQUENCE_COUNT
            ),
            expected=4_400,
        ),
        _validation_check(
            "selected-candidate-feasible",
            observed=selection.candidates[selection.selected_candidate_index].feasible,
            expected=True,
        ),
    )
    return HealthValidationV1(
        schema="ffb.health-validation/v1",
        intent_sha256=intent_sha256,
        checks=checks,
        all_checks_passed=all(check.passed for check in checks),
    )


def fit_health_benchmark(
    *,
    source_root: Path,
) -> HealthBenchmarkFit:
    """Run the exact frozen M4 clean fit and validation threshold selection.

    The public fit is intentionally not parameterized by sequence counts,
    threshold grids, seeds, profiles, or conditions.  It always loads the
    preregistered intent and enforces 200 train plus 200 validation sequences.
    """

    loaded = load_health_benchmark_intent(source_root=source_root)
    intent = loaded.intent
    if (
        intent.source_population.split_sequence_counts.train != TRAIN_SEQUENCE_COUNT
        or intent.source_population.split_sequence_counts.validation != VALIDATION_SEQUENCE_COUNT
    ):
        raise ValueError("M4 fit requires exactly 200 train and 200 validation sequences")
    profiles = load_health_population_profiles(intent, source_root=source_root)
    validation_cases = expand_validation_cases(intent)
    test_cases = expand_test_cases(intent)
    if len({case.value_id for case in (*validation_cases, *test_cases)}) != len(
        validation_cases
    ) + len(test_cases):
        raise ValueError("M4 value-level IDs must be globally unique")
    selection_cases = _selection_cases(intent, validation_cases)
    seed = intent.source_population.data_master_seed

    train_bases = generate_health_base_sequences(
        profiles.main_profile,
        split="train",
        sequence_count=TRAIN_SEQUENCE_COUNT,
        data_master_seed=seed,
    )
    train_ids = tuple(base.sequence_id for base in train_bases)
    train_traces = tuple(
        compute_health_feature_trace(
            generate_health_observations(
                base,
                fault=_IDENTITY_FAULT,
            ).health_frame_inputs()
        )
        for base in train_bases
    )
    calibration = fit_clean_health_calibration(train_traces)

    validation_bases = generate_health_base_sequences(
        profiles.main_profile,
        split="validation",
        sequence_count=VALIDATION_SEQUENCE_COUNT,
        data_master_seed=seed,
    )
    validation_ids = tuple(base.sequence_id for base in validation_bases)
    clean_cache = _cache_clean_validation(validation_bases, calibration)
    (
        clean_regression,
        clean_coverage,
        fixed_clean_coverage,
        false_alerts,
    ) = _evaluate_clean_candidates(clean_cache)
    (
        condition_regrets,
        identity_comparisons,
        identity_violations,
        oracle_violations,
    ) = _evaluate_utility_conditions(
        cases=selection_cases,
        bases=validation_bases,
        clean=clean_cache,
        calibration=calibration,
    )
    selection = select_health_thresholds(
        clean_regression_m2_by_candidate_sequence=clean_regression,
        clean_coverage_by_candidate_sequence=clean_coverage,
        fixed_clean_coverage_by_sequence=fixed_clean_coverage,
        false_alert_starts_by_candidate_sequence=false_alerts,
        condition_regrets=condition_regrets,
    )
    summary = HealthFitSummaryV1(
        schema="ffb.health-fit-summary/v1",
        intent_sha256=loaded.intent_sha256,
        main_profile_sha256=profiles.main_profile_sha256,
        edge_profile_sha256=profiles.edge_profile_sha256,
        train_sequence_count=TRAIN_SEQUENCE_COUNT,
        validation_sequence_count=VALIDATION_SEQUENCE_COUNT,
        ecdf_channel_count=8,
        ecdf_values_per_channel=ECDF_VALUES_PER_CHANNEL,
        candidate_count=CANDIDATE_COUNT,
        selected_candidate_index=selection.selected_candidate_index,
        selected_self_threshold=selection.selected_self_threshold,
        selected_cross_threshold=selection.selected_cross_threshold,
        selection_status="selected",
    )
    causality = _causality_checks(clean_cache[0])
    validation = _build_fit_validation(
        intent=intent,
        intent_sha256=loaded.intent_sha256,
        profiles=profiles,
        train_ids=train_ids,
        validation_ids=validation_ids,
        validation_cases=validation_cases,
        test_cases=test_cases,
        selection_cases=selection_cases,
        calibration=calibration,
        selection=selection,
        identity_comparisons=identity_comparisons,
        identity_violations=identity_violations,
        oracle_violations=oracle_violations,
        causality=causality,
    )
    return HealthBenchmarkFit(
        intent=intent,
        intent_sha256=loaded.intent_sha256,
        profiles=profiles,
        validation_cases=validation_cases,
        test_cases=test_cases,
        calibration=calibration,
        condition_regrets=condition_regrets,
        candidates=selection.candidates,
        selection=selection,
        summary=summary,
        validation=validation,
    )
