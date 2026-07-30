from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.contracts.health_result_v1 import (
    HealthMethod,
    HealthPolicy,
    HealthPolicyMethod,
    HealthWindow,
)
from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayHealthResultV1,
    ReplayHealthSequenceEventV1,
)
from fusion_fault_bench.contracts.replay_v1 import M5_SCENE_NAMES
from fusion_fault_bench.health import ExecutedAction, HealthLabel, RawEvidenceStatus
from fusion_fault_bench.replay_health import (
    ReplayHealthPolicyTrace,
    replay_health_schedule,
    replay_sequence_event_record,
)
from fusion_fault_bench.replay_health_population import (
    ReplayHealthPopulationMetric,
    aggregate_replay_health_contrasts,
    aggregate_replay_health_events,
    aggregate_replay_health_results,
    expected_replay_health_population_coordinates,
    validate_replay_health_population_grid,
)
from fusion_fault_bench.replay_inference import ReplayHealthSequenceContrast
from fusion_fault_bench.replay_plan import ReplayHealthCaseSpec, load_replay_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = load_replay_plan(source_root=ROOT)
SCENE_IDS = tuple(f"nuscenes:{name}" for name in M5_SCENE_NAMES)
IDENTITY_BOOTSTRAP = np.tile(np.arange(10, dtype=np.int64), (40, 1))
SUPPORT_DIGEST = "b" * 64
WINDOWS: tuple[HealthWindow, ...] = ("score", "event", "recovery")
POLICIES: tuple[HealthPolicy, ...] = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
)
ALL_POLICIES: tuple[HealthPolicyMethod, ...] = (
    *POLICIES,
    "combined-health-gate-abstain",
)
HEALTH_METHODS: tuple[HealthMethod, ...] = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    *POLICIES,
    "combined-health-gate-abstain",
    "fault-target-drop-policy",
    "frame-action-performance-oracle",
)
COMMON_MODE_METHODS = HEALTH_METHODS[:-2]


def _case(experiment_id: str, value: float) -> ReplayHealthCaseSpec:
    return next(
        case
        for case in PLAN.health_cases
        if case.identity.experiment_id == experiment_id and case.value == value
    )


def _result_rows(
    case: ReplayHealthCaseSpec,
    *,
    methods: tuple[HealthMethod, ...] | None = None,
    fixed_event_profile: tuple[tuple[float, int, int], ...] | None = None,
) -> tuple[ReplayHealthResultV1, ...]:
    selected_methods = (
        COMMON_MODE_METHODS
        if methods is None and case.family == "common-mode-position-bias"
        else HEALTH_METHODS
        if methods is None
        else methods
    )
    rows: list[ReplayHealthResultV1] = []
    for scene_index, sequence_id in enumerate(SCENE_IDS):
        for method in selected_methods:
            for window in WINDOWS:
                loss_sum, valid, eligible = 2.0, 2, 2
                if (
                    fixed_event_profile is not None
                    and method == "fixed-fusion"
                    and window == "event"
                ):
                    loss_sum, valid, eligible = fixed_event_profile[scene_index]
                rows.append(
                    ReplayHealthResultV1(
                        schema="ffb.replay-health-result/v1",
                        replay_experiment_identity_sha256=case.identity_sha256,
                        sequence_id=sequence_id,
                        condition_id=case.identity.experiment_id,
                        condition_selector=case.selector,
                        method=method,
                        window=window,
                        loss_sum_m2=loss_sum,
                        valid_object_frame_count=valid,
                        eligible_object_frame_count=eligible,
                    )
                )
    return tuple(rows)


def _contrast_rows(
    case: ReplayHealthCaseSpec,
    *,
    zero_recoverable_loss: bool = False,
    policy_loss_override: float | None = None,
) -> tuple[ReplayHealthSequenceContrast, ...]:
    rows: list[ReplayHealthSequenceContrast] = []
    for scene_index, sequence_id in enumerate(SCENE_IDS):
        count = scene_index + 1
        fixed_loss = 10.0
        policy_loss = (
            policy_loss_override
            if policy_loss_override is not None
            else 10.0
            if zero_recoverable_loss
            else 10.0 - scene_index
        )
        target_loss = policy_loss - 1.0
        oracle_loss = 10.0 if zero_recoverable_loss else 0.0
        target_applicable = case.family != "common-mode-position-bias"
        oracle_applicable = case.family != "common-mode-position-bias"
        for policy in ALL_POLICIES:
            for window in WINDOWS:
                oracle_support = (
                    "c" * 64
                    if case.family == "dropout" and window in {"score", "event"}
                    else SUPPORT_DIGEST
                )
                rows.append(
                    ReplayHealthSequenceContrast(
                        replay_experiment_identity_sha256=case.identity_sha256,
                        sequence_id=sequence_id,
                        condition_id=case.identity.experiment_id,
                        condition_selector=case.selector,
                        policy=policy,
                        window=window,
                        fixed_support_sha256=SUPPORT_DIGEST,
                        policy_support_sha256=SUPPORT_DIGEST,
                        fixed_policy_common_count=count,
                        fixed_on_common_loss_sum_m2=count * fixed_loss,
                        policy_on_fixed_common_loss_sum_m2=count * policy_loss,
                        target_drop_applicable=target_applicable,
                        policy_target_drop_common_count=count if target_applicable else None,
                        policy_on_target_common_loss_sum_m2=(
                            count * policy_loss if target_applicable else None
                        ),
                        target_drop_on_common_loss_sum_m2=(
                            count * target_loss if target_applicable else None
                        ),
                        target_drop_support_sha256=(SUPPORT_DIGEST if target_applicable else None),
                        frame_oracle_applicable=oracle_applicable,
                        policy_frame_oracle_common_count=count if oracle_applicable else None,
                        policy_on_oracle_common_loss_sum_m2=(
                            count * policy_loss if oracle_applicable else None
                        ),
                        frame_oracle_on_common_loss_sum_m2=(
                            count * oracle_loss if oracle_applicable else None
                        ),
                        frame_oracle_support_sha256=(oracle_support if oracle_applicable else None),
                    )
                )
    return tuple(rows)


def _event_rows(
    case: ReplayHealthCaseSpec,
    *,
    detected_scene_count: int,
) -> tuple[ReplayHealthSequenceEventV1, ...]:
    schedule = replay_health_schedule(16)
    rows: list[ReplayHealthSequenceEventV1] = []
    for scene_index, sequence_id in enumerate(SCENE_IDS):
        detected = scene_index < detected_scene_count
        scale = 0.5 + 0.1 * scene_index
        reference_times = tuple(scale * frame_index for frame_index in range(16))
        labels: tuple[HealthLabel, ...] = (
            (
                "healthy",
                "healthy",
                "healthy",
                "healthy",
                "healthy",
                "camera-fault",
                "camera-fault",
                "camera-fault",
                "camera-fault",
                "camera-fault",
                "camera-fault",
                "camera-fault",
                "camera-fault",
                "camera-fault",
                "healthy",
                "healthy",
            )
            if detected
            else ("healthy",) * 16
        )
        actions: tuple[ExecutedAction, ...] = tuple(
            "fixed-fusion" if label == "healthy" else "lidar-only" for label in labels
        )
        statuses: tuple[RawEvidenceStatus, ...] = ("update-eligible",) * 16
        camera_available = tuple(
            not (case.family == "dropout" and detected and frame_index == 6)
            for frame_index in range(16)
        )
        for policy in POLICIES:
            trace = ReplayHealthPolicyTrace(
                policy=policy,
                reference_times_s=reference_times,
                camera_available=camera_available,
                lidar_available=(True,) * 16,
                raw_labels=labels,
                evidence_statuses=statuses,
                latched_labels=labels,
                actions=actions,
            )
            rows.append(
                replay_sequence_event_record(
                    trace,
                    schedule=schedule,
                    replay_experiment_identity_sha256=case.identity_sha256,
                    sequence_id=sequence_id,
                    condition_id=case.identity.experiment_id,
                    condition_selector=case.selector,
                    fault_family=case.family,
                    fault_target=case.target,
                )
            )
    return tuple(rows)


def _metric(
    rows: tuple[ReplayHealthPopulationMetric, ...],
    *,
    method_id: str,
    metric_id: str,
    window: HealthWindow,
) -> ReplayHealthPopulationMetric:
    return next(
        row
        for row in rows
        if row.method_id == method_id and row.metric_id == metric_id and row.window == window
    )


@cache
def _full_population_grid() -> tuple[
    tuple[ReplayHealthPopulationMetric, ...],
    tuple[ReplayHealthSequenceContrast, ...],
]:
    metrics: list[ReplayHealthPopulationMetric] = []
    contrasts: list[ReplayHealthSequenceContrast] = []
    for case in PLAN.health_cases:
        case_contrasts = _contrast_rows(case)
        contrasts.extend(case_contrasts)
        metrics.extend(
            aggregate_replay_health_results(
                case,
                _result_rows(case),
                bootstrap_indices=IDENTITY_BOOTSTRAP,
            )
        )
        metrics.extend(
            aggregate_replay_health_contrasts(
                case,
                case_contrasts,
                bootstrap_indices=IDENTITY_BOOTSTRAP,
            )
        )
        metrics.extend(
            aggregate_replay_health_events(
                case,
                _event_rows(case, detected_scene_count=8),
                bootstrap_indices=IDENTITY_BOOTSTRAP,
            )
        )
    return tuple(metrics), tuple(contrasts)


def _as_not_applicable(
    metric: ReplayHealthPopulationMetric,
) -> ReplayHealthPopulationMetric:
    return replace(
        metric,
        status="not-applicable",
        interval=replace(
            metric.interval,
            estimate=None,
            lower=None,
            upper=None,
            defined_replicates=0,
        ),
        scene_numerators=(0.0,) * 10,
        scene_denominators=(0.0,) * 10,
        scene_defined=(False,) * 10,
    )


def test_result_losses_preserve_equal_scene_and_pooled_estimands() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    profile = tuple(
        (float(scene_index * count), count, count) for scene_index, count in enumerate(range(1, 11))
    )
    rows = aggregate_replay_health_results(
        case,
        _result_rows(case, fixed_event_profile=profile),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    full_support = _metric(
        rows,
        method_id="fixed-fusion",
        metric_id="matched-center-mse",
        window="event",
    )
    conditional = _metric(
        rows,
        method_id="fixed-fusion",
        metric_id="conditional-matched-center-mse",
        window="event",
    )
    assert full_support.interval.estimate == pytest.approx(4.5)
    assert full_support.status == "ok"
    assert full_support.scene_values == tuple(float(value) for value in range(10))
    assert conditional.interval.estimate == pytest.approx(6.0)


def test_partial_support_is_undefined_without_zero_imputation() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    profile = ((8.0, 2, 4),) + ((2.0, 2, 2),) * 9
    rows = aggregate_replay_health_results(
        case,
        _result_rows(case, fixed_event_profile=profile),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    full_support = _metric(
        rows,
        method_id="fixed-fusion",
        metric_id="matched-center-mse",
        window="event",
    )
    conditional = _metric(
        rows,
        method_id="fixed-fusion",
        metric_id="conditional-matched-center-mse",
        window="event",
    )
    coverage = _metric(
        rows,
        method_id="fixed-fusion",
        metric_id="coverage",
        window="event",
    )
    assert full_support.interval.estimate is None
    assert full_support.status == "undefined"
    assert full_support.scene_defined == (False,) + (True,) * 9
    assert full_support.scene_values == (None,) + (1.0,) * 9
    assert conditional.interval.estimate == pytest.approx(1.3)
    assert coverage.interval.estimate == pytest.approx(20.0 / 22.0)


def test_result_zero_denominators_remain_undefined() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    profile = ((0.0, 0, 0),) * 10
    rows = aggregate_replay_health_results(
        case,
        _result_rows(case, fixed_event_profile=profile),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    metrics = tuple(
        _metric(
            rows,
            method_id="fixed-fusion",
            metric_id=metric_id,
            window="event",
        )
        for metric_id in (
            "matched-center-mse",
            "conditional-matched-center-mse",
            "coverage",
            "undefined-output-rate",
            "scene-equal-coverage",
        )
    )
    assert all(row.interval.estimate is None for row in metrics)
    assert all(row.status == "undefined" for row in metrics)
    assert all(row.scene_values == (None,) * 10 for row in metrics)


def test_population_status_must_match_interval_and_not_applicable_support() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    rows = aggregate_replay_health_results(
        case,
        _result_rows(case),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    metric = _metric(
        rows,
        method_id="fixed-fusion",
        metric_id="matched-center-mse",
        window="event",
    )

    with pytest.raises(ValueError, match="status disagrees"):
        replace(metric, status="undefined")
    with pytest.raises(ValueError, match="cannot carry numeric support"):
        replace(
            metric,
            status="not-applicable",
            interval=replace(
                metric.interval,
                estimate=None,
                lower=None,
                upper=None,
                defined_replicates=0,
            ),
        )


def test_paired_contrasts_and_recovery_fraction_use_equal_scene_weight() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    rows = aggregate_replay_health_contrasts(
        case,
        _contrast_rows(case),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    gain = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="policy-gain-vs-fixed",
        window="event",
    )
    target_gap = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="gap-vs-fault-target-drop",
        window="event",
    )
    oracle_gap = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="gap-vs-frame-oracle",
        window="event",
    )
    recovery = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="frame-oracle-recoverable-loss-fraction",
        window="event",
    )
    assert gain.interval.estimate == pytest.approx(4.5)
    assert gain.scene_values == tuple(float(value) for value in range(10))
    assert target_gap.interval.estimate == pytest.approx(1.0)
    assert oracle_gap.interval.estimate == pytest.approx(5.5)
    assert recovery.aggregation == "unclipped-recovery-ratio"
    assert recovery.unit == "unitless"
    assert recovery.status == "ok"
    assert recovery.interval.estimate == pytest.approx(0.45)


def test_zero_recoverable_loss_denominator_is_undefined() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    rows = aggregate_replay_health_contrasts(
        case,
        _contrast_rows(case, zero_recoverable_loss=True),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    recovery = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="frame-oracle-recoverable-loss-fraction",
        window="event",
    )
    assert recovery.interval.estimate is None
    assert recovery.status == "undefined"
    assert recovery.scene_defined == (False,) * 10
    assert recovery.scene_values == (None,) * 10


def test_recovery_ratio_is_unitless_and_not_clipped_to_a_fraction_range() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    rows = aggregate_replay_health_contrasts(
        case,
        _contrast_rows(case, policy_loss_override=12.0),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    recovery = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="frame-oracle-recoverable-loss-fraction",
        window="event",
    )
    assert recovery.unit == "unitless"
    assert recovery.aggregation == "unclipped-recovery-ratio"
    assert recovery.interval.estimate == pytest.approx(-0.2)
    assert recovery.scene_values == pytest.approx((-0.2,) * 10)


def test_dropout_recovery_support_mismatches_are_explicitly_not_applicable() -> None:
    case = _case("replay-camera-dropout", 0.5)
    rows = aggregate_replay_health_contrasts(
        case,
        _contrast_rows(case),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    assert len(rows) == len(ALL_POLICIES) * len(WINDOWS) * 4
    recovery = tuple(
        row for row in rows if row.metric_id == "frame-oracle-recoverable-loss-fraction"
    )
    assert {(row.method_id, row.window) for row in recovery if row.status == "not-applicable"} == {
        (policy, window) for policy in ALL_POLICIES for window in ("score", "event")
    }
    applicable = tuple(row for row in recovery if row.window == "recovery")
    assert len(applicable) == len(ALL_POLICIES)
    assert all(row.status == "ok" for row in applicable)
    assert all(
        row.interval.estimate is None
        and row.scene_defined == (False,) * 10
        and row.scene_values == (None,) * 10
        for row in recovery
        if row.status == "not-applicable"
    )


def test_common_mode_structural_contrasts_are_explicitly_not_applicable() -> None:
    case = _case("replay-common-mode-x", 4.0)
    rows = aggregate_replay_health_contrasts(
        case,
        _contrast_rows(case),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    assert len(rows) == len(ALL_POLICIES) * len(WINDOWS) * 4
    assert sum(row.status == "ok" for row in rows) == len(ALL_POLICIES) * len(WINDOWS)
    not_applicable = tuple(row for row in rows if row.status == "not-applicable")
    assert len(not_applicable) == len(ALL_POLICIES) * len(WINDOWS) * 3
    assert {row.metric_id for row in not_applicable} == {
        "gap-vs-fault-target-drop",
        "gap-vs-frame-oracle",
        "frame-oracle-recoverable-loss-fraction",
    }
    assert all(row.interval.estimate is None for row in not_applicable)
    assert all(row.scene_values == (None,) * 10 for row in not_applicable)


def test_contrast_applicability_must_match_the_frozen_case_family() -> None:
    case = _case("replay-common-mode-x", 4.0)
    malformed = tuple(
        replace(
            row,
            target_drop_applicable=True,
            policy_target_drop_common_count=row.fixed_policy_common_count,
            policy_on_target_common_loss_sum_m2=row.policy_on_fixed_common_loss_sum_m2,
            target_drop_on_common_loss_sum_m2=row.fixed_on_common_loss_sum_m2,
            target_drop_support_sha256=SUPPORT_DIGEST,
            frame_oracle_applicable=True,
            policy_frame_oracle_common_count=row.fixed_policy_common_count,
            policy_on_oracle_common_loss_sum_m2=row.policy_on_fixed_common_loss_sum_m2,
            frame_oracle_on_common_loss_sum_m2=row.fixed_on_common_loss_sum_m2,
            frame_oracle_support_sha256=SUPPORT_DIGEST,
        )
        for row in _contrast_rows(case)
    )

    with pytest.raises(ValueError, match="applicability disagrees with the frozen case"):
        aggregate_replay_health_contrasts(
            case,
            malformed,
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )


def test_frozen_health_population_grid_has_exactly_14_988_coordinates() -> None:
    metrics, contrasts = _full_population_grid()
    expected_coordinates = expected_replay_health_population_coordinates(PLAN)

    assert len(metrics) == 14_988
    assert len(expected_coordinates) == 14_988
    assert {row.coordinate for row in metrics} == expected_coordinates
    validate_replay_health_population_grid(
        PLAN,
        metrics,
        contrasts=contrasts,
    )

    not_applicable = {
        (row.condition_selector, row.method_id, row.metric_id, row.window)
        for row in metrics
        if row.status == "not-applicable"
    }
    expected_not_applicable = {
        (case.selector, policy, metric_id, window)
        for case in PLAN.health_cases
        if case.family == "common-mode-position-bias"
        for policy in ALL_POLICIES
        for metric_id in (
            "gap-vs-fault-target-drop",
            "gap-vs-frame-oracle",
            "frame-oracle-recoverable-loss-fraction",
        )
        for window in WINDOWS
    } | {
        (
            case.selector,
            policy,
            "frame-oracle-recoverable-loss-fraction",
            window,
        )
        for case in PLAN.health_cases
        if case.family == "dropout"
        for policy in ALL_POLICIES
        for window in ("score", "event")
    }
    assert len(not_applicable) == 240
    assert not_applicable == expected_not_applicable


@pytest.mark.parametrize(
    "changes",
    (
        {"replay_experiment_identity_sha256": "f" * 64},
        {"condition_selector": "replay-camera-output-y-bias:not-frozen"},
        {"method_id": "not-a-method"},
        {"metric_id": "not-a-metric"},
        {"window": "not-a-window"},
        {"unit": "count"},
        {"aggregation": "pooled-valid-loss"},
    ),
)
def test_authority_rejects_wrong_but_unique_metric_coordinates(
    changes: dict[str, object],
) -> None:
    metrics, contrasts = _full_population_grid()
    base_index = next(
        index
        for index, row in enumerate(metrics)
        if row.condition_selector == "replay-camera-output-y-bias:+3"
        and row.method_id == "camera-only"
        and row.metric_id == "matched-center-mse"
        and row.window == "score"
    )
    malformed = list(metrics)
    malformed[base_index] = replace(metrics[base_index], **changes)

    with pytest.raises(ValueError, match="authoritative 14,988-row grid"):
        validate_replay_health_population_grid(
            PLAN,
            malformed,
            contrasts=contrasts,
        )


def test_authority_rejects_wrong_raw_contrast_identity_or_completeness() -> None:
    metrics, contrasts = _full_population_grid()
    malformed = list(contrasts)
    malformed[0] = replace(
        malformed[0],
        replay_experiment_identity_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="authoritative 6,450-row support grid"):
        validate_replay_health_population_grid(
            PLAN,
            metrics,
            contrasts=malformed,
        )
    with pytest.raises(ValueError, match="authoritative 6,450-row support grid"):
        validate_replay_health_population_grid(
            PLAN,
            metrics,
            contrasts=contrasts[:-1],
        )


def test_authority_reconciles_structural_and_support_derived_status() -> None:
    metrics, contrasts = _full_population_grid()

    common_index = next(
        index
        for index, row in enumerate(metrics)
        if row.condition_selector == "replay-common-mode-x:+4"
        and row.method_id == "combined-health-gate"
        and row.metric_id == "gap-vs-frame-oracle"
        and row.window == "event"
    )
    assert metrics[common_index].status == "not-applicable"
    malformed_common = list(metrics)
    malformed_common[common_index] = replace(
        metrics[common_index],
        status="undefined",
    )
    with pytest.raises(ValueError, match="applicability status"):
        validate_replay_health_population_grid(
            PLAN,
            malformed_common,
            contrasts=contrasts,
        )

    target_index = next(
        index
        for index, row in enumerate(metrics)
        if row.condition_selector == "replay-camera-output-y-bias:+3"
        and row.method_id == "combined-health-gate"
        and row.metric_id == "gap-vs-fault-target-drop"
        and row.window == "event"
    )
    assert metrics[target_index].status != "not-applicable"
    malformed_target = list(metrics)
    malformed_target[target_index] = _as_not_applicable(metrics[target_index])
    with pytest.raises(ValueError, match="applicability status"):
        validate_replay_health_population_grid(
            PLAN,
            malformed_target,
            contrasts=contrasts,
        )

    dropout_index = next(
        index
        for index, row in enumerate(metrics)
        if row.condition_selector == "replay-camera-dropout:0.5"
        and row.method_id == "combined-health-gate"
        and row.metric_id == "frame-oracle-recoverable-loss-fraction"
        and row.window == "event"
    )
    assert metrics[dropout_index].status == "not-applicable"
    malformed_dropout = list(metrics)
    malformed_dropout[dropout_index] = replace(
        metrics[dropout_index],
        status="undefined",
    )
    with pytest.raises(ValueError, match="applicability status"):
        validate_replay_health_population_grid(
            PLAN,
            malformed_dropout,
            contrasts=contrasts,
        )


def test_authority_recomputes_recovery_applicability_from_all_ten_scenes() -> None:
    metrics, contrasts = _full_population_grid()
    malformed = list(contrasts)
    contrast_index = next(
        index
        for index, row in enumerate(contrasts)
        if row.condition_selector == "replay-camera-output-y-bias:+3"
        and row.policy == "combined-health-gate"
        and row.window == "event"
        and row.sequence_id == SCENE_IDS[0]
    )
    assert contrasts[contrast_index].identical_support_recovery_applicable
    malformed[contrast_index] = replace(
        contrasts[contrast_index],
        frame_oracle_support_sha256="d" * 64,
    )
    assert not malformed[contrast_index].identical_support_recovery_applicable

    with pytest.raises(ValueError, match="applicability status"):
        validate_replay_health_population_grid(
            PLAN,
            metrics,
            contrasts=malformed,
        )


def test_authority_rejects_comparator_applicability_outside_frozen_structure() -> None:
    metrics, contrasts = _full_population_grid()
    malformed = list(contrasts)
    contrast_index = next(
        index
        for index, row in enumerate(contrasts)
        if row.condition_selector == "replay-camera-output-y-bias:+3"
    )
    malformed[contrast_index] = replace(
        malformed[contrast_index],
        target_drop_applicable=False,
        policy_target_drop_common_count=None,
        policy_on_target_common_loss_sum_m2=None,
        target_drop_on_common_loss_sum_m2=None,
        target_drop_support_sha256=None,
    )

    with pytest.raises(ValueError, match="comparator applicability"):
        validate_replay_health_population_grid(
            PLAN,
            metrics,
            contrasts=malformed,
        )


def test_event_steps_seconds_observed_fractions_and_pooled_occupancy() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    rows = aggregate_replay_health_events(
        case,
        _event_rows(case, detected_scene_count=8),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    method = "combined-health-gate"

    assert _metric(
        rows,
        method_id=method,
        metric_id="detection-fraction",
        window="event",
    ).interval.estimate == pytest.approx(0.8)
    assert _metric(
        rows,
        method_id=method,
        metric_id="detection-latency-steps-observed-fraction",
        window="event",
    ).interval.estimate == pytest.approx(0.8)
    detection_steps = _metric(
        rows,
        method_id=method,
        metric_id="detection-latency-steps",
        window="event",
    )
    assert detection_steps.unit == "observation-step"
    assert detection_steps.aggregation == "conditional-observed-scene-mean"
    assert detection_steps.interval.estimate == pytest.approx(1.0)
    assert _metric(
        rows,
        method_id=method,
        metric_id="detection-elapsed-reference-time",
        window="event",
    ).interval.estimate == pytest.approx(0.85)
    assert _metric(
        rows,
        method_id=method,
        metric_id="recovery-denominator-fraction",
        window="event",
    ).interval.estimate == pytest.approx(0.8)
    assert _metric(
        rows,
        method_id=method,
        metric_id="recovery-fraction",
        window="recovery",
    ).interval.estimate == pytest.approx(1.0)
    assert _metric(
        rows,
        method_id=method,
        metric_id="recovery-latency-steps-observed-fraction",
        window="recovery",
    ).interval.estimate == pytest.approx(0.8)
    assert _metric(
        rows,
        method_id=method,
        metric_id="recovery-latency-steps",
        window="recovery",
    ).interval.estimate == pytest.approx(2.0)
    assert _metric(
        rows,
        method_id=method,
        metric_id="recovery-elapsed-reference-time",
        window="recovery",
    ).interval.estimate == pytest.approx(1.7)

    state_values = tuple(
        _metric(
            rows,
            method_id=method,
            metric_id=f"state-{state}-occupancy",
            window="event",
        ).interval.estimate
        for state in ("healthy", "camera-fault", "lidar-fault", "ambiguous")
    )
    action_values = tuple(
        _metric(
            rows,
            method_id=method,
            metric_id=f"action-{action}-occupancy",
            window="event",
        ).interval.estimate
        for action in ("camera", "lidar", "fixed", "undefined")
    )
    assert state_values == pytest.approx((0.3, 0.7, 0.0, 0.0))
    assert action_values == pytest.approx((0.0, 0.7, 0.3, 0.0))
    assert sum(value for value in state_values if value is not None) == pytest.approx(1.0)
    assert sum(value for value in action_values if value is not None) == pytest.approx(1.0)


def test_dropout_first_missing_and_signed_response_use_recorded_time() -> None:
    case = _case("replay-camera-dropout", 0.5)
    rows = aggregate_replay_health_events(
        case,
        _event_rows(case, detected_scene_count=8),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    method = "combined-health-gate"

    assert _metric(
        rows,
        method_id="none",
        metric_id="realized-dropout-fraction",
        window="event",
    ).interval.estimate == pytest.approx(0.8)
    first_missing = _metric(
        rows,
        method_id="none",
        metric_id="first-missing-step",
        window="event",
    )
    assert first_missing.unit == "observation-step"
    assert first_missing.interval.estimate == pytest.approx(2.0)
    assert _metric(
        rows,
        method_id="none",
        metric_id="first-missing-elapsed-reference-time",
        window="event",
    ).interval.estimate == pytest.approx(1.7)
    assert _metric(
        rows,
        method_id=method,
        metric_id="detection-among-realized-dropout-fraction",
        window="event",
    ).interval.estimate == pytest.approx(1.0)
    response_steps = _metric(
        rows,
        method_id=method,
        metric_id="detection-minus-first-missing-steps",
        window="event",
    )
    assert response_steps.interval.estimate == pytest.approx(-1.0)
    assert response_steps.scene_values == (-1.0,) * 8 + (None,) * 2
    assert _metric(
        rows,
        method_id=method,
        metric_id="detection-minus-first-missing-elapsed-reference-time",
        window="event",
    ).interval.estimate == pytest.approx(-0.85)


def test_unobserved_event_latencies_and_recovery_denominator_stay_zero() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    rows = aggregate_replay_health_events(
        case,
        _event_rows(case, detected_scene_count=0),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    recovery = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="recovery-fraction",
        window="recovery",
    )
    latency = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="detection-latency-steps",
        window="event",
    )
    observed = _metric(
        rows,
        method_id="combined-health-gate",
        metric_id="detection-latency-steps-observed-fraction",
        window="event",
    )
    assert recovery.interval.estimate is None
    assert recovery.scene_values == (None,) * 10
    assert latency.interval.estimate is None
    assert latency.scene_values == (None,) * 10
    assert observed.interval.estimate == 0.0
    assert observed.scene_values == (0.0,) * 10


def test_every_population_input_requires_the_exact_ten_scene_matrix() -> None:
    case = _case("replay-camera-output-y-bias", 3.0)
    results = _result_rows(case)
    contrasts = _contrast_rows(case)
    events = _event_rows(case, detected_scene_count=8)

    with pytest.raises(ValueError, match="result matrix is incomplete"):
        aggregate_replay_health_results(
            case,
            results[:-1],
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )
    with pytest.raises(ValueError, match="result matrix is incomplete"):
        aggregate_replay_health_results(
            case,
            tuple(row for row in results if row.method == "fixed-fusion"),
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )
    with pytest.raises(ValueError, match="contrast matrix is incomplete"):
        aggregate_replay_health_contrasts(
            case,
            contrasts[:-1],
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )
    with pytest.raises(ValueError, match="event matrix is incomplete"):
        aggregate_replay_health_events(
            case,
            events[:-1],
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )


def test_common_mode_requires_exactly_the_eight_applicable_methods() -> None:
    case = _case("replay-common-mode-x", 4.0)
    aggregates = aggregate_replay_health_results(
        case,
        _result_rows(case),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    assert len(aggregates) == len(COMMON_MODE_METHODS) * len(WINDOWS) * 5

    with pytest.raises(ValueError, match="result matrix is incomplete"):
        aggregate_replay_health_results(
            case,
            _result_rows(case, methods=HEALTH_METHODS),
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )
