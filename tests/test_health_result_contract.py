from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.health_result_v1 import (
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
    HealthSequenceContrastV1,
    HealthSequenceEventV1,
    HealthSequenceLossV1,
    HealthThresholdCandidateV1,
    HealthValidationCheckV1,
    HealthValidationV1,
)

_DIGEST = "a" * 64


def _candidate(**updates: Any) -> HealthThresholdCandidateV1:
    payload: dict[str, Any] = {
        "schema": "ffb.health-threshold-candidate/v1",
        "candidate_index": 0,
        "self_threshold": 0.95,
        "cross_threshold": 0.95,
        "mean_clean_regression_m2": 0.001,
        "upper_95pct_clean_regression_m2": 0.004,
        "false_alert_episode_starts_per_sequence": 0.05,
        "clean_coverage": 1.0,
        "fixed_clean_coverage": 1.0,
        "feasible": True,
        "validation_regret_m2": 0.2,
    }
    payload.update(updates)
    return HealthThresholdCandidateV1.model_validate(payload)


def _event(**updates: Any) -> HealthSequenceEventV1:
    payload: dict[str, Any] = {
        "schema": "ffb.health-sequence-event/v1",
        "sequence_id": "sequence:0",
        "condition_id": "condition:0",
        "policy": "combined-health-gate",
        "detected": True,
        "detection_latency_frames": 1,
        "first_latch_label": "camera-fault",
        "outcome": "correct",
        "correctly_attributed": True,
        "attribution_latency_frames": 1,
        "realized_dropout": None,
        "first_missing_frame_minus_event_start": None,
        "detection_minus_first_missing_frames": None,
        "latch_episode_count": 1,
        "false_alert_episode_count": 0,
        "early_clear": False,
        "final_active_state": "camera-fault",
        "active_frame_count": 24,
        "active_healthy_frames": 1,
        "active_camera_fault_frames": 23,
        "active_lidar_fault_frames": 0,
        "active_ambiguous_frames": 0,
        "active_camera_action_frames": 0,
        "active_lidar_action_frames": 23,
        "active_fixed_action_frames": 1,
        "active_undefined_action_frames": 0,
        "recovery_eligible": True,
        "recovered": True,
        "recovery_latency_frames": 2,
    }
    payload.update(updates)
    return HealthSequenceEventV1.model_validate(payload)


def _aggregate(**updates: Any) -> HealthAggregateMetricV1:
    payload: dict[str, Any] = {
        "schema": "ffb.health-aggregate/v1",
        "condition_id": "condition:0",
        "method": "combined-health-gate",
        "metric_name": "policy-gain",
        "window": "event",
        "unit": "m^2",
        "status": "ok",
        "estimate": 0.2,
        "interval_lower": 0.1,
        "interval_upper": 0.3,
        "sequence_count": 200,
        "bootstrap_replicates": 2000,
        "defined_bootstrap_replicates": 2000,
    }
    payload.update(updates)
    return HealthAggregateMetricV1.model_validate(payload)


def _contrast(**updates: Any) -> HealthSequenceContrastV1:
    payload: dict[str, Any] = {
        "schema": "ffb.health-sequence-contrast/v1",
        "sequence_id": "sequence:0",
        "condition_id": "condition:0",
        "policy": "combined-health-gate",
        "window": "event",
        "fixed_support_sha256": _DIGEST,
        "policy_support_sha256": _DIGEST,
        "fixed_policy_common_count": 5,
        "fixed_on_common_loss_sum_m2": 2.0,
        "policy_on_fixed_common_loss_sum_m2": 1.0,
        "target_drop_applicable": True,
        "policy_target_drop_common_count": 5,
        "policy_on_target_common_loss_sum_m2": 1.0,
        "target_drop_on_common_loss_sum_m2": 0.5,
        "frame_oracle_applicable": True,
        "policy_frame_oracle_common_count": 5,
        "policy_on_oracle_common_loss_sum_m2": 1.0,
        "frame_oracle_on_common_loss_sum_m2": 0.25,
        "frame_oracle_support_sha256": _DIGEST,
    }
    payload.update(updates)
    return HealthSequenceContrastV1.model_validate(payload)


def test_threshold_candidate_grid_and_feasibility_are_literal() -> None:
    assert _candidate().feasible
    assert (
        _candidate(
            candidate_index=35,
            self_threshold=1.0,
            cross_threshold=1.0,
        ).candidate_index
        == 35
    )
    assert not _candidate(
        upper_95pct_clean_regression_m2=0.006,
        feasible=False,
    ).feasible

    for updates in (
        {"candidate_index": 1, "cross_threshold": 0.95},
        {"mean_clean_regression_m2": 0.003, "feasible": True},
        {"clean_coverage": 0.9, "feasible": True},
    ):
        with pytest.raises(ValidationError):
            _candidate(**updates)


def test_fit_summary_binds_selected_candidate_coordinate() -> None:
    payload = {
        "schema": "ffb.health-fit-summary/v1",
        "intent_sha256": _DIGEST,
        "main_profile_sha256": _DIGEST,
        "edge_profile_sha256": _DIGEST,
        "train_sequence_count": 200,
        "validation_sequence_count": 200,
        "ecdf_channel_count": 8,
        "ecdf_values_per_channel": 9200,
        "candidate_count": 36,
        "selected_candidate_index": 7,
        "selected_self_threshold": 0.975,
        "selected_cross_threshold": 0.975,
        "selection_status": "selected",
    }
    assert HealthFitSummaryV1.model_validate(payload).selected_candidate_index == 7
    payload["selected_cross_threshold"] = 0.99
    with pytest.raises(ValidationError):
        HealthFitSummaryV1.model_validate(payload)


def test_sequence_loss_requires_valid_subset_and_zero_for_no_support() -> None:
    payload = {
        "schema": "ffb.health-sequence-loss/v1",
        "sequence_id": "sequence:0",
        "condition_id": "condition:0",
        "method": "fixed-fusion",
        "window": "score",
        "loss_sum_m2": 1.0,
        "valid_object_frame_count": 5,
        "eligible_object_frame_count": 6,
    }
    assert HealthSequenceLossV1.model_validate(payload).valid_object_frame_count == 5
    for update in (
        {"valid_object_frame_count": 7},
        {"valid_object_frame_count": 0},
    ):
        with pytest.raises(ValidationError):
            HealthSequenceLossV1.model_validate(payload | update)


def test_sequence_contrast_requires_complete_applicability_and_hashed_recovery_support() -> None:
    record = _contrast()
    assert record.identical_support_recovery_applicable
    assert not _contrast(policy_support_sha256="b" * 64).identical_support_recovery_applicable

    mutations = (
        {"target_drop_applicable": False},
        {
            "fixed_policy_common_count": 0,
            "fixed_on_common_loss_sum_m2": 1.0,
        },
        {
            "policy_frame_oracle_common_count": 4,
        },
        {
            "policy_on_oracle_common_loss_sum_m2": 2.0,
        },
    )
    for mutation in mutations:
        with pytest.raises(ValidationError):
            _contrast(**mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda: {"detected": False},
        lambda: {"correctly_attributed": False},
        lambda: {"recovery_eligible": False},
        lambda: {"recovered": False},
        lambda: {"detected": False, "detection_latency_frames": None, "outcome": "correct"},
        lambda: {
            "realized_dropout": False,
            "first_missing_frame_minus_event_start": 0,
        },
        lambda: {
            "realized_dropout": True,
            "first_missing_frame_minus_event_start": None,
        },
        lambda: {
            "realized_dropout": True,
            "first_missing_frame_minus_event_start": 0,
            "detection_minus_first_missing_frames": None,
        },
        lambda: {"first_latch_label": "healthy"},
        lambda: {"attribution_latency_frames": 0},
        lambda: {"final_active_state": "lidar-fault"},
        lambda: {"latch_episode_count": 25},
        lambda: {"false_alert_episode_count": 25},
        lambda: {
            "final_active_state": "healthy",
            "recovery_eligible": False,
            "recovered": True,
        },
        lambda: {
            "realized_dropout": True,
            "first_missing_frame_minus_event_start": 5,
            "detection_minus_first_missing_frames": 1,
        },
        lambda: {"latch_episode_count": 0},
        lambda: {"final_active_state": "healthy"},
        lambda: {"detection_latency_frames": 24},
        lambda: {"attribution_latency_frames": 24},
        lambda: {"recovery_latency_frames": 24},
        lambda: {"active_frame_count": 25, "active_camera_fault_frames": 24},
        lambda: {"active_healthy_frames": 2},
        lambda: {"active_fixed_action_frames": 2},
    ],
)
def test_event_censoring_rejects_inconsistent_records(
    mutation: Callable[[], dict[str, object]],
) -> None:
    with pytest.raises(ValidationError):
        _event(**mutation())


def test_event_accepts_missed_censored_and_unrecovered_records() -> None:
    record = _event(
        detected=False,
        detection_latency_frames=None,
        first_latch_label=None,
        outcome="missed",
        correctly_attributed=False,
        attribution_latency_frames=None,
        final_active_state="healthy",
        recovery_eligible=False,
        recovered=False,
        recovery_latency_frames=None,
    )
    assert record.outcome == "missed"


def test_event_requires_a_distinct_later_episode_for_late_correct_attribution() -> None:
    record = _event(
        first_latch_label="ambiguous",
        outcome="ambiguous",
        attribution_latency_frames=3,
        latch_episode_count=2,
    )
    assert record.attribution_latency_frames == 3

    with pytest.raises(ValidationError, match="follow first detection"):
        _event(
            first_latch_label="ambiguous",
            outcome="ambiguous",
            attribution_latency_frames=1,
            latch_episode_count=2,
        )
    with pytest.raises(ValidationError, match="later latch episode"):
        _event(
            first_latch_label="ambiguous",
            outcome="ambiguous",
            attribution_latency_frames=3,
            latch_episode_count=1,
        )


def test_aggregate_requires_complete_ordered_interval_only_when_ok() -> None:
    assert _aggregate().estimate == 0.2
    with pytest.raises(ValidationError):
        _aggregate(interval_lower=0.4)
    with pytest.raises(ValidationError):
        _aggregate(status="undefined")
    undefined = _aggregate(
        status="undefined",
        estimate=None,
        interval_lower=None,
        interval_upper=None,
        defined_bootstrap_replicates=0,
    )
    assert undefined.estimate is None


def test_validation_is_a_unique_conjunction() -> None:
    check = HealthValidationCheckV1(
        check_id="identity",
        passed=True,
        observed=True,
        expected=True,
    )
    evidence = HealthValidationV1(
        schema="ffb.health-validation/v1",
        intent_sha256=_DIGEST,
        checks=(check,),
        all_checks_passed=True,
    )
    assert evidence.all_checks_passed
    with pytest.raises(ValidationError):
        HealthValidationV1(
            schema="ffb.health-validation/v1",
            intent_sha256=_DIGEST,
            checks=(check, check),
            all_checks_passed=True,
        )
    with pytest.raises(ValidationError):
        HealthValidationV1(
            schema="ffb.health-validation/v1",
            intent_sha256=_DIGEST,
            checks=(
                HealthValidationCheckV1(
                    check_id="failed",
                    passed=False,
                    observed=False,
                    expected=True,
                ),
            ),
            all_checks_passed=True,
        )
