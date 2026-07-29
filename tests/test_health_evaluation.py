from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.contracts.procedural_profile_v1 import load_procedural_profile
from fusion_fault_bench.experiments.health import generate_health_observations
from fusion_fault_bench.health import HealthCalibration, HealthThresholds
from fusion_fault_bench.health_evaluation import (
    HealthPolicyTrace,
    evaluate_health_sequence,
    fixed_fusion_values,
    health_support_mask_sha256,
    paired_common_support_sequence_contrast,
    sequence_contrast_records,
    sequence_event_record,
    sequence_loss_records,
)
from fusion_fault_bench.health_fit import (
    compute_health_feature_trace,
    rescore_health_feature_trace,
)
from fusion_fault_bench.scenarios.health import (
    HealthFaultSpec,
    generate_health_base_sequences,
)


def _base():
    profile = load_procedural_profile(Path("examples/profiles/constant-velocity-front-roi-v1.json"))
    return generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]


def _calibration() -> HealthCalibration:
    values = np.asarray([0.0, 1e12], dtype=np.float64)
    return HealthCalibration(
        camera_self_mean=values,
        camera_self_maximum=values,
        lidar_self_mean=values,
        lidar_self_maximum=values,
        camera_from_lidar_cross_mean=values,
        camera_from_lidar_cross_maximum=values,
        lidar_from_camera_cross_mean=values,
        lidar_from_camera_cross_maximum=values,
    )


def _evaluate(fault: HealthFaultSpec):
    observations = generate_health_observations(_base(), fault=fault)
    unscored = compute_health_feature_trace(observations.health_frame_inputs())
    evidence = rescore_health_feature_trace(unscored, _calibration()).frames
    evaluation = evaluate_health_sequence(
        observations,
        condition_id="test-condition",
        fault=fault,
        evidence=evidence,
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
    )
    return observations, evaluation


def test_identity_evaluates_all_methods_and_oracle_dominates() -> None:
    fault = HealthFaultSpec(
        family="identity",
        target="none",
        axis="none",
        unit="identity",
        value=0.0,
    )
    observations, evaluation = _evaluate(fault)
    assert len(evaluation.losses) == 10
    records = sequence_loss_records(
        evaluation,
        observations=observations,
        fault=fault,
    )
    assert len(records) == 30
    contrasts = sequence_contrast_records(evaluation, fault=fault)
    assert len(contrasts) == 15
    assert all(record.target_drop_applicable for record in contrasts)
    assert all(record.frame_oracle_applicable for record in contrasts)
    assert all(record.identical_support_recovery_applicable for record in contrasts)
    oracle = evaluation.loss("frame-action-performance-oracle")
    for trace in evaluation.losses:
        if np.array_equal(trace.valid_mask, oracle.valid_mask):
            assert trace.loss_m2.sum() + 1e-12 >= oracle.loss_m2.sum()
    assert np.array_equal(
        evaluation.loss("fixed-fusion").loss_m2,
        evaluation.loss("fault-target-drop-policy").loss_m2,
    )
    for policy in evaluation.policy_traces:
        event = sequence_event_record(
            policy,
            observations=observations,
            condition_id="test-condition",
            fault=fault,
        )
        assert not event.detected
        assert event.outcome == "missed"
        assert event.active_healthy_frames == event.active_frame_count


def test_full_dropout_is_coverage_first_and_direct_detection_recovers() -> None:
    fault = HealthFaultSpec(
        family="dropout",
        target="camera",
        axis="availability",
        unit="probability",
        value=1.0,
    )
    observations, evaluation = _evaluate(fault)
    fixed = evaluation.loss("fixed-fusion")
    combined = evaluation.loss("combined-health-gate")
    event_window = (12, 36)
    contrast, common_count = paired_common_support_sequence_contrast(
        fixed,
        combined,
        frame_window=event_window,
    )
    assert contrast is None
    assert common_count == 0
    contrast_row = next(
        row
        for row in sequence_contrast_records(evaluation, fault=fault)
        if row.policy == "combined-health-gate" and row.window == "event"
    )
    assert contrast_row.fixed_policy_common_count == 0
    assert not contrast_row.identical_support_recovery_applicable
    assert not np.any(fixed.valid_mask[12:36])
    assert np.all(combined.valid_mask[12:36] == observations.eligibility_mask[12:36])
    losses = {
        (row.method, row.window): row
        for row in sequence_loss_records(
            evaluation,
            observations=observations,
            fault=fault,
        )
    }
    assert tuple(
        losses[("fixed-fusion", window)].valid_object_frame_count
        for window in ("score", "event", "recovery")
    ) == (132, 0, 72)
    assert tuple(
        losses[("camera-only", window)].valid_object_frame_count
        for window in ("score", "event", "recovery")
    ) == (132, 0, 72)
    for policy in (
        "self-nis-gate",
        "cross-nis-gate",
        "direct-telemetry-gate",
        "combined-health-gate",
    ):
        assert losses[(policy, "event")].valid_object_frame_count == 144
        event = sequence_event_record(
            evaluation.policy_trace(policy),
            observations=observations,
            condition_id="test-condition",
            fault=fault,
        )
        assert event.realized_dropout is True
        assert event.first_missing_frame_minus_event_start == 0
        assert event.active_lidar_action_frames == 24
        assert event.active_camera_action_frames == 0
        assert event.active_fixed_action_frames == 0
        assert event.active_undefined_action_frames == 0

    direct = sequence_event_record(
        evaluation.policy_trace("direct-telemetry-gate"),
        observations=observations,
        condition_id="test-condition",
        fault=fault,
    )
    assert direct.realized_dropout
    assert direct.detected
    assert direct.outcome == "correct"
    assert direct.detection_latency_frames == 1
    assert direct.first_missing_frame_minus_event_start == 0
    assert direct.detection_minus_first_missing_frames == 1
    assert direct.recovery_eligible
    assert direct.recovered
    assert direct.recovery_latency_frames == 2


def test_partial_dropout_with_no_realization_forces_missed_regime_event() -> None:
    base = _base()
    active_uniform_minimum = float(base.dropout_uniform_by_frame[12:36].min())
    probability = max(np.nextafter(active_uniform_minimum, 0.0), 1e-12)
    if np.any(base.dropout_uniform_by_frame[12:36] < probability):
        pytest.skip("draw contains an exact zero, so no positive no-realization probability exists")
    fault = HealthFaultSpec(
        family="dropout",
        target="camera",
        axis="availability",
        unit="probability",
        value=probability,
    )
    observations = generate_health_observations(base, fault=fault)
    evidence = rescore_health_feature_trace(
        compute_health_feature_trace(observations.health_frame_inputs()),
        _calibration(),
    ).frames
    evaluation = evaluate_health_sequence(
        observations,
        condition_id="test-condition",
        fault=fault,
        evidence=evidence,
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
    )
    event = sequence_event_record(
        evaluation.policy_trace("direct-telemetry-gate"),
        observations=observations,
        condition_id="test-condition",
        fault=fault,
    )
    assert event.realized_dropout is False
    assert not event.detected
    assert event.outcome == "missed"
    assert event.first_missing_frame_minus_event_start is None
    assert event.detection_minus_first_missing_frames is None


def test_realized_dropout_first_missing_latency_does_not_require_detection() -> None:
    fault = HealthFaultSpec(
        family="dropout",
        target="camera",
        axis="availability",
        unit="probability",
        value=1.0,
    )
    observations = generate_health_observations(_base(), fault=fault)
    healthy_trace = HealthPolicyTrace(
        policy="combined-health-gate",
        raw_labels=("healthy",) * observations.frame_count,
        evidence_statuses=("update-eligible",) * observations.frame_count,
        latched_labels=("healthy",) * observations.frame_count,
        actions=("fixed-fusion",) * observations.frame_count,
    )
    event = sequence_event_record(
        healthy_trace,
        observations=observations,
        condition_id="test-condition",
        fault=fault,
    )
    assert event.realized_dropout is True
    assert not event.detected
    assert event.first_missing_frame_minus_event_start == 0
    assert event.detection_minus_first_missing_frames is None


def test_clean_predictor_mismatch_counts_false_alert_episodes() -> None:
    observations = generate_health_observations(
        _base(),
        fault=HealthFaultSpec(
            family="identity",
            target="none",
            axis="none",
            unit="identity",
            value=0.0,
        ),
    )
    labels = ("healthy",) * 13 + ("camera-fault",) * 3 + ("healthy",) * 32
    trace = HealthPolicyTrace(
        policy="combined-health-gate",
        raw_labels=labels,
        evidence_statuses=("update-eligible",) * observations.frame_count,
        latched_labels=labels,
        actions=("fixed-fusion",) * observations.frame_count,
    )
    event = sequence_event_record(
        trace,
        observations=observations,
        condition_id="motion",
        fault=HealthFaultSpec(
            family="clean-predictor-mismatch",
            target="none",
            axis="motion",
            unit="m/s^2",
            value=8.0,
        ),
    )
    assert event.false_alert_episode_count == 1


def test_common_mode_omits_hindsight_methods() -> None:
    edge_profile = load_procedural_profile(
        Path("examples/profiles/constant-velocity-fov-edge-v1.json")
    )
    base = generate_health_base_sequences(
        edge_profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    fault = HealthFaultSpec(
        family="common-mode-position-bias",
        target="both",
        axis="x",
        unit="m",
        value=1.0,
    )
    observations = generate_health_observations(base, fault=fault)
    evidence = rescore_health_feature_trace(
        compute_health_feature_trace(observations.health_frame_inputs()),
        _calibration(),
    ).frames
    evaluation = evaluate_health_sequence(
        observations,
        condition_id="common",
        fault=fault,
        evidence=evidence,
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
    )
    methods = {trace.method for trace in evaluation.losses}
    assert "fault-target-drop-policy" not in methods
    assert "frame-action-performance-oracle" not in methods
    contrasts = sequence_contrast_records(evaluation, fault=fault)
    assert len(contrasts) == 15
    assert all(not row.target_drop_applicable for row in contrasts)
    assert all(not row.frame_oracle_applicable for row in contrasts)


def test_cold_start_loss_windows_are_schedule_relative() -> None:
    fault = HealthFaultSpec(
        family="calibration-translation",
        target="camera",
        axis="x",
        unit="m",
        value=3.0,
        schedule="cold_start",
    )
    observations, evaluation = _evaluate(fault)
    fixed_records = [
        record
        for record in sequence_loss_records(
            evaluation,
            observations=observations,
            fault=fault,
        )
        if record.method == "fixed-fusion"
    ]
    assert [record.window for record in fixed_records] == ["score", "event", "recovery"]
    assert [record.eligible_object_frame_count for record in fixed_records] == [
        48 * 6,
        24 * 6,
        24 * 6,
    ]


def test_fixed_fusion_uses_reported_information_and_rejects_bad_support_window() -> None:
    fault = HealthFaultSpec(
        family="identity",
        target="none",
        axis="none",
        unit="identity",
        value=0.0,
    )
    observations, evaluation = _evaluate(fault)
    fused, available = fixed_fusion_values(observations)
    expected = (observations.camera_value_xy_m * 0.09 + observations.lidar_value_xy_m * 1.0) / 1.09
    assert np.allclose(fused[available], expected[available], rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError):
        paired_common_support_sequence_contrast(
            evaluation.loss("camera-only"),
            evaluation.loss("lidar-only"),
            frame_window=(10, 60),
        )
    digest = health_support_mask_sha256(
        evaluation.loss("fixed-fusion").valid_mask,
        frame_window=(12, 36),
    )
    assert len(digest) == 64
    assert digest != health_support_mask_sha256(
        evaluation.loss("fixed-fusion").valid_mask,
        frame_window=(13, 36),
    )


def test_event_reduction_uses_first_latch_but_later_correct_attribution() -> None:
    fault = HealthFaultSpec(
        family="additive-position-bias",
        target="camera",
        axis="y",
        unit="m",
        value=3.0,
    )
    observations, _ = _evaluate(fault)
    labels = ["healthy"] * 48
    labels[13:16] = ["ambiguous"] * 3
    labels[16:19] = ["healthy"] * 3
    labels[20:36] = ["camera-fault"] * 16
    labels[36:39] = ["camera-fault"] * 3
    actions = tuple("lidar-only" if label == "camera-fault" else "fixed-fusion" for label in labels)
    trace = HealthPolicyTrace(
        policy="combined-health-gate",
        raw_labels=tuple(labels),
        evidence_statuses=("update-eligible",) * 48,
        latched_labels=tuple(labels),
        actions=actions,
    )
    event = sequence_event_record(
        trace,
        observations=observations,
        condition_id="test-condition",
        fault=fault,
    )
    assert event.outcome == "ambiguous"
    assert event.first_latch_label == "ambiguous"
    assert event.correctly_attributed
    assert event.attribution_latency_frames == 8
    assert event.latch_episode_count == 2
    assert event.early_clear
