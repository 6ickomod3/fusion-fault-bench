from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.health import HealthThresholds
from fusion_fault_bench.replay_evaluation import evaluate_replay_health_sequence
from fusion_fault_bench.replay_experiments import (
    ReplayEstimateFrame,
    ReplayEstimateSequence,
    ReplayFaultCondition,
    ReplayObjectEstimate,
)
from fusion_fault_bench.replay_fit import (
    ReplayFitError,
    load_frozen_replay_health_fit,
)
from fusion_fault_bench.replay_geometry import ProjectedEstimate
from fusion_fault_bench.replay_inference import (
    H5_B_SELECTORS,
    equal_scene_contrast_interval,
    pooled_availability_interval,
    pooled_conditional_loss_interval,
    replay_sequence_contrast_values,
)
from fusion_fault_bench.replay_plan import (
    ReplayHealthCaseSpec,
    load_replay_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = load_replay_plan(source_root=ROOT)
FIT = load_frozen_replay_health_fit(ROOT)


def _projected(
    point: tuple[float, float],
    *,
    variance: float,
) -> ProjectedEstimate:
    return ProjectedEstimate(
        point_m=np.asarray(point, dtype=np.float64),
        jacobian=np.eye(2, dtype=np.float64),
        reported_covariance_m2=np.eye(2, dtype=np.float64) * variance,
    )


def _case(experiment_id: str, value: float) -> ReplayHealthCaseSpec:
    return next(
        case
        for case in PLAN.health_cases
        if case.identity.experiment_id == experiment_id and case.value == value
    )


def _sequence(
    *,
    condition: ReplayFaultCondition | None = None,
    local_camera_bias: float = 100.0,
    monitoring_camera_bias: float = 100.0,
    common_mode: bool = False,
    empty_frames: frozenset[int] = frozenset(),
    frame_count: int = 16,
    reference_times_s: tuple[float, ...] | None = None,
    camera_unavailable_frames: frozenset[int] = frozenset(),
    lidar_unavailable_frames: frozenset[int] = frozenset(),
) -> ReplayEstimateSequence:
    times = (
        tuple(float(frame_index) for frame_index in range(frame_count))
        if reference_times_s is None
        else reference_times_s
    )
    if len(times) != frame_count:
        raise ValueError("test reference times must align with frame_count")
    selected = (
        _case("replay-camera-output-y-bias", 3.0).for_frame_count(frame_count)
        if condition is None
        else condition
    )
    camera_variance = 1.0
    lidar_variance = 0.09
    fixed_variance = 1.0 / (1.0 / camera_variance + 1.0 / lidar_variance)
    frames: list[ReplayEstimateFrame] = []
    for frame_index, reference_time in enumerate(times):
        active = selected.fault_is_active(frame_index)
        objects: list[ReplayObjectEstimate] = []
        if frame_index not in empty_frames:
            for object_index in range(2):
                truth = np.asarray(
                    (10.0 + reference_time + object_index, 2.0 * object_index),
                    dtype=np.float64,
                )
                camera_local_bias = (
                    np.asarray(
                        (
                            local_camera_bias if common_mode else 0.0,
                            0.0 if common_mode else local_camera_bias,
                        )
                    )
                    if active
                    else np.zeros(2)
                )
                lidar_local_bias = (
                    np.asarray((local_camera_bias, 0.0)) if active and common_mode else np.zeros(2)
                )
                camera_monitoring_bias = (
                    np.asarray(
                        (
                            monitoring_camera_bias if common_mode else 0.0,
                            0.0 if common_mode else monitoring_camera_bias,
                        )
                    )
                    if active
                    else np.zeros(2)
                )
                lidar_monitoring_bias = (
                    np.asarray((monitoring_camera_bias, 0.0))
                    if active and common_mode
                    else np.zeros(2)
                )
                camera_local = truth + camera_local_bias
                lidar_local = truth + lidar_local_bias
                fixed_local = fixed_variance * (
                    camera_local / camera_variance + lidar_local / lidar_variance
                )
                objects.append(
                    ReplayObjectEstimate(
                        object_id=f"track:{object_index:04d}",
                        truth_current_ego_xy_m=truth,
                        camera_current_ego=_projected(
                            tuple(camera_local),
                            variance=camera_variance,
                        ),
                        lidar_current_ego=_projected(
                            tuple(lidar_local),
                            variance=lidar_variance,
                        ),
                        fixed_current_ego_xy_m=fixed_local,
                        fixed_reported_covariance_m2=np.eye(2) * fixed_variance,
                        camera_monitoring_scene=_projected(
                            tuple(truth + camera_monitoring_bias),
                            variance=camera_variance,
                        ),
                        lidar_monitoring_scene=_projected(
                            tuple(truth + lidar_monitoring_bias),
                            variance=lidar_variance,
                        ),
                        camera_reported_state_time_s=reference_time,
                        lidar_reported_state_time_s=reference_time,
                    )
                )
        frames.append(
            ReplayEstimateFrame(
                frame_index=frame_index,
                reference_time_s=reference_time,
                camera_available=frame_index not in camera_unavailable_frames,
                lidar_available=frame_index not in lidar_unavailable_frames,
                objects=tuple(objects),
            )
        )
    return ReplayEstimateSequence(
        sequence_id="nuscenes:scene-0061",
        condition=selected,
        frames=tuple(frames),
    )


def _case_for_sequence(sequence: ReplayEstimateSequence) -> ReplayHealthCaseSpec:
    return next(
        case
        for case in PLAN.health_cases
        if case.identity.experiment_id == sequence.condition.experiment_id
        and case.value == sequence.condition.value
    )


def _evaluate(sequence: ReplayEstimateSequence):
    return evaluate_replay_health_sequence(
        sequence,
        case=_case_for_sequence(sequence),
        fit=FIT,
    )


def test_health_evaluation_uses_complete_method_window_and_event_matrix() -> None:
    evaluated = _evaluate(_sequence())

    assert len(evaluated.losses) == 10
    assert len(evaluated.results) == 30
    assert len(evaluated.contrasts) == 15
    assert len(evaluated.events) == 4
    combined_event = next(
        row
        for row in evaluated.contrasts
        if row.policy == "combined-health-gate" and row.window == "event"
    )
    assert combined_event.fixed_policy_common_count == 16
    assert (
        combined_event.fixed_on_common_loss_sum_m2
        > combined_event.policy_on_fixed_common_loss_sum_m2
    )
    combined_event_record = next(
        row for row in evaluated.events if row.policy == "combined-health-gate"
    )
    assert combined_event_record.detected
    assert combined_event_record.outcome == "correct"
    assert combined_event_record.detection_latency_steps == 1
    assert combined_event_record.correctly_attributed


def test_health_features_never_use_localization_frame_values() -> None:
    evaluated = _evaluate(
        _sequence(
            local_camera_bias=100.0,
            monitoring_camera_bias=0.0,
        )
    )

    combined_event = next(row for row in evaluated.events if row.policy == "combined-health-gate")
    assert not combined_event.detected
    combined_actions = evaluated.policy_traces[3].actions
    assert set(combined_actions) == {"fixed-fusion"}
    fixed = evaluated.loss("fixed-fusion")
    policy = evaluated.loss("combined-health-gate")
    np.testing.assert_array_equal(fixed.valid, policy.valid)
    np.testing.assert_allclose(fixed.loss_m2, policy.loss_m2, rtol=0.0, atol=0.0)


def test_zero_object_frames_hold_numeric_latch_but_remain_event_steps() -> None:
    evaluated = _evaluate(_sequence(empty_frames=frozenset({6, 7})))

    for frame_index in (6, 7):
        evidence = evaluated.evidence[frame_index]
        assert evidence.object_count == 0
        assert evidence.camera_self.status == "insufficient-support"
        assert evidence.camera_from_lidar_cross.status == "insufficient-support"
    combined = evaluated.policy_traces[3]
    assert combined.latched_labels[6] == combined.latched_labels[5]
    assert combined.latched_labels[7] == combined.latched_labels[6]
    event = next(row for row in evaluated.events if row.policy == "combined-health-gate")
    assert (
        event.active_healthy_steps
        + event.active_camera_fault_steps
        + event.active_lidar_fault_steps
        + event.active_ambiguous_steps
        == 8
    )
    event_result = next(
        row for row in evaluated.results if row.method == "fixed-fusion" and row.window == "event"
    )
    assert event_result.eligible_object_frame_count == 12


def test_common_mode_omits_target_drop_and_oracle_without_reduced_claim() -> None:
    condition = _case("replay-common-mode-x", 4.0).for_frame_count(16)
    evaluated = _evaluate(
        _sequence(
            condition=condition,
            local_camera_bias=4.0,
            monitoring_camera_bias=4.0,
            common_mode=True,
        )
    )

    assert len(evaluated.losses) == 8
    assert len(evaluated.results) == 24
    assert len(evaluated.contrasts) == 15
    assert all(not row.target_drop_applicable for row in evaluated.contrasts)
    assert all(not row.frame_oracle_applicable for row in evaluated.contrasts)
    assert all(row.fault_target == "both" for row in evaluated.events)


def test_health_evaluator_rejects_wrong_case_and_forged_fit() -> None:
    sequence = _sequence()
    with pytest.raises(ValueError, match="exact frozen case"):
        evaluate_replay_health_sequence(
            sequence,
            case=_case("replay-lidar-output-y-bias", 3.0),
            fit=FIT,
        )

    with pytest.raises(ReplayFitError, match="does not match"):
        evaluate_replay_health_sequence(
            sequence,
            case=_case_for_sequence(sequence),
            fit=replace(
                FIT,
                thresholds=HealthThresholds(self_score=0.999, cross_score=0.999),
            ),
        )


def test_partial_dropout_contrast_uses_only_exact_paired_common_support() -> None:
    missing = frozenset({5, 6, 9})
    condition = _case("replay-camera-dropout", 0.5).for_frame_count(16)
    evaluated = _evaluate(
        _sequence(
            condition=condition,
            local_camera_bias=0.0,
            monitoring_camera_bias=0.0,
            camera_unavailable_frames=missing,
        )
    )
    fixed = evaluated.loss("fixed-fusion")
    policy = evaluated.loss("combined-health-gate")
    event_mask = np.asarray(
        [4 <= frame_index < 12 for frame_index, _ in fixed.row_keys],
        dtype=np.bool_,
    )
    common = event_mask & fixed.valid & policy.valid
    contrast = next(
        row
        for row in evaluated.contrasts
        if row.policy == "combined-health-gate" and row.window == "event"
    )

    assert np.any(event_mask & ~fixed.valid & policy.valid)
    assert contrast.fixed_support_sha256 != contrast.policy_support_sha256
    assert contrast.fixed_policy_common_count == int(np.count_nonzero(common)) == 10
    assert contrast.fixed_on_common_loss_sum_m2 == pytest.approx(
        math.fsum(float(value) for value in fixed.loss_m2[common])
    )
    assert contrast.policy_on_fixed_common_loss_sum_m2 == pytest.approx(
        math.fsum(float(value) for value in policy.loss_m2[common])
    )


def test_full_dropout_retains_coverage_but_never_zero_imputes_fixed_loss() -> None:
    active_frames = frozenset(range(4, 12))
    condition = _case("replay-camera-dropout", 1.0).for_frame_count(16)
    evaluated = _evaluate(
        _sequence(
            condition=condition,
            local_camera_bias=0.0,
            monitoring_camera_bias=0.0,
            camera_unavailable_frames=active_frames,
        )
    )
    contrast = next(
        row
        for row in evaluated.contrasts
        if row.policy == "combined-health-gate" and row.window == "event"
    )
    fixed = next(
        row for row in evaluated.results if row.method == "fixed-fusion" and row.window == "event"
    )
    target_drop = next(
        row
        for row in evaluated.results
        if row.method == "fault-target-drop-policy" and row.window == "event"
    )
    bootstrap = np.zeros((40, 1), dtype=np.int64)

    assert (fixed.loss_sum_m2, fixed.valid_object_frame_count) == (0.0, 0)
    assert fixed.eligible_object_frame_count == 16
    assert target_drop.valid_object_frame_count == target_drop.eligible_object_frame_count == 16
    assert contrast.fixed_policy_common_count == 0
    assert replay_sequence_contrast_values((contrast,)) == (None,)
    assert equal_scene_contrast_interval((contrast,), bootstrap).estimate is None
    coverage = pooled_availability_interval(
        [fixed.valid_object_frame_count],
        [fixed.eligible_object_frame_count],
        bootstrap,
    )
    assert coverage.estimate == 0.0
    conditional_loss = pooled_conditional_loss_interval(
        [fixed.loss_sum_m2],
        [fixed.valid_object_frame_count],
        bootstrap,
    )
    assert conditional_loss.estimate is None
    assert conditional_loss.defined_replicates == 0


def test_abstention_contrast_uses_fixed_abstain_intersection() -> None:
    condition = _case("replay-common-mode-x", 4.0).for_frame_count(16)
    evaluated = _evaluate(
        _sequence(
            condition=condition,
            local_camera_bias=4.0,
            monitoring_camera_bias=100.0,
            common_mode=True,
        )
    )
    fixed = evaluated.loss("fixed-fusion")
    abstain = evaluated.loss("combined-health-gate-abstain")
    event_mask = np.asarray(
        [4 <= frame_index < 12 for frame_index, _ in fixed.row_keys],
        dtype=np.bool_,
    )
    common = event_mask & fixed.valid & abstain.valid
    contrast = next(
        row
        for row in evaluated.contrasts
        if row.policy == "combined-health-gate-abstain" and row.window == "event"
    )

    assert np.any(event_mask & fixed.valid & ~abstain.valid)
    assert np.any(common)
    assert contrast.fixed_support_sha256 != contrast.policy_support_sha256
    assert contrast.fixed_policy_common_count == int(np.count_nonzero(common))
    assert contrast.fixed_on_common_loss_sum_m2 == pytest.approx(
        math.fsum(float(value) for value in fixed.loss_m2[common])
    )
    assert contrast.policy_on_fixed_common_loss_sum_m2 == pytest.approx(
        math.fsum(float(value) for value in abstain.loss_m2[common])
    )


def test_irregular_length_event_uses_dynamic_steps_and_reference_time() -> None:
    reference_times = (
        0.0,
        0.1,
        0.24,
        0.43,
        0.71,
        1.08,
        1.52,
        2.03,
        2.35,
        2.92,
        3.61,
        4.05,
        4.88,
        5.31,
        6.07,
        6.42,
        7.33,
        8.01,
        9.2,
    )
    condition = _case("replay-camera-output-y-bias", 3.0).for_frame_count(len(reference_times))
    evaluated = _evaluate(
        _sequence(
            condition=condition,
            frame_count=len(reference_times),
            reference_times_s=reference_times,
        )
    )
    event = next(row for row in evaluated.events if row.policy == "combined-health-gate")
    result = next(
        row
        for row in evaluated.results
        if row.method == "combined-health-gate" and row.window == "event"
    )

    assert event.schedule.frame_count == 19
    assert event.schedule.fault_active_frames == (4, 14)
    assert event.schedule.recovery_frames == (14, 19)
    assert event.detection_censor_bound_steps == 10
    assert event.detection_latency_steps == 1
    assert event.detection_latency_s == pytest.approx(reference_times[5] - reference_times[4])
    assert event.attribution_latency_steps == 1
    assert result.eligible_object_frame_count == 20
    assert (
        event.active_healthy_steps
        + event.active_camera_fault_steps
        + event.active_lidar_fault_steps
        + event.active_ambiguous_steps
        == 10
    )


def test_empty_unavailable_frames_keep_direct_priority_and_event_timing() -> None:
    active_frames = frozenset(range(4, 12))
    condition = _case("replay-camera-dropout", 1.0).for_frame_count(16)
    evaluated = _evaluate(
        _sequence(
            condition=condition,
            local_camera_bias=0.0,
            monitoring_camera_bias=0.0,
            empty_frames=frozenset({4, 5}),
            camera_unavailable_frames=active_frames,
        )
    )

    for frame_index in (4, 5):
        evidence = evaluated.evidence[frame_index]
        assert evidence.object_count == 0
        assert not evidence.camera_available
        assert not evidence.camera_timestamp_suspicious
        assert evidence.camera_self.status == "insufficient-support"
    for trace in (evaluated.policy_traces[2], evaluated.policy_traces[3]):
        assert trace.raw_labels[4:6] == ("camera-fault", "camera-fault")
        assert trace.evidence_statuses[4:6] == ("update-eligible", "update-eligible")
        assert trace.latched_labels[4] == "healthy"
        assert trace.latched_labels[5] == "camera-fault"
        assert trace.actions[4:6] == ("lidar-only", "lidar-only")
    event = next(row for row in evaluated.events if row.policy == "combined-health-gate")
    fixed = next(
        row for row in evaluated.results if row.method == "fixed-fusion" and row.window == "event"
    )
    assert event.detected and event.correctly_attributed
    assert event.detection_latency_steps == event.attribution_latency_steps == 1
    assert fixed.eligible_object_frame_count == 12


def test_h5_positive_selector_binds_the_declared_event_contrast_and_sign() -> None:
    evaluated = _evaluate(_sequence())
    selector = next(row for row in H5_B_SELECTORS if row.selector == evaluated.condition_selector)
    contrast = next(
        row
        for row in evaluated.contrasts
        if row.policy == selector.method and row.window == selector.window
    )
    values = replay_sequence_contrast_values((contrast,))

    assert selector.hypothesis_id == "h5-b1"
    assert selector.metric_name == "policy-gain-vs-fixed"
    assert selector.assessment_rule == "persistence"
    assert selector.expected_direction == "positive"
    assert values[0] is not None and values[0] > 0.0
