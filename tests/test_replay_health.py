from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayHealthEvidenceV1,
    ReplayHealthFrameInputV1,
    ReplayHealthResultV1,
    ReplayHealthScheduleV1,
    ReplayHealthSequenceEventV1,
    ReplayModalityMeasurementV1,
    ReplayNumericChannelEvidenceV1,
)
from fusion_fault_bench.health import (
    HealthCalibration,
    HealthFrameInput,
    HealthPolicy,
    HealthScorer,
    HealthThresholds,
    ModalityMeasurement,
    ObjectHealthInput,
)
from fusion_fault_bench.replay_health import (
    ReplayHealthFrameInput,
    ReplayHealthPolicy,
    ReplayHealthPolicyTrace,
    ReplayHealthScorer,
    ReplayNumericChannelEvidence,
    decide_replay_raw,
    evaluate_replay_policy_trace,
    replay_health_schedule,
    replay_sequence_event_record,
)

_DIGEST = "a" * 64


def _calibration() -> HealthCalibration:
    values = tuple(np.asarray([0.0, 0.1, 1.0, 10.0]) for _ in range(8))
    return HealthCalibration(*values)


def _measurement(
    value_xy: tuple[float, float],
    reference_time_s: float,
    *,
    reported_time_s: float | None = None,
) -> ModalityMeasurement:
    return ModalityMeasurement(
        value_xy_m=np.asarray(value_xy, dtype=np.float64),
        reported_covariance_xy_m2=np.eye(2, dtype=np.float64),
        reported_time_s=(reference_time_s if reported_time_s is None else reported_time_s),
    )


def _objects(
    value: float,
    reference_time_s: float,
    *,
    camera: bool = True,
    lidar: bool = True,
    camera_offset: float = 0.0,
    lidar_offset: float = 0.0,
) -> tuple[ObjectHealthInput, ...]:
    return tuple(
        ObjectHealthInput(
            object_id=f"object-{index}",
            camera=(
                _measurement(
                    (value + index + camera_offset, 0.0),
                    reference_time_s,
                )
                if camera
                else None
            ),
            lidar=(
                _measurement(
                    (value + index + lidar_offset, 0.0),
                    reference_time_s,
                )
                if lidar
                else None
            ),
        )
        for index in range(2)
    )


def _replay_frame(
    reference_time_s: float,
    *,
    value: float = 0.0,
    camera_available: bool = True,
    lidar_available: bool = True,
    empty: bool = False,
) -> ReplayHealthFrameInput:
    return ReplayHealthFrameInput(
        reference_time_s=reference_time_s,
        camera_available=camera_available,
        lidar_available=lidar_available,
        objects=(
            ()
            if empty
            else _objects(
                value,
                reference_time_s,
                camera=camera_available,
                lidar=lidar_available,
            )
        ),
    )


def _event_trace(
    *,
    camera_missing_frame: int | None = 6,
) -> ReplayHealthPolicyTrace:
    labels = (
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
    camera_available = tuple(
        not (camera_missing_frame is not None and index == camera_missing_frame)
        for index in range(16)
    )
    return ReplayHealthPolicyTrace(
        policy="combined-health-gate",
        reference_times_s=(
            0.0,
            0.1,
            0.25,
            0.4,
            0.6,
            0.9,
            1.1,
            1.4,
            1.8,
            2.0,
            2.3,
            2.7,
            3.0,
            3.4,
            3.9,
            4.5,
        ),
        camera_available=camera_available,
        lidar_available=(True,) * 16,
        raw_labels=labels,
        evidence_statuses=("update-eligible",) * 16,
        latched_labels=labels,
        actions=tuple("fixed-fusion" if label == "healthy" else "lidar-only" for label in labels),
    )


def _measurement_payload() -> dict[str, object]:
    return {
        "value_xy_m": (0.0, 0.0),
        "reported_covariance_xy_m2": ((1.0, 0.0), (0.0, 1.0)),
        "reported_time_s": 0.0,
    }


def _numeric_payload(
    *,
    current: int = 2,
    mature: int = 2,
) -> dict[str, object]:
    return {
        "status": "defined",
        "mature_object_count": mature,
        "current_object_count": current,
        "mature_fraction": mature / current,
        "mean_nis": 1.0,
        "maximum_nis": 2.0,
        "score": 0.5,
    }


def _valid_event() -> ReplayHealthSequenceEventV1:
    return replay_sequence_event_record(
        _event_trace(camera_missing_frame=6),
        schedule=replay_health_schedule(16),
        replay_experiment_identity_sha256=_DIGEST,
        sequence_id="nuscenes:scene-0061",
        condition_id="replay-camera-dropout",
        condition_selector="replay-camera-dropout:+0.5",
        fault_family="dropout",
        fault_target="camera",
    )


def _validate_event_update(**updates: object) -> ReplayHealthSequenceEventV1:
    payload = _valid_event().model_dump(mode="python", by_alias=True)
    payload.update(updates)
    return ReplayHealthSequenceEventV1.model_validate(payload)


def test_variable_schedule_uses_only_frame_count_and_frozen_equations() -> None:
    schedule = replay_health_schedule(17)
    assert schedule.predictor_initialization_frames == (0, 2)
    assert schedule.clean_prefix_frames == (0, 4)
    assert schedule.score_frames == (2, 17)
    assert schedule.fault_active_frames == (4, 12)
    assert schedule.recovery_frames == (12, 17)
    assert schedule.active_frame_count == 8
    assert schedule.recovery_frame_count == 5

    with pytest.raises(ValidationError):
        replay_health_schedule(15)
    with pytest.raises(TypeError):
        replay_health_schedule(True)


def test_replay_contract_rejects_schedule_and_covariance_rebinding() -> None:
    schedule = replay_health_schedule(16)
    payload = schedule.model_dump(mode="python", by_alias=True)
    payload["clean_prefix_frames"] = (0, 5)
    with pytest.raises(ValidationError, match="frozen frame-count equations"):
        ReplayHealthScheduleV1.model_validate(payload)

    measurement = _measurement_payload()
    measurement["reported_covariance_xy_m2"] = ((1.0, 0.25), (0.0, 1.0))
    with pytest.raises(ValidationError, match="symmetric"):
        ReplayModalityMeasurementV1.model_validate(measurement)
    measurement["reported_covariance_xy_m2"] = ((1.0, 2.0), (2.0, 1.0))
    with pytest.raises(ValidationError, match="positive definite"):
        ReplayModalityMeasurementV1.model_validate(measurement)


def test_replay_frame_contract_enforces_population_order_and_availability() -> None:
    base: dict[str, object] = {
        "schema": "ffb.replay-health-frame-input/v1",
        "sequence_id": "nuscenes:scene-0061",
        "frame_index": 0,
        "reference_time_s": 0.0,
        "camera_available": True,
        "lidar_available": True,
        "objects": (
            {
                "object_id": "track:0001",
                "camera": _measurement_payload(),
                "lidar": _measurement_payload(),
            },
            {
                "object_id": "track:0002",
                "camera": _measurement_payload(),
                "lidar": _measurement_payload(),
            },
        ),
    }
    assert len(ReplayHealthFrameInputV1.model_validate(base).objects) == 2

    invalid_scene = {**base, "sequence_id": "nuscenes:scene-9999"}
    with pytest.raises(ValidationError, match="frozen scene population"):
        ReplayHealthFrameInputV1.model_validate(invalid_scene)

    objects = cast(tuple[dict[str, object], ...], base["objects"])
    duplicate = {**base, "objects": (objects[0], objects[0])}
    with pytest.raises(ValidationError, match="unique"):
        ReplayHealthFrameInputV1.model_validate(duplicate)

    reversed_objects = {**base, "objects": tuple(reversed(objects))}
    with pytest.raises(ValidationError, match="UTF-8"):
        ReplayHealthFrameInputV1.model_validate(reversed_objects)

    camera_unavailable = {**base, "camera_available": False}
    with pytest.raises(ValidationError, match="camera measurement"):
        ReplayHealthFrameInputV1.model_validate(camera_unavailable)
    lidar_unavailable = {**base, "lidar_available": False}
    with pytest.raises(ValidationError, match="lidar measurement"):
        ReplayHealthFrameInputV1.model_validate(lidar_unavailable)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"mature_object_count": 3}, "mature support"),
        ({"mature_fraction": 0.5}, "mature fraction"),
        ({"mature_object_count": 1, "mature_fraction": 0.5}, "two mature objects"),
        (
            {
                "status": "insufficient-support",
                "mean_nis": None,
                "maximum_nis": None,
                "score": None,
            },
            "null statistics",
        ),
    ],
)
def test_numeric_contract_enforces_support_and_definition(
    updates: dict[str, object],
    message: str,
) -> None:
    payload = {**_numeric_payload(), **updates}
    with pytest.raises(ValidationError, match=message):
        ReplayNumericChannelEvidenceV1.model_validate(payload)


def test_health_evidence_contract_aligns_channels_and_empty_frame_semantics() -> None:
    channel = _numeric_payload()
    payload: dict[str, object] = {
        "schema": "ffb.replay-health-evidence/v1",
        "sequence_id": "nuscenes:scene-0061",
        "frame_index": 2,
        "reference_time_s": 0.2,
        "camera_available": True,
        "lidar_available": True,
        "camera_timestamp_suspicious": False,
        "lidar_timestamp_suspicious": False,
        "camera_missing_fraction_last_four": 0.0,
        "lidar_missing_fraction_last_four": 0.0,
        "camera_self": channel,
        "lidar_self": channel,
        "camera_from_lidar_cross": channel,
        "lidar_from_camera_cross": channel,
    }
    assert ReplayHealthEvidenceV1.model_validate(payload).camera_self.status == "defined"

    with pytest.raises(ValidationError, match="frozen scene population"):
        ReplayHealthEvidenceV1.model_validate({**payload, "sequence_id": "nuscenes:scene-9999"})
    with pytest.raises(ValidationError, match="share current object support"):
        ReplayHealthEvidenceV1.model_validate(
            {**payload, "camera_self": _numeric_payload(current=3, mature=2)}
        )

    empty = {
        "status": "insufficient-support",
        "mature_object_count": 0,
        "current_object_count": 0,
        "mature_fraction": 0.0,
        "mean_nis": None,
        "maximum_nis": None,
        "score": None,
    }
    empty_payload = {
        **payload,
        "camera_timestamp_suspicious": True,
        "camera_self": empty,
        "lidar_self": empty,
        "camera_from_lidar_cross": empty,
        "lidar_from_camera_cross": empty,
    }
    with pytest.raises(ValidationError, match="empty replay frame"):
        ReplayHealthEvidenceV1.model_validate(empty_payload)


def test_replay_contract_accepts_empty_frames_and_plus_signed_selectors() -> None:
    frame = ReplayHealthFrameInputV1.model_validate_json(
        """
        {
          "schema": "ffb.replay-health-frame-input/v1",
          "sequence_id": "nuscenes:scene-0061",
          "frame_index": 3,
          "reference_time_s": 0.25,
          "camera_available": true,
          "lidar_available": true,
          "objects": []
        }
        """
    )
    assert frame.objects == ()

    row = ReplayHealthResultV1.model_validate(
        {
            "schema": "ffb.replay-health-result/v1",
            "replay_experiment_identity_sha256": _DIGEST,
            "sequence_id": "nuscenes:scene-0061",
            "condition_id": "replay-lidar-output-y-bias",
            "condition_selector": "replay-lidar-output-y-bias:+3",
            "method": "combined-health-gate",
            "window": "event",
            "loss_sum_m2": 0.0,
            "valid_object_frame_count": 0,
            "eligible_object_frame_count": 0,
        }
    )
    assert row.condition_selector.endswith(":+3")

    payload = row.model_dump(mode="json", by_alias=True)
    payload["condition_selector"] = "replay-camera-output-y-bias:+3"
    with pytest.raises(ValidationError, match="base condition"):
        ReplayHealthResultV1.model_validate(payload)
    payload["condition_selector"] = "replay-lidar-output-y-bias:+3/path"
    with pytest.raises(ValidationError):
        ReplayHealthResultV1.model_validate(payload)


def test_m4_nonempty_scoring_is_value_for_value_unchanged() -> None:
    m4 = HealthScorer(_calibration())
    replay = ReplayHealthScorer(_calibration())
    for frame_index in range(4):
        reference_time = float(frame_index)
        objects = _objects(reference_time, reference_time)
        m4_evidence = m4.process_frame(
            HealthFrameInput(
                reference_time_s=reference_time,
                camera_available=True,
                lidar_available=True,
                objects=objects,
            )
        )
        replay_evidence = replay.process_frame(
            ReplayHealthFrameInput(
                reference_time_s=reference_time,
                camera_available=True,
                lidar_available=True,
                objects=objects,
            )
        )
        assert asdict(replay_evidence) == asdict(m4_evidence)

    with pytest.raises(ValueError, match="nonempty tuple"):
        HealthFrameInput(
            reference_time_s=5.0,
            camera_available=True,
            lidar_available=True,
            objects=(),
        )


@pytest.mark.parametrize(
    "method",
    (
        "self-nis-gate",
        "cross-nis-gate",
        "direct-telemetry-gate",
        "combined-health-gate",
    ),
)
def test_m4_nonempty_decision_latch_and_action_are_value_for_value_unchanged(
    method: Any,
) -> None:
    thresholds = HealthThresholds(self_score=0.5, cross_score=0.5)
    m4_scorer = HealthScorer(_calibration())
    replay_scorer = ReplayHealthScorer(_calibration())
    m4_policy = HealthPolicy(method=method, thresholds=thresholds)
    replay_policy = ReplayHealthPolicy(method=method, thresholds=thresholds)
    for frame_index in range(6):
        reference_time = float(frame_index)
        objects = _objects(
            reference_time,
            reference_time,
            camera_offset=10.0 if frame_index in {3, 4} else 0.0,
        )
        m4_evidence = m4_scorer.process_frame(
            HealthFrameInput(
                reference_time_s=reference_time,
                camera_available=True,
                lidar_available=True,
                objects=objects,
            )
        )
        replay_evidence = replay_scorer.process_frame(
            ReplayHealthFrameInput(
                reference_time_s=reference_time,
                camera_available=True,
                lidar_available=True,
                objects=objects,
            )
        )
        m4_output = m4_policy.apply(m4_evidence)
        replay_output = replay_policy.apply(replay_evidence)
        assert asdict(replay_output.raw_decision) == asdict(m4_output.raw_decision)
        assert replay_output.latched_state == m4_output.latched_state
        assert replay_output.executed_action == m4_output.executed_action


def test_empty_frame_has_exact_zero_numeric_support_and_direct_priority() -> None:
    scorer = ReplayHealthScorer(_calibration())
    scorer.process_frame(_replay_frame(0.0, value=0.0))
    scorer.process_frame(_replay_frame(1.0, value=1.0))
    empty = scorer.process_frame(
        _replay_frame(
            2.0,
            camera_available=False,
            lidar_available=True,
            empty=True,
        )
    )

    assert empty.object_count == 0
    assert not empty.camera_timestamp_suspicious
    assert not empty.lidar_timestamp_suspicious
    assert empty.camera_missing_fraction_last_four == pytest.approx(1.0 / 3.0)
    for channel in (
        empty.camera_self,
        empty.lidar_self,
        empty.camera_from_lidar_cross,
        empty.lidar_from_camera_cross,
    ):
        assert asdict(channel) == {
            "status": "insufficient-support",
            "mature_object_count": 0,
            "current_object_count": 0,
            "mature_fraction": 0.0,
            "mean_nis": None,
            "maximum_nis": None,
            "score": None,
        }

    thresholds = HealthThresholds(self_score=0.999, cross_score=0.995)
    direct = decide_replay_raw(
        method="direct-telemetry-gate",
        evidence=empty,
        thresholds=thresholds,
    )
    combined = decide_replay_raw(
        method="combined-health-gate",
        evidence=empty,
        thresholds=thresholds,
    )
    self_only = decide_replay_raw(
        method="self-nis-gate",
        evidence=empty,
        thresholds=thresholds,
    )
    assert (direct.label, direct.evidence_status) == (
        "camera-fault",
        "update-eligible",
    )
    assert (combined.label, combined.evidence_status) == (
        "camera-fault",
        "update-eligible",
    )
    assert (self_only.label, self_only.evidence_status) == (
        "ambiguous",
        "insufficient-support",
    )


def test_empty_numeric_support_holds_latch_but_direct_evidence_uses_m4_recurrence() -> None:
    thresholds = HealthThresholds(self_score=0.999, cross_score=0.995)
    scorer = ReplayHealthScorer(_calibration())
    camera_missing = tuple(
        scorer.process_frame(
            _replay_frame(
                float(index),
                camera_available=False,
                lidar_available=True,
                empty=True,
            )
        )
        for index in range(2)
    )
    direct = ReplayHealthPolicy(
        method="direct-telemetry-gate",
        thresholds=thresholds,
    )
    first = direct.apply(camera_missing[0])
    second = direct.apply(camera_missing[1])
    assert first.latched_state.label == "healthy"
    assert first.executed_action == "lidar-only"
    assert second.latched_state.label == "camera-fault"
    assert second.executed_action == "lidar-only"

    healthy_empty = ReplayHealthScorer(_calibration()).process_frame(_replay_frame(0.0, empty=True))
    combined = ReplayHealthPolicy(
        method="combined-health-gate",
        thresholds=thresholds,
    )
    combined.apply(camera_missing[0])
    activated = combined.apply(camera_missing[1])
    held = combined.apply(
        type(healthy_empty)(
            reference_time_s=2.0,
            camera_available=healthy_empty.camera_available,
            lidar_available=healthy_empty.lidar_available,
            camera_timestamp_suspicious=False,
            lidar_timestamp_suspicious=False,
            camera_missing_fraction_last_four=healthy_empty.camera_missing_fraction_last_four,
            lidar_missing_fraction_last_four=healthy_empty.lidar_missing_fraction_last_four,
            camera_self=healthy_empty.camera_self,
            lidar_self=healthy_empty.lidar_self,
            camera_from_lidar_cross=healthy_empty.camera_from_lidar_cross,
            lidar_from_camera_cross=healthy_empty.lidar_from_camera_cross,
        )
    )
    assert activated.latched_state.label == "camera-fault"
    assert held.raw_decision.evidence_status == "insufficient-support"
    assert held.latched_state == activated.latched_state
    assert held.executed_action == "lidar-only"


def test_empty_frame_advances_clock_without_changing_object_history() -> None:
    scorer = ReplayHealthScorer(_calibration())
    scorer.process_frame(_replay_frame(0.0, value=0.0))
    scorer.process_frame(_replay_frame(1.0, value=1.0))
    scorer.process_frame(_replay_frame(2.0, empty=True))
    evidence = scorer.process_frame(_replay_frame(3.0, value=3.0))
    assert evidence.camera_self.status == "defined"
    assert evidence.lidar_self.status == "defined"
    assert evidence.camera_self.mean_nis == pytest.approx(0.0)
    assert evidence.lidar_self.mean_nis == pytest.approx(0.0)


def test_runtime_replay_inputs_reject_order_availability_and_nonfinite_time() -> None:
    objects = _objects(0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        ReplayHealthFrameInput(
            reference_time_s=float("inf"),
            camera_available=True,
            lidar_available=True,
            objects=objects,
        )
    with pytest.raises(ValueError, match="unique"):
        ReplayHealthFrameInput(
            reference_time_s=0.0,
            camera_available=True,
            lidar_available=True,
            objects=(objects[0], objects[0]),
        )
    with pytest.raises(ValueError, match="UTF-8"):
        ReplayHealthFrameInput(
            reference_time_s=0.0,
            camera_available=True,
            lidar_available=True,
            objects=tuple(reversed(objects)),
        )
    with pytest.raises(ValueError, match="camera measurement"):
        ReplayHealthFrameInput(
            reference_time_s=0.0,
            camera_available=False,
            lidar_available=True,
            objects=objects,
        )
    with pytest.raises(ValueError, match="lidar measurement"):
        ReplayHealthFrameInput(
            reference_time_s=0.0,
            camera_available=True,
            lidar_available=False,
            objects=objects,
        )


def test_runtime_numeric_evidence_rejects_inconsistent_support_and_statistics() -> None:
    defined = ReplayNumericChannelEvidence(
        status="defined",
        mature_object_count=2,
        current_object_count=2,
        mature_fraction=1.0,
        mean_nis=1.0,
        maximum_nis=2.0,
        score=0.5,
    )
    with pytest.raises(ValueError, match="unknown replay numeric status"):
        replace(defined, status="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="within current support"):
        replace(defined, mature_object_count=3)
    with pytest.raises(ValueError, match="mature_fraction"):
        replace(defined, mature_fraction=0.5)
    with pytest.raises(ValueError, match="mature numeric statistics"):
        replace(defined, mature_object_count=1, mature_fraction=0.5)
    with pytest.raises(ValueError, match="null statistics"):
        replace(defined, status="insufficient-support")


def test_runtime_frame_evidence_enforces_ranges_alignment_and_empty_semantics() -> None:
    scorer = ReplayHealthScorer(_calibration())
    evidence = scorer.process_frame(_replay_frame(0.0))
    with pytest.raises(ValueError, match="finite"):
        replace(evidence, reference_time_s=float("nan"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(evidence, camera_missing_fraction_last_four=-0.1)

    one_object = ReplayNumericChannelEvidence(
        status="insufficient-support",
        mature_object_count=0,
        current_object_count=1,
        mature_fraction=0.0,
        mean_nis=None,
        maximum_nis=None,
        score=None,
    )
    with pytest.raises(ValueError, match="share current support"):
        replace(evidence, camera_self=one_object)

    empty = ReplayHealthScorer(_calibration()).process_frame(_replay_frame(0.0, empty=True))
    with pytest.raises(ValueError, match="cannot contain timestamp evidence"):
        replace(empty, camera_timestamp_suspicious=True)


def test_policy_and_trace_contracts_enforce_method_causality_and_alignment() -> None:
    thresholds = HealthThresholds(self_score=0.5, cross_score=0.5)
    with pytest.raises(ValueError, match="unknown health method"):
        ReplayHealthPolicy(method="other", thresholds=thresholds)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="combined gate"):
        ReplayHealthPolicy(
            method="self-nis-gate",
            thresholds=thresholds,
            abstain_on_ambiguous=True,
        )

    policy = ReplayHealthPolicy(method="combined-health-gate", thresholds=thresholds)
    assert policy.state.label == "healthy"

    valid = _event_trace()
    with pytest.raises(ValueError, match="nonempty and aligned"):
        replace(valid, actions=valid.actions[:-1])
    with pytest.raises(ValueError, match="finite"):
        replace(
            valid,
            reference_times_s=(float("nan"), *valid.reference_times_s[1:]),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(
            valid,
            reference_times_s=(
                valid.reference_times_s[0],
                valid.reference_times_s[0],
                *valid.reference_times_s[2:],
            ),
        )
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_replay_policy_trace((), method="combined-health-gate", thresholds=thresholds)

    scorer = ReplayHealthScorer(_calibration())
    evidence = tuple(
        scorer.process_frame(_replay_frame(float(index), value=float(index))) for index in range(3)
    )
    evaluated = evaluate_replay_policy_trace(
        evidence,
        method="combined-health-gate",
        thresholds=thresholds,
    )
    assert evaluated.reference_times_s == (0.0, 1.0, 2.0)
    assert len(evaluated.actions) == 3


def test_dynamic_event_record_uses_steps_elapsed_time_and_signed_dropout_response() -> None:
    schedule = replay_health_schedule(16)
    trace = _event_trace(camera_missing_frame=6)
    event = replay_sequence_event_record(
        trace,
        schedule=schedule,
        replay_experiment_identity_sha256=_DIGEST,
        sequence_id="nuscenes:scene-0061",
        condition_id="replay-camera-dropout",
        condition_selector="replay-camera-dropout:+0.5",
        fault_family="dropout",
        fault_target="camera",
    )
    assert event.detected
    assert event.detection_latency_steps == 1
    assert event.detection_latency_s == pytest.approx(0.3)
    assert event.correctly_attributed
    assert event.attribution_latency_steps == 1
    assert event.realized_dropout
    assert event.first_missing_step == 2
    assert event.first_missing_latency_s == pytest.approx(0.5)
    assert event.detection_minus_first_missing_steps == -1
    assert event.detection_minus_first_missing_s == pytest.approx(-0.2)
    assert event.recovery_eligible
    assert event.recovered
    assert event.recovery_latency_steps == 2
    assert event.recovery_latency_s == pytest.approx(0.9)
    assert (
        event.active_healthy_steps
        + event.active_camera_fault_steps
        + event.active_lidar_fault_steps
        + event.active_ambiguous_steps
        == schedule.active_frame_count
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sequence_id": "nuscenes:scene-9999"}, "frozen scene population"),
        ({"condition_id": "replay-not-preregistered"}, "frozen M5-B matrix"),
        (
            {"condition_selector": "replay-lidar-dropout:+0.5"},
            "bind the base condition_id",
        ),
        ({"fault_family": "identity"}, "identity replay events require target none"),
        (
            {"fault_family": "common-mode-position-bias"},
            "common-mode replay events require target both",
        ),
        (
            {"fault_family": "calibration-yaw", "fault_target": "lidar"},
            "calibration faults require target camera",
        ),
        (
            {"fault_family": "additive-position-bias", "fault_target": "both"},
            "single-target replay faults",
        ),
        ({"detection_censor_bound_steps": 9}, "censor bounds"),
        ({"detection_latency_steps": 8}, "outside the dynamic active window"),
        ({"recovery_latency_steps": 4}, "outside the recovery window"),
        ({"detected": False}, "detection step"),
        ({"detection_latency_s": None}, "detection seconds"),
        ({"first_latch_label": None}, "first latch label"),
        ({"first_latch_label": "healthy"}, "cannot first latch healthy"),
        ({"correctly_attributed": False}, "attribution step"),
        ({"attribution_latency_s": None}, "attribution seconds"),
        ({"attribution_latency_steps": 0}, "cannot precede detection"),
        ({"outcome": "wrong-sensor"}, "later latch episode"),
        ({"latch_episode_count": 0}, "require a latch episode"),
        (
            {"fault_family": "additive-position-bias"},
            "realized_dropout must be present exactly",
        ),
        ({"first_missing_step": None}, "first-missing step"),
        ({"first_missing_latency_s": None}, "first-missing seconds"),
        (
            {"detection_minus_first_missing_steps": 0},
            "detection minus first missing",
        ),
        (
            {
                "fault_family": "common-mode-position-bias",
                "fault_target": "both",
                "realized_dropout": None,
                "first_missing_step": None,
                "first_missing_latency_s": None,
                "detection_minus_first_missing_steps": None,
                "detection_minus_first_missing_s": None,
            },
            "targetless replay events cannot be correctly attributed",
        ),
        (
            {
                "fault_family": "common-mode-position-bias",
                "fault_target": "both",
                "outcome": "wrong-sensor",
                "correctly_attributed": False,
                "attribution_latency_steps": None,
                "attribution_latency_s": None,
                "realized_dropout": None,
                "first_missing_step": None,
                "first_missing_latency_s": None,
                "detection_minus_first_missing_steps": None,
                "detection_minus_first_missing_s": None,
            },
            "targetless detected event must be ambiguous",
        ),
        (
            {
                "outcome": "ambiguous",
                "correctly_attributed": False,
                "attribution_latency_steps": None,
                "attribution_latency_s": None,
            },
            "outcome must match the first latch label",
        ),
        ({"active_healthy_steps": 2}, "state occupancy"),
        ({"active_fixed_action_steps": 2}, "action occupancy"),
        ({"final_active_state": "lidar-fault"}, "final active state must occur"),
        ({"recovery_eligible": False}, "recovery eligibility"),
        (
            {"final_active_state": "healthy", "recovery_eligible": False},
            "recovery requires a nonhealthy",
        ),
        ({"recovered": False}, "recovery step"),
        ({"recovery_latency_s": None}, "recovery seconds"),
        ({"latch_episode_count": 9}, "latch episode count"),
        ({"false_alert_episode_count": 9}, "false-alert count"),
        ({"false_alert_episode_count": 1}, "zero false alerts"),
    ],
)
def test_event_contract_rejects_inconsistent_dynamic_semantics(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _validate_event_update(**updates)


def test_event_reducer_covers_wrong_then_correct_ambiguous_and_censored_recovery() -> None:
    schedule = replay_health_schedule(16)
    base = _event_trace(camera_missing_frame=None)

    later_correct_labels = (
        *("healthy",) * 5,
        "lidar-fault",
        "healthy",
        *("camera-fault",) * 7,
        "healthy",
        "healthy",
    )
    later_correct = replace(
        base,
        raw_labels=later_correct_labels,
        latched_labels=later_correct_labels,
        actions=tuple(
            "lidar-only" if label == "camera-fault" else "fixed-fusion"
            for label in later_correct_labels
        ),
    )
    wrong_then_correct = replay_sequence_event_record(
        later_correct,
        schedule=schedule,
        replay_experiment_identity_sha256=_DIGEST,
        sequence_id="nuscenes:scene-0061",
        condition_id="replay-camera-output-y-bias",
        condition_selector="replay-camera-output-y-bias:+3",
        fault_family="additive-position-bias",
        fault_target="camera",
    )
    assert wrong_then_correct.outcome == "wrong-sensor"
    assert wrong_then_correct.correctly_attributed
    assert wrong_then_correct.attribution_latency_steps == 3
    assert wrong_then_correct.latch_episode_count == 2
    assert wrong_then_correct.early_clear

    ambiguous_labels = (*("healthy",) * 5, *("ambiguous",) * 9, "healthy", "healthy")
    ambiguous = replay_sequence_event_record(
        replace(
            base,
            raw_labels=ambiguous_labels,
            latched_labels=ambiguous_labels,
            actions=("fixed-fusion",) * 16,
        ),
        schedule=schedule,
        replay_experiment_identity_sha256=_DIGEST,
        sequence_id="nuscenes:scene-0061",
        condition_id="replay-common-mode-x",
        condition_selector="replay-common-mode-x:+4",
        fault_family="common-mode-position-bias",
        fault_target="both",
    )
    assert ambiguous.outcome == "ambiguous"
    assert not ambiguous.correctly_attributed

    unrecovered_labels = (*("healthy",) * 5, *("camera-fault",) * 11)
    unrecovered = replay_sequence_event_record(
        replace(
            base,
            raw_labels=unrecovered_labels,
            latched_labels=unrecovered_labels,
            actions=tuple(
                "lidar-only" if label == "camera-fault" else "fixed-fusion"
                for label in unrecovered_labels
            ),
        ),
        schedule=schedule,
        replay_experiment_identity_sha256=_DIGEST,
        sequence_id="nuscenes:scene-0061",
        condition_id="replay-camera-output-y-bias",
        condition_selector="replay-camera-output-y-bias:+3",
        fault_family="additive-position-bias",
        fault_target="camera",
    )
    assert unrecovered.recovery_eligible
    assert not unrecovered.recovered
    assert unrecovered.recovery_latency_steps is None


@pytest.mark.parametrize(
    ("fault_family", "fault_target", "message"),
    [
        ("identity", "camera", "identity replay events require target none"),
        (
            "common-mode-position-bias",
            "camera",
            "common-mode replay events require target both",
        ),
        ("additive-position-bias", "both", "single-target replay faults"),
    ],
)
def test_event_reducer_rejects_invalid_family_target_bindings(
    fault_family: Any,
    fault_target: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replay_sequence_event_record(
            _event_trace(),
            schedule=replay_health_schedule(16),
            replay_experiment_identity_sha256=_DIGEST,
            sequence_id="nuscenes:scene-0061",
            condition_id="replay-camera-dropout",
            condition_selector="replay-camera-dropout:+0.5",
            fault_family=fault_family,
            fault_target=fault_target,
        )

    with pytest.raises(ValueError, match="trace and dynamic schedule"):
        replay_sequence_event_record(
            _event_trace(),
            schedule=replay_health_schedule(17),
            replay_experiment_identity_sha256=_DIGEST,
            sequence_id="nuscenes:scene-0061",
            condition_id="replay-camera-dropout",
            condition_selector="replay-camera-dropout:+0.5",
            fault_family="dropout",
            fault_target="camera",
        )


def test_result_contract_rejects_population_condition_and_support_mismatches() -> None:
    payload: dict[str, object] = {
        "schema": "ffb.replay-health-result/v1",
        "replay_experiment_identity_sha256": _DIGEST,
        "sequence_id": "nuscenes:scene-0061",
        "condition_id": "replay-lidar-output-y-bias",
        "condition_selector": "replay-lidar-output-y-bias:+3",
        "method": "combined-health-gate",
        "window": "event",
        "loss_sum_m2": 1.0,
        "valid_object_frame_count": 1,
        "eligible_object_frame_count": 1,
    }
    with pytest.raises(ValidationError, match="frozen scene population"):
        ReplayHealthResultV1.model_validate({**payload, "sequence_id": "nuscenes:scene-9999"})
    with pytest.raises(ValidationError, match="frozen M5-B matrix"):
        ReplayHealthResultV1.model_validate(
            {
                **payload,
                "condition_id": "replay-not-preregistered",
                "condition_selector": "replay-not-preregistered:1",
            }
        )
    with pytest.raises(ValidationError, match="cannot exceed eligible"):
        ReplayHealthResultV1.model_validate({**payload, "valid_object_frame_count": 2})
    with pytest.raises(ValidationError, match="zero accumulated loss"):
        ReplayHealthResultV1.model_validate({**payload, "valid_object_frame_count": 0})


@pytest.mark.parametrize(
    ("fault_family", "fault_target", "selector"),
    [
        ("identity", "none", "replay-clean:0"),
        ("dropout", "camera", "replay-camera-dropout:+0.1"),
    ],
)
def test_identity_and_unrealized_dropout_are_forced_missed_false_alert_controls(
    fault_family: Any,
    fault_target: Any,
    selector: str,
) -> None:
    trace = _event_trace(camera_missing_frame=None)
    event = replay_sequence_event_record(
        trace,
        schedule=replay_health_schedule(16),
        replay_experiment_identity_sha256=_DIGEST,
        sequence_id="nuscenes:scene-0061",
        condition_id=selector.split(":", maxsplit=1)[0],
        condition_selector=selector,
        fault_family=fault_family,
        fault_target=fault_target,
    )
    assert not event.detected
    assert event.outcome == "missed"
    assert event.detection_latency_steps is None
    assert event.false_alert_episode_count == 1
    if fault_family == "dropout":
        assert event.realized_dropout is False
