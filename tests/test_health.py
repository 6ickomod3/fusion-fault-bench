from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from fusion_fault_bench.health import (
    HealthCalibration,
    HealthFrameEvidence,
    HealthFrameInput,
    HealthLatchState,
    HealthPolicy,
    HealthScorer,
    HealthThresholds,
    ModalityMeasurement,
    NumericChannelEvidence,
    ObjectHealthInput,
    RawHealthDecision,
    advance_latch,
    choose_executed_action,
    decide_raw,
    ecdf_rank,
)


def _calibration(
    values: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0),
) -> HealthCalibration:
    arrays = [np.asarray(values, dtype=np.float64) for _ in range(8)]
    return HealthCalibration(*arrays)


def _measurement(
    x: float,
    y: float,
    reference_time_s: float,
    *,
    reported_time_s: float | None = None,
    covariance_scale: float = 1.0,
) -> ModalityMeasurement:
    return ModalityMeasurement(
        value_xy_m=np.asarray([x, y], dtype=np.float64),
        reported_covariance_xy_m2=np.eye(2, dtype=np.float64) * covariance_scale,
        reported_time_s=(reference_time_s if reported_time_s is None else reported_time_s),
    )


def _frame(
    reference_time_s: float,
    *,
    camera_xy: tuple[float, float] | None,
    lidar_xy: tuple[float, float] | None,
    camera_available: bool = True,
    lidar_available: bool = True,
    camera_reported_time_s: float | None = None,
    lidar_reported_time_s: float | None = None,
    object_count: int = 2,
) -> HealthFrameInput:
    objects = tuple(
        ObjectHealthInput(
            object_id=f"object-{index}",
            camera=(
                None
                if camera_xy is None
                else _measurement(
                    camera_xy[0],
                    camera_xy[1],
                    reference_time_s,
                    reported_time_s=camera_reported_time_s,
                )
            ),
            lidar=(
                None
                if lidar_xy is None
                else _measurement(
                    lidar_xy[0],
                    lidar_xy[1],
                    reference_time_s,
                    reported_time_s=lidar_reported_time_s,
                )
            ),
        )
        for index in range(object_count)
    )
    return HealthFrameInput(
        reference_time_s=reference_time_s,
        camera_available=camera_available,
        lidar_available=lidar_available,
        objects=objects,
    )


def _numeric(
    score: float | None,
    *,
    mature_count: int = 2,
) -> NumericChannelEvidence:
    if score is None:
        insufficient_count = min(mature_count, 1)
        return NumericChannelEvidence(
            status="insufficient-support",
            mature_object_count=insufficient_count,
            current_object_count=2,
            mature_fraction=insufficient_count / 2,
            mean_nis=None,
            maximum_nis=None,
            score=None,
        )
    return NumericChannelEvidence(
        status="defined",
        mature_object_count=mature_count,
        current_object_count=2,
        mature_fraction=mature_count / 2,
        mean_nis=score,
        maximum_nis=score,
        score=score,
    )


def _evidence(
    *,
    camera_self: float | None = 0.0,
    lidar_self: float | None = 0.0,
    camera_from_lidar: float | None = 0.0,
    lidar_from_camera: float | None = 0.0,
    camera_available: bool = True,
    lidar_available: bool = True,
    camera_timestamp_suspicious: bool = False,
    lidar_timestamp_suspicious: bool = False,
) -> HealthFrameEvidence:
    return HealthFrameEvidence(
        reference_time_s=2.0,
        camera_available=camera_available,
        lidar_available=lidar_available,
        camera_timestamp_suspicious=camera_timestamp_suspicious,
        lidar_timestamp_suspicious=lidar_timestamp_suspicious,
        camera_missing_fraction_last_four=float(not camera_available),
        lidar_missing_fraction_last_four=float(not lidar_available),
        camera_self=_numeric(camera_self),
        lidar_self=_numeric(lidar_self),
        camera_from_lidar_cross=_numeric(camera_from_lidar),
        lidar_from_camera_cross=_numeric(lidar_from_camera),
    )


def _decision(
    label: str,
    *,
    status: str = "update-eligible",
) -> RawHealthDecision:
    return RawHealthDecision(
        method="direct-telemetry-gate",
        label=label,  # type: ignore[arg-type]
        evidence_status=status,  # type: ignore[arg-type]
        camera_alarm=None,
        lidar_alarm=None,
        any_cross_alarm=None,
    )


def test_measurement_and_calibration_make_defensive_immutable_float64_copies() -> None:
    value = np.asarray([1.0, 2.0], dtype=np.float32)
    covariance = np.eye(2, dtype=np.float32)
    measurement = ModalityMeasurement(
        value_xy_m=value,
        reported_covariance_xy_m2=covariance,
        reported_time_s=0.0,
    )
    calibration_source = np.asarray([0.0, 1.0], dtype=np.float64)
    calibration = HealthCalibration(*(calibration_source for _ in range(8)))

    value[0] = 99.0
    covariance[0, 0] = 99.0
    calibration_source[0] = 99.0

    assert measurement.value_xy_m.tolist() == [1.0, 2.0]
    assert measurement.reported_covariance_xy_m2.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert measurement.value_xy_m.dtype == np.float64
    assert not measurement.value_xy_m.flags.writeable
    assert not measurement.reported_covariance_xy_m2.flags.writeable
    assert calibration.camera_self_mean.tolist() == [0.0, 1.0]
    assert not calibration.camera_self_mean.flags.writeable


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "value_xy_m": [1.0],
                "reported_covariance_xy_m2": np.eye(2),
                "reported_time_s": 0.0,
            },
            "shape",
        ),
        (
            {
                "value_xy_m": [1.0, np.nan],
                "reported_covariance_xy_m2": np.eye(2),
                "reported_time_s": 0.0,
            },
            "finite",
        ),
        (
            {
                "value_xy_m": [1.0, 2.0],
                "reported_covariance_xy_m2": [1.0, 1.0],
                "reported_time_s": 0.0,
            },
            "shape",
        ),
        (
            {
                "value_xy_m": [1.0, 2.0],
                "reported_covariance_xy_m2": [[1.0, 0.0], [0.0, np.nan]],
                "reported_time_s": 0.0,
            },
            "finite",
        ),
        (
            {
                "value_xy_m": [1.0, 2.0],
                "reported_covariance_xy_m2": [[1.0, 0.5], [0.0, 1.0]],
                "reported_time_s": 0.0,
            },
            "symmetric",
        ),
        (
            {
                "value_xy_m": [1.0, 2.0],
                "reported_covariance_xy_m2": [[1.0, 0.0], [0.0, 0.0]],
                "reported_time_s": 0.0,
            },
            "positive definite",
        ),
        (
            {
                "value_xy_m": [1.0, 2.0],
                "reported_covariance_xy_m2": np.eye(2),
                "reported_time_s": np.inf,
            },
            "finite",
        ),
    ],
)
def test_measurement_rejects_invalid_geometry(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModalityMeasurement(**arguments)  # type: ignore[arg-type]


def test_frame_rejects_duplicate_objects_and_measurements_from_unavailable_sensor() -> None:
    item = ObjectHealthInput(
        object_id="same",
        camera=_measurement(0.0, 0.0, 0.0),
        lidar=None,
    )
    with pytest.raises(ValueError, match="unique"):
        HealthFrameInput(
            reference_time_s=0.0,
            camera_available=True,
            lidar_available=False,
            objects=(item, item),
        )
    with pytest.raises(ValueError, match="camera measurement"):
        HealthFrameInput(
            reference_time_s=0.0,
            camera_available=False,
            lidar_available=False,
            objects=(item,),
        )
    lidar_item = ObjectHealthInput(
        object_id="lidar",
        camera=None,
        lidar=_measurement(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="lidar measurement"):
        HealthFrameInput(
            reference_time_s=0.0,
            camera_available=False,
            lidar_available=False,
            objects=(lidar_item,),
        )


def test_input_contract_rejects_empty_object_id_support_and_nonfinite_frame_time() -> None:
    with pytest.raises(ValueError, match="nonempty opaque"):
        ObjectHealthInput(object_id="", camera=None, lidar=None)
    with pytest.raises(ValueError, match="nonempty tuple"):
        HealthFrameInput(
            reference_time_s=0.0,
            camera_available=True,
            lidar_available=True,
            objects=(),
        )
    item = ObjectHealthInput(object_id="object", camera=None, lidar=None)
    with pytest.raises(ValueError, match="finite"):
        HealthFrameInput(
            reference_time_s=np.nan,
            camera_available=True,
            lidar_available=True,
            objects=(item,),
        )


def test_calibration_threshold_and_numeric_evidence_validation() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _calibration(())
    with pytest.raises(ValueError, match="finite"):
        _calibration((0.0, np.nan))
    with pytest.raises(ValueError, match="sorted"):
        _calibration((1.0, 0.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        HealthThresholds(self_score=-0.1, cross_score=0.5)
    with pytest.raises(ValueError, match="at least two"):
        NumericChannelEvidence(
            status="defined",
            mature_object_count=1,
            current_object_count=2,
            mature_fraction=0.5,
            mean_nis=0.0,
            maximum_nis=0.0,
            score=0.0,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "status": "bogus",
                "mature_object_count": 0,
                "current_object_count": 2,
                "mature_fraction": 0.0,
                "mean_nis": None,
                "maximum_nis": None,
                "score": None,
            },
            "unknown",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 0,
                "current_object_count": 0,
                "mature_fraction": 0.0,
                "mean_nis": None,
                "maximum_nis": None,
                "score": None,
            },
            "positive",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 3,
                "current_object_count": 2,
                "mature_fraction": 1.5,
                "mean_nis": None,
                "maximum_nis": None,
                "score": None,
            },
            "within",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 1,
                "current_object_count": 2,
                "mature_fraction": 0.0,
                "mean_nis": None,
                "maximum_nis": None,
                "score": None,
            },
            "fraction",
        ),
        (
            {
                "status": "defined",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": None,
                "maximum_nis": 0.0,
                "score": 0.0,
            },
            "requires finite",
        ),
        (
            {
                "status": "defined",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": -1.0,
                "maximum_nis": 0.0,
                "score": 0.0,
            },
            "nonnegative",
        ),
        (
            {
                "status": "defined",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": 0.0,
                "maximum_nis": 0.0,
                "score": 1.1,
            },
            "exceed",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": None,
                "maximum_nis": None,
                "score": None,
            },
            "fewer than two",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 1,
                "current_object_count": 2,
                "mature_fraction": 0.5,
                "mean_nis": 0.0,
                "maximum_nis": None,
                "score": None,
            },
            "cannot contain",
        ),
    ],
)
def test_numeric_channel_rejects_internally_inconsistent_outputs(
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NumericChannelEvidence(**arguments)  # type: ignore[arg-type]


def test_frame_evidence_validates_time_and_missing_fraction_diagnostics() -> None:
    arguments = {
        "reference_time_s": 0.0,
        "camera_available": True,
        "lidar_available": True,
        "camera_timestamp_suspicious": False,
        "lidar_timestamp_suspicious": False,
        "camera_missing_fraction_last_four": 0.0,
        "lidar_missing_fraction_last_four": 0.0,
        "camera_self": _numeric(0.0),
        "lidar_self": _numeric(0.0),
        "camera_from_lidar_cross": _numeric(0.0),
        "lidar_from_camera_cross": _numeric(0.0),
    }
    with pytest.raises(ValueError, match="finite"):
        HealthFrameEvidence(**(arguments | {"reference_time_s": np.inf}))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        HealthFrameEvidence(**(arguments | {"camera_missing_fraction_last_four": 1.1}))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.25),
        (2.0, 0.5),
        (2.5, 1.0),
        (4.0, 1.0),
    ],
)
def test_ecdf_rank_uses_strict_less_than_ties(value: float, expected: float) -> None:
    clean = np.asarray([0.0, 1.0, 2.0, 2.0], dtype=np.float64)
    assert ecdf_rank(clean, value) == expected


def test_two_prior_observation_predictor_matches_constant_velocity_analytic_oracle() -> None:
    scorer = HealthScorer(_calibration())
    first = scorer.process_frame(_frame(0.0, camera_xy=(0.0, 0.0), lidar_xy=(0.0, 0.0)))
    second = scorer.process_frame(_frame(1.0, camera_xy=(1.0, -2.0), lidar_xy=(1.0, -2.0)))
    third = scorer.process_frame(_frame(2.0, camera_xy=(2.0, -4.0), lidar_xy=(2.0, -4.0)))

    assert first.camera_self.status == "insufficient-support"
    assert second.camera_self.status == "insufficient-support"
    for channel in (
        third.camera_self,
        third.lidar_self,
        third.camera_from_lidar_cross,
        third.lidar_from_camera_cross,
    ):
        assert channel.status == "defined"
        assert channel.mature_object_count == 2
        assert channel.mean_nis == pytest.approx(0.0)
        assert channel.maximum_nis == pytest.approx(0.0)
        assert channel.score == 0.0


def test_nis_covariance_propagation_matches_closed_form() -> None:
    scorer = HealthScorer(_calibration((0.0, 1.0, 10.0, 20.0)))
    scorer.process_frame(_frame(0.0, camera_xy=(0.0, 0.0), lidar_xy=(0.0, 0.0)))
    scorer.process_frame(_frame(1.0, camera_xy=(1.0, 0.0), lidar_xy=(1.0, 0.0)))
    evidence = scorer.process_frame(_frame(2.0, camera_xy=(10.0, 0.0), lidar_xy=(2.0, 0.0)))

    # h=1, P^-=(1+h)^2 I+h^2 I=5I and S=P^-+R_current=6I.
    assert evidence.camera_self.mean_nis == pytest.approx(64.0 / 6.0)
    assert evidence.camera_self.maximum_nis == pytest.approx(64.0 / 6.0)
    assert evidence.camera_from_lidar_cross.mean_nis == pytest.approx(64.0 / 6.0)


def test_predictions_are_committed_before_current_measurements_update_either_history() -> None:
    scorer = HealthScorer(_calibration((0.0, 1.0, 10.0, 20.0)))
    scorer.process_frame(_frame(0.0, camera_xy=(0.0, 0.0), lidar_xy=(0.0, 0.0)))
    scorer.process_frame(_frame(1.0, camera_xy=(1.0, 0.0), lidar_xy=(1.0, 0.0)))
    current = scorer.process_frame(_frame(2.0, camera_xy=(10.0, 0.0), lidar_xy=(2.0, 0.0)))

    # The current camera outlier cannot contaminate the current lidar<-camera
    # prediction. Both histories still predict x=2 before either update.
    assert current.lidar_from_camera_cross.mean_nis == pytest.approx(0.0)
    following = scorer.process_frame(_frame(3.0, camera_xy=(3.0, 0.0), lidar_xy=(3.0, 0.0)))
    assert following.lidar_from_camera_cross.mean_nis is not None
    assert following.lidar_from_camera_cross.mean_nis > 0.0


def test_missing_measurements_do_not_update_independent_modality_history() -> None:
    scorer = HealthScorer(_calibration())
    scorer.process_frame(_frame(0.0, camera_xy=(0.0, 0.0), lidar_xy=(0.0, 0.0)))
    scorer.process_frame(_frame(1.0, camera_xy=(1.0, 0.0), lidar_xy=(1.0, 0.0)))
    missing = scorer.process_frame(
        _frame(
            2.0,
            camera_xy=None,
            lidar_xy=(2.0, 0.0),
            camera_available=False,
        )
    )
    recovered = scorer.process_frame(_frame(3.0, camera_xy=(3.0, 0.0), lidar_xy=(3.0, 0.0)))

    assert missing.camera_direct_suspicious
    assert missing.camera_self.status == "insufficient-support"
    assert recovered.camera_self.mean_nis == pytest.approx(0.0)
    assert recovered.camera_missing_fraction_last_four == pytest.approx(0.25)


def test_reported_timestamp_drives_only_direct_residual_not_prediction_clock() -> None:
    scorer = HealthScorer(_calibration())
    for time in (0.0, 1.0):
        scorer.process_frame(
            _frame(
                time,
                camera_xy=(time, 0.0),
                lidar_xy=(time, 0.0),
                camera_reported_time_s=time + 100.0,
            )
        )
    evidence = scorer.process_frame(
        _frame(
            2.0,
            camera_xy=(2.0, 0.0),
            lidar_xy=(2.0, 0.0),
            camera_reported_time_s=102.0,
        )
    )

    assert evidence.camera_timestamp_suspicious
    assert evidence.camera_self.mean_nis == pytest.approx(0.0)
    assert evidence.lidar_from_camera_cross.mean_nis == pytest.approx(0.0)


def test_scorer_rejects_noncausal_reference_time_order() -> None:
    scorer = HealthScorer(_calibration())
    scorer.process_frame(_frame(1.0, camera_xy=(0.0, 0.0), lidar_xy=(0.0, 0.0)))
    with pytest.raises(ValueError, match="strictly increasing"):
        scorer.process_frame(_frame(1.0, camera_xy=(1.0, 0.0), lidar_xy=(1.0, 0.0)))


def test_future_mutation_cannot_change_an_already_scored_prefix() -> None:
    first = HealthScorer(_calibration())
    second = HealthScorer(_calibration())
    prefix_first: list[HealthFrameEvidence] = []
    prefix_second: list[HealthFrameEvidence] = []
    for time in (0.0, 1.0, 2.0):
        frame = _frame(
            time,
            camera_xy=(time, 0.0),
            lidar_xy=(time, 0.0),
        )
        prefix_first.append(first.process_frame(frame))
        prefix_second.append(second.process_frame(frame))

    first.process_frame(_frame(3.0, camera_xy=(3.0, 0.0), lidar_xy=(3.0, 0.0)))
    second.process_frame(_frame(3.0, camera_xy=(3000.0, 0.0), lidar_xy=(-3000.0, 0.0)))

    assert prefix_first == prefix_second
    assert prefix_first[-1].camera_self.mean_nis == pytest.approx(0.0)


@pytest.mark.parametrize(
    (
        "method",
        "evidence",
        "expected_label",
        "expected_status",
    ),
    [
        (
            "direct-telemetry-gate",
            _evidence(camera_available=False),
            "camera-fault",
            "update-eligible",
        ),
        (
            "direct-telemetry-gate",
            _evidence(
                camera_available=False,
                lidar_timestamp_suspicious=True,
            ),
            "ambiguous",
            "update-eligible",
        ),
        (
            "self-nis-gate",
            _evidence(camera_self=0.9),
            "camera-fault",
            "update-eligible",
        ),
        (
            "self-nis-gate",
            _evidence(camera_self=0.9, lidar_self=0.9),
            "ambiguous",
            "update-eligible",
        ),
        (
            "cross-nis-gate",
            _evidence(camera_from_lidar=0.9),
            "ambiguous",
            "update-eligible",
        ),
        (
            "cross-nis-gate",
            _evidence(),
            "healthy",
            "update-eligible",
        ),
        (
            "self-nis-gate",
            _evidence(camera_self=None),
            "ambiguous",
            "insufficient-support",
        ),
        (
            "cross-nis-gate",
            _evidence(lidar_from_camera=None),
            "ambiguous",
            "insufficient-support",
        ),
        (
            "combined-health-gate",
            _evidence(camera_from_lidar=None),
            "ambiguous",
            "insufficient-support",
        ),
    ],
)
def test_exact_raw_decision_rules(
    method: str,
    evidence: HealthFrameEvidence,
    expected_label: str,
    expected_status: str,
) -> None:
    decision = decide_raw(
        method=method,  # type: ignore[arg-type]
        evidence=evidence,
        thresholds=HealthThresholds(self_score=0.5, cross_score=0.5),
    )
    assert decision.label == expected_label
    assert decision.evidence_status == expected_status


def test_combined_rule_prioritizes_direct_then_self_then_cross() -> None:
    thresholds = HealthThresholds(self_score=0.5, cross_score=0.5)
    direct = decide_raw(
        method="combined-health-gate",
        evidence=_evidence(
            camera_self=0.9,
            lidar_timestamp_suspicious=True,
        ),
        thresholds=thresholds,
    )
    self_before_cross = decide_raw(
        method="combined-health-gate",
        evidence=_evidence(
            camera_self=0.9,
            lidar_from_camera=0.9,
        ),
        thresholds=thresholds,
    )
    cross = decide_raw(
        method="combined-health-gate",
        evidence=_evidence(lidar_from_camera=0.9),
        thresholds=thresholds,
    )

    assert direct.label == "lidar-fault"
    assert self_before_cross.label == "camera-fault"
    assert cross.label == "ambiguous"


def test_direct_evidence_remains_eligible_when_numeric_support_is_insufficient() -> None:
    evidence = _evidence(
        camera_self=None,
        lidar_self=None,
        camera_from_lidar=None,
        lidar_from_camera=None,
        camera_available=False,
    )
    combined = decide_raw(
        method="combined-health-gate",
        evidence=evidence,
        thresholds=HealthThresholds(self_score=0.5, cross_score=0.5),
    )
    direct = decide_raw(
        method="direct-telemetry-gate",
        evidence=evidence,
        thresholds=HealthThresholds(self_score=0.5, cross_score=0.5),
    )

    assert combined.label == direct.label == "camera-fault"
    assert combined.evidence_status == direct.evidence_status == "update-eligible"


def test_threshold_one_disables_numeric_alarm_even_for_rank_one() -> None:
    decision = decide_raw(
        method="self-nis-gate",
        evidence=_evidence(camera_self=1.0, lidar_self=1.0),
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
    )
    assert decision.label == "healthy"


def test_healthy_raw_evidence_clears_an_activation_candidate() -> None:
    state = HealthLatchState(
        activation_candidate="camera-fault",
        activation_count=1,
    )
    assert advance_latch(state, _decision("healthy")) == HealthLatchState()


def test_activation_requires_two_identical_labels_and_contrary_label_restarts_count() -> None:
    state = advance_latch(HealthLatchState(), _decision("camera-fault"))
    assert state == HealthLatchState(
        activation_candidate="camera-fault",
        activation_count=1,
    )
    state = advance_latch(state, _decision("lidar-fault"))
    assert state == HealthLatchState(
        activation_candidate="lidar-fault",
        activation_count=1,
    )
    state = advance_latch(state, _decision("lidar-fault"))
    assert state == HealthLatchState(label="lidar-fault")


def test_nonhealthy_latch_never_switches_and_recovers_on_third_healthy_frame() -> None:
    state = HealthLatchState(label="camera-fault")
    state = advance_latch(state, _decision("lidar-fault"))
    assert state == HealthLatchState(label="camera-fault")
    state = advance_latch(state, _decision("healthy"))
    assert state.recovery_count == 1
    state = advance_latch(state, _decision("ambiguous"))
    assert state == HealthLatchState(label="camera-fault")
    for expected_count in (1, 2):
        state = advance_latch(state, _decision("healthy"))
        assert state.recovery_count == expected_count
    state = advance_latch(state, _decision("healthy"))
    assert state == HealthLatchState()


@pytest.mark.parametrize(
    "initial",
    [
        HealthLatchState(
            activation_candidate="camera-fault",
            activation_count=1,
        ),
        HealthLatchState(label="lidar-fault", recovery_count=2),
    ],
)
def test_insufficient_support_holds_every_counter_and_latched_state(
    initial: HealthLatchState,
) -> None:
    assert advance_latch(initial, _decision("ambiguous", status="insufficient-support")) is initial


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"label": "bogus"}, "unknown latched"),
        ({"activation_candidate": "healthy", "activation_count": 1}, "nonhealthy"),
        ({"activation_count": 2}, "zero or one"),
        ({"recovery_count": 3}, r"\{0, 1, 2\}"),
        ({"label": "healthy", "recovery_count": 1}, "recovery count"),
        ({"label": "healthy", "activation_candidate": "camera-fault"}, "inconsistent"),
        (
            {
                "label": "camera-fault",
                "activation_candidate": "lidar-fault",
                "activation_count": 1,
            },
            "cannot retain",
        ),
    ],
)
def test_latch_state_rejects_impossible_recurrence_states(
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HealthLatchState(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    (
        "camera_available",
        "lidar_available",
        "label",
        "abstain",
        "expected",
    ),
    [
        (False, False, "healthy", False, "undefined"),
        (True, False, "lidar-fault", False, "camera-only"),
        (False, True, "camera-fault", False, "lidar-only"),
        (True, True, "healthy", False, "fixed-fusion"),
        (True, True, "camera-fault", False, "lidar-only"),
        (True, True, "lidar-fault", False, "camera-only"),
        (True, True, "ambiguous", False, "fixed-fusion"),
        (True, True, "ambiguous", True, "undefined"),
    ],
)
def test_exact_executed_action_mapping(
    camera_available: bool,
    lidar_available: bool,
    label: str,
    abstain: bool,
    expected: str,
) -> None:
    assert (
        choose_executed_action(
            camera_available=camera_available,
            lidar_available=lidar_available,
            latched_label=label,  # type: ignore[arg-type]
            abstain_on_ambiguous=abstain,
        )
        == expected
    )


def test_action_and_raw_decision_reject_unknown_labels_or_methods() -> None:
    with pytest.raises(ValueError, match="unknown latched"):
        choose_executed_action(
            camera_available=True,
            lidar_available=True,
            latched_label="bogus",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown health method"):
        decide_raw(
            method="bogus",  # type: ignore[arg-type]
            evidence=_evidence(),
            thresholds=HealthThresholds(self_score=0.5, cross_score=0.5),
        )
    with pytest.raises(ValueError, match="unknown raw health label"):
        _decision("bogus")
    with pytest.raises(ValueError, match="unknown raw evidence status"):
        _decision("healthy", status="bogus")


def test_immediate_missing_input_action_is_not_detection() -> None:
    policy = HealthPolicy(
        method="direct-telemetry-gate",
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
    )
    first = policy.apply(_evidence(camera_available=False))
    second = policy.apply(_evidence(camera_available=False))

    assert first.raw_decision.label == "camera-fault"
    assert first.latched_state.label == "healthy"
    assert first.executed_action == "lidar-only"
    assert second.latched_state.label == "camera-fault"
    assert second.executed_action == "lidar-only"


def test_combined_abstaining_policy_only_abstains_after_ambiguous_latch() -> None:
    policy = HealthPolicy(
        method="combined-health-gate",
        thresholds=HealthThresholds(self_score=0.5, cross_score=0.5),
        abstain_on_ambiguous=True,
    )
    evidence = _evidence(camera_from_lidar=0.9)
    first = policy.apply(evidence)
    second = policy.apply(evidence)

    assert first.executed_action == "fixed-fusion"
    assert second.latched_state.label == "ambiguous"
    assert second.executed_action == "undefined"


def test_policy_rejects_unknown_method_and_abstention_on_noncombined_method() -> None:
    thresholds = HealthThresholds(self_score=0.5, cross_score=0.5)
    with pytest.raises(ValueError, match="unknown health method"):
        HealthPolicy(method="bogus", thresholds=thresholds)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only for the combined"):
        HealthPolicy(
            method="self-nis-gate",
            thresholds=thresholds,
            abstain_on_ambiguous=True,
        )
    policy = HealthPolicy(method="self-nis-gate", thresholds=thresholds)
    assert policy.state == HealthLatchState()


def test_observable_input_contract_has_no_scenario_or_fault_metadata_fields() -> None:
    assert {field.name for field in fields(HealthFrameInput)} == {
        "reference_time_s",
        "camera_available",
        "lidar_available",
        "objects",
    }
    assert {field.name for field in fields(ObjectHealthInput)} == {
        "object_id",
        "camera",
        "lidar",
    }
    assert {field.name for field in fields(ModalityMeasurement)} == {
        "value_xy_m",
        "reported_covariance_xy_m2",
        "reported_time_s",
    }
    prohibited = {
        "truth",
        "latent_velocity",
        "fault_family",
        "fault_target",
        "severity",
        "direction",
        "event_phase",
        "event_boundary",
        "seed",
        "split",
        "sequence_id",
        "manifest",
        "frame_index",
    }
    all_fields = {
        field.name
        for contract in (HealthFrameInput, ObjectHealthInput, ModalityMeasurement)
        for field in fields(contract)
    }
    assert prohibited.isdisjoint(all_fields)
