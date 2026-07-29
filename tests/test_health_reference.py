from __future__ import annotations

import numpy as np
import pytest

from fusion_fault_bench.health import (
    HealthCalibration,
    HealthFrameInput,
    HealthLatchState,
    HealthScorer,
    ModalityMeasurement,
    ObjectHealthInput,
    RawHealthDecision,
    advance_latch,
)
from fusion_fault_bench.health import (
    ecdf_rank as production_ecdf_rank,
)
from fusion_fault_bench.reference.health import (
    constant_velocity_prediction,
    ecdf_rank,
    frame_oracle_action,
    latch_trace,
    nis,
)


def _calibration() -> HealthCalibration:
    values = np.asarray([0.0, 1.0], dtype=np.float64)
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


def _measurement(x: float, time_s: float) -> ModalityMeasurement:
    return ModalityMeasurement(
        value_xy_m=np.asarray([x, 0.0]),
        reported_covariance_xy_m2=np.eye(2),
        reported_time_s=time_s,
    )


def _frame(x: float, time_s: float) -> HealthFrameInput:
    return HealthFrameInput(
        reference_time_s=time_s,
        camera_available=True,
        lidar_available=True,
        objects=tuple(
            ObjectHealthInput(
                object_id=f"object:{index}",
                camera=_measurement(x, time_s),
                lidar=_measurement(x, time_s),
            )
            for index in range(2)
        ),
    )


def test_scalar_predictor_and_nis_match_production_trace() -> None:
    prediction, covariance = constant_velocity_prediction(
        first_value_xy=[0.0, 0.0],
        first_covariance_xy=np.eye(2),
        first_time_s=0.0,
        second_value_xy=[1.0, 0.0],
        second_covariance_xy=np.eye(2),
        second_time_s=0.1,
        reference_time_s=0.2,
    )
    assert np.array_equal(prediction, [2.0, 0.0])
    assert np.array_equal(covariance, 5.0 * np.eye(2))
    expected_nis = nis(
        current_value_xy=[2.5, 0.0],
        current_covariance_xy=np.eye(2),
        predicted_value_xy=prediction,
        predicted_covariance_xy=covariance,
    )

    scorer = HealthScorer(_calibration())
    scorer.process_frame(_frame(0.0, 0.0))
    scorer.process_frame(_frame(1.0, 0.1))
    evidence = scorer.process_frame(_frame(2.5, 0.2))
    assert evidence.camera_self.mean_nis == pytest.approx(expected_nis)
    assert evidence.camera_self.maximum_nis == pytest.approx(expected_nis)


def test_reference_ecdf_matches_production_at_ties_and_endpoints() -> None:
    clean = [0.0, 1.0, 1.0, 3.0]
    for value in (-1.0, 0.0, 1.0, 2.0, 4.0):
        assert ecdf_rank(clean, value) == production_ecdf_rank(clean, value)
    with pytest.raises(ValueError):
        ecdf_rank([], 1.0)


def test_scalar_latch_trace_matches_production_recurrence() -> None:
    labels = (
        "camera-fault",
        "lidar-fault",
        "lidar-fault",
        "healthy",
        "ambiguous",
        "healthy",
        "healthy",
        "healthy",
    )
    statuses = (
        "update-eligible",
        "update-eligible",
        "update-eligible",
        "insufficient-support",
        "update-eligible",
        "update-eligible",
        "update-eligible",
        "update-eligible",
    )
    state = HealthLatchState()
    production: list[str] = []
    for label, status in zip(labels, statuses, strict=True):
        state = advance_latch(
            state,
            RawHealthDecision(
                method="combined-health-gate",
                label=label,
                evidence_status=status,
                camera_alarm=None,
                lidar_alarm=None,
                any_cross_alarm=None,
            ),
        )
        production.append(state.label)
    assert latch_trace(labels, statuses) == tuple(production)
    with pytest.raises(ValueError):
        latch_trace(("healthy",), ())


def test_frame_oracle_is_one_action_and_uses_declared_tie_order() -> None:
    truth = np.zeros((2, 2))
    assert (
        frame_oracle_action(
            truth_xy=truth,
            camera_xy=np.ones((2, 2)),
            lidar_xy=np.ones((2, 2)),
            fixed_xy=np.ones((2, 2)),
        )
        == "camera-only"
    )
    assert (
        frame_oracle_action(
            truth_xy=truth,
            camera_xy=None,
            lidar_xy=2.0 * np.ones((2, 2)),
            fixed_xy=np.ones((2, 2)),
        )
        == "fixed-fusion"
    )
    with pytest.raises(ValueError):
        frame_oracle_action(
            truth_xy=truth,
            camera_xy=None,
            lidar_xy=None,
            fixed_xy=None,
        )
    with pytest.raises(ValueError):
        frame_oracle_action(
            truth_xy=np.zeros(2),
            camera_xy=np.zeros((1, 2)),
            lidar_xy=None,
            fixed_xy=None,
        )
    with pytest.raises(ValueError):
        frame_oracle_action(
            truth_xy=truth,
            camera_xy=np.zeros((1, 2)),
            lidar_xy=None,
            fixed_xy=None,
        )


def test_reference_rejects_bad_prediction_and_nis_shapes() -> None:
    with pytest.raises(ValueError):
        constant_velocity_prediction(
            first_value_xy=[0.0],
            first_covariance_xy=np.eye(2),
            first_time_s=0.0,
            second_value_xy=[1.0, 0.0],
            second_covariance_xy=np.eye(2),
            second_time_s=0.1,
            reference_time_s=0.2,
        )
    with pytest.raises(ValueError):
        constant_velocity_prediction(
            first_value_xy=[0.0, 0.0],
            first_covariance_xy=np.eye(2),
            first_time_s=0.1,
            second_value_xy=[1.0, 0.0],
            second_covariance_xy=np.eye(2),
            second_time_s=0.1,
            reference_time_s=0.2,
        )
    with pytest.raises(ValueError):
        nis(
            current_value_xy=[0.0],
            current_covariance_xy=np.eye(2),
            predicted_value_xy=[0.0, 0.0],
            predicted_covariance_xy=np.eye(2),
        )
