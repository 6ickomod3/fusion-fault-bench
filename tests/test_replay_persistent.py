from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import fusion_fault_bench.replay_persistent as replay_persistent
from fusion_fault_bench.replay_experiments import (
    ReplayEstimateFrame,
    ReplayEstimateSequence,
    ReplayObjectEstimate,
)
from fusion_fault_bench.replay_geometry import ProjectedEstimate
from fusion_fault_bench.replay_persistent import (
    ReplayPersistentMethodResult,
    evaluate_replay_persistent_sequence,
)
from fusion_fault_bench.replay_plan import ReplayPersistentCase, load_replay_plan

ROOT = Path(__file__).resolve().parents[1]


def _projected(point: tuple[float, float]) -> ProjectedEstimate:
    return ProjectedEstimate(
        point_m=np.asarray(point, dtype=np.float64),
        jacobian=np.eye(2, dtype=np.float64),
        reported_covariance_m2=np.eye(2, dtype=np.float64),
    )


def _case(experiment_id: str, selector: str) -> ReplayPersistentCase:
    plan = load_replay_plan(source_root=ROOT)
    return next(
        case
        for case in plan.persistent_cases
        if case.identity.experiment_id == experiment_id
        and case.fault_condition.selector == selector
    )


def _sequence(
    case: ReplayPersistentCase,
    *,
    availability: tuple[tuple[bool, bool], ...] = ((True, True), (True, True)),
) -> ReplayEstimateSequence:
    frames: list[ReplayEstimateFrame] = []
    for frame_index, (camera_available, lidar_available) in enumerate(availability):
        truth = np.asarray((10.0 + frame_index, 0.0), dtype=np.float64)
        frames.append(
            ReplayEstimateFrame(
                frame_index=frame_index,
                reference_time_s=float(frame_index),
                camera_available=camera_available,
                lidar_available=lidar_available,
                objects=(
                    ReplayObjectEstimate(
                        object_id="track:00",
                        truth_current_ego_xy_m=truth,
                        camera_current_ego=_projected((truth[0] + 2.0, truth[1])),
                        lidar_current_ego=_projected((truth[0] + 1.0, truth[1])),
                        fixed_current_ego_xy_m=np.asarray(
                            (truth[0] + 0.5, truth[1]),
                            dtype=np.float64,
                        ),
                        fixed_reported_covariance_m2=np.eye(2, dtype=np.float64),
                        camera_monitoring_scene=_projected((truth[0] + 2.0, truth[1])),
                        lidar_monitoring_scene=_projected((truth[0] + 1.0, truth[1])),
                        camera_reported_state_time_s=float(frame_index),
                        lidar_reported_state_time_s=float(frame_index),
                    ),
                ),
            )
        )
    return ReplayEstimateSequence(
        sequence_id="nuscenes:scene-0061",
        condition=case.fault_condition,
        frames=tuple(frames),
    )


def test_persistent_single_target_preserves_m3_target_drop_and_oracle() -> None:
    case = _case("replay-lidar-y-bias", "replay-lidar-y-bias:+4")
    evaluated = evaluate_replay_persistent_sequence(case, _sequence(case))

    assert tuple(row.method for row in evaluated.results) == tuple(case.source_manifest.methods)
    assert evaluated.result("camera-only").conditional_loss_m2 == 4.0
    assert evaluated.result("lidar-only").conditional_loss_m2 == 1.0
    assert evaluated.result("fixed-fusion").conditional_loss_m2 == 0.25
    assert evaluated.result("fault-target-drop-policy").conditional_loss_m2 == 4.0
    assert evaluated.result("performance-oracle").conditional_loss_m2 == 0.25


def test_persistent_identity_target_drop_uses_fixed_fusion() -> None:
    case = _case("replay-lidar-y-bias", "replay-lidar-y-bias:0")
    evaluated = evaluate_replay_persistent_sequence(case, _sequence(case))

    assert evaluated.result("fault-target-drop-policy").conditional_loss_m2 == 0.25


def test_persistent_dropout_keeps_eligible_denominator_and_missing_output() -> None:
    case = _case("replay-camera-dropout", "replay-camera-dropout:1")
    evaluated = evaluate_replay_persistent_sequence(
        case,
        _sequence(case, availability=((False, True), (False, True))),
    )

    camera = evaluated.result("camera-only")
    fixed = evaluated.result("fixed-fusion")
    target_drop = evaluated.result("fault-target-drop-policy")
    assert camera.coverage == 0.0
    assert camera.conditional_loss_m2 is None
    assert fixed.metric("undefined-output-rate") == 1.0
    assert fixed.conditional_loss_m2 is None
    assert target_drop.coverage == 1.0
    assert target_drop.conditional_loss_m2 == 1.0


def test_persistent_common_mode_has_no_target_drop_or_oracle() -> None:
    case = _case("replay-common-mode-x", "replay-common-mode-x:+4")
    evaluated = evaluate_replay_persistent_sequence(case, _sequence(case))

    assert tuple(row.method for row in evaluated.results) == (
        "camera-only",
        "lidar-only",
        "fixed-fusion",
    )


def test_persistent_sufficient_statistics_reject_invalid_support_and_coordinates() -> None:
    valid = ReplayPersistentMethodResult(
        method="camera-only",
        loss_sum_m2=2.0,
        valid_object_frame_count=2,
        eligible_object_frame_count=2,
    )
    assert valid.coverage == 1.0
    assert valid.conditional_loss_m2 == 1.0
    assert valid.metric("matched-center-mse") == 1.0
    assert valid.metric("conditional-matched-center-mse") == 1.0
    assert valid.metric("coverage") == 1.0
    assert valid.metric("undefined-output-rate") == 0.0
    assert (
        ReplayPersistentMethodResult(
            method="camera-only",
            loss_sum_m2=0.0,
            valid_object_frame_count=0,
            eligible_object_frame_count=2,
        ).conditional_loss_m2
        is None
    )
    for update in (
        {"loss_sum_m2": float("nan")},
        {"loss_sum_m2": -1.0},
        {"valid_object_frame_count": True},
        {"eligible_object_frame_count": True},
        {"valid_object_frame_count": -1},
        {"eligible_object_frame_count": 0},
        {"valid_object_frame_count": 3},
        {"valid_object_frame_count": 0, "loss_sum_m2": 1.0},
    ):
        with pytest.raises(ValueError):
            ReplayPersistentMethodResult(
                method="camera-only",
                loss_sum_m2=update.get("loss_sum_m2", 2.0),  # type: ignore[arg-type]
                valid_object_frame_count=update.get(  # type: ignore[arg-type]
                    "valid_object_frame_count", 2
                ),
                eligible_object_frame_count=update.get(  # type: ignore[arg-type]
                    "eligible_object_frame_count", 2
                ),
            )

    case = _case("replay-lidar-y-bias", "replay-lidar-y-bias:+4")
    evaluated = evaluate_replay_persistent_sequence(case, _sequence(case))
    with pytest.raises(KeyError):
        evaluated.result("combined-health-gate")
    for update in (
        {"replay_experiment_identity_sha256": "short"},
        {"sequence_id": ""},
        {"condition_id": ""},
        {"condition_selector": ""},
        {"results": ()},
        {"results": (evaluated.results[0], evaluated.results[0])},
        {
            "results": (
                evaluated.results[0],
                replace(evaluated.results[1], eligible_object_frame_count=3),
                *evaluated.results[2:],
            )
        },
        {"cross_modal_disagreement_sum_m2": float("nan")},
        {"cross_modal_disagreement_sum_m2": -1.0},
        {"cross_modal_common_count": True},
        {"cross_modal_common_count": -1},
        {"cross_modal_common_count": 3},
        {"cross_modal_common_count": 0, "cross_modal_disagreement_sum_m2": 1.0},
    ):
        with pytest.raises(ValueError):
            replace(evaluated, **update)


def test_persistent_evaluation_rejects_fault_support_and_method_mismatches() -> None:
    case = _case("replay-lidar-y-bias", "replay-lidar-y-bias:+4")
    sequence = _sequence(case)
    with pytest.raises(ValueError, match="wrong fault coordinate"):
        evaluate_replay_persistent_sequence(
            _case("replay-lidar-y-bias", "replay-lidar-y-bias:0"),
            sequence,
        )

    active_condition = replace(case.fault_condition, active_frames=(0, 1))
    with pytest.raises(ValueError, match="complete scene"):
        evaluate_replay_persistent_sequence(
            replace(case, fault_condition=active_condition),
            replace(sequence, condition=active_condition),
        )

    empty_sequence = replace(
        sequence,
        frames=tuple(replace(frame, objects=()) for frame in sequence.frames),
    )
    with pytest.raises(ValueError, match="nonempty frozen base support"):
        evaluate_replay_persistent_sequence(case, empty_sequence)

    with pytest.raises(ValueError, match="performance oracle"):
        evaluate_replay_persistent_sequence(
            case,
            _sequence(case, availability=((True, False), (True, False))),
        )

    common = _case("replay-common-mode-x", "replay-common-mode-x:+4")
    with pytest.raises(ValueError, match="non-dropout"):
        evaluate_replay_persistent_sequence(
            common,
            _sequence(common, availability=((False, True), (False, True))),
        )

    item = sequence.frames[0].objects[0]
    with pytest.raises(ValueError, match="requires a base method"):
        replay_persistent._value(item, "combined-health-gate")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a base method"):
        replay_persistent._available(  # type: ignore[arg-type]
            sequence.frames[0],
            "combined-health-gate",
        )
