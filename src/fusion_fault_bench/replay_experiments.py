"""Paired estimator-output generation for both frozen M5 replay panels.

The same scene-level random draws feed M5-A and M5-B.  Physical camera proxy
generation always uses the true state, pose, and extrinsic.  Faulted metadata
is used only during reconstruction.  Localization values remain in the current
LiDAR-time ego BEV; the yaw-anchored scene copy is monitoring-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.replay_geometry import (
    ProjectedEstimate,
    RigidTransform3,
    SceneBevAnchor,
    calibration_perturbation,
    monitoring_scene_projection,
    persistent_panel_projection,
    reconstruct_camera,
    reconstruct_lidar,
    reported_camera_extrinsic,
)
from fusion_fault_bench.replay_source import ReplayObjectFrame, ReplayScene
from fusion_fault_bench.rng import draw_fault_uniforms, draw_standard_normal_xy

type FloatArray = npt.NDArray[np.float64]
type FaultFamily = Literal[
    "identity",
    "additive-position-bias",
    "increased-noise-underreported",
    "increased-noise-correctly-reported",
    "timestamp-offset",
    "dropout",
    "calibration-translation",
    "calibration-yaw",
    "common-mode-position-bias",
]
type FaultTarget = Literal["camera", "lidar", "both", "none"]

_CAMERA_STD_XY_M = np.asarray((1.0, 1.0), dtype=np.float64)
_LIDAR_STD_XY_M = np.asarray((0.3, 0.3), dtype=np.float64)
M5_DATA_MASTER_SEED = 1729


def _finite_array(
    value: npt.ArrayLike,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def _finite_scalar(value: float, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _selector_value(value: float, *, signed: bool) -> str:
    rendered = format(float(value), ".15g")
    if signed and value > 0.0:
        return f"+{rendered}"
    return rendered


@dataclass(frozen=True, slots=True)
class ReplayFaultCondition:
    """One exact replay fault coordinate and its active observation window."""

    experiment_id: str
    family: FaultFamily
    target: FaultTarget
    axis: str
    unit: str
    value: float
    identity: bool
    active_frames: tuple[int, int] | None

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must be nonempty")
        value = _finite_scalar(self.value, field_name="value")
        object.__setattr__(self, "value", value)
        if self.family == "identity":
            if not self.identity or self.target != "none" or value != 0.0:
                raise ValueError("identity family requires a zero targetless coordinate")
        elif self.identity:
            expected_identity = (
                1.0
                if self.family
                in {
                    "increased-noise-underreported",
                    "increased-noise-correctly-reported",
                }
                else 0.0
            )
            if value != expected_identity:
                raise ValueError("identity coordinate disagrees with its fault family")
        elif self.target == "none":
            raise ValueError("nonidentity replay faults require a target")
        if self.family == "common-mode-position-bias" and self.target != "both":
            raise ValueError("common-mode replay faults require target both")
        if self.family not in {"identity", "common-mode-position-bias"} and self.target not in {
            "camera",
            "lidar",
        }:
            raise ValueError("ordinary replay faults require one sensor target")
        if (
            self.family
            in {
                "increased-noise-underreported",
                "increased-noise-correctly-reported",
            }
            and value < 1.0
        ):
            raise ValueError("noise scales must be at least one")
        if self.family == "dropout" and not 0.0 <= value <= 1.0:
            raise ValueError("dropout probability must lie in [0, 1]")
        if self.active_frames is not None:
            start, end = self.active_frames
            if type(start) is not int or type(end) is not int or start < 0 or end <= start:
                raise ValueError("active_frames must be a nonempty half-open integer interval")

    @property
    def selector(self) -> str:
        """Return the exact public selector coordinate used by M5-B hypotheses."""

        signed = self.family in {
            "additive-position-bias",
            "timestamp-offset",
            "calibration-translation",
            "calibration-yaw",
            "common-mode-position-bias",
        }
        return f"{self.experiment_id}:{_selector_value(self.value, signed=signed)}"

    def fault_is_active(self, frame_index: int) -> bool:
        """Return whether the nonidentity effect is active at one frame."""

        if self.identity:
            return False
        if self.active_frames is None:
            return True
        start, end = self.active_frames
        return start <= frame_index < end


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ReplaySceneDraws:
    """One immutable draw set shared across panels, faults, and severities."""

    data_master_seed: int
    row_keys: tuple[tuple[int, str], ...]
    camera_standard_normal_xy: FloatArray
    lidar_standard_normal_xy: FloatArray
    dropout_uniform_by_frame: FloatArray

    def __post_init__(self) -> None:
        if self.data_master_seed != M5_DATA_MASTER_SEED:
            raise ValueError("replay draws must bind the frozen M5 data master seed")
        if not self.row_keys:
            raise ValueError("a replay draw set requires eligible object-frame support")
        if self.row_keys != tuple(
            sorted(self.row_keys, key=lambda item: (item[0], item[1].encode("utf-8")))
        ):
            raise ValueError("replay draw rows must use frame/object order")
        if len(set(self.row_keys)) != len(self.row_keys):
            raise ValueError("replay draw row keys must be unique")
        row_count = len(self.row_keys)
        object.__setattr__(
            self,
            "camera_standard_normal_xy",
            _finite_array(
                self.camera_standard_normal_xy,
                shape=(row_count, 2),
                field_name="camera_standard_normal_xy",
            ),
        )
        object.__setattr__(
            self,
            "lidar_standard_normal_xy",
            _finite_array(
                self.lidar_standard_normal_xy,
                shape=(row_count, 2),
                field_name="lidar_standard_normal_xy",
            ),
        )
        raw_uniforms = np.asarray(self.dropout_uniform_by_frame, dtype=np.float64)
        if raw_uniforms.ndim != 1 or raw_uniforms.size == 0:
            raise ValueError("dropout_uniform_by_frame must be a nonempty vector")
        if not bool(np.all(np.isfinite(raw_uniforms))) or bool(
            np.any((raw_uniforms < 0.0) | (raw_uniforms >= 1.0))
        ):
            raise ValueError("dropout uniforms must lie in [0, 1)")
        object.__setattr__(
            self,
            "dropout_uniform_by_frame",
            immutable_float64_copy(raw_uniforms),
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ReplayObjectEstimate:
    """One eligible known-object estimate in localization and monitoring frames."""

    object_id: str
    truth_current_ego_xy_m: FloatArray
    camera_current_ego: ProjectedEstimate
    lidar_current_ego: ProjectedEstimate
    fixed_current_ego_xy_m: FloatArray
    fixed_reported_covariance_m2: FloatArray
    camera_monitoring_scene: ProjectedEstimate
    lidar_monitoring_scene: ProjectedEstimate
    camera_reported_state_time_s: float
    lidar_reported_state_time_s: float

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be nonempty")
        object.__setattr__(
            self,
            "truth_current_ego_xy_m",
            _finite_array(
                self.truth_current_ego_xy_m,
                shape=(2,),
                field_name="truth_current_ego_xy_m",
            ),
        )
        object.__setattr__(
            self,
            "fixed_current_ego_xy_m",
            _finite_array(
                self.fixed_current_ego_xy_m,
                shape=(2,),
                field_name="fixed_current_ego_xy_m",
            ),
        )
        object.__setattr__(
            self,
            "fixed_reported_covariance_m2",
            _finite_array(
                self.fixed_reported_covariance_m2,
                shape=(2, 2),
                field_name="fixed_reported_covariance_m2",
            ),
        )
        for field_name in (
            "camera_reported_state_time_s",
            "lidar_reported_state_time_s",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_scalar(getattr(self, field_name), field_name=field_name),
            )


@dataclass(frozen=True, slots=True, repr=False)
class ReplayEstimateFrame:
    """One replay frame, including exact frame-level availability."""

    frame_index: int
    reference_time_s: float
    camera_available: bool
    lidar_available: bool
    objects: tuple[ReplayObjectEstimate, ...]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        object.__setattr__(
            self,
            "reference_time_s",
            _finite_scalar(self.reference_time_s, field_name="reference_time_s"),
        )
        identifiers = tuple(item.object_id for item in self.objects)
        if identifiers != tuple(sorted(identifiers, key=lambda value: value.encode("utf-8"))):
            raise ValueError("estimated objects must use canonical opaque-ID order")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("estimated objects must be unique within a frame")


@dataclass(frozen=True, slots=True, repr=False)
class ReplayEstimateSequence:
    """One complete condition realization for a replay scene."""

    sequence_id: str
    condition: ReplayFaultCondition
    frames: tuple[ReplayEstimateFrame, ...]

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must be nonempty")
        if not self.frames:
            raise ValueError("replay estimate sequence must be nonempty")
        if tuple(frame.frame_index for frame in self.frames) != tuple(range(len(self.frames))):
            raise ValueError("estimated frame indices must be contiguous")
        reference_times = tuple(frame.reference_time_s for frame in self.frames)
        if reference_times[0] != 0.0 or any(
            right <= left for left, right in pairwise(reference_times)
        ):
            raise ValueError("estimated reference times must start at zero and increase")

    @property
    def eligible_object_frame_count(self) -> int:
        """Return the frozen pre-fault support count."""

        return sum(len(frame.objects) for frame in self.frames)


def draw_replay_scene_randomness(
    scene: ReplayScene,
) -> ReplaySceneDraws:
    """Draw the exact paired camera, LiDAR, and frame-dropout streams."""

    row_keys = tuple(
        (frame.frame_index, item.object_id)
        for frame in scene.frames
        for item in frame.eligible_objects
    )
    if not row_keys:
        raise ValueError("replay scene has no eligible object-frame support")
    return ReplaySceneDraws(
        data_master_seed=M5_DATA_MASTER_SEED,
        row_keys=row_keys,
        camera_standard_normal_xy=draw_standard_normal_xy(
            data_master_seed=M5_DATA_MASTER_SEED,
            stream_name="camera",
            sequence_id=scene.sequence_id,
            object_frame_count=len(row_keys),
        ),
        lidar_standard_normal_xy=draw_standard_normal_xy(
            data_master_seed=M5_DATA_MASTER_SEED,
            stream_name="lidar",
            sequence_id=scene.sequence_id,
            object_frame_count=len(row_keys),
        ),
        dropout_uniform_by_frame=draw_fault_uniforms(
            data_master_seed=M5_DATA_MASTER_SEED,
            sequence_id=scene.sequence_id,
            frame_count=len(scene.frames),
        ),
    )


def _fault_axis_vector(condition: ReplayFaultCondition) -> FloatArray:
    result = np.zeros(2, dtype=np.float64)
    if condition.axis == "x":
        result[0] = condition.value
    elif condition.axis == "y":
        result[1] = condition.value
    elif condition.family not in {
        "identity",
        "increased-noise-underreported",
        "increased-noise-correctly-reported",
        "timestamp-offset",
        "dropout",
        "calibration-yaw",
    }:
        raise ValueError("replay position fault uses an unsupported axis")
    return immutable_float64_copy(result)


def _base_covariance(std_xy_m: FloatArray, *, scale: float) -> FloatArray:
    variances = np.square(std_xy_m * scale)
    return immutable_float64_copy(np.diag(variances))


def _fuse_full_information(
    first: ProjectedEstimate,
    second: ProjectedEstimate,
) -> tuple[FloatArray, FloatArray]:
    first_information = np.linalg.inv(first.reported_covariance_m2)
    second_information = np.linalg.inv(second.reported_covariance_m2)
    covariance = np.linalg.inv(first_information + second_information)
    value = covariance @ (first_information @ first.point_m + second_information @ second.point_m)
    if (
        not bool(np.all(np.isfinite(value)))
        or not bool(np.all(np.isfinite(covariance)))
        or not bool(np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12))
        or float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0
    ):
        raise ValueError("replay information fusion produced an invalid result")
    return immutable_float64_copy(value), immutable_float64_copy(covariance)


def _availability(
    condition: ReplayFaultCondition,
    *,
    frame_index: int,
    dropout_uniform: float,
) -> tuple[bool, bool]:
    camera_available = True
    lidar_available = True
    if condition.family == "dropout" and condition.fault_is_active(frame_index):
        available = dropout_uniform >= condition.value
        if condition.target == "camera":
            camera_available = available
        elif condition.target == "lidar":
            lidar_available = available
        else:
            raise ValueError("dropout requires a single target")
    return camera_available, lidar_available


def _estimate_object(
    source: ReplayObjectFrame,
    *,
    condition: ReplayFaultCondition,
    frame_index: int,
    reference_time_s: float,
    camera_time_s: float,
    camera_standard_normal_xy: FloatArray,
    lidar_standard_normal_xy: FloatArray,
    global_from_reference_ego: RigidTransform3,
    global_from_camera_ego: RigidTransform3,
    true_camera_ego_from_camera: RigidTransform3,
    scene_anchor: SceneBevAnchor,
) -> ReplayObjectEstimate:
    active = condition.fault_is_active(frame_index)
    camera_actual_scale = 1.0
    lidar_actual_scale = 1.0
    camera_reported_scale = 1.0
    lidar_reported_scale = 1.0
    if active and condition.family in {
        "increased-noise-underreported",
        "increased-noise-correctly-reported",
    }:
        if condition.target == "camera":
            camera_actual_scale = condition.value
            if condition.family == "increased-noise-correctly-reported":
                camera_reported_scale = condition.value
        elif condition.target == "lidar":
            lidar_actual_scale = condition.value
            if condition.family == "increased-noise-correctly-reported":
                lidar_reported_scale = condition.value

    camera_bias = np.zeros(2, dtype=np.float64)
    lidar_bias = np.zeros(2, dtype=np.float64)
    axis_bias = _fault_axis_vector(condition) if active else np.zeros(2, dtype=np.float64)
    if active and condition.family == "additive-position-bias":
        if condition.target == "camera":
            camera_bias = np.asarray(axis_bias, dtype=np.float64)
        elif condition.target == "lidar":
            lidar_bias = np.asarray(axis_bias, dtype=np.float64)
    elif active and condition.family == "common-mode-position-bias":
        camera_bias = np.asarray(axis_bias, dtype=np.float64)
        lidar_bias = np.asarray(axis_bias, dtype=np.float64)

    timestamp_fault_s = (
        condition.value if active and condition.family == "timestamp-offset" else 0.0
    )
    camera_timestamp_fault_s = timestamp_fault_s if condition.target == "camera" else 0.0
    lidar_timestamp_fault_s = timestamp_fault_s if condition.target == "lidar" else 0.0

    perturbation = RigidTransform3.identity()
    if active and condition.family == "calibration-translation":
        translation = np.zeros(3, dtype=np.float64)
        translation[0 if condition.axis == "x" else 1] = condition.value
        perturbation = calibration_perturbation(
            translation_camera_ego_m=translation,
        )
    elif active and condition.family == "calibration-yaw":
        perturbation = calibration_perturbation(
            yaw_camera_ego_rad=condition.value,
        )
    reported_extrinsic = reported_camera_extrinsic(
        true_camera_ego_from_camera=true_camera_ego_from_camera,
        perturbation_camera_ego=perturbation,
    )

    camera_reconstruction = reconstruct_camera(
        truth_global_at_reference_m=source.center_global_m,
        velocity_global_mps=source.velocity_global_mps,
        reference_time_s=reference_time_s,
        camera_time_s=camera_time_s,
        base_error_reference_bev_m=(
            camera_standard_normal_xy * _CAMERA_STD_XY_M * camera_actual_scale
        ),
        timestamp_fault_s=camera_timestamp_fault_s,
        additive_bias_reference_bev_m=camera_bias,
        global_from_reference_ego=global_from_reference_ego,
        global_from_camera_ego=global_from_camera_ego,
        true_camera_ego_from_camera=true_camera_ego_from_camera,
        reported_camera_ego_from_camera=reported_extrinsic,
    )
    lidar_reconstruction = reconstruct_lidar(
        truth_global_at_reference_m=source.center_global_m,
        velocity_global_mps=source.velocity_global_mps,
        reference_time_s=reference_time_s,
        base_error_reference_bev_m=(
            lidar_standard_normal_xy * _LIDAR_STD_XY_M * lidar_actual_scale
        ),
        timestamp_fault_s=lidar_timestamp_fault_s,
        additive_bias_reference_bev_m=lidar_bias,
        global_from_reference_ego=global_from_reference_ego,
    )
    camera_base_covariance = _base_covariance(
        _CAMERA_STD_XY_M,
        scale=camera_reported_scale,
    )
    lidar_base_covariance = _base_covariance(
        _LIDAR_STD_XY_M,
        scale=lidar_reported_scale,
    )
    camera_current = persistent_panel_projection(
        camera_reconstruction,
        reported_base_covariance_m2=camera_base_covariance,
    )
    lidar_current = persistent_panel_projection(
        lidar_reconstruction,
        reported_base_covariance_m2=lidar_base_covariance,
    )
    fixed_value, fixed_covariance = _fuse_full_information(
        camera_current,
        lidar_current,
    )
    camera_monitoring = monitoring_scene_projection(
        camera_reconstruction,
        reported_base_covariance_m2=camera_base_covariance,
        global_from_current_ego=global_from_reference_ego,
        scene_anchor=scene_anchor,
    )
    lidar_monitoring = monitoring_scene_projection(
        lidar_reconstruction,
        reported_base_covariance_m2=lidar_base_covariance,
        global_from_current_ego=global_from_reference_ego,
        scene_anchor=scene_anchor,
    )
    return ReplayObjectEstimate(
        object_id=source.object_id,
        truth_current_ego_xy_m=source.support.center_reference_ego_m[:2],
        camera_current_ego=camera_current,
        lidar_current_ego=lidar_current,
        fixed_current_ego_xy_m=fixed_value,
        fixed_reported_covariance_m2=fixed_covariance,
        camera_monitoring_scene=camera_monitoring,
        lidar_monitoring_scene=lidar_monitoring,
        camera_reported_state_time_s=camera_reconstruction.reported_state_time_s,
        lidar_reported_state_time_s=lidar_reconstruction.reported_state_time_s,
    )


def generate_replay_condition_sequence(
    scene: ReplayScene,
    *,
    condition: ReplayFaultCondition,
    draws: ReplaySceneDraws,
) -> ReplayEstimateSequence:
    """Generate one complete, paired replay condition for a recorded scene."""

    if len(draws.dropout_uniform_by_frame) != len(scene.frames):
        raise ValueError("replay draw frame count does not match the scene")
    expected_row_keys = tuple(
        (frame.frame_index, item.object_id)
        for frame in scene.frames
        for item in frame.eligible_objects
    )
    if draws.row_keys != expected_row_keys:
        raise ValueError("replay draw rows do not match the frozen scene support")
    if condition.active_frames is not None and condition.active_frames[1] > len(scene.frames):
        raise ValueError("fault active window exceeds the replay scene")

    scene_anchor = SceneBevAnchor.from_first_reference_pose(scene.frames[0].lidar.global_from_ego)
    draw_index = 0
    frames: list[ReplayEstimateFrame] = []
    for frame in scene.frames:
        camera_available, lidar_available = _availability(
            condition,
            frame_index=frame.frame_index,
            dropout_uniform=float(draws.dropout_uniform_by_frame[frame.frame_index]),
        )
        camera_time_s = (
            frame.reference_time_s
            + (frame.camera.timestamp_us - frame.lidar.timestamp_us) / 1_000_000.0
        )
        estimates: list[ReplayObjectEstimate] = []
        for source in frame.eligible_objects:
            estimates.append(
                _estimate_object(
                    source,
                    condition=condition,
                    frame_index=frame.frame_index,
                    reference_time_s=frame.reference_time_s,
                    camera_time_s=camera_time_s,
                    camera_standard_normal_xy=draws.camera_standard_normal_xy[draw_index],
                    lidar_standard_normal_xy=draws.lidar_standard_normal_xy[draw_index],
                    global_from_reference_ego=frame.lidar.global_from_ego,
                    global_from_camera_ego=frame.camera.global_from_ego,
                    true_camera_ego_from_camera=frame.camera.ego_from_sensor,
                    scene_anchor=scene_anchor,
                )
            )
            draw_index += 1
        frames.append(
            ReplayEstimateFrame(
                frame_index=frame.frame_index,
                reference_time_s=frame.reference_time_s,
                camera_available=camera_available,
                lidar_available=lidar_available,
                objects=tuple(estimates),
            )
        )
    if draw_index != len(draws.row_keys):
        raise AssertionError("replay draw consumption is incomplete")
    return ReplayEstimateSequence(
        sequence_id=scene.sequence_id,
        condition=condition,
        frames=tuple(frames),
    )
