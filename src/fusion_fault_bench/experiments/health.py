"""Transient M4 estimator-output observations in persistent scene BEV.

Physical proxy observations are always generated from true state, true time,
and the true camera extrinsic.  Calibration and timing faults affect only the
metadata consumed by reconstruction/alignment.  The resulting scorer adapter
intentionally omits truth, actual covariance, sequence identity, and all fault
metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.health import (
    HealthFrameInput,
    ModalityMeasurement,
    ObjectHealthInput,
)
from fusion_fault_bench.scenarios.health import (
    HealthBaseSequence,
    HealthFaultSpec,
    health_event_schedule,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

_CAMERA_STD_XY_M = np.asarray((1.0, 1.0), dtype=np.float64)
_LIDAR_STD_XY_M = np.asarray((0.3, 0.3), dtype=np.float64)
_SYMMETRY_TOLERANCE = 1e-12


def _immutable_int64(value: npt.ArrayLike) -> IntArray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int64).reshape(array.shape)


def _immutable_bool(value: npt.ArrayLike) -> BoolArray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.bool_))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(array.shape)


def _immutable_finite(
    value: npt.ArrayLike,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def _immutable_covariances(
    value: npt.ArrayLike,
    *,
    frame_count: int,
    object_count: int,
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    expected_shape = (frame_count, object_count, 2, 2)
    if array.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    if not np.allclose(
        array,
        np.swapaxes(array, -1, -2),
        rtol=0.0,
        atol=_SYMMETRY_TOLERANCE,
    ):
        raise ValueError(f"{field_name} must contain symmetric matrices")
    symmetric = np.asarray(
        (array + np.swapaxes(array, -1, -2)) / 2.0,
        dtype=np.float64,
    )
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{field_name} must contain positive-definite matrices") from error
    return immutable_float64_copy(symmetric)


def _diagonal_covariance_grid(
    std_xy_m: FloatArray,
    scale_by_frame: FloatArray,
    *,
    object_count: int,
) -> FloatArray:
    frame_count = scale_by_frame.size
    variances = np.square(std_xy_m)[np.newaxis, :] * np.square(scale_by_frame)[:, np.newaxis]
    result = np.zeros((frame_count, object_count, 2, 2), dtype=np.float64)
    result[:, :, 0, 0] = variances[:, 0, np.newaxis]
    result[:, :, 1, 1] = variances[:, 1, np.newaxis]
    return immutable_float64_copy(result)


@dataclass(frozen=True, slots=True)
class HealthObservationSequence:
    """Immutable observations plus evaluation-only truth and covariance roles."""

    sequence_id: str
    object_ids: tuple[str, ...]
    frame_indices: IntArray
    reference_times_s: FloatArray
    truth_xy_m: FloatArray
    eligibility_mask: BoolArray
    camera_value_xy_m: FloatArray
    lidar_value_xy_m: FloatArray
    camera_actual_covariance_xy_m2: FloatArray
    lidar_actual_covariance_xy_m2: FloatArray
    camera_reported_covariance_xy_m2: FloatArray
    lidar_reported_covariance_xy_m2: FloatArray
    camera_available: BoolArray
    lidar_available: BoolArray
    camera_reported_times_s: FloatArray
    lidar_reported_times_s: FloatArray
    health_frame: Literal["persistent-scene-bev"] = "persistent-scene-bev"

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must be nonempty")
        if self.health_frame != "persistent-scene-bev":
            raise ValueError("M4 observations require persistent scene BEV")
        frame_count = len(self.frame_indices)
        object_count = len(self.object_ids)
        if frame_count == 0 or object_count == 0:
            raise ValueError("health observation sequences must be nonempty")
        if len(set(self.object_ids)) != object_count or self.object_ids != tuple(
            sorted(self.object_ids, key=lambda item: item.encode("utf-8"))
        ):
            raise ValueError("object IDs must be unique and in UTF-8 byte order")

        raw_indices = np.asarray(self.frame_indices)
        if raw_indices.shape != (frame_count,) or raw_indices.dtype.kind not in {"i", "u"}:
            raise ValueError("frame_indices must be an integer vector")
        indices = np.asarray(raw_indices, dtype=np.int64)
        if not np.array_equal(indices, np.arange(frame_count, dtype=np.int64)):
            raise ValueError("frame_indices must be zero-based and contiguous")
        object.__setattr__(self, "frame_indices", _immutable_int64(indices))

        for field_name in (
            "reference_times_s",
            "camera_reported_times_s",
            "lidar_reported_times_s",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_finite(
                    getattr(self, field_name),
                    shape=(frame_count,),
                    field_name=field_name,
                ),
            )
        if np.any(np.diff(self.reference_times_s) <= 0.0):
            raise ValueError("reference_times_s must be strictly increasing")

        for field_name in (
            "truth_xy_m",
            "camera_value_xy_m",
            "lidar_value_xy_m",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_finite(
                    getattr(self, field_name),
                    shape=(frame_count, object_count, 2),
                    field_name=field_name,
                ),
            )
        for field_name in (
            "camera_actual_covariance_xy_m2",
            "lidar_actual_covariance_xy_m2",
            "camera_reported_covariance_xy_m2",
            "lidar_reported_covariance_xy_m2",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_covariances(
                    getattr(self, field_name),
                    frame_count=frame_count,
                    object_count=object_count,
                    field_name=field_name,
                ),
            )

        eligibility = np.asarray(self.eligibility_mask, dtype=np.bool_)
        if eligibility.shape != (frame_count, object_count):
            raise ValueError("eligibility_mask has an invalid shape")
        if np.any(np.count_nonzero(eligibility, axis=1) == 0):
            raise ValueError("each health frame requires an eligible object")
        object.__setattr__(self, "eligibility_mask", _immutable_bool(eligibility))
        for field_name in ("camera_available", "lidar_available"):
            array = np.asarray(getattr(self, field_name), dtype=np.bool_)
            if array.shape != (frame_count,):
                raise ValueError(f"{field_name} must have shape ({frame_count},)")
            object.__setattr__(self, field_name, _immutable_bool(array))

    @property
    def frame_count(self) -> int:
        return int(self.frame_indices.size)

    @property
    def object_count(self) -> int:
        return len(self.object_ids)

    def health_frame_inputs(self) -> tuple[HealthFrameInput, ...]:
        """Build the only leakage-bounded objects accepted by ``HealthScorer``."""

        frames: list[HealthFrameInput] = []
        for frame_index in range(self.frame_count):
            camera_available = bool(self.camera_available[frame_index])
            lidar_available = bool(self.lidar_available[frame_index])
            objects: list[ObjectHealthInput] = []
            for object_index, object_id in enumerate(self.object_ids):
                if not self.eligibility_mask[frame_index, object_index]:
                    continue
                camera = (
                    ModalityMeasurement(
                        value_xy_m=self.camera_value_xy_m[frame_index, object_index],
                        reported_covariance_xy_m2=self.camera_reported_covariance_xy_m2[
                            frame_index,
                            object_index,
                        ],
                        reported_time_s=float(self.camera_reported_times_s[frame_index]),
                    )
                    if camera_available
                    else None
                )
                lidar = (
                    ModalityMeasurement(
                        value_xy_m=self.lidar_value_xy_m[frame_index, object_index],
                        reported_covariance_xy_m2=self.lidar_reported_covariance_xy_m2[
                            frame_index,
                            object_index,
                        ],
                        reported_time_s=float(self.lidar_reported_times_s[frame_index]),
                    )
                    if lidar_available
                    else None
                )
                objects.append(
                    ObjectHealthInput(
                        object_id=object_id,
                        camera=camera,
                        lidar=lidar,
                    )
                )
            frames.append(
                HealthFrameInput(
                    reference_time_s=float(self.reference_times_s[frame_index]),
                    camera_available=camera_available,
                    lidar_available=lidar_available,
                    objects=tuple(objects),
                )
            )
        return tuple(frames)


def _camera_physical_proxy(
    *,
    noisy_truth_xy_m: FloatArray,
    true_translation_m: FloatArray,
    true_rotation: FloatArray,
) -> FloatArray:
    frame_count, object_count, _ = noisy_truth_xy_m.shape
    noisy_truth_xyz = np.concatenate(
        (
            noisy_truth_xy_m,
            np.zeros((frame_count, object_count, 1), dtype=np.float64),
        ),
        axis=2,
    )
    # Row-vector storage implements the canonical column-vector equation:
    # q_camera = R_true^T (p_scene - t_true).
    return immutable_float64_copy((noisy_truth_xyz - true_translation_m) @ true_rotation)


def _camera_reconstruction(
    *,
    physical_proxy_camera: FloatArray,
    used_translation_m: FloatArray,
    used_rotation: FloatArray,
) -> FloatArray:
    reconstructed = physical_proxy_camera @ used_rotation.T + used_translation_m
    return immutable_float64_copy(reconstructed[:, :, :2])


def _active_operator_mask(
    base: HealthBaseSequence,
    fault: HealthFaultSpec,
) -> BoolArray:
    schedule = health_event_schedule(fault.schedule)
    if schedule.frame_count != base.frame_count:
        raise ValueError("fault schedule and base sequence frame counts disagree")
    if fault.family in {"identity", "clean-predictor-mismatch"}:
        return _immutable_bool(np.zeros(base.frame_count, dtype=np.bool_))
    return schedule.active_mask()


def _validate_population_operator(
    base: HealthBaseSequence,
    fault: HealthFaultSpec,
) -> None:
    if fault.family == "clean-predictor-mismatch":
        if base.motion_model != "bounded-acceleration-control":
            raise ValueError("clean predictor mismatch requires the maneuver trajectory")
    elif base.motion_model == "bounded-acceleration-control":
        raise ValueError("the maneuver trajectory is reserved for its clean control")
    if fault.family == "common-mode-position-bias":
        if base.profile_id != "constant-velocity-fov-edge-v1" or base.split != "test":
            raise ValueError("common-mode bias requires the frozen edge test population")
    elif fault.family != "identity" and base.profile_id != "constant-velocity-front-roi-v1":
        raise ValueError("non-common M4 faults require the frozen main population")
    if fault.schedule == "cold_start" and base.split != "test":
        raise ValueError("cold-start controls require the main test split")


def generate_health_observations(
    base: HealthBaseSequence,
    *,
    fault: HealthFaultSpec,
) -> HealthObservationSequence:
    """Generate one paired transient condition without exposing its metadata."""

    _validate_population_operator(base, fault)
    frame_count = base.frame_count
    object_count = base.object_count
    active = _active_operator_mask(base, fault)

    camera_actual_scale = np.ones(frame_count, dtype=np.float64)
    lidar_actual_scale = np.ones(frame_count, dtype=np.float64)
    camera_reported_scale = np.ones(frame_count, dtype=np.float64)
    lidar_reported_scale = np.ones(frame_count, dtype=np.float64)
    if fault.family in {
        "increased-noise-underreported",
        "increased-noise-correctly-reported",
    }:
        actual_scale = camera_actual_scale if fault.target == "camera" else lidar_actual_scale
        actual_scale[active] = fault.value
        if fault.family == "increased-noise-correctly-reported":
            reported_scale = (
                camera_reported_scale if fault.target == "camera" else lidar_reported_scale
            )
            reported_scale[active] = fault.value

    camera_error = (
        base.camera_standard_normal_xy
        * _CAMERA_STD_XY_M[np.newaxis, np.newaxis, :]
        * camera_actual_scale[:, np.newaxis, np.newaxis]
    )
    lidar_error = (
        base.lidar_standard_normal_xy
        * _LIDAR_STD_XY_M[np.newaxis, np.newaxis, :]
        * lidar_actual_scale[:, np.newaxis, np.newaxis]
    )
    physical_proxy = _camera_physical_proxy(
        noisy_truth_xy_m=immutable_float64_copy(base.truth_xy_m + camera_error),
        true_translation_m=base.camera_true_translation_m,
        true_rotation=base.camera_true_rotation,
    )
    camera = np.array(
        _camera_reconstruction(
            physical_proxy_camera=physical_proxy,
            used_translation_m=base.camera_true_translation_m,
            used_rotation=base.camera_true_rotation,
        ),
        dtype=np.float64,
        copy=True,
    )
    lidar = np.asarray(base.truth_xy_m, dtype=np.float64) + lidar_error

    if fault.family == "calibration-translation":
        used_translation = np.array(
            base.camera_true_translation_m,
            dtype=np.float64,
            copy=True,
        )
        axis_index = 0 if fault.axis == "x" else 1
        used_translation[axis_index] += fault.value
        faulted_camera = _camera_reconstruction(
            physical_proxy_camera=physical_proxy,
            used_translation_m=immutable_float64_copy(used_translation),
            used_rotation=base.camera_true_rotation,
        )
        camera[active] = faulted_camera[active]
    elif fault.family == "calibration-yaw":
        cosine = math.cos(fault.value)
        sine = math.sin(fault.value)
        delta_rotation = np.asarray(
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        used_rotation = delta_rotation @ base.camera_true_rotation
        used_translation = delta_rotation @ base.camera_true_translation_m
        faulted_camera = _camera_reconstruction(
            physical_proxy_camera=physical_proxy,
            used_translation_m=immutable_float64_copy(used_translation),
            used_rotation=immutable_float64_copy(used_rotation),
        )
        camera[active] = faulted_camera[active]
    elif fault.family == "additive-position-bias":
        axis_index = 0 if fault.axis == "x" else 1
        target = camera if fault.target == "camera" else lidar
        target[active, :, axis_index] += fault.value
    elif fault.family == "common-mode-position-bias":
        axis_index = 0 if fault.axis == "x" else 1
        camera[active, :, axis_index] += fault.value
        lidar[active, :, axis_index] += fault.value
    elif fault.family == "timestamp-offset":
        target = camera if fault.target == "camera" else lidar
        target[active] -= fault.value * base.velocity_xy_mps[active]

    camera_reported_times = np.array(
        base.reference_times_s,
        dtype=np.float64,
        copy=True,
    )
    lidar_reported_times = np.array(
        base.reference_times_s,
        dtype=np.float64,
        copy=True,
    )
    if fault.family == "timestamp-offset":
        reported_times = camera_reported_times if fault.target == "camera" else lidar_reported_times
        reported_times[active] += fault.value

    camera_available = np.ones(frame_count, dtype=np.bool_)
    lidar_available = np.ones(frame_count, dtype=np.bool_)
    if fault.family == "dropout":
        dropped = active & (base.dropout_uniform_by_frame < fault.value)
        if fault.target == "camera":
            camera_available[dropped] = False
        else:
            lidar_available[dropped] = False

    return HealthObservationSequence(
        sequence_id=base.sequence_id,
        object_ids=base.object_ids,
        frame_indices=base.frame_indices,
        reference_times_s=base.reference_times_s,
        truth_xy_m=base.truth_xy_m,
        eligibility_mask=base.eligibility_mask,
        camera_value_xy_m=camera,
        lidar_value_xy_m=lidar,
        camera_actual_covariance_xy_m2=_diagonal_covariance_grid(
            _CAMERA_STD_XY_M,
            immutable_float64_copy(camera_actual_scale),
            object_count=object_count,
        ),
        lidar_actual_covariance_xy_m2=_diagonal_covariance_grid(
            _LIDAR_STD_XY_M,
            immutable_float64_copy(lidar_actual_scale),
            object_count=object_count,
        ),
        camera_reported_covariance_xy_m2=_diagonal_covariance_grid(
            _CAMERA_STD_XY_M,
            immutable_float64_copy(camera_reported_scale),
            object_count=object_count,
        ),
        lidar_reported_covariance_xy_m2=_diagonal_covariance_grid(
            _LIDAR_STD_XY_M,
            immutable_float64_copy(lidar_reported_scale),
            object_count=object_count,
        ),
        camera_available=camera_available,
        lidar_available=lidar_available,
        camera_reported_times_s=camera_reported_times,
        lidar_reported_times_s=lidar_reported_times,
    )
