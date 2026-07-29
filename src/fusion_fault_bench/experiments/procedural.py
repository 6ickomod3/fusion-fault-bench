"""Procedural estimator-output faults and ordered sequence-row generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AvailabilityControlManifest,
    CalibrationTranslationFault,
    CalibrationYawFault,
    CommonModeControlManifest,
    CommonModePositionBiasFault,
    CorrectlyReportedNoiseFault,
    GeometryCrossoverManifest,
    MethodId,
    ProceduralSource,
    SeverityDirection,
    SeverityUnit,
    TimestampOffsetFault,
    UnderreportedNoiseFault,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    ProceduralProfileV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RateMetricRecord,
    SeverityCoordinate,
)
from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.se3 import quaternion_wxyz_to_rotation
from fusion_fault_bench.metrics.localization import matched_center_mse
from fusion_fault_bench.scenarios.procedural import generate_procedural_sequences

type ProceduralManifest = (
    GeometryCrossoverManifest | CommonModeControlManifest | AvailabilityControlManifest
)
type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]


def _immutable_bool(value: npt.ArrayLike) -> BoolArray:
    source = np.ascontiguousarray(np.asarray(value, dtype=np.bool_))
    return np.frombuffer(source.tobytes(order="C"), dtype=np.bool_).reshape(source.shape)


def _float_matrix(
    value: npt.ArrayLike,
    *,
    shape: tuple[int, int],
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def _float_vector(
    value: npt.ArrayLike,
    *,
    length: int,
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{field_name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def _frame_indices(value: npt.ArrayLike, *, object_frame_count: int) -> IntArray:
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"} or array.shape != (object_frame_count,):
        raise ValueError(
            "eligible_frame_indices must be an integer vector aligned with eligible rows"
        )
    result = np.asarray(array, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError("eligible_frame_indices cannot contain negative values")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.int64)


def _direction_sign(condition: ConditionKey) -> float:
    if condition.direction == "identity":
        return 0.0
    if condition.direction == "negative":
        return -1.0
    if condition.direction in {"positive", "increase"}:
        return 1.0
    raise ValueError(f"unknown severity direction: {condition.direction!r}")


def _severity(condition: ConditionKey) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=cast(SeverityDirection, condition.direction),
        unit=cast(SeverityUnit, condition.unit),
    )


def _reported_variances(
    manifest: ProceduralManifest,
    condition: ConditionKey,
) -> tuple[FloatArray, FloatArray]:
    camera = np.square(np.asarray(manifest.observations.camera.reported_std_xy_m, dtype=np.float64))
    lidar = np.square(np.asarray(manifest.observations.lidar.reported_std_xy_m, dtype=np.float64))
    fault = manifest.fault_sweep
    if isinstance(fault, CorrectlyReportedNoiseFault):
        target = camera if fault.target == "camera" else lidar
        target *= condition.magnitude * condition.magnitude
    return immutable_float64_copy(camera), immutable_float64_copy(lidar)


def _camera_reconstruction(
    *,
    truth_xy_m: FloatArray,
    base_error_xy_m: FloatArray,
    true_translation_m: FloatArray,
    true_rotation: FloatArray,
    condition: ConditionKey,
    fault: object,
) -> FloatArray:
    noisy_ego_xy = np.asarray(truth_xy_m) + np.asarray(base_error_xy_m)
    truth_xyz = np.column_stack(
        (
            noisy_ego_xy,
            np.zeros(truth_xy_m.shape[0], dtype=np.float64),
        )
    )
    # The physical proxy is formed once with the true transform. Calibration
    # faults affect only the metadata used for the reconstruction below.
    camera_points = (truth_xyz - true_translation_m) @ true_rotation
    used_rotation = np.array(true_rotation, dtype=np.float64, copy=True)
    used_translation = np.array(true_translation_m, dtype=np.float64, copy=True)
    signed_magnitude = _direction_sign(condition) * condition.magnitude
    if isinstance(fault, CalibrationTranslationFault):
        axis_index = 0 if fault.axis == "x" else 1
        used_translation[axis_index] += signed_magnitude
    elif isinstance(fault, CalibrationYawFault):
        cosine = np.cos(signed_magnitude)
        sine = np.sin(signed_magnitude)
        delta_rotation = np.asarray(
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        used_rotation = delta_rotation @ used_rotation
        used_translation = delta_rotation @ used_translation
    reconstructed_xyz = camera_points @ used_rotation.T + used_translation
    return immutable_float64_copy(reconstructed_xyz[:, :2])


@dataclass(frozen=True, slots=True)
class ProceduralConditionOutputs:
    """Per-object-frame estimator values, covariance reports, and availability."""

    camera_value_xy_m: FloatArray
    lidar_value_xy_m: FloatArray
    fixed_fusion_value_xy_m: FloatArray
    camera_reported_variance_xy_m2: FloatArray
    lidar_reported_variance_xy_m2: FloatArray
    camera_available: BoolArray
    lidar_available: BoolArray
    fixed_fusion_available: BoolArray


def generate_procedural_condition_outputs(
    manifest: ProceduralManifest,
    *,
    condition: ConditionKey,
    truth_xy_m: npt.ArrayLike,
    velocity_xy_mps: npt.ArrayLike,
    eligible_frame_indices: npt.ArrayLike,
    camera_standard_normal_xy: npt.ArrayLike,
    lidar_standard_normal_xy: npt.ArrayLike,
    fault_uniform_by_frame: npt.ArrayLike,
    camera_true_translation_m: npt.ArrayLike,
    camera_true_quaternion_wxyz: npt.ArrayLike,
) -> ProceduralConditionOutputs:
    """Apply one frozen fault while keeping physical truth and base draws paired."""

    if condition not in expected_conditions(manifest):
        raise ValueError("condition is not part of the manifest fault grid")
    raw_truth = np.asarray(truth_xy_m, dtype=np.float64)
    if raw_truth.ndim != 2 or raw_truth.shape[1:] != (2,) or raw_truth.shape[0] == 0:
        raise ValueError("truth_xy_m must have nonempty shape (object_frame_count, 2)")
    object_frame_count = raw_truth.shape[0]
    truth = _float_matrix(
        raw_truth,
        shape=(object_frame_count, 2),
        field_name="truth_xy_m",
    )
    velocity = _float_matrix(
        velocity_xy_mps,
        shape=(object_frame_count, 2),
        field_name="velocity_xy_mps",
    )
    camera_normal = _float_matrix(
        camera_standard_normal_xy,
        shape=(object_frame_count, 2),
        field_name="camera_standard_normal_xy",
    )
    lidar_normal = _float_matrix(
        lidar_standard_normal_xy,
        shape=(object_frame_count, 2),
        field_name="lidar_standard_normal_xy",
    )
    frame_indices = _frame_indices(
        eligible_frame_indices,
        object_frame_count=object_frame_count,
    )
    raw_uniforms = np.asarray(fault_uniform_by_frame, dtype=np.float64)
    if raw_uniforms.ndim != 1 or raw_uniforms.size == 0:
        raise ValueError("fault_uniform_by_frame must be a nonempty vector")
    uniforms = _float_vector(
        raw_uniforms,
        length=raw_uniforms.size,
        field_name="fault_uniform_by_frame",
    )
    if np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("fault_uniform_by_frame values must lie in [0, 1)")
    if int(frame_indices.max()) >= uniforms.size:
        raise ValueError("eligible_frame_indices exceed the fault-uniform vector")
    translation = _float_vector(
        camera_true_translation_m,
        length=3,
        field_name="camera_true_translation_m",
    )
    quaternion = _float_vector(
        camera_true_quaternion_wxyz,
        length=4,
        field_name="camera_true_quaternion_wxyz",
    )
    rotation = quaternion_wxyz_to_rotation(quaternion)

    fault = manifest.fault_sweep
    camera_std = np.asarray(
        manifest.observations.camera.actual_std_xy_m,
        dtype=np.float64,
    )
    lidar_std = np.asarray(
        manifest.observations.lidar.actual_std_xy_m,
        dtype=np.float64,
    )
    if isinstance(fault, (CorrectlyReportedNoiseFault, UnderreportedNoiseFault)):
        if fault.target == "camera":
            camera_std = camera_std * condition.magnitude
        else:
            lidar_std = lidar_std * condition.magnitude

    if isinstance(fault, CalibrationYawFault) and (
        camera_std[0] != camera_std[1]
        or manifest.observations.camera.reported_std_xy_m[0]
        != manifest.observations.camera.reported_std_xy_m[1]
    ):
        raise ValueError("calibration yaw requires isotropic camera covariance in M3")
    camera = np.array(
        _camera_reconstruction(
            truth_xy_m=truth,
            base_error_xy_m=immutable_float64_copy(camera_normal * camera_std),
            true_translation_m=translation,
            true_rotation=rotation,
            condition=condition,
            fault=fault,
        ),
        dtype=np.float64,
        copy=True,
    )
    lidar = np.asarray(truth, dtype=np.float64) + lidar_normal * lidar_std
    signed_magnitude = _direction_sign(condition) * condition.magnitude
    if isinstance(fault, TimestampOffsetFault):
        target = camera if fault.target == "camera" else lidar
        target -= signed_magnitude * velocity
    if isinstance(fault, AdditivePositionBiasFault):
        axis_index = 0 if fault.axis == "x" else 1
        target = camera if fault.target == "camera" else lidar
        target[:, axis_index] += signed_magnitude
    elif isinstance(fault, CommonModePositionBiasFault):
        axis_index = 0 if fault.axis == "x" else 1
        camera[:, axis_index] += signed_magnitude
        lidar[:, axis_index] += signed_magnitude

    camera_variance, lidar_variance = _reported_variances(manifest, condition)
    camera_information = 1.0 / camera_variance
    lidar_information = 1.0 / lidar_variance
    fused_variance = 1.0 / (camera_information + lidar_information)
    fusion = fused_variance * (camera_information * camera + lidar_information * lidar)
    if not np.all(np.isfinite(fusion)):
        raise ValueError("procedural information fusion produced non-finite values")

    camera_available = np.ones(object_frame_count, dtype=np.bool_)
    lidar_available = np.ones(object_frame_count, dtype=np.bool_)
    if isinstance(manifest, AvailabilityControlManifest):
        available = uniforms[frame_indices] >= condition.magnitude
        if fault.target == "camera":
            camera_available = available
        else:
            lidar_available = available
    fixed_available = camera_available & lidar_available
    return ProceduralConditionOutputs(
        camera_value_xy_m=immutable_float64_copy(camera),
        lidar_value_xy_m=immutable_float64_copy(lidar),
        fixed_fusion_value_xy_m=immutable_float64_copy(fusion),
        camera_reported_variance_xy_m2=camera_variance,
        lidar_reported_variance_xy_m2=lidar_variance,
        camera_available=_immutable_bool(camera_available),
        lidar_available=_immutable_bool(lidar_available),
        fixed_fusion_available=_immutable_bool(fixed_available),
    )


def _conditional_loss(
    values_xy_m: FloatArray,
    truth_xy_m: FloatArray,
    available: BoolArray,
) -> float | None:
    if not np.any(available):
        return None
    return matched_center_mse(values_xy_m[available], truth_xy_m[available])


def _localization_row(
    *,
    run_id: str,
    digest: str,
    sequence_id: str,
    condition: ConditionKey,
    method: MethodId,
    eligible_count: int,
    valid_count: int,
    metric_name: str,
    value: float | None,
) -> LocalizationMetricRecord:
    return LocalizationMetricRecord.model_validate(
        {
            "schema": "ffb.sequence-metric/v1alpha1",
            "record_level": "sequence",
            "run_id": run_id,
            "manifest_sha256": digest,
            "sequence_id": sequence_id,
            "fault_family": condition.fault_family,
            "fault_axis": condition.fault_axis,
            "severity": _severity(condition),
            "method_id": method,
            "eligible_object_frame_count": eligible_count,
            "valid_object_frame_count": valid_count,
            "metric_name": metric_name,
            "status": "ok" if value is not None else "undefined",
            "value": value,
            "unit": "m^2",
        }
    )


def _rate_row(
    *,
    run_id: str,
    digest: str,
    sequence_id: str,
    condition: ConditionKey,
    method: MethodId,
    eligible_count: int,
    valid_count: int,
    metric_name: str,
    value: float,
) -> RateMetricRecord:
    return RateMetricRecord.model_validate(
        {
            "schema": "ffb.sequence-metric/v1alpha1",
            "record_level": "sequence",
            "run_id": run_id,
            "manifest_sha256": digest,
            "sequence_id": sequence_id,
            "fault_family": condition.fault_family,
            "fault_axis": condition.fault_axis,
            "severity": _severity(condition),
            "method_id": method,
            "eligible_object_frame_count": eligible_count,
            "valid_object_frame_count": valid_count,
            "metric_name": metric_name,
            "status": "ok",
            "value": value,
            "unit": "fraction",
        }
    )


def generate_procedural_sequence_metric_rows(
    manifest: ProceduralManifest,
    *,
    run_id: str,
    sequence_id: str,
    truth_xy_m: npt.ArrayLike,
    velocity_xy_mps: npt.ArrayLike,
    eligible_frame_indices: npt.ArrayLike,
    camera_standard_normal_xy: npt.ArrayLike,
    lidar_standard_normal_xy: npt.ArrayLike,
    fault_uniform_by_frame: npt.ArrayLike,
    camera_true_translation_m: npt.ArrayLike,
    camera_true_quaternion_wxyz: npt.ArrayLike,
) -> tuple[MetricRecordV1Alpha1, ...]:
    """Generate all manifest-ordered rows for one complete procedural sequence."""

    truth = np.asarray(truth_xy_m, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1:] != (2,) or truth.shape[0] == 0:
        raise ValueError("truth_xy_m must have nonempty shape (object_frame_count, 2)")
    eligible_count = truth.shape[0]
    digest = sha256_digest(manifest)
    records: list[MetricRecordV1Alpha1] = []
    for condition in expected_conditions(manifest):
        outputs = generate_procedural_condition_outputs(
            manifest,
            condition=condition,
            truth_xy_m=truth,
            velocity_xy_mps=velocity_xy_mps,
            eligible_frame_indices=eligible_frame_indices,
            camera_standard_normal_xy=camera_standard_normal_xy,
            lidar_standard_normal_xy=lidar_standard_normal_xy,
            fault_uniform_by_frame=fault_uniform_by_frame,
            camera_true_translation_m=camera_true_translation_m,
            camera_true_quaternion_wxyz=camera_true_quaternion_wxyz,
        )
        values = {
            "camera-only": outputs.camera_value_xy_m,
            "lidar-only": outputs.lidar_value_xy_m,
            "fixed-fusion": outputs.fixed_fusion_value_xy_m,
        }
        masks = {
            "camera-only": outputs.camera_available,
            "lidar-only": outputs.lidar_available,
            "fixed-fusion": outputs.fixed_fusion_available,
        }

        if isinstance(manifest, AvailabilityControlManifest):
            healthy_method: MethodId = (
                "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
            )
            source_method: MethodId = (
                "fixed-fusion" if condition.severity_index == 0 else healthy_method
            )
            values["fault-target-drop-policy"] = values[source_method]
            masks["fault-target-drop-policy"] = masks[source_method]
            for method in manifest.methods:
                available = masks[method]
                valid_count = int(np.count_nonzero(available))
                conditional = _conditional_loss(values[method], truth, available)
                for metric_name in manifest.evaluation.metrics:
                    if metric_name == "coverage":
                        records.append(
                            _rate_row(
                                run_id=run_id,
                                digest=digest,
                                sequence_id=sequence_id,
                                condition=condition,
                                method=method,
                                eligible_count=eligible_count,
                                valid_count=valid_count,
                                metric_name=metric_name,
                                value=valid_count / eligible_count,
                            )
                        )
                    elif metric_name == "undefined-output-rate":
                        records.append(
                            _rate_row(
                                run_id=run_id,
                                digest=digest,
                                sequence_id=sequence_id,
                                condition=condition,
                                method=method,
                                eligible_count=eligible_count,
                                valid_count=valid_count,
                                metric_name=metric_name,
                                value=(eligible_count - valid_count) / eligible_count,
                            )
                        )
                    else:
                        records.append(
                            _localization_row(
                                run_id=run_id,
                                digest=digest,
                                sequence_id=sequence_id,
                                condition=condition,
                                method=method,
                                eligible_count=eligible_count,
                                valid_count=valid_count,
                                metric_name=metric_name,
                                value=conditional,
                            )
                        )
            continue

        losses = {method: matched_center_mse(value, truth) for method, value in values.items()}
        if isinstance(manifest, GeometryCrossoverManifest):
            healthy_method = (
                "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
            )
            losses["fault-target-drop-policy"] = (
                losses["fixed-fusion"] if condition.severity_index == 0 else losses[healthy_method]
            )
            losses["performance-oracle"] = min(
                losses["camera-only"],
                losses["lidar-only"],
                losses["fixed-fusion"],
            )
        for method in manifest.methods:
            records.append(
                _localization_row(
                    run_id=run_id,
                    digest=digest,
                    sequence_id=sequence_id,
                    condition=condition,
                    method=method,
                    eligible_count=eligible_count,
                    valid_count=eligible_count,
                    metric_name="matched-center-mse",
                    value=losses[method],
                )
            )
    return tuple(records)


def generate_procedural_sequence_metrics(
    manifest: ProceduralManifest,
    *,
    profile: ProceduralProfileV1,
    run_id: str,
) -> tuple[MetricRecordV1Alpha1, ...]:
    """Generate every ordered sequence row from one content-addressed profile."""

    source = manifest.source
    if not isinstance(source, ProceduralSource):
        raise TypeError("procedural generation requires a procedural source")
    if profile.profile_id != source.profile_id:
        raise ValueError("profile ID does not match the manifest source")
    if sha256_digest(profile) != source.profile_sha256:
        raise ValueError("profile digest does not match the manifest source")
    profile_roi = (
        profile.eligibility.x_min_m,
        profile.eligibility.x_max_m,
        profile.eligibility.abs_y_max_m,
        profile.eligibility.camera_half_fov_rad,
    )
    manifest_roi = (
        manifest.roi.x_min_m,
        manifest.roi.x_max_m,
        manifest.roi.abs_y_max_m,
        manifest.roi.camera_half_fov_rad,
    )
    if profile_roi != manifest_roi:
        raise ValueError("profile eligibility ROI does not match the manifest")

    extrinsic = profile.rig.camera_true_extrinsic
    sequences = generate_procedural_sequences(
        profile,
        split=source.split,
        sequence_count=source.sequence_count,
        data_master_seed=manifest.rng.data_master_seed,
    )
    records: list[MetricRecordV1Alpha1] = []
    for sequence in sequences:
        records.extend(
            generate_procedural_sequence_metric_rows(
                manifest,
                run_id=run_id,
                sequence_id=sequence.sequence_id,
                truth_xy_m=sequence.eligible_truth_xy_m,
                velocity_xy_mps=sequence.eligible_velocity_xy_mps,
                eligible_frame_indices=sequence.eligible_frame_indices,
                camera_standard_normal_xy=sequence.camera_standard_normal_xy,
                lidar_standard_normal_xy=sequence.lidar_standard_normal_xy,
                fault_uniform_by_frame=sequence.fault_uniform_by_frame,
                camera_true_translation_m=extrinsic.translation_m,
                camera_true_quaternion_wxyz=extrinsic.quaternion_wxyz,
            )
        )
    return tuple(records)
