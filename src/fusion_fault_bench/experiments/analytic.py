"""One-object Gaussian analytic experiment generation."""

from __future__ import annotations

from typing import cast

import numpy as np

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AnalyticCrossoverManifest,
    CorrectlyReportedNoiseFault,
    FaultAxis,
    FaultFamily,
    MethodId,
    SeverityDirection,
    SeverityUnit,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.fusion.information import fuse_diagonal_information
from fusion_fault_bench.metrics.localization import squared_center_error
from fusion_fault_bench.rng import draw_standard_normal_xy


def _reported_variances(
    manifest: AnalyticCrossoverManifest,
    condition: ConditionKey,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.square(np.asarray(manifest.observations.camera.reported_std_xy_m, dtype=np.float64))
    lidar = np.square(np.asarray(manifest.observations.lidar.reported_std_xy_m, dtype=np.float64))
    fault = manifest.fault_sweep
    if isinstance(fault, CorrectlyReportedNoiseFault):
        target = camera if fault.target == "camera" else lidar
        target *= condition.magnitude * condition.magnitude
    return camera, lidar


def _condition_errors(
    manifest: AnalyticCrossoverManifest,
    condition: ConditionKey,
    camera_standard_normal: np.ndarray,
    lidar_standard_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = camera_standard_normal * np.asarray(
        manifest.observations.camera.actual_std_xy_m,
        dtype=np.float64,
    )
    lidar = lidar_standard_normal * np.asarray(
        manifest.observations.lidar.actual_std_xy_m,
        dtype=np.float64,
    )
    fault = manifest.fault_sweep
    target = camera if fault.target == "camera" else lidar
    if isinstance(fault, AdditivePositionBiasFault):
        if condition.direction != "identity":
            sign = -1.0 if condition.direction == "negative" else 1.0
            axis_index = 0 if fault.axis == "x" else 1
            target[axis_index] += sign * condition.magnitude
    else:
        target *= condition.magnitude
    return camera, lidar


def _severity(condition: ConditionKey) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=cast(SeverityDirection, condition.direction),
        unit=cast(SeverityUnit, condition.unit),
    )


def generate_analytic_sequence_metrics(
    manifest: AnalyticCrossoverManifest,
    *,
    run_id: str,
) -> tuple[MetricRecordV1Alpha1, ...]:
    """Generate deterministic paired sequence losses in contractual order."""

    digest = sha256_digest(manifest)
    zero = np.zeros(2, dtype=np.float64)
    conditions = expected_conditions(manifest)
    records: list[MetricRecordV1Alpha1] = []
    for sequence_id in expected_sequence_ids(manifest):
        camera_normal = draw_standard_normal_xy(
            data_master_seed=manifest.rng.data_master_seed,
            stream_name="camera",
            sequence_id=sequence_id,
            object_frame_count=1,
        )[0]
        lidar_normal = draw_standard_normal_xy(
            data_master_seed=manifest.rng.data_master_seed,
            stream_name="lidar",
            sequence_id=sequence_id,
            object_frame_count=1,
        )[0]
        for condition in conditions:
            camera, lidar = _condition_errors(
                manifest,
                condition,
                camera_normal,
                lidar_normal,
            )
            camera_variance, lidar_variance = _reported_variances(manifest, condition)
            fusion = fuse_diagonal_information(
                first_value_xy=camera,
                first_reported_variance_xy=camera_variance,
                second_value_xy=lidar,
                second_reported_variance_xy=lidar_variance,
            )
            camera_loss = squared_center_error(camera, zero)
            lidar_loss = squared_center_error(lidar, zero)
            fused_loss = squared_center_error(fusion.value_xy, zero)
            healthy_loss = lidar_loss if manifest.fault_sweep.target == "camera" else camera_loss
            values: dict[MethodId, float] = {
                "camera-only": camera_loss,
                "lidar-only": lidar_loss,
                "fixed-fusion": fused_loss,
                "fault-target-drop-policy": (
                    fused_loss if condition.severity_index == 0 else healthy_loss
                ),
                "performance-oracle": min(camera_loss, lidar_loss, fused_loss),
            }
            severity = _severity(condition)
            for method in manifest.methods:
                records.append(
                    LocalizationMetricRecord(
                        schema="ffb.sequence-metric/v1alpha1",
                        record_level="sequence",
                        run_id=run_id,
                        manifest_sha256=digest,
                        sequence_id=sequence_id,
                        fault_family=cast(FaultFamily, condition.fault_family),
                        fault_axis=cast(FaultAxis, condition.fault_axis),
                        severity=severity,
                        method_id=method,
                        eligible_object_frame_count=1,
                        valid_object_frame_count=1,
                        metric_name="matched-center-mse",
                        status="ok",
                        value=values[method],
                        unit="m^2",
                    )
                )
    return tuple(records)
