"""Token-free production geometry for the local M2 nuScenes diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.adapters.nuscenes import (
    CAMERA_CHANNEL,
    NuScenesMiniMetadata,
    SampleDataRow,
)
from fusion_fault_bench.geometry.camera import (
    PinholeCamera,
    box_corners,
    devkit_box_any_visible,
    project_point,
)
from fusion_fault_bench.geometry.frames import FrameId
from fusion_fault_bench.geometry.se3 import SE3
from fusion_fault_bench.reference.nuscenes_projection import (
    ScalarProjectionDiagnostic,
)

type Vec2 = tuple[float, float]
type Vec3 = tuple[float, float, float]


class _PrivateRepr:
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class ProductionAnnotationProjection(_PrivateRepr):
    """One identifier-free annotation projection from production geometry."""

    center_camera_m: Vec3
    center_depth_m: float
    center_uv_px: Vec2 | None
    center_projection_valid: bool
    center_strict_image_inside: bool
    corners_camera_m: tuple[Vec3, ...]
    corner_uv_px: tuple[Vec2 | None, ...]
    box_any_visible: bool


@dataclass(frozen=True, slots=True, repr=False)
class ProductionProjectionDiagnostic(_PrivateRepr):
    """Identifier-free production output retained only for a local cross-check."""

    width_px: int
    height_px: int
    annotation_count: int
    finite_positive_depth_center_count: int
    annotations: tuple[ProductionAnnotationProjection, ...]


def _vec2(value: npt.ArrayLike) -> Vec2:
    array = np.asarray(value, dtype=np.float64)
    return (float(array[0]), float(array[1]))


def _vec3(value: npt.ArrayLike) -> Vec3:
    array = np.asarray(value, dtype=np.float64)
    return (float(array[0]), float(array[1]), float(array[2]))


def _camera_keyframe(
    metadata: NuScenesMiniMetadata,
    *,
    sample_token: str,
) -> SampleDataRow:
    candidates: list[SampleDataRow] = []
    for sample_data in metadata.sample_data.values():
        if sample_data.sample_token != sample_token or not sample_data.is_key_frame:
            continue
        calibrated = metadata.calibrated_sensors[sample_data.calibrated_sensor_token]
        sensor = metadata.sensors[calibrated.sensor_token]
        if sensor.channel == CAMERA_CHANNEL:
            candidates.append(sample_data)
    if len(candidates) != 1:
        raise ValueError("production local diagnostic selection failed")
    return candidates[0]


def build_production_projection_diagnostic(
    metadata: NuScenesMiniMetadata,
) -> ProductionProjectionDiagnostic:
    """Project the frozen local sample using adapter records and named SE(3)."""

    try:
        scene = min(metadata.scenes.values(), key=lambda row: row.name)
        sample = metadata.samples[scene.first_sample_token]
        camera_data = _camera_keyframe(metadata, sample_token=sample.token)
        calibration = metadata.calibrated_sensors[camera_data.calibrated_sensor_token]
        camera_pose = metadata.ego_poses[camera_data.ego_pose_token]

        global_frame = FrameId.global_frame(log_namespace=scene.log_token)
        camera_ego_frame = FrameId.ego(
            log_namespace=scene.log_token,
            timestamp_qualifier=str(camera_data.timestamp),
        )
        camera_frame = FrameId.camera(
            channel=CAMERA_CHANNEL,
            calibration_instance=calibration.token,
            timestamp_qualifier=str(camera_data.timestamp),
        )
        global_from_camera_ego = SE3.from_quaternion_wxyz(
            target_frame=global_frame,
            source_frame=camera_ego_frame,
            translation_m=camera_pose.translation,
            quaternion_wxyz=camera_pose.rotation,
        )
        camera_ego_from_camera = SE3.from_quaternion_wxyz(
            target_frame=camera_ego_frame,
            source_frame=camera_frame,
            translation_m=calibration.translation,
            quaternion_wxyz=calibration.rotation,
        )
        camera_from_global = camera_ego_from_camera.inverse().compose(
            global_from_camera_ego.inverse()
        )
        camera = PinholeCamera(
            intrinsic=np.asarray(calibration.camera_intrinsic, dtype=np.float64),
            width_px=camera_data.width,
            height_px=camera_data.height,
        )

        selected_annotations = sorted(
            (
                annotation
                for annotation in metadata.sample_annotations.values()
                if annotation.sample_token == sample.token
            ),
            key=lambda row: row.token.encode("utf-8"),
        )
        projected: list[ProductionAnnotationProjection] = []
        positive_depth_count = 0
        for annotation in selected_annotations:
            center_camera = camera_from_global.apply(
                annotation.translation,
                source_frame=camera_from_global.source_frame,
            )
            center_projection = project_point(center_camera, camera)
            if center_projection.valid:
                positive_depth_count += 1
            center_inside = bool(
                center_projection.uv_px is not None
                and camera.contains_strict(
                    center_projection.uv_px,
                    depth_m=center_projection.depth_m,
                    minimum_depth_m=0.1,
                )
            )

            corners_global = box_corners(
                center_m=annotation.translation,
                size_width_length_height_m=annotation.size,
                orientation_wxyz=annotation.rotation,
            )
            corners_camera = camera_from_global.apply(
                corners_global,
                source_frame=camera_from_global.source_frame,
            )
            corner_pixels: list[Vec2 | None] = []
            for index in range(corners_camera.shape[1]):
                projection = project_point(corners_camera[:, index], camera)
                corner_pixels.append(None if projection.uv_px is None else _vec2(projection.uv_px))
            projected.append(
                ProductionAnnotationProjection(
                    center_camera_m=_vec3(center_camera),
                    center_depth_m=center_projection.depth_m,
                    center_uv_px=(
                        None if center_projection.uv_px is None else _vec2(center_projection.uv_px)
                    ),
                    center_projection_valid=center_projection.valid,
                    center_strict_image_inside=center_inside,
                    corners_camera_m=tuple(
                        _vec3(corners_camera[:, index]) for index in range(corners_camera.shape[1])
                    ),
                    corner_uv_px=tuple(corner_pixels),
                    box_any_visible=devkit_box_any_visible(
                        corners_camera,
                        camera,
                        visible_corner_min_depth_m=1.0,
                        all_corners_min_depth_m=0.1,
                    ),
                )
            )
        if not projected or positive_depth_count < 1:
            raise ValueError("production local diagnostic is vacuous")
        return ProductionProjectionDiagnostic(
            width_px=camera.width_px,
            height_px=camera.height_px,
            annotation_count=len(projected),
            finite_positive_depth_center_count=positive_depth_count,
            annotations=tuple(projected),
        )
    except ValueError:
        raise
    except (KeyError, TypeError):
        raise ValueError("production local diagnostic failed") from None


def _finite_close(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    absolute_tolerance: float,
) -> bool:
    return len(left) == len(right) and all(
        math.isfinite(first) and math.isfinite(second) and abs(first - second) <= absolute_tolerance
        for first, second in zip(left, right, strict=True)
    )


def projection_crosscheck_passes(
    production: ProductionProjectionDiagnostic,
    scalar: ScalarProjectionDiagnostic,
    *,
    pixel_tolerance_px: float,
    depth_tolerance_m: float,
    minimum_annotation_count: int,
    minimum_finite_positive_depth_center_count: int,
) -> bool:
    """Compare independent diagnostic paths without returning local residuals."""

    if (
        not math.isfinite(pixel_tolerance_px)
        or pixel_tolerance_px < 0.0
        or not math.isfinite(depth_tolerance_m)
        or depth_tolerance_m < 0.0
    ):
        raise ValueError("projection cross-check tolerances must be finite and non-negative")
    if (
        production.width_px != scalar.width_px
        or production.height_px != scalar.height_px
        or production.annotation_count != scalar.annotation_count
        or production.annotation_count < minimum_annotation_count
        or production.finite_positive_depth_center_count
        != scalar.finite_positive_depth_center_count
        or production.finite_positive_depth_center_count
        < minimum_finite_positive_depth_center_count
        or len(production.annotations) != len(scalar.annotations)
    ):
        return False

    for observed, expected in zip(
        production.annotations,
        scalar.annotations,
        strict=True,
    ):
        if (
            not _finite_close(
                observed.center_camera_m,
                expected.center_camera_m,
                absolute_tolerance=depth_tolerance_m,
            )
            or abs(observed.center_depth_m - expected.center_depth_m) > depth_tolerance_m
            or observed.center_projection_valid != expected.center_projection_valid
            or observed.center_strict_image_inside != expected.center_strict_image_inside
            or observed.box_any_visible != expected.box_any_visible
            or len(observed.corners_camera_m) != len(expected.corners_camera_m)
            or len(observed.corner_uv_px) != len(expected.corner_uv_px)
        ):
            return False
        if (observed.center_uv_px is None) != (expected.center_uv_px is None):
            return False
        if (
            observed.center_uv_px is not None
            and expected.center_uv_px is not None
            and not _finite_close(
                observed.center_uv_px,
                expected.center_uv_px,
                absolute_tolerance=pixel_tolerance_px,
            )
        ):
            return False
        for observed_corner, expected_corner in zip(
            observed.corners_camera_m,
            expected.corners_camera_m,
            strict=True,
        ):
            if not _finite_close(
                observed_corner,
                expected_corner,
                absolute_tolerance=depth_tolerance_m,
            ):
                return False
        for observed_uv, expected_uv in zip(
            observed.corner_uv_px,
            expected.corner_uv_px,
            strict=True,
        ):
            if (observed_uv is None) != (expected_uv is None):
                return False
            if (
                observed_uv is not None
                and expected_uv is not None
                and not _finite_close(
                    observed_uv,
                    expected_uv,
                    absolute_tolerance=pixel_tolerance_px,
                )
            ):
                return False
    return True
