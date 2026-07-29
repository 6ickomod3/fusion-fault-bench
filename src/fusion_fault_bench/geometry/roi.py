"""Pre-fault common-support predicates with typed transform roles."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.geometry.camera import PinholeCamera, project_point
from fusion_fault_bench.geometry.se3 import SE3


@dataclass(frozen=True, slots=True)
class EligibilityTransform:
    """A nominal transform authorized for pre-fault eligibility decisions."""

    transform: SE3


@dataclass(frozen=True, slots=True)
class ReportedReconstructionTransform:
    """A possibly corrupted transform authorized only for reconstruction."""

    transform: SE3


@dataclass(frozen=True, slots=True)
class CommonRoi:
    """Inclusive ego-forward and lateral limits from the frozen manifest."""

    x_min_m: float
    x_max_m: float
    abs_y_max_m: float

    def __post_init__(self) -> None:
        values = (self.x_min_m, self.x_max_m, self.abs_y_max_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ROI bounds must be finite")
        if self.x_min_m < 0.0:
            raise ValueError("x_min_m must be non-negative")
        if self.x_max_m <= self.x_min_m:
            raise ValueError("x_max_m must exceed x_min_m")
        if self.abs_y_max_m <= 0.0:
            raise ValueError("abs_y_max_m must be strictly positive")

    def contains_ego_xy(self, center_ego_m: npt.ArrayLike) -> bool:
        """Apply strict positive-forward and inclusive manifest bounds."""

        center = np.asarray(center_ego_m, dtype=np.float64)
        if center.shape not in {(2,), (3,)}:
            raise ValueError("center_ego_m must have shape (2,) or (3,)")
        if not np.all(np.isfinite(center)):
            raise ValueError("center_ego_m must contain only finite values")
        x_m = float(center[0])
        y_m = float(center[1])
        return bool(
            x_m > 0.0 and self.x_min_m <= x_m <= self.x_max_m and abs(y_m) <= self.abs_y_max_m
        )


def _availability(value: bool, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")
    return value


def _require_eligibility_transform(value: object) -> EligibilityTransform:
    if not isinstance(value, EligibilityTransform):
        raise TypeError("ROI eligibility requires EligibilityTransform values")
    return value


def _require_reported_transform(value: object) -> ReportedReconstructionTransform:
    if not isinstance(value, ReportedReconstructionTransform):
        raise TypeError("reconstruction requires ReportedReconstructionTransform")
    return value


def calibrated_center_eligible(
    *,
    center_global_m: npt.ArrayLike,
    reference_ego_from_global: EligibilityTransform,
    camera_from_global: EligibilityTransform,
    camera: PinholeCamera,
    roi: CommonRoi,
    lidar_support_available: bool,
    camera_estimator_available: bool = True,
    lidar_estimator_available: bool = True,
    center_minimum_depth_m: float = 0.1,
) -> bool:
    """Evaluate common support using only nominal eligibility transforms."""

    reference_transform = _require_eligibility_transform(reference_ego_from_global)
    camera_transform = _require_eligibility_transform(camera_from_global)
    if (
        reference_transform.transform.source_frame != camera_transform.transform.source_frame
        or reference_transform.transform.source_frame.kind != "global"
        or reference_transform.transform.target_frame.kind != "ego"
        or camera_transform.transform.target_frame.kind != "camera"
    ):
        raise ValueError("calibrated ROI transforms must map one global frame to ego and camera")
    lidar_support = _availability(
        lidar_support_available,
        field_name="lidar_support_available",
    )
    camera_available = _availability(
        camera_estimator_available,
        field_name="camera_estimator_available",
    )
    lidar_available = _availability(
        lidar_estimator_available,
        field_name="lidar_estimator_available",
    )
    if not lidar_support or not camera_available or not lidar_available:
        return False
    center_reference = reference_transform.transform.apply(
        center_global_m,
        source_frame=reference_transform.transform.source_frame,
    )
    if not roi.contains_ego_xy(center_reference):
        return False
    center_camera = camera_transform.transform.apply(
        center_global_m,
        source_frame=camera_transform.transform.source_frame,
    )
    projection = project_point(center_camera, camera)
    return bool(
        projection.uv_px is not None
        and camera.contains_strict(
            projection.uv_px,
            depth_m=projection.depth_m,
            minimum_depth_m=center_minimum_depth_m,
        )
    )


def procedural_center_eligible(
    *,
    center_ego_m: npt.ArrayLike,
    roi: CommonRoi,
    camera_half_fov_rad: float,
    lidar_support_available: bool,
    camera_estimator_available: bool = True,
    lidar_estimator_available: bool = True,
) -> bool:
    """Evaluate inclusive symmetric half-FOV support without camera intrinsics."""

    if (
        not math.isfinite(camera_half_fov_rad)
        or camera_half_fov_rad <= 0.0
        or camera_half_fov_rad >= math.pi / 2.0
    ):
        raise ValueError("camera_half_fov_rad must be finite and in (0, pi/2)")
    if not (
        _availability(lidar_support_available, field_name="lidar_support_available")
        and _availability(
            camera_estimator_available,
            field_name="camera_estimator_available",
        )
        and _availability(
            lidar_estimator_available,
            field_name="lidar_estimator_available",
        )
    ):
        return False
    center = np.asarray(center_ego_m, dtype=np.float64)
    if not roi.contains_ego_xy(center):
        return False
    bearing = math.atan2(float(center[1]), float(center[0]))
    return abs(bearing) <= camera_half_fov_rad


def reconstruct_with_reported_transform(
    *,
    point_source_m: npt.ArrayLike,
    reported_transform: ReportedReconstructionTransform,
) -> npt.NDArray[np.float64]:
    """Apply a reported transform without granting it eligibility authority."""

    transform = _require_reported_transform(reported_transform)
    return transform.transform.apply(
        point_source_m,
        source_frame=transform.transform.source_frame,
    )
