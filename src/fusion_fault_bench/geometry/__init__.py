"""CPU-only named geometry and covariance utilities."""

from fusion_fault_bench.geometry.camera import (
    PinholeCamera,
    Projection,
    box_corners,
    devkit_box_any_visible,
    project_point,
)
from fusion_fault_bench.geometry.covariance import (
    BearingDepthCovariance,
    PropagatedCovariance,
    bearing_depth_jacobian_camera,
    bearing_depth_jacobian_reference_ego_bev,
    bearing_depth_point_camera,
    bearing_depth_point_reference_ego_bev,
    propagate_bearing_depth_covariance,
)
from fusion_fault_bench.geometry.frames import FrameId
from fusion_fault_bench.geometry.roi import (
    CommonRoi,
    EligibilityTransform,
    ReportedReconstructionTransform,
    calibrated_center_eligible,
    procedural_center_eligible,
    reconstruct_with_reported_transform,
)
from fusion_fault_bench.geometry.se3 import SE3, quaternion_wxyz_to_rotation

__all__ = [
    "SE3",
    "BearingDepthCovariance",
    "CommonRoi",
    "EligibilityTransform",
    "FrameId",
    "PinholeCamera",
    "Projection",
    "PropagatedCovariance",
    "ReportedReconstructionTransform",
    "bearing_depth_jacobian_camera",
    "bearing_depth_jacobian_reference_ego_bev",
    "bearing_depth_point_camera",
    "bearing_depth_point_reference_ego_bev",
    "box_corners",
    "calibrated_center_eligible",
    "devkit_box_any_visible",
    "procedural_center_eligible",
    "project_point",
    "propagate_bearing_depth_covariance",
    "quaternion_wxyz_to_rotation",
    "reconstruct_with_reported_transform",
]
