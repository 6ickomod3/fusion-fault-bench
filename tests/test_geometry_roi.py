from __future__ import annotations

import math

import numpy as np
import pytest

from fusion_fault_bench.geometry import (
    SE3,
    CommonRoi,
    EligibilityTransform,
    FrameId,
    PinholeCamera,
    ReportedReconstructionTransform,
    calibrated_center_eligible,
    procedural_center_eligible,
    reconstruct_with_reported_transform,
)


def _calibrated_setup() -> tuple[
    EligibilityTransform,
    EligibilityTransform,
    PinholeCamera,
    CommonRoi,
]:
    global_frame = FrameId.global_frame(log_namespace="roi-log")
    reference_ego = FrameId.ego(
        log_namespace="roi-log",
        timestamp_qualifier="lidar-time",
    )
    camera_frame = FrameId.camera(
        channel="CAM_FRONT",
        calibration_instance="roi-calibration",
        timestamp_qualifier="camera-time",
    )
    reference_from_global = EligibilityTransform(
        SE3(
            target_frame=reference_ego,
            source_frame=global_frame,
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        )
    )
    camera_from_global = EligibilityTransform(
        SE3(
            target_frame=camera_frame,
            source_frame=global_frame,
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        )
    )
    camera = PinholeCamera(
        intrinsic=np.asarray([[10.0, 0.0, 50.0], [0.0, 10.0, 50.0], [0.0, 0.0, 1.0]]),
        width_px=100,
        height_px=100,
    )
    return reference_from_global, camera_from_global, camera, CommonRoi(1.0, 10.0, 5.0)


def test_calibrated_center_uses_shared_pre_fault_support() -> None:
    reference_from_global, camera_from_global, camera, roi = _calibrated_setup()

    assert calibrated_center_eligible(
        center_global_m=(5.0, 0.0, 10.0),
        reference_ego_from_global=reference_from_global,
        camera_from_global=camera_from_global,
        camera=camera,
        roi=roi,
        lidar_support_available=True,
    )
    for field in (
        "lidar_support_available",
        "camera_estimator_available",
        "lidar_estimator_available",
    ):
        arguments = {
            "center_global_m": (5.0, 0.0, 10.0),
            "reference_ego_from_global": reference_from_global,
            "camera_from_global": camera_from_global,
            "camera": camera,
            "roi": roi,
            "lidar_support_available": True,
            "camera_estimator_available": True,
            "lidar_estimator_available": True,
        }
        arguments[field] = False
        assert not calibrated_center_eligible(**arguments)


def test_reported_calibration_cannot_control_roi_membership() -> None:
    reference_from_global, camera_from_global, camera, roi = _calibrated_setup()
    corrupted = ReportedReconstructionTransform(
        SE3(
            target_frame=camera_from_global.transform.target_frame,
            source_frame=camera_from_global.transform.source_frame,
            rotation=np.eye(3),
            translation_m=np.asarray([1_000.0, 0.0, 0.0]),
        )
    )
    nominal_membership = calibrated_center_eligible(
        center_global_m=(5.0, 0.0, 10.0),
        reference_ego_from_global=reference_from_global,
        camera_from_global=camera_from_global,
        camera=camera,
        roi=roi,
        lidar_support_available=True,
    )

    with pytest.raises(TypeError, match="EligibilityTransform"):
        calibrated_center_eligible(
            center_global_m=(5.0, 0.0, 10.0),
            reference_ego_from_global=reference_from_global,
            camera_from_global=corrupted,  # type: ignore[arg-type]
            camera=camera,
            roi=roi,
            lidar_support_available=True,
        )
    reconstructed = reconstruct_with_reported_transform(
        point_source_m=(5.0, 0.0, 10.0),
        reported_transform=corrupted,
    )
    assert nominal_membership
    assert reconstructed == pytest.approx((1005.0, 0.0, 10.0))


def test_calibrated_roi_checks_frame_roles_and_shared_global_namespace() -> None:
    reference_from_global, camera_from_global, camera, roi = _calibrated_setup()
    other_global = FrameId.global_frame(log_namespace="other-log")
    wrong_camera = EligibilityTransform(
        SE3(
            target_frame=camera_from_global.transform.target_frame,
            source_frame=other_global,
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        )
    )

    with pytest.raises(ValueError, match="one global frame"):
        calibrated_center_eligible(
            center_global_m=(5.0, 0.0, 10.0),
            reference_ego_from_global=reference_from_global,
            camera_from_global=wrong_camera,
            camera=camera,
            roi=roi,
            lidar_support_available=True,
        )


def test_longitudinal_lateral_and_fov_boundaries_are_frozen() -> None:
    roi = CommonRoi(x_min_m=1.0, x_max_m=10.0, abs_y_max_m=5.0)

    assert roi.contains_ego_xy((1.0, 5.0))
    assert roi.contains_ego_xy((10.0, -5.0))
    assert not CommonRoi(0.0, 10.0, 5.0).contains_ego_xy((0.0, 0.0))
    assert not roi.contains_ego_xy((10.0 + np.finfo(np.float64).eps * 10.0, 0.0))
    assert not roi.contains_ego_xy((5.0, np.nextafter(5.0, math.inf)))

    assert procedural_center_eligible(
        center_ego_m=(1.0, 1.0),
        roi=roi,
        camera_half_fov_rad=math.pi / 4.0,
        lidar_support_available=True,
    )
    assert not procedural_center_eligible(
        center_ego_m=(1.0, np.nextafter(1.0, math.inf)),
        roi=roi,
        camera_half_fov_rad=math.pi / 4.0,
        lidar_support_available=True,
    )


def test_calibrated_center_depth_and_image_bounds_are_strict() -> None:
    reference_from_global, camera_from_global, camera, roi = _calibrated_setup()

    assert not calibrated_center_eligible(
        center_global_m=(1.0, 0.0, 0.1),
        reference_ego_from_global=reference_from_global,
        camera_from_global=camera_from_global,
        camera=camera,
        roi=roi,
        lidar_support_available=True,
    )
    assert not calibrated_center_eligible(
        center_global_m=(10.0, 0.0, 2.0),
        reference_ego_from_global=reference_from_global,
        camera_from_global=camera_from_global,
        camera=camera,
        roi=roi,
        lidar_support_available=True,
    )
    assert not calibrated_center_eligible(
        center_global_m=(11.0, 0.0, 20.0),
        reference_ego_from_global=reference_from_global,
        camera_from_global=camera_from_global,
        camera=camera,
        roi=roi,
        lidar_support_available=True,
    )


@pytest.mark.parametrize(
    "roi",
    [
        CommonRoi(1.0, 2.0, 1.0),
    ],
)
def test_roi_and_availability_reject_wrong_types_and_values(roi: CommonRoi) -> None:
    with pytest.raises(TypeError, match="bool"):
        procedural_center_eligible(
            center_ego_m=(1.5, 0.0),
            roi=roi,
            camera_half_fov_rad=0.5,
            lidar_support_available=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="pi/2"):
        procedural_center_eligible(
            center_ego_m=(1.5, 0.0),
            roi=roi,
            camera_half_fov_rad=math.pi / 2.0,
            lidar_support_available=True,
        )
    with pytest.raises(ValueError):
        CommonRoi(2.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        CommonRoi(float("nan"), 2.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        CommonRoi(-1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        CommonRoi(1.0, 2.0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        roi.contains_ego_xy((1.0,))
    with pytest.raises(ValueError, match="finite"):
        roi.contains_ego_xy((1.0, float("inf")))
    assert not procedural_center_eligible(
        center_ego_m=(3.0, 0.0),
        roi=roi,
        camera_half_fov_rad=0.5,
        lidar_support_available=False,
    )
    assert not procedural_center_eligible(
        center_ego_m=(3.0, 0.0),
        roi=roi,
        camera_half_fov_rad=0.5,
        lidar_support_available=True,
    )
