from __future__ import annotations

from dataclasses import replace

from fusion_fault_bench.adapters.nuscenes import (
    DATASET_AUTHENTICATION,
    CalibratedSensorRow,
    EgoPoseRow,
    NuScenesMiniMetadata,
    NuScenesMiniValidation,
    SampleAnnotationRow,
    SampleDataRow,
    SampleRow,
    SceneRow,
    SensorRow,
)
from fusion_fault_bench.nuscenes_geometry import (
    build_production_projection_diagnostic,
    projection_crosscheck_passes,
)
from fusion_fault_bench.reference.nuscenes_projection import (
    ScalarAnnotationProjection,
    ScalarProjectionDiagnostic,
)


def _metadata() -> NuScenesMiniMetadata:
    sample_token = "private-sample"
    camera_data = SampleDataRow(
        token="private-camera-data",
        sample_token=sample_token,
        ego_pose_token="private-camera-pose",
        calibrated_sensor_token="private-camera-calibration",
        filename="samples/CAM_FRONT/private.jpg",
        fileformat="jpg",
        width=100,
        height=100,
        timestamp=1,
        is_key_frame=True,
        next="",
        prev="",
    )
    annotation = SampleAnnotationRow(
        token="private-annotation",
        sample_token=sample_token,
        instance_token="private-instance",
        attribute_tokens=(),
        visibility_token="",
        translation=(0.0, 0.0, 10.0),
        size=(2.0, 2.0, 2.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        num_lidar_pts=1,
        num_radar_pts=0,
        next="",
        prev="",
    )
    return NuScenesMiniMetadata(
        attributes={},
        calibrated_sensors={
            "private-camera-calibration": CalibratedSensorRow(
                token="private-camera-calibration",
                sensor_token="private-camera-sensor",
                translation=(0.0, 0.0, 0.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
                camera_intrinsic=(
                    (10.0, 0.0, 50.0),
                    (0.0, 10.0, 50.0),
                    (0.0, 0.0, 1.0),
                ),
            )
        },
        categories={},
        ego_poses={
            "private-camera-pose": EgoPoseRow(
                token="private-camera-pose",
                translation=(0.0, 0.0, 0.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
                timestamp=1,
            )
        },
        instances={},
        logs={},
        samples={
            sample_token: SampleRow(
                token=sample_token,
                timestamp=1,
                scene_token="private-scene",
                next="",
                prev="",
            )
        },
        sample_annotations={annotation.token: annotation},
        sample_data={camera_data.token: camera_data},
        scenes={
            "private-scene": SceneRow(
                token="private-scene",
                name="scene-synthetic",
                description="",
                log_token="private-log",
                nbr_samples=1,
                first_sample_token=sample_token,
                last_sample_token=sample_token,
            )
        },
        sensors={
            "private-camera-sensor": SensorRow(
                token="private-camera-sensor",
                channel="CAM_FRONT",
                modality="camera",
            )
        },
        visibility={},
        validation=NuScenesMiniValidation(
            headline_profile_passed_attested=False,
            structural_integrity_passed_attested=True,
            keyframe_blob_check_count=0,
            keyframe_blob_validation_passed_attested=True,
            dataset_authentication=DATASET_AUTHENTICATION,
        ),
    )


def _scalar_from_production() -> ScalarProjectionDiagnostic:
    production = build_production_projection_diagnostic(_metadata())
    annotation = production.annotations[0]
    return ScalarProjectionDiagnostic(
        width_px=production.width_px,
        height_px=production.height_px,
        annotation_count=production.annotation_count,
        finite_positive_depth_center_count=(production.finite_positive_depth_center_count),
        annotations=(
            ScalarAnnotationProjection(
                center_camera_m=annotation.center_camera_m,
                center_depth_m=annotation.center_depth_m,
                center_uv_px=annotation.center_uv_px,
                center_projection_valid=annotation.center_projection_valid,
                center_strict_image_inside=annotation.center_strict_image_inside,
                corners_camera_m=annotation.corners_camera_m,
                corner_uv_px=annotation.corner_uv_px,
                box_any_visible=annotation.box_any_visible,
            ),
        ),
    )


def test_production_geometry_projects_with_camera_time_pose() -> None:
    diagnostic = build_production_projection_diagnostic(_metadata())

    assert diagnostic.annotation_count == 1
    assert diagnostic.finite_positive_depth_center_count == 1
    assert diagnostic.annotations[0].center_camera_m == (0.0, 0.0, 10.0)
    assert diagnostic.annotations[0].center_uv_px == (50.0, 50.0)
    assert diagnostic.annotations[0].center_strict_image_inside
    assert diagnostic.annotations[0].box_any_visible
    assert "private-" not in repr(diagnostic)


def test_crosscheck_accepts_matching_independent_shape_and_decisions() -> None:
    production = build_production_projection_diagnostic(_metadata())
    scalar = _scalar_from_production()

    assert projection_crosscheck_passes(
        production,
        scalar,
        pixel_tolerance_px=1e-9,
        depth_tolerance_m=1e-10,
        minimum_annotation_count=1,
        minimum_finite_positive_depth_center_count=1,
    )


def test_crosscheck_rejects_numeric_or_decision_disagreement() -> None:
    production = build_production_projection_diagnostic(_metadata())
    scalar = _scalar_from_production()
    annotation = scalar.annotations[0]
    shifted = replace(
        scalar,
        annotations=(
            replace(
                annotation,
                center_uv_px=(50.0 + 2e-9, 50.0),
            ),
        ),
    )
    flipped = replace(
        scalar,
        annotations=(
            replace(
                annotation,
                box_any_visible=not annotation.box_any_visible,
            ),
        ),
    )

    for candidate in (shifted, flipped):
        assert not projection_crosscheck_passes(
            production,
            candidate,
            pixel_tolerance_px=1e-9,
            depth_tolerance_m=1e-10,
            minimum_annotation_count=1,
            minimum_finite_positive_depth_center_count=1,
        )
