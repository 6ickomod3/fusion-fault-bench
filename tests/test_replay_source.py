from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fusion_fault_bench.adapters.nuscenes import (
    CalibratedSensorRow,
    CategoryRow,
    EgoPoseRow,
    InstanceRow,
    LogRow,
    NuScenesMiniMetadata,
    NuScenesMiniValidation,
    SampleAnnotationRow,
    SampleDataRow,
    SampleRow,
    SceneRow,
    SensorRow,
    VisibilityRow,
)
from fusion_fault_bench.contracts.replay_v1 import M5_SCENE_NAMES
from fusion_fault_bench.replay_source import (
    DescriptorQuantiles,
    FiniteDifferenceProxy,
    ReplayPopulation,
    ReplaySourceError,
    ReplaySourceErrorCode,
    SupportWaterfall,
    build_scene_descriptor_primitives,
    descriptor_quantiles,
    extract_m5_replay_source,
    finite_difference_proxy,
    log_group_count,
    support_waterfall,
)


def _metadata() -> NuScenesMiniMetadata:
    identity = (1.0, 0.0, 0.0, 0.0)
    sensors = {
        "private-camera-sensor": SensorRow(
            token="private-camera-sensor",
            channel="CAM_FRONT",
            modality="camera",
        ),
        "private-lidar-sensor": SensorRow(
            token="private-lidar-sensor",
            channel="LIDAR_TOP",
            modality="lidar",
        ),
    }
    calibrated_sensors = {
        "private-camera-calibration": CalibratedSensorRow(
            token="private-camera-calibration",
            sensor_token="private-camera-sensor",
            translation=(0.0, 0.0, 0.0),
            rotation=identity,
            camera_intrinsic=(
                (10.0, 0.0, 50.0),
                (0.0, 10.0, 50.0),
                (0.0, 0.0, 1.0),
            ),
        ),
        "private-lidar-calibration": CalibratedSensorRow(
            token="private-lidar-calibration",
            sensor_token="private-lidar-sensor",
            translation=(0.0, 0.0, 0.0),
            rotation=identity,
            camera_intrinsic=(),
        ),
    }
    categories = {
        "private-category": CategoryRow(
            token="private-category",
            name="vehicle.car",
            description="",
            index=1,
        )
    }
    visibility = {
        "private-visibility": VisibilityRow(
            token="private-visibility",
            level="v80-100",
            description="",
        )
    }
    logs = {
        "private-log-a": LogRow(
            token="private-log-a",
            logfile="",
            vehicle="",
            date_captured="",
            location="",
        ),
        "private-log-b": LogRow(
            token="private-log-b",
            logfile="",
            vehicle="",
            date_captured="",
            location="",
        ),
    }
    scenes: dict[str, SceneRow] = {}
    samples: dict[str, SampleRow] = {}
    sample_data: dict[str, SampleDataRow] = {}
    ego_poses: dict[str, EgoPoseRow] = {}
    annotations: dict[str, SampleAnnotationRow] = {}
    instances: dict[str, InstanceRow] = {}

    for scene_index, scene_name in enumerate(M5_SCENE_NAMES):
        scene_token = f"private-scene-{scene_index}"
        frame_count = 2 if scene_index == 0 else 1
        sample_tokens = tuple(
            f"private-sample-{scene_index}-{frame_index}" for frame_index in range(frame_count)
        )
        for frame_index, sample_token in enumerate(sample_tokens):
            sample_timestamp = 1_000_000 + scene_index * 100_000_000 + frame_index * 1_000_000
            camera_timestamp = sample_timestamp + 50
            lidar_timestamp = sample_timestamp + 100
            samples[sample_token] = SampleRow(
                token=sample_token,
                timestamp=sample_timestamp,
                scene_token=scene_token,
                prev="" if frame_index == 0 else sample_tokens[frame_index - 1],
                next=("" if frame_index == frame_count - 1 else sample_tokens[frame_index + 1]),
            )
            camera_pose_token = f"private-camera-pose-{scene_index}-{frame_index}"
            lidar_pose_token = f"private-lidar-pose-{scene_index}-{frame_index}"
            ego_poses[camera_pose_token] = EgoPoseRow(
                token=camera_pose_token,
                translation=(0.0, 0.0, 0.0),
                rotation=identity,
                timestamp=camera_timestamp,
            )
            ego_poses[lidar_pose_token] = EgoPoseRow(
                token=lidar_pose_token,
                translation=(0.0, 0.0, 0.0),
                rotation=identity,
                timestamp=lidar_timestamp,
            )
            camera_data_token = f"private-camera-data-{scene_index}-{frame_index}"
            lidar_data_token = f"private-lidar-data-{scene_index}-{frame_index}"
            sample_data[camera_data_token] = SampleDataRow(
                token=camera_data_token,
                sample_token=sample_token,
                ego_pose_token=camera_pose_token,
                calibrated_sensor_token="private-camera-calibration",
                filename=f"samples/CAM_FRONT/private-{scene_index}-{frame_index}.jpg",
                fileformat="jpg",
                width=100,
                height=100,
                timestamp=camera_timestamp,
                is_key_frame=True,
                prev="",
                next="",
            )
            sample_data[lidar_data_token] = SampleDataRow(
                token=lidar_data_token,
                sample_token=sample_token,
                ego_pose_token=lidar_pose_token,
                calibrated_sensor_token="private-lidar-calibration",
                filename=f"samples/LIDAR_TOP/private-{scene_index}-{frame_index}.bin",
                fileformat="bin",
                width=0,
                height=0,
                timestamp=lidar_timestamp,
                is_key_frame=True,
                prev="",
                next="",
            )
        scenes[scene_token] = SceneRow(
            token=scene_token,
            name=scene_name,
            description="",
            log_token="private-log-b" if scene_index < 5 else "private-log-a",
            nbr_samples=frame_count,
            first_sample_token=sample_tokens[0],
            last_sample_token=sample_tokens[-1],
        )
        suffixes = ("z", "a") if scene_index == 0 else ("only",)
        for suffix_index, suffix in enumerate(suffixes):
            instance_token = f"private-instance-{scene_index}-{suffix}"
            annotation_token = f"private-annotation-{scene_index}-{suffix}"
            instances[instance_token] = InstanceRow(
                token=instance_token,
                category_token="private-category",
                nbr_annotations=1,
                first_annotation_token=annotation_token,
                last_annotation_token=annotation_token,
            )
            annotations[annotation_token] = SampleAnnotationRow(
                token=annotation_token,
                sample_token=sample_tokens[0],
                instance_token=instance_token,
                attribute_tokens=(),
                visibility_token=("private-visibility" if suffix_index == 0 else ""),
                translation=(12.0 if suffix == "z" else 10.0, 0.0, 10.0),
                size=(2.0, 4.0, 1.5),
                rotation=identity,
                num_lidar_pts=5,
                num_radar_pts=0,
                prev="",
                next="",
            )
    return NuScenesMiniMetadata(
        attributes={},
        calibrated_sensors=calibrated_sensors,
        categories=categories,
        ego_poses=ego_poses,
        instances=instances,
        logs=logs,
        samples=samples,
        sample_annotations=annotations,
        sample_data=sample_data,
        scenes=scenes,
        sensors=sensors,
        visibility=visibility,
        validation=NuScenesMiniValidation(),
    )


def test_extracts_exact_scene_frame_object_and_opaque_group_order() -> None:
    population = extract_m5_replay_source(_metadata())
    first = population.scenes[0]

    assert tuple(scene.scene_name for scene in population.scenes) == M5_SCENE_NAMES
    assert tuple(scene.sequence_id for scene in population.scenes) == tuple(
        f"nuscenes:{name}" for name in M5_SCENE_NAMES
    )
    assert first.log_group_id == "log-group:01"
    assert population.scenes[-1].log_group_id == "log-group:00"
    assert log_group_count(population) == 2
    assert tuple(frame.frame_index for frame in first.frames) == (0, 1)
    assert tuple(frame.reference_time_s for frame in first.frames) == (0.0, 1.0)
    assert tuple(item.object_id for item in first.frames[0].objects) == (
        "track:0000",
        "track:0001",
    )
    assert tuple(float(item.center_global_m[0]) for item in first.frames[0].objects) == (
        10.0,
        12.0,
    )
    assert first.frames[1].objects == ()
    assert first.frames[1].eligible_objects == ()
    assert all(item.support.eligible for item in first.frames[0].objects)
    assert "private-" not in repr(population)


def test_source_table_insertion_order_cannot_change_replay_rows() -> None:
    metadata = _metadata()
    shuffled = replace(
        metadata,
        ego_poses=dict(reversed(tuple(metadata.ego_poses.items()))),
        instances=dict(reversed(tuple(metadata.instances.items()))),
        samples=dict(reversed(tuple(metadata.samples.items()))),
        sample_annotations=dict(reversed(tuple(metadata.sample_annotations.items()))),
        sample_data=dict(reversed(tuple(metadata.sample_data.items()))),
        scenes=dict(reversed(tuple(metadata.scenes.items()))),
    )

    def signature(source: NuScenesMiniMetadata) -> tuple[object, ...]:
        population = extract_m5_replay_source(source)
        return tuple(
            (
                scene.scene_name,
                scene.log_group_id,
                tuple(
                    (
                        frame.frame_index,
                        frame.reference_time_s,
                        tuple(
                            (
                                item.object_id,
                                tuple(float(value) for value in item.center_global_m),
                                item.velocity_method,
                                tuple(float(value) for value in item.velocity_global_mps),
                                item.support.eligible,
                            )
                            for item in frame.objects
                        ),
                    )
                    for frame in scene.frames
                ),
            )
            for scene in population.scenes
        )

    assert signature(metadata) == signature(shuffled)


def test_velocity_proxy_uses_centered_endpoint_and_zero_hold_gap_rules() -> None:
    centered = finite_difference_proxy(
        current_time_us=2_000_000,
        current_value=(100.0, 100.0, 100.0),
        previous=(1_000_000, (0.0, 0.0, 0.0)),
        following=(3_000_000, (4.0, -2.0, 1.0)),
    )
    endpoint = finite_difference_proxy(
        current_time_us=2_000_000,
        current_value=(2.0, -1.0, 0.5),
        previous=(1_000_000, (0.0, 0.0, 0.0)),
        following=None,
    )
    leading_endpoint = finite_difference_proxy(
        current_time_us=1_000_000,
        current_value=(0.0, 0.0, 0.0),
        previous=None,
        following=(2_000_000, (2.0, -1.0, 0.5)),
    )
    too_wide = finite_difference_proxy(
        current_time_us=3_000_000,
        current_value=(2.0, 0.0, 0.0),
        previous=(1_000_000, (0.0, 0.0, 0.0)),
        following=None,
    )

    assert centered.method == "centered"
    assert np.allclose(centered.vector_per_s, (2.0, -1.0, 0.5))
    assert endpoint.method == "one-sided"
    assert np.allclose(endpoint.vector_per_s, (2.0, -1.0, 0.5))
    assert leading_endpoint.method == "one-sided"
    assert np.allclose(leading_endpoint.vector_per_s, (2.0, -1.0, 0.5))
    assert too_wide.method == "zero-order-hold"
    assert np.array_equal(too_wide.vector_per_s, np.zeros(3))


def test_support_and_descriptor_primitives_preserve_empty_frames() -> None:
    first = extract_m5_replay_source(_metadata()).scenes[0]
    waterfall = support_waterfall(first)
    descriptors = build_scene_descriptor_primitives(first)
    quantiles = descriptor_quantiles((0.0, 10.0))

    assert (
        waterfall.all_annotations,
        waterfall.roi_pass,
        waterfall.camera_center_pass,
        waterfall.lidar_points_positive,
        waterfall.final_eligible,
    ) == (2, 2, 2, 2, 2)
    assert descriptors.sample_count == 2
    assert descriptors.reference_time_delta_s == (1.0,)
    assert descriptors.camera_minus_lidar_offset_s == pytest.approx((-0.00005, -0.00005))
    assert descriptors.eligible_object_frame_count == 2
    assert descriptors.unique_eligible_track_count == 2
    assert descriptors.eligible_track_length_frames == (1, 1)
    assert descriptors.zero_order_hold_velocity_fraction == 1.0
    assert descriptors.category_counts == (("vehicle.car", 2),)
    assert descriptors.visibility_counts == (("unknown", 1), ("v80-100", 1))
    assert (
        quantiles.minimum,
        quantiles.lower_quartile,
        quantiles.median,
        quantiles.upper_quartile,
        quantiles.maximum,
    ) == (0.0, 2.5, 5.0, 7.5, 10.0)


def test_missing_or_duplicate_keyframes_fail_with_sanitized_error() -> None:
    metadata = _metadata()
    first_sample = next(
        sample
        for sample in metadata.samples.values()
        if sample.scene_token == metadata.scenes["private-scene-0"].token
    )
    camera_rows = [
        row
        for row in metadata.sample_data.values()
        if row.sample_token == first_sample.token
        and row.calibrated_sensor_token == "private-camera-calibration"
    ]
    duplicate = replace(
        camera_rows[0],
        token="private-duplicate-camera-data",
        filename="samples/CAM_FRONT/private-duplicate.jpg",
    )
    changed = replace(
        metadata,
        sample_data={**metadata.sample_data, duplicate.token: duplicate},
    )

    with pytest.raises(ReplaySourceError) as caught:
        extract_m5_replay_source(changed)
    assert caught.value.code is ReplaySourceErrorCode.KEYFRAME_INVALID
    rendered = f"{caught.value!s} {caught.value!r}"
    assert "private-" not in rendered


def test_acceleration_neighbor_keyerror_is_sanitized() -> None:
    metadata = _metadata()
    current_token = "private-annotation-0-z"
    foreign_token = "private-annotation-1-only"
    current = metadata.sample_annotations[current_token]
    foreign = metadata.sample_annotations[foreign_token]
    changed = replace(
        metadata,
        sample_annotations={
            **metadata.sample_annotations,
            current_token: replace(current, prev=foreign_token),
            foreign_token: replace(
                foreign,
                instance_token=current.instance_token,
                next=current_token,
            ),
        },
    )

    with pytest.raises(ReplaySourceError) as caught:
        extract_m5_replay_source(changed)
    assert caught.value.code is ReplaySourceErrorCode.RELATION_INVALID
    rendered = f"{caught.value!s} {caught.value!r}"
    assert "private-" not in rendered


def test_source_value_contracts_reject_nonfinite_and_noncanonical_rows() -> None:
    population = extract_m5_replay_source(_metadata())
    scene = population.scenes[0]
    frame = scene.frames[0]
    first, second = frame.objects

    invalid_constructors = (
        lambda: FiniteDifferenceProxy(
            vector_per_s=np.zeros(2),
            method="centered",
        ),
        lambda: FiniteDifferenceProxy(
            vector_per_s=np.asarray((0.0, np.nan, 0.0)),
            method="centered",
        ),
        lambda: replace(frame.lidar, timestamp_us=0),
        lambda: replace(first, object_id="private-object"),
        lambda: replace(first, size_width_length_height_m=(0.0, 1.0, 1.0)),
        lambda: replace(first, category_name=""),
        lambda: replace(first, visibility_level=""),
        lambda: replace(first, num_lidar_points=-1),
        lambda: replace(frame, frame_index=-1),
        lambda: replace(frame, sample_timestamp_us=0),
        lambda: replace(frame, reference_time_s=np.nan),
        lambda: replace(frame, objects=(second, first)),
        lambda: replace(frame, objects=(first, first)),
        lambda: replace(scene, sequence_id="private-sequence"),
        lambda: replace(scene, log_group_id="private-log"),
        lambda: replace(scene, frames=()),
        lambda: replace(scene, frames=(replace(frame, frame_index=1),)),
        lambda: replace(scene, frames=(replace(frame, reference_time_s=1.0),)),
        lambda: ReplayPopulation(scenes=tuple(reversed(population.scenes))),
        lambda: SupportWaterfall(-1, 0, 0, 0, 0),
        lambda: SupportWaterfall(1, 2, 1, 1, 1),
        lambda: DescriptorQuantiles(0.0, 1.0, np.nan, 3.0, 4.0),
        lambda: DescriptorQuantiles(0.0, 2.0, 1.0, 3.0, 4.0),
    )

    for construct in invalid_constructors:
        with pytest.raises(ValueError):
            construct()


def test_finite_difference_contract_and_fallback_branches() -> None:
    zero = np.zeros(3)

    with pytest.raises(ValueError, match="current_time_us"):
        finite_difference_proxy(
            current_time_us=0,
            current_value=zero,
            previous=None,
            following=None,
        )
    with pytest.raises(ValueError, match="neighbor timestamp"):
        finite_difference_proxy(
            current_time_us=1_000_000,
            current_value=zero,
            previous=("private", zero),  # type: ignore[arg-type]
            following=None,
        )

    no_neighbor = finite_difference_proxy(
        current_time_us=1_000_000,
        current_value=zero,
        previous=None,
        following=None,
    )
    invalid_center = finite_difference_proxy(
        current_time_us=2_000_000,
        current_value=zero,
        previous=(3_000_000, zero),
        following=(1_000_000, zero),
    )
    same_time = finite_difference_proxy(
        current_time_us=1_000_000,
        current_value=zero,
        previous=(1_000_000, zero),
        following=None,
    )

    assert no_neighbor.method == "zero-order-hold"
    assert invalid_center.method == "zero-order-hold"
    assert same_time.method == "zero-order-hold"


def test_descriptor_contracts_reject_invalid_inputs_and_empty_support() -> None:
    population = extract_m5_replay_source(_metadata())
    scene = population.scenes[1]
    empty_frame = replace(scene.frames[0], objects=())
    empty_scene = replace(scene, frames=(empty_frame,))

    for values in ((), (0.0, np.inf), ((0.0, 1.0),)):
        with pytest.raises(ValueError, match="descriptor values"):
            descriptor_quantiles(values)
    with pytest.raises(ValueError, match="no final-eligible"):
        build_scene_descriptor_primitives(empty_scene)


def _assert_source_error(
    metadata: NuScenesMiniMetadata,
    code: ReplaySourceErrorCode,
) -> None:
    with pytest.raises(ReplaySourceError) as caught:
        extract_m5_replay_source(metadata)
    assert caught.value.code is code
    assert "private-" not in f"{caught.value!s} {caught.value!r}"


def test_scene_selection_and_log_relations_fail_closed() -> None:
    metadata = _metadata()
    first_scene = metadata.scenes["private-scene-0"]
    duplicate = replace(
        first_scene,
        token="private-duplicate-scene",
        first_sample_token="private-sample-1-0",
        last_sample_token="private-sample-1-0",
        nbr_samples=1,
    )
    _assert_source_error(
        replace(
            metadata,
            scenes={**metadata.scenes, duplicate.token: duplicate},
        ),
        ReplaySourceErrorCode.SCENE_SELECTION_INVALID,
    )
    _assert_source_error(
        replace(
            metadata,
            scenes={
                token: row for token, row in metadata.scenes.items() if token != "private-scene-0"
            },
        ),
        ReplaySourceErrorCode.SCENE_SELECTION_INVALID,
    )
    _assert_source_error(
        replace(
            metadata,
            logs={
                token: row for token, row in metadata.logs.items() if token != first_scene.log_token
            },
        ),
        ReplaySourceErrorCode.RELATION_INVALID,
    )


def test_sample_chain_failures_are_sanitized() -> None:
    metadata = _metadata()
    first_token = "private-sample-0-0"
    second_token = "private-sample-0-1"
    first = metadata.samples[first_token]
    second = metadata.samples[second_token]

    mutations = (
        (
            replace(
                metadata,
                samples={
                    **metadata.samples,
                    first_token: replace(first, next=first_token),
                },
            ),
            ReplaySourceErrorCode.CHAIN_INVALID,
        ),
        (
            replace(
                metadata,
                samples={
                    token: row for token, row in metadata.samples.items() if token != first_token
                },
            ),
            ReplaySourceErrorCode.RELATION_INVALID,
        ),
        (
            replace(
                metadata,
                samples={
                    **metadata.samples,
                    first_token: replace(first, scene_token="private-scene-1"),
                },
            ),
            ReplaySourceErrorCode.CHAIN_INVALID,
        ),
        (
            replace(
                metadata,
                samples={
                    **metadata.samples,
                    second_token: replace(second, prev=""),
                },
            ),
            ReplaySourceErrorCode.CHAIN_INVALID,
        ),
        (
            replace(
                metadata,
                scenes={
                    **metadata.scenes,
                    "private-scene-0": replace(
                        metadata.scenes["private-scene-0"],
                        nbr_samples=3,
                    ),
                },
            ),
            ReplaySourceErrorCode.CHAIN_INVALID,
        ),
    )

    for changed, code in mutations:
        _assert_source_error(changed, code)


def test_keyframe_relation_timestamp_and_geometry_failures_are_sanitized() -> None:
    metadata = _metadata()
    camera_data_token = "private-camera-data-0-0"
    camera_row = metadata.sample_data[camera_data_token]
    camera_pose_token = camera_row.ego_pose_token

    mutations = (
        replace(
            metadata,
            sample_data={
                **metadata.sample_data,
                camera_data_token: replace(camera_row, is_key_frame=False),
            },
        ),
        replace(
            metadata,
            calibrated_sensors={
                token: row
                for token, row in metadata.calibrated_sensors.items()
                if token != camera_row.calibrated_sensor_token
            },
        ),
        replace(
            metadata,
            ego_poses={
                token: row
                for token, row in metadata.ego_poses.items()
                if token != camera_pose_token
            },
        ),
        replace(
            metadata,
            ego_poses={
                **metadata.ego_poses,
                camera_pose_token: replace(
                    metadata.ego_poses[camera_pose_token],
                    timestamp=camera_row.timestamp + 1,
                ),
            },
        ),
        replace(
            metadata,
            calibrated_sensors={
                **metadata.calibrated_sensors,
                camera_row.calibrated_sensor_token: replace(
                    metadata.calibrated_sensors[camera_row.calibrated_sensor_token],
                    camera_intrinsic=((1.0,),),
                ),
            },
        ),
    )
    expected = (
        ReplaySourceErrorCode.KEYFRAME_INVALID,
        ReplaySourceErrorCode.RELATION_INVALID,
        ReplaySourceErrorCode.RELATION_INVALID,
        ReplaySourceErrorCode.RELATION_INVALID,
        ReplaySourceErrorCode.GEOMETRY_INVALID,
    )

    for changed, code in zip(mutations, expected, strict=True):
        _assert_source_error(changed, code)


def test_annotation_relation_and_frame_invariants_fail_closed() -> None:
    metadata = _metadata()
    annotation_token = "private-annotation-0-z"
    annotation = metadata.sample_annotations[annotation_token]
    camera_data_token = "private-camera-data-0-0"
    lidar_second_token = "private-lidar-data-0-1"

    missing_visibility = replace(
        metadata,
        visibility={},
    )
    missing_category = replace(
        metadata,
        categories={},
    )
    duplicate_instance_token = "private-annotation-0-a"
    duplicate_instance = replace(
        metadata,
        sample_annotations={
            **metadata.sample_annotations,
            duplicate_instance_token: replace(
                metadata.sample_annotations[duplicate_instance_token],
                instance_token=annotation.instance_token,
            ),
        },
    )
    nonincreasing_lidar = replace(
        metadata,
        sample_data={
            **metadata.sample_data,
            lidar_second_token: replace(
                metadata.sample_data[lidar_second_token],
                timestamp=metadata.sample_data["private-lidar-data-0-0"].timestamp,
            ),
        },
        ego_poses={
            **metadata.ego_poses,
            metadata.sample_data[lidar_second_token].ego_pose_token: replace(
                metadata.ego_poses[metadata.sample_data[lidar_second_token].ego_pose_token],
                timestamp=metadata.sample_data["private-lidar-data-0-0"].timestamp,
            ),
        },
    )
    missing_sensor = replace(
        metadata,
        sensors={
            token: row
            for token, row in metadata.sensors.items()
            if token
            != metadata.calibrated_sensors[
                metadata.sample_data[camera_data_token].calibrated_sensor_token
            ].sensor_token
        },
    )

    for changed, code in (
        (missing_visibility, ReplaySourceErrorCode.RELATION_INVALID),
        (missing_category, ReplaySourceErrorCode.RELATION_INVALID),
        (duplicate_instance, ReplaySourceErrorCode.RELATION_INVALID),
        (nonincreasing_lidar, ReplaySourceErrorCode.CHAIN_INVALID),
        (missing_sensor, ReplaySourceErrorCode.RELATION_INVALID),
    ):
        _assert_source_error(changed, code)
