"""Token-free metadata extraction for the frozen nuScenes-mini replay source."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import Literal, Never

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.adapters.nuscenes import (
    CAMERA_CHANNEL,
    LIDAR_CHANNEL,
    NuScenesMiniMetadata,
    SampleAnnotationRow,
    SampleDataRow,
    SampleRow,
    SceneRow,
)
from fusion_fault_bench.contracts.replay_v1 import M5_SCENE_NAMES
from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.camera import PinholeCamera
from fusion_fault_bench.replay_geometry import (
    NominalEligibility,
    RigidTransform3,
    evaluate_nominal_eligibility,
)

type FloatArray = npt.NDArray[np.float64]
type DifferenceMethod = Literal["centered", "one-sided", "zero-order-hold"]

M5_REPLAY_SCENE_NAMES = M5_SCENE_NAMES
DESCRIPTOR_QUANTILE_PROBABILITIES = (0.0, 0.25, 0.5, 0.75, 1.0)

_CENTERED_MAXIMUM_SPAN_S = 3.0
_ONE_SIDED_MAXIMUM_SPAN_S = 1.5


class ReplaySourceErrorCode(StrEnum):
    """Stable failure classes that cannot expose private dataset identifiers."""

    SCENE_SELECTION_INVALID = "scene-selection-invalid"
    RELATION_INVALID = "metadata-relation-invalid"
    CHAIN_INVALID = "metadata-chain-invalid"
    KEYFRAME_INVALID = "keyframe-selection-invalid"
    GEOMETRY_INVALID = "nominal-geometry-invalid"


_ERROR_MESSAGES: Mapping[ReplaySourceErrorCode, str] = MappingProxyType(
    {
        ReplaySourceErrorCode.SCENE_SELECTION_INVALID: (
            "replay source scene selection validation failed"
        ),
        ReplaySourceErrorCode.RELATION_INVALID: (
            "replay source metadata relationship validation failed"
        ),
        ReplaySourceErrorCode.CHAIN_INVALID: ("replay source metadata chain validation failed"),
        ReplaySourceErrorCode.KEYFRAME_INVALID: (
            "replay source key-frame selection validation failed"
        ),
        ReplaySourceErrorCode.GEOMETRY_INVALID: (
            "replay source nominal geometry validation failed"
        ),
    }
)


class ReplaySourceError(RuntimeError):
    """One fixed-message replay extraction failure."""

    code: ReplaySourceErrorCode

    def __init__(self, code: ReplaySourceErrorCode) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value!r})"


class _PrivateRepr:
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _raise(code: ReplaySourceErrorCode) -> Never:
    raise ReplaySourceError(code) from None


def _finite_vector(
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


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class FiniteDifferenceProxy(_PrivateRepr):
    """One official-style annotation finite-difference proxy."""

    vector_per_s: FloatArray
    method: DifferenceMethod

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector_per_s",
            _finite_vector(
                self.vector_per_s,
                shape=(3,),
                field_name="vector_per_s",
            ),
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ReplaySensorSnapshot(_PrivateRepr):
    """Token-free pose and calibration for one selected keyframe."""

    timestamp_us: int
    global_from_ego: RigidTransform3
    ego_from_sensor: RigidTransform3
    camera: PinholeCamera | None

    def __post_init__(self) -> None:
        if type(self.timestamp_us) is not int or self.timestamp_us <= 0:
            raise ValueError("timestamp_us must be a positive integer")


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ReplayObjectFrame(_PrivateRepr):
    """One local-only annotation-derived object row with an opaque identity."""

    object_id: str
    center_global_m: FloatArray
    size_width_length_height_m: FloatArray
    orientation_global_wxyz: FloatArray
    velocity_global_mps: FloatArray
    velocity_method: DifferenceMethod
    acceleration_global_mps2: FloatArray
    acceleration_method: DifferenceMethod
    category_name: str
    visibility_level: str
    num_lidar_points: int
    support: NominalEligibility

    def __post_init__(self) -> None:
        ordinal = self.object_id.removeprefix("track:")
        if not self.object_id.startswith("track:") or len(ordinal) < 4 or not ordinal.isdigit():
            raise ValueError("object_id must use the frozen opaque track format")
        for field_name, shape in (
            ("center_global_m", (3,)),
            ("size_width_length_height_m", (3,)),
            ("orientation_global_wxyz", (4,)),
            ("velocity_global_mps", (3,)),
            ("acceleration_global_mps2", (3,)),
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_vector(
                    getattr(self, field_name),
                    shape=shape,
                    field_name=field_name,
                ),
            )
        if np.any(self.size_width_length_height_m <= 0.0):
            raise ValueError("recorded box dimensions must be positive")
        if not self.category_name:
            raise ValueError("category_name must be non-empty")
        if not self.visibility_level:
            raise ValueError("visibility_level must be non-empty")
        if type(self.num_lidar_points) is not int or self.num_lidar_points < 0:
            raise ValueError("num_lidar_points must be a non-negative integer")


@dataclass(frozen=True, slots=True, repr=False)
class ReplayFrame(_PrivateRepr):
    """One frame in exact sample-chain order, including empty eligible frames."""

    frame_index: int
    sample_timestamp_us: int
    reference_time_s: float
    lidar: ReplaySensorSnapshot
    camera: ReplaySensorSnapshot
    objects: tuple[ReplayObjectFrame, ...]

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if type(self.sample_timestamp_us) is not int or self.sample_timestamp_us <= 0:
            raise ValueError("sample_timestamp_us must be a positive integer")
        if not math.isfinite(self.reference_time_s) or self.reference_time_s < 0.0:
            raise ValueError("reference_time_s must be finite and non-negative")
        object_ids = tuple(item.object_id for item in self.objects)
        if object_ids != tuple(sorted(object_ids, key=lambda item: item.encode("utf-8"))):
            raise ValueError("objects must use canonical opaque-ID order")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("objects must be unique within a frame")

    @property
    def eligible_objects(self) -> tuple[ReplayObjectFrame, ...]:
        """Return final-eligible rows without changing their canonical order."""

        return tuple(item for item in self.objects if item.support.eligible)


@dataclass(frozen=True, slots=True, repr=False)
class ReplayScene(_PrivateRepr):
    """One complete scene with public sequence and opaque log-group IDs."""

    scene_name: str
    sequence_id: str
    log_group_id: str
    frames: tuple[ReplayFrame, ...]

    def __post_init__(self) -> None:
        if self.sequence_id != f"nuscenes:{self.scene_name}":
            raise ValueError("sequence_id must be derived from scene_name")
        log_ordinal = self.log_group_id.removeprefix("log-group:")
        if (
            not self.log_group_id.startswith("log-group:")
            or len(log_ordinal) < 2
            or not log_ordinal.isdigit()
        ):
            raise ValueError("log_group_id must be opaque")
        if not self.frames:
            raise ValueError("a replay scene must contain at least one frame")
        if tuple(frame.frame_index for frame in self.frames) != tuple(range(len(self.frames))):
            raise ValueError("frame indices must be contiguous from zero")
        reference_times = tuple(frame.reference_time_s for frame in self.frames)
        if reference_times[0] != 0.0 or any(
            right <= left for left, right in pairwise(reference_times)
        ):
            raise ValueError("reference times must start at zero and increase")


@dataclass(frozen=True, slots=True, repr=False)
class ReplayPopulation(_PrivateRepr):
    """The exact ordered ten-scene M5 metadata replay population."""

    scenes: tuple[ReplayScene, ...]

    def __post_init__(self) -> None:
        if tuple(scene.scene_name for scene in self.scenes) != M5_REPLAY_SCENE_NAMES:
            raise ValueError("replay population does not contain the frozen scene order")


@dataclass(frozen=True, slots=True)
class SupportWaterfall:
    """Cumulative counts in the preregistered support-filter order."""

    all_annotations: int
    roi_pass: int
    camera_center_pass: int
    lidar_points_positive: int
    final_eligible: int

    def __post_init__(self) -> None:
        counts = (
            self.all_annotations,
            self.roi_pass,
            self.camera_center_pass,
            self.lidar_points_positive,
            self.final_eligible,
        )
        if any(type(item) is not int or item < 0 for item in counts):
            raise ValueError("support waterfall counts must be non-negative integers")
        if any(right > left for left, right in pairwise(counts)):
            raise ValueError("support waterfall must be cumulative")


@dataclass(frozen=True, slots=True)
class DescriptorQuantiles:
    """Five linear quantiles at probabilities ``0,.25,.5,.75,1``."""

    minimum: float
    lower_quartile: float
    median: float
    upper_quartile: float
    maximum: float

    def __post_init__(self) -> None:
        values = (
            self.minimum,
            self.lower_quartile,
            self.median,
            self.upper_quartile,
            self.maximum,
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("descriptor quantiles must be finite")
        if any(right < left for left, right in pairwise(values)):
            raise ValueError("descriptor quantiles must be ordered")


@dataclass(frozen=True, slots=True, repr=False)
class SceneDescriptorPrimitives(_PrivateRepr):
    """Aggregate-safe primitives; no frame, annotation, or private token IDs."""

    sample_count: int
    reference_time_delta_s: tuple[float, ...]
    camera_minus_lidar_offset_s: tuple[float, ...]
    support_waterfall: SupportWaterfall
    eligible_object_frame_count: int
    unique_eligible_track_count: int
    eligible_track_length_frames: tuple[int, ...]
    ego_range_m: tuple[float, ...]
    ego_bearing_rad: tuple[float, ...]
    box_width_m: tuple[float, ...]
    box_length_m: tuple[float, ...]
    box_height_m: tuple[float, ...]
    finite_difference_speed_mps: tuple[float, ...]
    finite_difference_acceleration_mps2: tuple[float, ...]
    visibility_counts: tuple[tuple[str, int], ...]
    num_lidar_points: tuple[int, ...]
    zero_order_hold_velocity_fraction: float
    category_counts: tuple[tuple[str, int], ...]


def finite_difference_proxy(
    *,
    current_time_us: int,
    current_value: npt.ArrayLike,
    previous: tuple[int, npt.ArrayLike] | None,
    following: tuple[int, npt.ArrayLike] | None,
) -> FiniteDifferenceProxy:
    """Apply the frozen centered/endpoint secant and gap-limit rule."""

    if type(current_time_us) is not int or current_time_us <= 0:
        raise ValueError("current_time_us must be a positive integer")
    current = _finite_vector(current_value, shape=(3,), field_name="current_value")
    zero = np.zeros(3, dtype=np.float64)
    if previous is not None and following is not None:
        previous_time, previous_value = previous
        following_time, following_value = following
        span_s = (following_time - previous_time) / 1_000_000.0
        if (
            type(previous_time) is int
            and type(following_time) is int
            and previous_time < current_time_us < following_time
            and span_s <= _CENTERED_MAXIMUM_SPAN_S
        ):
            before = _finite_vector(
                previous_value,
                shape=(3,),
                field_name="previous_value",
            )
            after = _finite_vector(
                following_value,
                shape=(3,),
                field_name="following_value",
            )
            return FiniteDifferenceProxy(
                vector_per_s=(after - before) / span_s,
                method="centered",
            )
        return FiniteDifferenceProxy(
            vector_per_s=zero,
            method="zero-order-hold",
        )
    neighbor = previous if previous is not None else following
    if neighbor is None:
        return FiniteDifferenceProxy(
            vector_per_s=zero,
            method="zero-order-hold",
        )
    neighbor_time, neighbor_value = neighbor
    if type(neighbor_time) is not int:
        raise ValueError("neighbor timestamp must be an integer")
    span_s = abs(neighbor_time - current_time_us) / 1_000_000.0
    if span_s == 0.0 or span_s > _ONE_SIDED_MAXIMUM_SPAN_S:
        return FiniteDifferenceProxy(
            vector_per_s=zero,
            method="zero-order-hold",
        )
    neighbor_vector = _finite_vector(
        neighbor_value,
        shape=(3,),
        field_name="neighbor_value",
    )
    if neighbor_time < current_time_us:
        difference = current - neighbor_vector
    else:
        difference = neighbor_vector - current
    return FiniteDifferenceProxy(
        vector_per_s=difference / span_s,
        method="one-sided",
    )


def _sample_chain(
    metadata: NuScenesMiniMetadata,
    scene: SceneRow,
) -> tuple[SampleRow, ...]:
    rows: list[SampleRow] = []
    current_token = scene.first_sample_token
    previous_token = ""
    visited: set[str] = set()
    while current_token:
        if current_token in visited:
            _raise(ReplaySourceErrorCode.CHAIN_INVALID)
        visited.add(current_token)
        try:
            row = metadata.samples[current_token]
        except KeyError:
            _raise(ReplaySourceErrorCode.RELATION_INVALID)
        if row.scene_token != scene.token or row.prev != previous_token:
            _raise(ReplaySourceErrorCode.CHAIN_INVALID)
        rows.append(row)
        previous_token = row.token
        current_token = row.next
    if len(rows) != scene.nbr_samples or not rows or rows[-1].token != scene.last_sample_token:
        _raise(ReplaySourceErrorCode.CHAIN_INVALID)
    return tuple(rows)


def _selected_keyframes(
    metadata: NuScenesMiniMetadata,
) -> Mapping[tuple[str, str], tuple[SampleDataRow, ...]]:
    selected: dict[tuple[str, str], list[SampleDataRow]] = defaultdict(list)
    try:
        for row in metadata.sample_data.values():
            if not row.is_key_frame:
                continue
            calibration = metadata.calibrated_sensors[row.calibrated_sensor_token]
            sensor = metadata.sensors[calibration.sensor_token]
            if sensor.channel in {CAMERA_CHANNEL, LIDAR_CHANNEL}:
                selected[(row.sample_token, sensor.channel)].append(row)
    except KeyError:
        _raise(ReplaySourceErrorCode.RELATION_INVALID)
    return MappingProxyType({key: tuple(value) for key, value in selected.items()})


def _sensor_snapshot(
    metadata: NuScenesMiniMetadata,
    row: SampleDataRow,
    *,
    is_camera: bool,
) -> ReplaySensorSnapshot:
    try:
        calibration = metadata.calibrated_sensors[row.calibrated_sensor_token]
        pose = metadata.ego_poses[row.ego_pose_token]
    except KeyError:
        _raise(ReplaySourceErrorCode.RELATION_INVALID)
    if pose.timestamp != row.timestamp:
        _raise(ReplaySourceErrorCode.RELATION_INVALID)
    try:
        camera = (
            PinholeCamera(
                intrinsic=np.asarray(
                    calibration.camera_intrinsic,
                    dtype=np.float64,
                ),
                width_px=row.width,
                height_px=row.height,
            )
            if is_camera
            else None
        )
        return ReplaySensorSnapshot(
            timestamp_us=row.timestamp,
            global_from_ego=RigidTransform3.from_quaternion_wxyz(
                translation_m=pose.translation,
                quaternion_wxyz=pose.rotation,
            ),
            ego_from_sensor=RigidTransform3.from_quaternion_wxyz(
                translation_m=calibration.translation,
                quaternion_wxyz=calibration.rotation,
            ),
            camera=camera,
        )
    except ValueError:
        _raise(ReplaySourceErrorCode.GEOMETRY_INVALID)


def _annotation_neighbor(
    metadata: NuScenesMiniMetadata,
    annotation: SampleAnnotationRow,
    *,
    previous: bool,
) -> tuple[int, npt.ArrayLike] | None:
    token = annotation.prev if previous else annotation.next
    if not token:
        return None
    try:
        neighbor = metadata.sample_annotations[token]
        sample = metadata.samples[neighbor.sample_token]
    except KeyError:
        _raise(ReplaySourceErrorCode.RELATION_INVALID)
    if neighbor.instance_token != annotation.instance_token:
        _raise(ReplaySourceErrorCode.CHAIN_INVALID)
    return (sample.timestamp, neighbor.translation)


def _motion_proxies(
    metadata: NuScenesMiniMetadata,
    annotations: Sequence[SampleAnnotationRow],
) -> tuple[
    Mapping[str, FiniteDifferenceProxy],
    Mapping[str, FiniteDifferenceProxy],
]:
    velocities: dict[str, FiniteDifferenceProxy] = {}
    for annotation in annotations:
        try:
            current_time = metadata.samples[annotation.sample_token].timestamp
        except KeyError:
            _raise(ReplaySourceErrorCode.RELATION_INVALID)
        velocities[annotation.token] = finite_difference_proxy(
            current_time_us=current_time,
            current_value=annotation.translation,
            previous=_annotation_neighbor(
                metadata,
                annotation,
                previous=True,
            ),
            following=_annotation_neighbor(
                metadata,
                annotation,
                previous=False,
            ),
        )
    accelerations: dict[str, FiniteDifferenceProxy] = {}
    for annotation in annotations:
        try:
            current_time = metadata.samples[annotation.sample_token].timestamp
            previous: tuple[int, npt.ArrayLike] | None = None
            following: tuple[int, npt.ArrayLike] | None = None
            if annotation.prev:
                previous_annotation = metadata.sample_annotations[annotation.prev]
                previous = (
                    metadata.samples[previous_annotation.sample_token].timestamp,
                    velocities[annotation.prev].vector_per_s,
                )
            if annotation.next:
                following_annotation = metadata.sample_annotations[annotation.next]
                following = (
                    metadata.samples[following_annotation.sample_token].timestamp,
                    velocities[annotation.next].vector_per_s,
                )
        except KeyError:
            _raise(ReplaySourceErrorCode.RELATION_INVALID)
        accelerations[annotation.token] = finite_difference_proxy(
            current_time_us=current_time,
            current_value=velocities[annotation.token].vector_per_s,
            previous=previous,
            following=following,
        )
    return MappingProxyType(velocities), MappingProxyType(accelerations)


def _visibility_level(
    metadata: NuScenesMiniMetadata,
    annotation: SampleAnnotationRow,
) -> str:
    if not annotation.visibility_token:
        return "unknown"
    try:
        return metadata.visibility[annotation.visibility_token].level
    except KeyError:
        _raise(ReplaySourceErrorCode.RELATION_INVALID)


def _category_name(
    metadata: NuScenesMiniMetadata,
    annotation: SampleAnnotationRow,
) -> str:
    try:
        instance = metadata.instances[annotation.instance_token]
        return metadata.categories[instance.category_token].name
    except KeyError:
        _raise(ReplaySourceErrorCode.RELATION_INVALID)


def _object_row(
    metadata: NuScenesMiniMetadata,
    annotation: SampleAnnotationRow,
    *,
    object_id: str,
    lidar: ReplaySensorSnapshot,
    camera_snapshot: ReplaySensorSnapshot,
    velocities: Mapping[str, FiniteDifferenceProxy],
    accelerations: Mapping[str, FiniteDifferenceProxy],
) -> ReplayObjectFrame:
    camera = camera_snapshot.camera
    if camera is None:
        _raise(ReplaySourceErrorCode.KEYFRAME_INVALID)
    try:
        support = evaluate_nominal_eligibility(
            center_global_m=annotation.translation,
            global_from_reference_ego=lidar.global_from_ego,
            global_from_camera_ego=camera_snapshot.global_from_ego,
            true_camera_ego_from_camera=camera_snapshot.ego_from_sensor,
            camera=camera,
            num_lidar_points=annotation.num_lidar_pts,
        )
        velocity = velocities[annotation.token]
        acceleration = accelerations[annotation.token]
        return ReplayObjectFrame(
            object_id=object_id,
            center_global_m=np.asarray(annotation.translation, dtype=np.float64),
            size_width_length_height_m=np.asarray(
                annotation.size,
                dtype=np.float64,
            ),
            orientation_global_wxyz=np.asarray(
                annotation.rotation,
                dtype=np.float64,
            ),
            velocity_global_mps=velocity.vector_per_s,
            velocity_method=velocity.method,
            acceleration_global_mps2=acceleration.vector_per_s,
            acceleration_method=acceleration.method,
            category_name=_category_name(metadata, annotation),
            visibility_level=_visibility_level(metadata, annotation),
            num_lidar_points=annotation.num_lidar_pts,
            support=support,
        )
    except ReplaySourceError:
        raise
    except (KeyError, TypeError, ValueError):
        _raise(ReplaySourceErrorCode.GEOMETRY_INVALID)


def extract_m5_replay_source(
    metadata: NuScenesMiniMetadata,
) -> ReplayPopulation:
    """Extract the exact ten scenes without opening raw sensor payloads."""

    scenes_by_name: dict[str, SceneRow] = {}
    for scene in metadata.scenes.values():
        if scene.name in scenes_by_name:
            _raise(ReplaySourceErrorCode.SCENE_SELECTION_INVALID)
        scenes_by_name[scene.name] = scene
    if (
        tuple(sorted(scenes_by_name, key=lambda item: item.encode("utf-8")))
        != M5_REPLAY_SCENE_NAMES
    ):
        _raise(ReplaySourceErrorCode.SCENE_SELECTION_INVALID)
    try:
        if any(scene.log_token not in metadata.logs for scene in scenes_by_name.values()):
            _raise(ReplaySourceErrorCode.RELATION_INVALID)
    except TypeError:
        _raise(ReplaySourceErrorCode.RELATION_INVALID)
    log_tokens = sorted(
        {scene.log_token for scene in scenes_by_name.values()},
        key=lambda item: item.encode("utf-8"),
    )
    opaque_log_groups = {token: f"log-group:{index:02d}" for index, token in enumerate(log_tokens)}
    keyframes = _selected_keyframes(metadata)
    annotations_by_sample: dict[str, list[SampleAnnotationRow]] = defaultdict(list)
    for annotation in metadata.sample_annotations.values():
        annotations_by_sample[annotation.sample_token].append(annotation)

    replay_scenes: list[ReplayScene] = []
    for scene_name in M5_REPLAY_SCENE_NAMES:
        scene = scenes_by_name[scene_name]
        samples = _sample_chain(metadata, scene)
        scene_sample_tokens = {sample.token for sample in samples}
        annotations = tuple(
            annotation
            for annotation in metadata.sample_annotations.values()
            if annotation.sample_token in scene_sample_tokens
        )
        instance_tokens = sorted(
            {annotation.instance_token for annotation in annotations},
            key=lambda item: item.encode("utf-8"),
        )
        object_ids = {token: f"track:{index:04d}" for index, token in enumerate(instance_tokens)}
        velocities, accelerations = _motion_proxies(metadata, annotations)

        first_lidar_rows = keyframes.get(
            (samples[0].token, LIDAR_CHANNEL),
            (),
        )
        if len(first_lidar_rows) != 1:
            _raise(ReplaySourceErrorCode.KEYFRAME_INVALID)
        first_reference_timestamp_us = first_lidar_rows[0].timestamp
        previous_reference_timestamp_us: int | None = None
        frames: list[ReplayFrame] = []
        for frame_index, sample in enumerate(samples):
            lidar_rows = keyframes.get((sample.token, LIDAR_CHANNEL), ())
            camera_rows = keyframes.get((sample.token, CAMERA_CHANNEL), ())
            if len(lidar_rows) != 1 or len(camera_rows) != 1:
                _raise(ReplaySourceErrorCode.KEYFRAME_INVALID)
            lidar = _sensor_snapshot(
                metadata,
                lidar_rows[0],
                is_camera=False,
            )
            camera = _sensor_snapshot(
                metadata,
                camera_rows[0],
                is_camera=True,
            )
            if (
                previous_reference_timestamp_us is not None
                and lidar.timestamp_us <= previous_reference_timestamp_us
            ):
                _raise(ReplaySourceErrorCode.CHAIN_INVALID)
            previous_reference_timestamp_us = lidar.timestamp_us

            sample_annotations = annotations_by_sample.get(sample.token, [])
            if len({row.instance_token for row in sample_annotations}) != len(sample_annotations):
                _raise(ReplaySourceErrorCode.RELATION_INVALID)
            ordered_annotations = sorted(
                sample_annotations,
                key=lambda row: object_ids[row.instance_token].encode("utf-8"),
            )
            objects = tuple(
                _object_row(
                    metadata,
                    annotation,
                    object_id=object_ids[annotation.instance_token],
                    lidar=lidar,
                    camera_snapshot=camera,
                    velocities=velocities,
                    accelerations=accelerations,
                )
                for annotation in ordered_annotations
            )
            frames.append(
                ReplayFrame(
                    frame_index=frame_index,
                    sample_timestamp_us=sample.timestamp,
                    reference_time_s=(lidar.timestamp_us - first_reference_timestamp_us)
                    / 1_000_000.0,
                    lidar=lidar,
                    camera=camera,
                    objects=objects,
                )
            )
        replay_scenes.append(
            ReplayScene(
                scene_name=scene_name,
                sequence_id=f"nuscenes:{scene_name}",
                log_group_id=opaque_log_groups[scene.log_token],
                frames=tuple(frames),
            )
        )
    return ReplayPopulation(scenes=tuple(replay_scenes))


def support_waterfall(scene: ReplayScene) -> SupportWaterfall:
    """Count cumulative support filters without changing the frozen mask."""

    objects = tuple(item for frame in scene.frames for item in frame.objects)
    roi = tuple(item for item in objects if item.support.roi_pass)
    camera = tuple(item for item in roi if item.support.camera_center_pass)
    lidar = tuple(item for item in camera if item.support.lidar_points_pass)
    eligible = tuple(
        item
        for item in lidar
        if item.support.camera_estimator_available and item.support.lidar_estimator_available
    )
    return SupportWaterfall(
        all_annotations=len(objects),
        roi_pass=len(roi),
        camera_center_pass=len(camera),
        lidar_points_positive=len(lidar),
        final_eligible=len(eligible),
    )


def descriptor_quantiles(values: Sequence[float]) -> DescriptorQuantiles:
    """Compute the frozen NumPy-linear five-number descriptor summary."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError("descriptor values must be one non-empty finite vector")
    quantiles = np.quantile(
        array,
        np.asarray(DESCRIPTOR_QUANTILE_PROBABILITIES, dtype=np.float64),
        method="linear",
    )
    return DescriptorQuantiles(
        minimum=float(quantiles[0]),
        lower_quartile=float(quantiles[1]),
        median=float(quantiles[2]),
        upper_quartile=float(quantiles[3]),
        maximum=float(quantiles[4]),
    )


def build_scene_descriptor_primitives(
    scene: ReplayScene,
) -> SceneDescriptorPrimitives:
    """Build preregistered, aggregate-safe descriptor inputs for one scene."""

    eligible = tuple(
        item for frame in scene.frames for item in frame.objects if item.support.eligible
    )
    if not eligible:
        raise ValueError("scene has no final-eligible object frames")
    track_counts = Counter(item.object_id for item in eligible)
    visibility = Counter(item.visibility_level for item in eligible)
    categories = Counter(item.category_name for item in eligible)
    ranges = tuple(
        math.hypot(
            float(item.support.center_reference_ego_m[0]),
            float(item.support.center_reference_ego_m[1]),
        )
        for item in eligible
    )
    bearings = tuple(
        math.atan2(
            float(item.support.center_reference_ego_m[1]),
            float(item.support.center_reference_ego_m[0]),
        )
        for item in eligible
    )
    zero_hold_count = sum(item.velocity_method == "zero-order-hold" for item in eligible)
    return SceneDescriptorPrimitives(
        sample_count=len(scene.frames),
        reference_time_delta_s=tuple(
            right.reference_time_s - left.reference_time_s for left, right in pairwise(scene.frames)
        ),
        camera_minus_lidar_offset_s=tuple(
            (frame.camera.timestamp_us - frame.lidar.timestamp_us) / 1_000_000.0
            for frame in scene.frames
        ),
        support_waterfall=support_waterfall(scene),
        eligible_object_frame_count=len(eligible),
        unique_eligible_track_count=len(track_counts),
        eligible_track_length_frames=tuple(
            track_counts[key] for key in sorted(track_counts, key=lambda item: item.encode("utf-8"))
        ),
        ego_range_m=ranges,
        ego_bearing_rad=bearings,
        box_width_m=tuple(float(item.size_width_length_height_m[0]) for item in eligible),
        box_length_m=tuple(float(item.size_width_length_height_m[1]) for item in eligible),
        box_height_m=tuple(float(item.size_width_length_height_m[2]) for item in eligible),
        finite_difference_speed_mps=tuple(
            float(np.linalg.norm(item.velocity_global_mps[:2])) for item in eligible
        ),
        finite_difference_acceleration_mps2=tuple(
            float(np.linalg.norm(item.acceleration_global_mps2[:2])) for item in eligible
        ),
        visibility_counts=tuple(
            (key, visibility[key])
            for key in sorted(visibility, key=lambda item: item.encode("utf-8"))
        ),
        num_lidar_points=tuple(item.num_lidar_points for item in eligible),
        zero_order_hold_velocity_fraction=zero_hold_count / len(eligible),
        category_counts=tuple(
            (key, categories[key])
            for key in sorted(categories, key=lambda item: item.encode("utf-8"))
        ),
    )


def log_group_count(population: ReplayPopulation) -> int:
    """Count opaque log clusters without returning private tokens."""

    return len({scene.log_group_id for scene in population.scenes})
