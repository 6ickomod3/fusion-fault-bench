"""Strict, local-only nuScenes-mini metadata loading and validation.

The adapter intentionally keeps dataset paths out of returned objects and
sanitizes all public failures. It reads JSON metadata and inspects key-frame
blob filesystem metadata, but it never opens image or point-cloud payloads.
"""

from __future__ import annotations

import json
import math
import stat
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Never, Protocol, cast

VERSION_DIRECTORY = "v1.0-mini"
PROFILE = "official-nuscenes-v1.0-mini"
CAMERA_CHANNEL = "CAM_FRONT"
LIDAR_CHANNEL = "LIDAR_TOP"
DATASET_AUTHENTICATION = "summary-does-not-authenticate-dataset-bytes"
QUATERNION_UNIT_NORM_TOLERANCE = 1e-6

_EXPECTED_HEADLINE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "scene": 10,
        "sample": 404,
        "sample_annotation": 18_538,
    }
)

type Vector3 = tuple[float, float, float]
type QuaternionWxyz = tuple[float, float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]
type CameraIntrinsic = Matrix3 | tuple[()]


class NuScenesAdapterErrorCode(StrEnum):
    """Stable, privacy-safe adapter failure categories."""

    ROOT_INVALID = "dataset-root-invalid"
    TABLE_INVALID = "metadata-table-invalid"
    ROW_INVALID = "metadata-row-invalid"
    RELATION_INVALID = "metadata-relation-invalid"
    CHAIN_INVALID = "metadata-chain-invalid"
    KEYFRAME_INVALID = "keyframe-selection-invalid"
    BLOB_INVALID = "keyframe-blob-invalid"


_ERROR_MESSAGES: Mapping[NuScenesAdapterErrorCode, str] = MappingProxyType(
    {
        NuScenesAdapterErrorCode.ROOT_INVALID: "nuScenes dataset root validation failed",
        NuScenesAdapterErrorCode.TABLE_INVALID: "nuScenes metadata table validation failed",
        NuScenesAdapterErrorCode.ROW_INVALID: "nuScenes metadata row validation failed",
        NuScenesAdapterErrorCode.RELATION_INVALID: (
            "nuScenes metadata relationship validation failed"
        ),
        NuScenesAdapterErrorCode.CHAIN_INVALID: "nuScenes metadata chain validation failed",
        NuScenesAdapterErrorCode.KEYFRAME_INVALID: (
            "nuScenes key-frame selection validation failed"
        ),
        NuScenesAdapterErrorCode.BLOB_INVALID: "nuScenes key-frame blob validation failed",
    }
)


class NuScenesAdapterError(RuntimeError):
    """A fixed-message error that cannot disclose local dataset details."""

    code: NuScenesAdapterErrorCode

    def __init__(self, code: NuScenesAdapterErrorCode) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value!r})"


class _TokenRow(Protocol):
    @property
    def token(self) -> str: ...


class _PrivateRepr:
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class AttributeRow(_PrivateRepr):
    token: str
    name: str
    description: str


@dataclass(frozen=True, slots=True, repr=False)
class CalibratedSensorRow(_PrivateRepr):
    token: str
    sensor_token: str
    translation: Vector3
    rotation: QuaternionWxyz
    camera_intrinsic: CameraIntrinsic


@dataclass(frozen=True, slots=True, repr=False)
class CategoryRow(_PrivateRepr):
    token: str
    name: str
    description: str
    index: int | None


@dataclass(frozen=True, slots=True, repr=False)
class EgoPoseRow(_PrivateRepr):
    token: str
    translation: Vector3
    rotation: QuaternionWxyz
    timestamp: int


@dataclass(frozen=True, slots=True, repr=False)
class InstanceRow(_PrivateRepr):
    token: str
    category_token: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str


@dataclass(frozen=True, slots=True, repr=False)
class LogRow(_PrivateRepr):
    token: str
    logfile: str
    vehicle: str
    date_captured: str
    location: str


@dataclass(frozen=True, slots=True, repr=False)
class SampleRow(_PrivateRepr):
    token: str
    timestamp: int
    scene_token: str
    next: str
    prev: str


@dataclass(frozen=True, slots=True, repr=False)
class SampleAnnotationRow(_PrivateRepr):
    token: str
    sample_token: str
    instance_token: str
    attribute_tokens: tuple[str, ...]
    visibility_token: str
    translation: Vector3
    size: Vector3
    rotation: QuaternionWxyz
    num_lidar_pts: int
    num_radar_pts: int
    next: str
    prev: str


@dataclass(frozen=True, slots=True, repr=False)
class SampleDataRow(_PrivateRepr):
    token: str
    sample_token: str
    ego_pose_token: str
    calibrated_sensor_token: str
    filename: str
    fileformat: str
    width: int
    height: int
    timestamp: int
    is_key_frame: bool
    next: str
    prev: str


@dataclass(frozen=True, slots=True, repr=False)
class SceneRow(_PrivateRepr):
    token: str
    name: str
    description: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str


@dataclass(frozen=True, slots=True, repr=False)
class SensorRow(_PrivateRepr):
    token: str
    channel: str
    modality: str


@dataclass(frozen=True, slots=True, repr=False)
class VisibilityRow(_PrivateRepr):
    token: str
    level: str
    description: str


@dataclass(frozen=True, slots=True)
class NuScenesMiniValidation:
    """Sanitized adapter evidence matching the public result allowlist subset."""

    profile: str = PROFILE
    expected_headline_counts: Mapping[str, int] = field(
        default_factory=lambda: _EXPECTED_HEADLINE_COUNTS
    )
    headline_profile_passed_attested: bool = False
    structural_integrity_passed_attested: bool = True
    keyframe_blob_check_count: int = 0
    keyframe_blob_validation_passed_attested: bool = True
    dataset_authentication: str = DATASET_AUTHENTICATION

    def to_public_mapping(self) -> dict[str, object]:
        """Return only fields approved for the public geometry-validation record."""

        return {
            "profile": self.profile,
            "expected_headline_counts": dict(self.expected_headline_counts),
            "headline_profile_passed_attested": self.headline_profile_passed_attested,
            "structural_integrity_passed_attested": (self.structural_integrity_passed_attested),
            "keyframe_blob_check_count": self.keyframe_blob_check_count,
            "keyframe_blob_validation_passed_attested": (
                self.keyframe_blob_validation_passed_attested
            ),
            "dataset_authentication": self.dataset_authentication,
        }


@dataclass(frozen=True, slots=True, repr=False)
class NuScenesMiniMetadata(_PrivateRepr):
    """Validated, local-only typed indexes for later geometry integration."""

    attributes: Mapping[str, AttributeRow]
    calibrated_sensors: Mapping[str, CalibratedSensorRow]
    categories: Mapping[str, CategoryRow]
    ego_poses: Mapping[str, EgoPoseRow]
    instances: Mapping[str, InstanceRow]
    logs: Mapping[str, LogRow]
    samples: Mapping[str, SampleRow]
    sample_annotations: Mapping[str, SampleAnnotationRow]
    sample_data: Mapping[str, SampleDataRow]
    scenes: Mapping[str, SceneRow]
    sensors: Mapping[str, SensorRow]
    visibility: Mapping[str, VisibilityRow]
    validation: NuScenesMiniValidation


_JsonObject = Mapping[str, Any]


def _raise(code: NuScenesAdapterErrorCode) -> Never:
    raise NuScenesAdapterError(code) from None


def _required(value: _JsonObject, key: str) -> Any:
    if key not in value:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return value[key]


def _string(value: Any, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return value


def _token(value: Any, *, allow_empty: bool = False) -> str:
    return _string(value, allow_empty=allow_empty)


def _integer(value: Any, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return value


def _timestamp(value: Any) -> int:
    return _integer(value, minimum=1)


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    result = float(value)
    if not math.isfinite(result):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return result


def _vector3(value: Any, *, positive: bool = False) -> Vector3:
    if not isinstance(value, list):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    items = cast("list[Any]", value)
    if len(items) != 3:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    result = (_number(items[0]), _number(items[1]), _number(items[2]))
    if positive and any(component <= 0.0 for component in result):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return result


def _quaternion(value: Any) -> QuaternionWxyz:
    if not isinstance(value, list):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    items = cast("list[Any]", value)
    if len(items) != 4:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    result = (
        _number(items[0]),
        _number(items[1]),
        _number(items[2]),
        _number(items[3]),
    )
    norm = math.sqrt(math.fsum(component * component for component in result))
    if abs(norm - 1.0) > QUATERNION_UNIT_NORM_TOLERANCE:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return result


def _camera_intrinsic(value: Any) -> CameraIntrinsic:
    if not isinstance(value, list):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    items = cast("list[Any]", value)
    if not items:
        return ()
    if len(items) != 3:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return (
        _vector3(items[0]),
        _vector3(items[1]),
        _vector3(items[2]),
    )


def _token_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return tuple(_token(item) for item in cast("list[Any]", value))


def _parse_attribute(value: _JsonObject) -> AttributeRow:
    return AttributeRow(
        token=_token(_required(value, "token")),
        name=_string(_required(value, "name"), allow_empty=False),
        description=_string(_required(value, "description")),
    )


def _parse_calibrated_sensor(value: _JsonObject) -> CalibratedSensorRow:
    return CalibratedSensorRow(
        token=_token(_required(value, "token")),
        sensor_token=_token(_required(value, "sensor_token")),
        translation=_vector3(_required(value, "translation")),
        rotation=_quaternion(_required(value, "rotation")),
        camera_intrinsic=_camera_intrinsic(_required(value, "camera_intrinsic")),
    )


def _parse_category(value: _JsonObject) -> CategoryRow:
    raw_index = value.get("index")
    return CategoryRow(
        token=_token(_required(value, "token")),
        name=_string(_required(value, "name"), allow_empty=False),
        description=_string(_required(value, "description")),
        index=None if raw_index is None else _integer(raw_index, minimum=0),
    )


def _parse_ego_pose(value: _JsonObject) -> EgoPoseRow:
    return EgoPoseRow(
        token=_token(_required(value, "token")),
        translation=_vector3(_required(value, "translation")),
        rotation=_quaternion(_required(value, "rotation")),
        timestamp=_timestamp(_required(value, "timestamp")),
    )


def _parse_instance(value: _JsonObject) -> InstanceRow:
    return InstanceRow(
        token=_token(_required(value, "token")),
        category_token=_token(_required(value, "category_token")),
        nbr_annotations=_integer(_required(value, "nbr_annotations"), minimum=1),
        first_annotation_token=_token(_required(value, "first_annotation_token")),
        last_annotation_token=_token(_required(value, "last_annotation_token")),
    )


def _parse_log(value: _JsonObject) -> LogRow:
    return LogRow(
        token=_token(_required(value, "token")),
        logfile=_string(_required(value, "logfile"), allow_empty=False),
        vehicle=_string(_required(value, "vehicle"), allow_empty=False),
        date_captured=_string(_required(value, "date_captured"), allow_empty=False),
        location=_string(_required(value, "location"), allow_empty=False),
    )


def _parse_sample(value: _JsonObject) -> SampleRow:
    return SampleRow(
        token=_token(_required(value, "token")),
        timestamp=_timestamp(_required(value, "timestamp")),
        scene_token=_token(_required(value, "scene_token")),
        next=_token(_required(value, "next"), allow_empty=True),
        prev=_token(_required(value, "prev"), allow_empty=True),
    )


def _parse_sample_annotation(value: _JsonObject) -> SampleAnnotationRow:
    return SampleAnnotationRow(
        token=_token(_required(value, "token")),
        sample_token=_token(_required(value, "sample_token")),
        instance_token=_token(_required(value, "instance_token")),
        attribute_tokens=_token_list(_required(value, "attribute_tokens")),
        visibility_token=_token(_required(value, "visibility_token"), allow_empty=True),
        translation=_vector3(_required(value, "translation")),
        size=_vector3(_required(value, "size"), positive=True),
        rotation=_quaternion(_required(value, "rotation")),
        num_lidar_pts=_integer(_required(value, "num_lidar_pts"), minimum=0),
        num_radar_pts=_integer(_required(value, "num_radar_pts"), minimum=0),
        next=_token(_required(value, "next"), allow_empty=True),
        prev=_token(_required(value, "prev"), allow_empty=True),
    )


def _parse_sample_data(value: _JsonObject) -> SampleDataRow:
    is_key_frame = _required(value, "is_key_frame")
    if type(is_key_frame) is not bool:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return SampleDataRow(
        token=_token(_required(value, "token")),
        sample_token=_token(_required(value, "sample_token")),
        ego_pose_token=_token(_required(value, "ego_pose_token")),
        calibrated_sensor_token=_token(_required(value, "calibrated_sensor_token")),
        filename=_string(_required(value, "filename"), allow_empty=False),
        fileformat=_string(_required(value, "fileformat"), allow_empty=False),
        width=_integer(_required(value, "width"), minimum=0),
        height=_integer(_required(value, "height"), minimum=0),
        timestamp=_timestamp(_required(value, "timestamp")),
        is_key_frame=is_key_frame,
        next=_token(_required(value, "next"), allow_empty=True),
        prev=_token(_required(value, "prev"), allow_empty=True),
    )


def _parse_scene(value: _JsonObject) -> SceneRow:
    return SceneRow(
        token=_token(_required(value, "token")),
        name=_string(_required(value, "name"), allow_empty=False),
        description=_string(_required(value, "description")),
        log_token=_token(_required(value, "log_token")),
        nbr_samples=_integer(_required(value, "nbr_samples"), minimum=1),
        first_sample_token=_token(_required(value, "first_sample_token")),
        last_sample_token=_token(_required(value, "last_sample_token")),
    )


def _parse_sensor(value: _JsonObject) -> SensorRow:
    modality = _string(_required(value, "modality"), allow_empty=False)
    if modality not in {"camera", "lidar", "radar"}:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    return SensorRow(
        token=_token(_required(value, "token")),
        channel=_string(_required(value, "channel"), allow_empty=False),
        modality=modality,
    )


def _parse_visibility(value: _JsonObject) -> VisibilityRow:
    return VisibilityRow(
        token=_token(_required(value, "token")),
        level=_string(_required(value, "level"), allow_empty=False),
        description=_string(_required(value, "description")),
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _raise(NuScenesAdapterErrorCode.TABLE_INVALID)
        result[key] = value
    return result


def _reject_nonstandard_number(_: str) -> Never:
    _raise(NuScenesAdapterErrorCode.TABLE_INVALID)


def _resolve_version_root(root: object) -> tuple[Path, Path]:
    if not isinstance(root, Path):
        _raise(NuScenesAdapterErrorCode.ROOT_INVALID)
    try:
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            _raise(NuScenesAdapterErrorCode.ROOT_INVALID)
        version_candidate = resolved_root / VERSION_DIRECTORY
        version_stat = version_candidate.lstat()
        if stat.S_ISLNK(version_stat.st_mode) or not stat.S_ISDIR(version_stat.st_mode):
            _raise(NuScenesAdapterErrorCode.ROOT_INVALID)
        resolved_version = version_candidate.resolve(strict=True)
        resolved_version.relative_to(resolved_root)
    except NuScenesAdapterError:
        raise
    except (OSError, RuntimeError, ValueError):
        _raise(NuScenesAdapterErrorCode.ROOT_INVALID)
    return resolved_root, resolved_version


def _read_table(version_root: Path, table: str) -> list[Any]:
    path = version_root / f"{table}.json"
    try:
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            _raise(NuScenesAdapterErrorCode.TABLE_INVALID)
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_nonstandard_number,
            )
    except NuScenesAdapterError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        _raise(NuScenesAdapterErrorCode.TABLE_INVALID)
    if not isinstance(value, list):
        _raise(NuScenesAdapterErrorCode.TABLE_INVALID)
    return cast("list[Any]", value)


def _load_index[Row: _TokenRow](
    version_root: Path,
    table: str,
    parser: Callable[[_JsonObject], Row],
) -> Mapping[str, Row]:
    index: dict[str, Row] = {}
    for raw_row in _read_table(version_root, table):
        if not isinstance(raw_row, dict):
            _raise(NuScenesAdapterErrorCode.ROW_INVALID)
        row = parser(cast("_JsonObject", raw_row))
        if row.token in index:
            _raise(NuScenesAdapterErrorCode.ROW_INVALID)
        index[row.token] = row
    return MappingProxyType(index)


def _require_reference(token: str, index: Mapping[str, object]) -> None:
    if token not in index:
        _raise(NuScenesAdapterErrorCode.RELATION_INVALID)


def _validate_pinhole(intrinsic: CameraIntrinsic) -> None:
    if not intrinsic:
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)
    row0, row1, row2 = intrinsic
    if row0[0] <= 0.0 or row1[1] <= 0.0 or row1[0] != 0.0 or row2 != (0.0, 0.0, 1.0):
        _raise(NuScenesAdapterErrorCode.ROW_INVALID)


def _expected_modality(channel: str) -> str | None:
    if channel.startswith("CAM_"):
        return "camera"
    if channel.startswith("LIDAR_"):
        return "lidar"
    if channel.startswith("RADAR_"):
        return "radar"
    return None


def _validate_sensor_relations(
    calibrated_sensors: Mapping[str, CalibratedSensorRow],
    sensors: Mapping[str, SensorRow],
) -> None:
    for sensor in sensors.values():
        expected = _expected_modality(sensor.channel)
        if expected is not None and sensor.modality != expected:
            _raise(NuScenesAdapterErrorCode.RELATION_INVALID)

    for calibrated in calibrated_sensors.values():
        _require_reference(calibrated.sensor_token, sensors)
        sensor = sensors[calibrated.sensor_token]
        if sensor.modality == "camera":
            _validate_pinhole(calibrated.camera_intrinsic)
        elif calibrated.camera_intrinsic != ():
            _raise(NuScenesAdapterErrorCode.ROW_INVALID)


def _validate_sample_chains(
    scenes: Mapping[str, SceneRow],
    samples: Mapping[str, SampleRow],
    logs: Mapping[str, LogRow],
) -> None:
    assigned: defaultdict[str, set[str]] = defaultdict(set)
    for sample in samples.values():
        _require_reference(sample.scene_token, scenes)
        assigned[sample.scene_token].add(sample.token)
        if sample.prev:
            _require_reference(sample.prev, samples)
            previous = samples[sample.prev]
            if previous.next != sample.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            if previous.scene_token != sample.scene_token or sample.timestamp <= previous.timestamp:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
        if sample.next:
            _require_reference(sample.next, samples)
            if samples[sample.next].prev != sample.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)

    for scene in scenes.values():
        _require_reference(scene.log_token, logs)
        _require_reference(scene.first_sample_token, samples)
        _require_reference(scene.last_sample_token, samples)
        first = samples[scene.first_sample_token]
        last = samples[scene.last_sample_token]
        if (
            first.scene_token != scene.token
            or last.scene_token != scene.token
            or first.prev
            or last.next
        ):
            _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)

        visited: set[str] = set()
        current_token = first.token
        while True:
            if current_token in visited:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            current = samples[current_token]
            if current.scene_token != scene.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            visited.add(current_token)
            if current_token == last.token:
                break
            if not current.next:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            current_token = current.next

        if len(visited) != scene.nbr_samples or visited != assigned[scene.token]:
            _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)


def _validate_annotation_chains(
    annotations: Mapping[str, SampleAnnotationRow],
    instances: Mapping[str, InstanceRow],
    categories: Mapping[str, CategoryRow],
    attributes: Mapping[str, AttributeRow],
    visibility: Mapping[str, VisibilityRow],
    samples: Mapping[str, SampleRow],
) -> None:
    assigned: defaultdict[str, set[str]] = defaultdict(set)
    for instance in instances.values():
        _require_reference(instance.category_token, categories)

    for annotation in annotations.values():
        _require_reference(annotation.sample_token, samples)
        _require_reference(annotation.instance_token, instances)
        assigned[annotation.instance_token].add(annotation.token)
        for attribute_token in annotation.attribute_tokens:
            _require_reference(attribute_token, attributes)
        if annotation.visibility_token:
            _require_reference(annotation.visibility_token, visibility)
        if annotation.prev:
            _require_reference(annotation.prev, annotations)
            previous = annotations[annotation.prev]
            if previous.next != annotation.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            previous_sample = samples[previous.sample_token]
            current_sample = samples[annotation.sample_token]
            if (
                previous.instance_token != annotation.instance_token
                or previous_sample.scene_token != current_sample.scene_token
                or current_sample.timestamp <= previous_sample.timestamp
            ):
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
        if annotation.next:
            _require_reference(annotation.next, annotations)
            if annotations[annotation.next].prev != annotation.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)

    for instance in instances.values():
        _require_reference(instance.first_annotation_token, annotations)
        _require_reference(instance.last_annotation_token, annotations)
        first = annotations[instance.first_annotation_token]
        last = annotations[instance.last_annotation_token]
        if (
            first.instance_token != instance.token
            or last.instance_token != instance.token
            or first.prev
            or last.next
        ):
            _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)

        visited: set[str] = set()
        current_token = first.token
        while True:
            if current_token in visited:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            current = annotations[current_token]
            if current.instance_token != instance.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            visited.add(current_token)
            if current_token == last.token:
                break
            if not current.next:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            current_token = current.next

        if len(visited) != instance.nbr_annotations or visited != assigned[instance.token]:
            _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)


def _validate_sample_data_chains(
    sample_data: Mapping[str, SampleDataRow],
    samples: Mapping[str, SampleRow],
    ego_poses: Mapping[str, EgoPoseRow],
    calibrated_sensors: Mapping[str, CalibratedSensorRow],
    sensors: Mapping[str, SensorRow],
) -> Mapping[str, tuple[str, str]]:
    channel_by_token: dict[str, tuple[str, str]] = {}
    groups: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    calibration_tokens: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for record in sample_data.values():
        _require_reference(record.sample_token, samples)
        _require_reference(record.ego_pose_token, ego_poses)
        _require_reference(record.calibrated_sensor_token, calibrated_sensors)
        pose = ego_poses[record.ego_pose_token]
        if pose.timestamp != record.timestamp:
            _raise(NuScenesAdapterErrorCode.RELATION_INVALID)
        calibrated = calibrated_sensors[record.calibrated_sensor_token]
        sensor = sensors[calibrated.sensor_token]
        if sensor.modality == "camera":
            if record.width <= 0 or record.height <= 0:
                _raise(NuScenesAdapterErrorCode.ROW_INVALID)
        elif record.width < 0 or record.height < 0:
            _raise(NuScenesAdapterErrorCode.ROW_INVALID)

        scene_token = samples[record.sample_token].scene_token
        channel_by_token[record.token] = (sensor.channel, sensor.modality)
        group = (scene_token, sensor.channel)
        groups[group].add(record.token)
        calibration_tokens[group].add(record.calibrated_sensor_token)

        if record.prev:
            _require_reference(record.prev, sample_data)
            previous = sample_data[record.prev]
            if previous.next != record.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            previous_calibrated = calibrated_sensors[previous.calibrated_sensor_token]
            previous_sensor = sensors[previous_calibrated.sensor_token]
            previous_scene = samples[previous.sample_token].scene_token
            if (
                previous_scene != scene_token
                or previous_sensor.channel != sensor.channel
                or previous.calibrated_sensor_token != record.calibrated_sensor_token
                or record.timestamp <= previous.timestamp
            ):
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
        if record.next:
            _require_reference(record.next, sample_data)
            if sample_data[record.next].prev != record.token:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)

    for group, members in groups.items():
        if len(calibration_tokens[group]) != 1:
            _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
        heads = [token for token in members if not sample_data[token].prev]
        tails = [token for token in members if not sample_data[token].next]
        if len(heads) != 1 or len(tails) != 1:
            _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
        visited: set[str] = set()
        current_token = heads[0]
        while True:
            if current_token in visited or current_token not in members:
                _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)
            visited.add(current_token)
            current = sample_data[current_token]
            if not current.next:
                break
            current_token = current.next
        if visited != members:
            _raise(NuScenesAdapterErrorCode.CHAIN_INVALID)

    return MappingProxyType(channel_by_token)


def _validate_blob(dataset_root: Path, filename: str) -> None:
    relative = PurePosixPath(filename)
    if (
        not filename
        or relative.is_absolute()
        or "\\" in filename
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        _raise(NuScenesAdapterErrorCode.BLOB_INVALID)

    current = dataset_root
    try:
        for index, part in enumerate(relative.parts):
            current = current / part
            item_stat = current.lstat()
            if stat.S_ISLNK(item_stat.st_mode):
                _raise(NuScenesAdapterErrorCode.BLOB_INVALID)
            is_last = index == len(relative.parts) - 1
            if is_last:
                if not stat.S_ISREG(item_stat.st_mode) or item_stat.st_size <= 0:
                    _raise(NuScenesAdapterErrorCode.BLOB_INVALID)
            elif not stat.S_ISDIR(item_stat.st_mode):
                _raise(NuScenesAdapterErrorCode.BLOB_INVALID)
        current.resolve(strict=True).relative_to(dataset_root)
    except NuScenesAdapterError:
        raise
    except (OSError, RuntimeError, ValueError):
        _raise(NuScenesAdapterErrorCode.BLOB_INVALID)


def _validate_keyframes(
    dataset_root: Path,
    samples: Mapping[str, SampleRow],
    sample_data: Mapping[str, SampleDataRow],
    channel_by_token: Mapping[str, tuple[str, str]],
) -> int:
    keyframes_by_sample: defaultdict[str, defaultdict[str, list[SampleDataRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in sample_data.values():
        if record.is_key_frame:
            channel, _ = channel_by_token[record.token]
            keyframes_by_sample[record.sample_token][channel].append(record)

    selected: list[SampleDataRow] = []
    for sample in samples.values():
        by_channel = keyframes_by_sample[sample.token]
        camera = by_channel[CAMERA_CHANNEL]
        lidar = by_channel[LIDAR_CHANNEL]
        if len(camera) != 1 or len(lidar) != 1:
            _raise(NuScenesAdapterErrorCode.KEYFRAME_INVALID)
        if (
            channel_by_token[camera[0].token][1] != "camera"
            or channel_by_token[lidar[0].token][1] != "lidar"
        ):
            _raise(NuScenesAdapterErrorCode.KEYFRAME_INVALID)
        selected.extend((camera[0], lidar[0]))

    for record in selected:
        _validate_blob(dataset_root, record.filename)
    return len(selected)


def _load_validated(root: Path) -> NuScenesMiniMetadata:
    dataset_root, version_root = _resolve_version_root(root)
    attributes = _load_index(version_root, "attribute", _parse_attribute)
    calibrated_sensors = _load_index(version_root, "calibrated_sensor", _parse_calibrated_sensor)
    categories = _load_index(version_root, "category", _parse_category)
    ego_poses = _load_index(version_root, "ego_pose", _parse_ego_pose)
    instances = _load_index(version_root, "instance", _parse_instance)
    logs = _load_index(version_root, "log", _parse_log)
    samples = _load_index(version_root, "sample", _parse_sample)
    sample_annotations = _load_index(version_root, "sample_annotation", _parse_sample_annotation)
    sample_data = _load_index(version_root, "sample_data", _parse_sample_data)
    scenes = _load_index(version_root, "scene", _parse_scene)
    sensors = _load_index(version_root, "sensor", _parse_sensor)
    visibility = _load_index(version_root, "visibility", _parse_visibility)

    _validate_sensor_relations(calibrated_sensors, sensors)
    _validate_sample_chains(scenes, samples, logs)
    _validate_annotation_chains(
        sample_annotations,
        instances,
        categories,
        attributes,
        visibility,
        samples,
    )
    channel_by_token = _validate_sample_data_chains(
        sample_data,
        samples,
        ego_poses,
        calibrated_sensors,
        sensors,
    )
    blob_check_count = _validate_keyframes(
        dataset_root,
        samples,
        sample_data,
        channel_by_token,
    )
    headline_passed = (
        len(scenes) == _EXPECTED_HEADLINE_COUNTS["scene"]
        and len(samples) == _EXPECTED_HEADLINE_COUNTS["sample"]
        and len(sample_annotations) == _EXPECTED_HEADLINE_COUNTS["sample_annotation"]
    )
    validation = NuScenesMiniValidation(
        headline_profile_passed_attested=headline_passed,
        keyframe_blob_check_count=blob_check_count,
    )
    return NuScenesMiniMetadata(
        attributes=attributes,
        calibrated_sensors=calibrated_sensors,
        categories=categories,
        ego_poses=ego_poses,
        instances=instances,
        logs=logs,
        samples=samples,
        sample_annotations=sample_annotations,
        sample_data=sample_data,
        scenes=scenes,
        sensors=sensors,
        visibility=visibility,
        validation=validation,
    )


def load_nuscenes_mini(root: Path) -> NuScenesMiniMetadata:
    """Load, index, and fully validate local nuScenes-mini metadata."""

    return _load_validated(root)


def validate_nuscenes_mini(root: Path) -> NuScenesMiniValidation:
    """Return only sanitized aggregate attestations for a local mini dataset."""

    return _load_validated(root).validation
