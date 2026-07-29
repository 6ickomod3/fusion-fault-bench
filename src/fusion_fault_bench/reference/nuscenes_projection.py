"""Independent scalar nuScenes projection path for the private M2 diagnostic.

This module intentionally imports neither the production geometry layer nor the
production nuScenes adapter. It independently loads the minimum raw metadata,
selects the frozen diagnostic sample, and applies scalar rigid-transform and
pinhole equations. Returned values contain no dataset identifiers or paths.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

type Vec2 = tuple[float, float]
type Vec3 = tuple[float, float, float]
type Mat3 = tuple[Vec3, Vec3, Vec3]

_VERSION_DIRECTORY = "v1.0-mini"
_CAMERA_CHANNEL = "CAM_FRONT"
_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


class ScalarProjectionReferenceError(ValueError):
    """The local scalar reference could not produce a sanitized diagnostic."""


@dataclass(frozen=True, slots=True)
class ScalarAnnotationProjection:
    """One identifier-free annotation projection retained only in local memory."""

    center_camera_m: Vec3
    center_depth_m: float
    center_uv_px: Vec2 | None
    center_projection_valid: bool
    center_strict_image_inside: bool
    corners_camera_m: tuple[Vec3, ...]
    corner_uv_px: tuple[Vec2 | None, ...]
    box_any_visible: bool


@dataclass(frozen=True, slots=True)
class ScalarProjectionDiagnostic:
    """Identifier-free scalar output used to cross-check production geometry."""

    width_px: int
    height_px: int
    annotation_count: int
    finite_positive_depth_center_count: int
    annotations: tuple[ScalarAnnotationProjection, ...]


def _as_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return cast(dict[str, object], value)


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return cast(list[object], value)


def _as_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return value


def _as_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return value


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return result


def _as_vec3(value: object) -> Vec3:
    items = _as_list(value)
    if len(items) != 3:
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return (_as_float(items[0]), _as_float(items[1]), _as_float(items[2]))


def _as_quaternion_wxyz(value: object) -> tuple[float, float, float, float]:
    items = _as_list(value)
    if len(items) != 4:
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    quaternion = tuple(_as_float(item) for item in items)
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return (w / norm, x / norm, y / norm, z / norm)


def _as_mat3(value: object) -> Mat3:
    rows = _as_list(value)
    if len(rows) != 3:
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    matrix_rows: list[Vec3] = []
    for row in rows:
        entries = _as_list(row)
        if len(entries) != 3:
            raise ScalarProjectionReferenceError("local scalar metadata is invalid")
        matrix_rows.append((_as_float(entries[0]), _as_float(entries[1]), _as_float(entries[2])))
    return (matrix_rows[0], matrix_rows[1], matrix_rows[2])


def _load_table(root: Path, table: str) -> tuple[dict[str, object], ...]:
    try:
        raw: object = json.loads(
            (root / _VERSION_DIRECTORY / f"{table}.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ScalarProjectionReferenceError(
            "local scalar reference could not load required metadata"
        ) from None
    return tuple(_as_object(row) for row in _as_list(raw))


def _index(rows: tuple[dict[str, object], ...]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        token = _as_string(row.get("token"))
        if token in result:
            raise ScalarProjectionReferenceError("local scalar metadata is invalid")
        result[token] = row
    return result


def _transpose(matrix: Mat3) -> Mat3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _mat_vec(matrix: Mat3, vector: Vec3) -> Vec3:
    return (
        math.fsum(matrix[0][index] * vector[index] for index in range(3)),
        math.fsum(matrix[1][index] * vector[index] for index in range(3)),
        math.fsum(matrix[2][index] * vector[index] for index in range(3)),
    )


def _add(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _subtract(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _quaternion_rotation(quaternion: tuple[float, float, float, float]) -> Mat3:
    w, x, y, z = quaternion
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _global_to_camera(
    point_global: Vec3,
    *,
    global_from_ego_rotation: Mat3,
    global_from_ego_translation: Vec3,
    ego_from_camera_rotation: Mat3,
    ego_from_camera_translation: Vec3,
) -> Vec3:
    point_ego = _mat_vec(
        _transpose(global_from_ego_rotation),
        _subtract(point_global, global_from_ego_translation),
    )
    return _mat_vec(
        _transpose(ego_from_camera_rotation),
        _subtract(point_ego, ego_from_camera_translation),
    )


def _project(point_camera: Vec3, intrinsic: Mat3) -> Vec2 | None:
    depth = point_camera[2]
    if depth <= 0.0:
        return None
    homogeneous = _mat_vec(intrinsic, point_camera)
    if homogeneous[2] <= 0.0:
        return None
    uv = (homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2])
    if not all(math.isfinite(component) for component in uv):
        return None
    return uv


def _strict_inside(uv: Vec2 | None, *, width_px: int, height_px: int) -> bool:
    return uv is not None and 0.0 < uv[0] < float(width_px) and 0.0 < uv[1] < float(height_px)


def _box_corners_global(annotation: dict[str, object]) -> tuple[Vec3, ...]:
    center = _as_vec3(annotation.get("translation"))
    width, length, height = _as_vec3(annotation.get("size"))
    if width <= 0.0 or length <= 0.0 or height <= 0.0:
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    rotation = _quaternion_rotation(_as_quaternion_wxyz(annotation.get("rotation")))
    local_corners = (
        (length / 2.0, width / 2.0, height / 2.0),
        (length / 2.0, -width / 2.0, height / 2.0),
        (length / 2.0, -width / 2.0, -height / 2.0),
        (length / 2.0, width / 2.0, -height / 2.0),
        (-length / 2.0, width / 2.0, height / 2.0),
        (-length / 2.0, -width / 2.0, height / 2.0),
        (-length / 2.0, -width / 2.0, -height / 2.0),
        (-length / 2.0, width / 2.0, -height / 2.0),
    )
    return tuple(_add(_mat_vec(rotation, corner), center) for corner in local_corners)


def _resolve_camera_records(
    sample_token: str,
    *,
    sample_data_rows: tuple[dict[str, object], ...],
    calibrated_sensor_index: dict[str, dict[str, object]],
    sensor_index: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    candidates: list[tuple[dict[str, object], dict[str, object]]] = []
    for sample_data in sample_data_rows:
        if _as_string(sample_data.get("sample_token")) != sample_token:
            continue
        if not _as_bool(sample_data.get("is_key_frame")):
            continue
        calibration = calibrated_sensor_index[
            _as_string(sample_data.get("calibrated_sensor_token"))
        ]
        sensor = sensor_index[_as_string(calibration.get("sensor_token"))]
        if _as_string(sensor.get("channel")) == _CAMERA_CHANNEL:
            candidates.append((sample_data, calibration))
    if len(candidates) != 1:
        raise ScalarProjectionReferenceError("local scalar metadata is invalid")
    return candidates[0]


def build_scalar_projection_diagnostic(dataset_root: Path) -> ScalarProjectionDiagnostic:
    """Build the frozen one-sample scalar reference without exposing identifiers."""

    try:
        scenes = _load_table(dataset_root, "scene")
        samples = _index(_load_table(dataset_root, "sample"))
        sample_data_rows = _load_table(dataset_root, "sample_data")
        calibrated_sensors = _index(_load_table(dataset_root, "calibrated_sensor"))
        sensors = _index(_load_table(dataset_root, "sensor"))
        ego_poses = _index(_load_table(dataset_root, "ego_pose"))
        annotations = _load_table(dataset_root, "sample_annotation")

        selected_scene = min(scenes, key=lambda row: _as_string(row.get("name")))
        sample_token = _as_string(selected_scene.get("first_sample_token"))
        if sample_token not in samples:
            raise ScalarProjectionReferenceError("local scalar metadata is invalid")
        camera_data, calibration = _resolve_camera_records(
            sample_token,
            sample_data_rows=sample_data_rows,
            calibrated_sensor_index=calibrated_sensors,
            sensor_index=sensors,
        )
        ego_pose = ego_poses[_as_string(camera_data.get("ego_pose_token"))]

        width_px = _as_int(camera_data.get("width"))
        height_px = _as_int(camera_data.get("height"))
        if width_px <= 0 or height_px <= 0:
            raise ScalarProjectionReferenceError("local scalar metadata is invalid")
        intrinsic = _as_mat3(calibration.get("camera_intrinsic"))
        global_from_ego_rotation = _quaternion_rotation(
            _as_quaternion_wxyz(ego_pose.get("rotation"))
        )
        global_from_ego_translation = _as_vec3(ego_pose.get("translation"))
        ego_from_camera_rotation = _quaternion_rotation(
            _as_quaternion_wxyz(calibration.get("rotation"))
        )
        ego_from_camera_translation = _as_vec3(calibration.get("translation"))

        selected_annotations = sorted(
            (
                annotation
                for annotation in annotations
                if _as_string(annotation.get("sample_token")) == sample_token
            ),
            key=lambda row: _as_string(row.get("token")).encode("utf-8"),
        )
        projected: list[ScalarAnnotationProjection] = []
        positive_depth_count = 0
        for annotation in selected_annotations:
            center_camera = _global_to_camera(
                _as_vec3(annotation.get("translation")),
                global_from_ego_rotation=global_from_ego_rotation,
                global_from_ego_translation=global_from_ego_translation,
                ego_from_camera_rotation=ego_from_camera_rotation,
                ego_from_camera_translation=ego_from_camera_translation,
            )
            center_uv = _project(center_camera, intrinsic)
            center_valid = center_camera[2] > 0.0 and center_uv is not None
            if center_valid:
                positive_depth_count += 1
            center_inside = center_camera[2] > 0.1 and _strict_inside(
                center_uv, width_px=width_px, height_px=height_px
            )

            corners_camera = tuple(
                _global_to_camera(
                    corner,
                    global_from_ego_rotation=global_from_ego_rotation,
                    global_from_ego_translation=global_from_ego_translation,
                    ego_from_camera_rotation=ego_from_camera_rotation,
                    ego_from_camera_translation=ego_from_camera_translation,
                )
                for corner in _box_corners_global(annotation)
            )
            corner_uv = tuple(_project(corner, intrinsic) for corner in corners_camera)
            visible = tuple(
                corner[2] > 1.0 and _strict_inside(uv, width_px=width_px, height_px=height_px)
                for corner, uv in zip(corners_camera, corner_uv, strict=True)
            )
            box_any_visible = any(visible) and all(corner[2] > 0.1 for corner in corners_camera)
            projected.append(
                ScalarAnnotationProjection(
                    center_camera_m=center_camera,
                    center_depth_m=center_camera[2],
                    center_uv_px=center_uv,
                    center_projection_valid=center_valid,
                    center_strict_image_inside=center_inside,
                    corners_camera_m=corners_camera,
                    corner_uv_px=corner_uv,
                    box_any_visible=box_any_visible,
                )
            )

        if not projected or positive_depth_count < 1:
            raise ScalarProjectionReferenceError("local scalar diagnostic is vacuous")
        return ScalarProjectionDiagnostic(
            width_px=width_px,
            height_px=height_px,
            annotation_count=len(projected),
            finite_positive_depth_center_count=positive_depth_count,
            annotations=tuple(projected),
        )
    except ScalarProjectionReferenceError:
        raise
    except (KeyError, TypeError, ValueError):
        raise ScalarProjectionReferenceError("local scalar metadata is invalid") from None


def render_scalar_diagnostic_svg(diagnostic: ScalarProjectionDiagnostic) -> str:
    """Render a token-free blank-canvas diagnostic; the caller keeps it local-only."""

    width = diagnostic.width_px
    height = diagnostic.height_px
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
        ),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#111827"/>',
    ]
    for annotation in diagnostic.annotations:
        color = "#22c55e" if annotation.box_any_visible else "#64748b"
        for first, second in _BOX_EDGES:
            left = annotation.corner_uv_px[first]
            right = annotation.corner_uv_px[second]
            if left is None or right is None:
                continue
            lines.append(
                f'<line x1="{left[0]:.6f}" y1="{left[1]:.6f}" '
                f'x2="{right[0]:.6f}" y2="{right[1]:.6f}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        if annotation.center_uv_px is not None:
            uv = annotation.center_uv_px
            lines.append(f'<circle cx="{uv[0]:.6f}" cy="{uv[1]:.6f}" r="3" fill="#f8fafc"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
