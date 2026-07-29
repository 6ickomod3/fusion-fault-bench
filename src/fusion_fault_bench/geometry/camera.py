"""Pinhole projection and nuScenes-compatible 3D box visibility helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.se3 import (
    FloatArray,
    quaternion_wxyz_to_rotation,
)

type ArrayLike = Sequence[float] | FloatArray

_PINHOLE_ROW_TOLERANCE = 1e-12


def _positive_integer(value: int, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _finite_array(
    value: npt.ArrayLike,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


@dataclass(frozen=True, slots=True)
class PinholeCamera:
    """Finite calibrated pinhole intrinsics and image dimensions."""

    intrinsic: FloatArray
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        intrinsic = _finite_array(
            self.intrinsic,
            shape=(3, 3),
            field_name="intrinsic",
        )
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be strictly positive")
        if not np.allclose(
            intrinsic[2],
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            rtol=0.0,
            atol=_PINHOLE_ROW_TOLERANCE,
        ):
            raise ValueError("camera intrinsic third row must be [0, 0, 1]")
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(
            self,
            "width_px",
            _positive_integer(self.width_px, field_name="width_px"),
        )
        object.__setattr__(
            self,
            "height_px",
            _positive_integer(self.height_px, field_name="height_px"),
        )

    def contains_strict(
        self,
        uv_px: npt.ArrayLike,
        *,
        depth_m: float,
        minimum_depth_m: float,
    ) -> bool:
        """Apply strict image and depth bounds without pixel rounding."""

        uv = _finite_array(uv_px, shape=(2,), field_name="uv_px")
        if not math.isfinite(depth_m):
            raise ValueError("depth_m must be finite")
        if not math.isfinite(minimum_depth_m) or minimum_depth_m < 0.0:
            raise ValueError("minimum_depth_m must be finite and non-negative")
        return bool(
            depth_m > minimum_depth_m
            and 0.0 < uv[0] < self.width_px
            and 0.0 < uv[1] < self.height_px
        )


@dataclass(frozen=True, slots=True)
class Projection:
    """One point projection; invalid depth never produces divided pixels."""

    depth_m: float
    uv_px: FloatArray | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.depth_m):
            raise ValueError("projection depth_m must be finite")
        if self.uv_px is None:
            return
        if self.depth_m <= 0.0:
            raise ValueError("projection pixels require strictly positive depth")
        object.__setattr__(
            self,
            "uv_px",
            _finite_array(self.uv_px, shape=(2,), field_name="uv_px"),
        )

    @property
    def valid(self) -> bool:
        return self.uv_px is not None


def project_point(point_camera_m: npt.ArrayLike, camera: PinholeCamera) -> Projection:
    """Project one finite camera-frame point with optical-axis depth ``z``."""

    point = _finite_array(
        point_camera_m,
        shape=(3,),
        field_name="point_camera_m",
    )
    depth = float(point[2])
    if depth <= 0.0:
        return Projection(depth_m=depth, uv_px=None)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        homogeneous_pixel = camera.intrinsic @ point
        uv = np.asarray(homogeneous_pixel[:2] / homogeneous_pixel[2], dtype=np.float64)
    if not np.all(np.isfinite(uv)):
        raise ValueError("projection produced non-finite pixels")
    return Projection(depth_m=depth, uv_px=immutable_float64_copy(uv))


def box_corners(
    *,
    center_m: npt.ArrayLike,
    size_width_length_height_m: npt.ArrayLike,
    orientation_wxyz: ArrayLike,
) -> FloatArray:
    """Return official nuScenes-order box corners as a ``3 x 8`` matrix."""

    center = _finite_array(center_m, shape=(3,), field_name="center_m")
    size = _finite_array(
        size_width_length_height_m,
        shape=(3,),
        field_name="size_width_length_height_m",
    )
    if np.any(size <= 0.0):
        raise ValueError("box dimensions must be strictly positive")
    width, length, height = (float(component) for component in size)
    local = np.vstack(
        (
            length / 2.0 * np.asarray([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64),
            width / 2.0 * np.asarray([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float64),
            height / 2.0 * np.asarray([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64),
        )
    )
    rotation = quaternion_wxyz_to_rotation(orientation_wxyz)
    corners = rotation @ local + center[:, np.newaxis]
    return immutable_float64_copy(corners)


def devkit_box_any_visible(
    corners_camera_m: npt.ArrayLike,
    camera: PinholeCamera,
    *,
    visible_corner_min_depth_m: float = 1.0,
    all_corners_min_depth_m: float = 0.1,
) -> bool:
    """Apply the official devkit ``BoxVisibility.ANY`` center-free rule."""

    corners = _finite_array(
        corners_camera_m,
        shape=(3, 8),
        field_name="corners_camera_m",
    )
    if (
        not math.isfinite(visible_corner_min_depth_m)
        or not math.isfinite(all_corners_min_depth_m)
        or visible_corner_min_depth_m < 0.0
        or all_corners_min_depth_m < 0.0
    ):
        raise ValueError("box visibility depth thresholds must be finite and non-negative")
    depths = corners[2]
    if not bool(np.all(depths > all_corners_min_depth_m)):
        return False
    for index in range(corners.shape[1]):
        depth = float(depths[index])
        if depth <= visible_corner_min_depth_m:
            continue
        projection = project_point(corners[:, index], camera)
        assert projection.uv_px is not None
        if camera.contains_strict(
            projection.uv_px,
            depth_m=depth,
            minimum_depth_m=visible_corner_min_depth_m,
        ):
            return True
    return False
