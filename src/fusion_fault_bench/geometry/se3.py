"""Immutable named SE(3) transforms using the benchmark's column-vector convention."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.frames import FrameId

type FloatArray = npt.NDArray[np.float64]
type ArrayLike = Sequence[float] | FloatArray

_QUATERNION_NORM_TOLERANCE = 1e-6
_ROTATION_ORTHOGONALITY_TOLERANCE = 1e-10
_ROTATION_DETERMINANT_TOLERANCE = 1e-10


def _readonly_copy(value: npt.ArrayLike, *, shape: tuple[int, ...], field_name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def quaternion_wxyz_to_rotation(quaternion_wxyz: ArrayLike) -> FloatArray:
    """Convert one explicit scalar-first unit quaternion to an active rotation."""

    quaternion = _readonly_copy(
        quaternion_wxyz,
        shape=(4,),
        field_name="quaternion_wxyz",
    )
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion_wxyz must have nonzero finite norm")
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise ValueError("quaternion_wxyz norm must be within 1e-6 of one")
    w, x, y, z = (float(component) / norm for component in quaternion)
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    return immutable_float64_copy(rotation)


def _validated_rotation(rotation: npt.ArrayLike) -> FloatArray:
    result = _readonly_copy(rotation, shape=(3, 3), field_name="rotation")
    identity_error = float(np.linalg.norm(result.T @ result - np.eye(3), ord=np.inf))
    if identity_error > _ROTATION_ORTHOGONALITY_TOLERANCE:
        raise ValueError("rotation must be orthogonal within 1e-10")
    determinant_error = abs(float(np.linalg.det(result)) - 1.0)
    if determinant_error > _ROTATION_DETERMINANT_TOLERANCE:
        raise ValueError("rotation must be right-handed with determinant one")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class SE3:
    """One named transform ``T_target<-source`` with immutable float64 storage."""

    target_frame: FrameId
    source_frame: FrameId
    rotation: FloatArray
    translation_m: FloatArray

    def __post_init__(self) -> None:
        target_log = (
            self.target_frame.qualifiers[0] if self.target_frame.kind in {"global", "ego"} else None
        )
        source_log = (
            self.source_frame.qualifiers[0] if self.source_frame.kind in {"global", "ego"} else None
        )
        if target_log is not None and source_log is not None and target_log != source_log:
            raise ValueError("cross-log transforms require an explicit bridge outside M2")
        object.__setattr__(self, "rotation", _validated_rotation(self.rotation))
        object.__setattr__(
            self,
            "translation_m",
            _readonly_copy(
                self.translation_m,
                shape=(3,),
                field_name="translation_m",
            ),
        )

    @classmethod
    def from_quaternion_wxyz(
        cls,
        *,
        target_frame: FrameId,
        source_frame: FrameId,
        translation_m: ArrayLike,
        quaternion_wxyz: ArrayLike,
    ) -> SE3:
        """Construct ``T_target<-source`` from nuScenes-order pose components."""

        return cls(
            target_frame=target_frame,
            source_frame=source_frame,
            rotation=quaternion_wxyz_to_rotation(quaternion_wxyz),
            translation_m=np.asarray(translation_m, dtype=np.float64),
        )

    @classmethod
    def identity(cls, frame: FrameId) -> SE3:
        """Construct the identity mapping for exactly one qualified frame."""

        return cls(
            target_frame=frame,
            source_frame=frame,
            rotation=np.eye(3, dtype=np.float64),
            translation_m=np.zeros(3, dtype=np.float64),
        )

    def inverse(self) -> SE3:
        """Return ``T_source<-target`` using ``-R.T @ t``."""

        inverse_rotation = self.rotation.T
        return SE3(
            target_frame=self.source_frame,
            source_frame=self.target_frame,
            rotation=inverse_rotation,
            translation_m=-(inverse_rotation @ self.translation_m),
        )

    def compose(self, right: SE3) -> SE3:
        """Return ``self @ right`` after exact intermediate-frame validation."""

        if self.source_frame != right.target_frame:
            raise ValueError("cannot compose transforms with mismatched intermediate frames")
        return SE3(
            target_frame=self.target_frame,
            source_frame=right.source_frame,
            rotation=self.rotation @ right.rotation,
            translation_m=self.rotation @ right.translation_m + self.translation_m,
        )

    def apply(self, points: npt.ArrayLike, *, source_frame: FrameId) -> FloatArray:
        """Apply the transform to one point or a ``3 x N`` column-point matrix."""

        if source_frame != self.source_frame:
            raise ValueError("point source frame does not match transform source frame")
        array = np.asarray(points, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("points must contain only finite values")
        if array.ndim == 1:
            if array.shape != (3,):
                raise ValueError("point must have shape (3,)")
            result = self.rotation @ array + self.translation_m
        elif array.ndim == 2:
            if array.shape[0] != 3:
                raise ValueError("column-point matrix must have shape (3, N)")
            result = self.rotation @ array + self.translation_m[:, np.newaxis]
        else:
            raise ValueError("points must be one vector or a 3 x N column matrix")
        return immutable_float64_copy(result)

    def homogeneous_matrix(self) -> FloatArray:
        """Export the conventional 4 x 4 homogeneous matrix."""

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation_m
        return immutable_float64_copy(matrix)

    def is_close(
        self,
        other: SE3,
        *,
        rotation_atol: float,
        translation_atol_m: float,
    ) -> bool:
        """Compare transforms only when both named mappings agree exactly."""

        if self.target_frame != other.target_frame or self.source_frame != other.source_frame:
            raise ValueError("cannot compare transforms with different named mappings")
        if (
            not math.isfinite(rotation_atol)
            or not math.isfinite(translation_atol_m)
            or rotation_atol < 0.0
            or translation_atol_m < 0.0
        ):
            raise ValueError("comparison tolerances must be finite and non-negative")
        return bool(
            np.allclose(self.rotation, other.rotation, rtol=0.0, atol=rotation_atol)
            and np.allclose(
                self.translation_m,
                other.translation_m,
                rtol=0.0,
                atol=translation_atol_m,
            )
        )
