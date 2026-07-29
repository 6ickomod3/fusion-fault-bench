"""Bearing/depth reconstruction and role-preserving covariance propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.roi import EligibilityTransform
from fusion_fault_bench.geometry.se3 import FloatArray

type CovarianceRole = Literal["actual", "reported"]
type BearingDepthParameter = Literal["bearing_rad", "optical_depth_m"]

_EXPECTED_PARAMETER_ORDER: tuple[BearingDepthParameter, BearingDepthParameter] = (
    "bearing_rad",
    "optical_depth_m",
)
_SYMMETRY_TOLERANCE = 1e-12


def _require_eligibility_transform(value: object) -> EligibilityTransform:
    if not isinstance(value, EligibilityTransform):
        raise TypeError("bearing/depth mapping requires an EligibilityTransform")
    return value


def _require_covariance(value: object) -> BearingDepthCovariance:
    if not isinstance(value, BearingDepthCovariance):
        raise TypeError("covariance must be a BearingDepthCovariance")
    return value


def _finite_scalar(value: float, *, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _validate_observation(*, bearing_rad: float, optical_depth_m: float) -> None:
    _finite_scalar(bearing_rad, field_name="bearing_rad")
    _finite_scalar(optical_depth_m, field_name="optical_depth_m")
    if not -math.pi / 2.0 < bearing_rad < math.pi / 2.0:
        raise ValueError("bearing_rad must lie strictly inside (-pi/2, pi/2)")
    if optical_depth_m <= 0.0:
        raise ValueError("optical_depth_m must be strictly positive")


@dataclass(frozen=True, slots=True)
class BearingDepthCovariance:
    """A full SPD covariance whose actual/reported role cannot be discarded."""

    role: CovarianceRole
    matrix: FloatArray
    parameter_order: tuple[
        BearingDepthParameter,
        BearingDepthParameter,
    ] = _EXPECTED_PARAMETER_ORDER

    def __post_init__(self) -> None:
        if self.role not in {"actual", "reported"}:
            raise ValueError("covariance role must be actual or reported")
        if self.parameter_order != _EXPECTED_PARAMETER_ORDER:
            raise ValueError("covariance parameter order must be bearing_rad, optical_depth_m")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (2, 2):
            raise ValueError("bearing/depth covariance must have shape (2, 2)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("bearing/depth covariance must contain only finite values")
        if not np.allclose(
            matrix,
            matrix.T,
            rtol=0.0,
            atol=_SYMMETRY_TOLERANCE,
        ):
            raise ValueError("bearing/depth covariance must be symmetric")
        symmetric = np.asarray((matrix + matrix.T) / 2.0, dtype=np.float64)
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as error:
            raise ValueError("bearing/depth covariance must be positive definite") from error
        object.__setattr__(self, "matrix", immutable_float64_copy(symmetric))


@dataclass(frozen=True, slots=True)
class PropagatedCovariance:
    """One role-preserving full covariance in reference-ego BEV."""

    role: CovarianceRole
    matrix_xy_m2: FloatArray
    jacobian_xy: FloatArray

    def __post_init__(self) -> None:
        if self.role not in {"actual", "reported"}:
            raise ValueError("propagated covariance role must be actual or reported")
        matrix = np.asarray(self.matrix_xy_m2, dtype=np.float64)
        jacobian = np.asarray(self.jacobian_xy, dtype=np.float64)
        if matrix.shape != (2, 2) or jacobian.shape != (2, 2):
            raise ValueError("propagated covariance and Jacobian must have shape (2, 2)")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(jacobian)):
            raise ValueError("propagated covariance and Jacobian must be finite")
        if not np.allclose(
            matrix,
            matrix.T,
            rtol=0.0,
            atol=_SYMMETRY_TOLERANCE,
        ):
            raise ValueError("propagated covariance must be symmetric")
        symmetric = np.asarray((matrix + matrix.T) / 2.0, dtype=np.float64)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
        roundoff_scale = float(np.max(np.abs(symmetric)))
        if minimum_eigenvalue < (-np.finfo(np.float64).eps * max(1.0, roundoff_scale) * 32.0):
            raise ValueError("propagated covariance must be positive semidefinite")
        object.__setattr__(
            self,
            "matrix_xy_m2",
            immutable_float64_copy(symmetric),
        )
        object.__setattr__(
            self,
            "jacobian_xy",
            immutable_float64_copy(jacobian),
        )


def bearing_depth_point_camera(
    *,
    bearing_rad: float,
    optical_depth_m: float,
    camera_vertical_coordinate_m: float,
) -> FloatArray:
    """Map horizontal bearing and optical depth into the camera frame."""

    _validate_observation(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
    )
    _finite_scalar(
        camera_vertical_coordinate_m,
        field_name="camera_vertical_coordinate_m",
    )
    point = np.asarray(
        [
            optical_depth_m * math.tan(bearing_rad),
            camera_vertical_coordinate_m,
            optical_depth_m,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(point)):
        raise ValueError("bearing/depth reconstruction produced a non-finite point")
    return immutable_float64_copy(point)


def bearing_depth_jacobian_camera(
    *,
    bearing_rad: float,
    optical_depth_m: float,
) -> FloatArray:
    """Return the analytic ``d p_camera / d (bearing, depth)`` Jacobian."""

    _validate_observation(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
    )
    tangent = math.tan(bearing_rad)
    secant_squared = 1.0 + tangent * tangent
    jacobian = np.asarray(
        [
            [optical_depth_m * secant_squared, tangent],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("bearing/depth Jacobian is non-finite")
    return immutable_float64_copy(jacobian)


def bearing_depth_point_reference_ego_bev(
    *,
    bearing_rad: float,
    optical_depth_m: float,
    camera_vertical_coordinate_m: float,
    reference_ego_from_camera: EligibilityTransform,
) -> FloatArray:
    """Map one camera observation into the nominal reference-ego BEV frame."""

    transform = _require_eligibility_transform(reference_ego_from_camera)
    point_camera = bearing_depth_point_camera(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
        camera_vertical_coordinate_m=camera_vertical_coordinate_m,
    )
    if (
        transform.transform.source_frame.kind != "camera"
        or transform.transform.target_frame.kind != "ego"
    ):
        raise ValueError("bearing/depth transform must map camera into reference ego")
    point_ego = transform.transform.apply(
        point_camera,
        source_frame=transform.transform.source_frame,
    )
    result = np.asarray(point_ego[:2], dtype=np.float64)
    return immutable_float64_copy(result)


def bearing_depth_jacobian_reference_ego_bev(
    *,
    bearing_rad: float,
    optical_depth_m: float,
    reference_ego_from_camera: EligibilityTransform,
) -> FloatArray:
    """Rotate the camera Jacobian into the reference-ego BEV output."""

    transform = _require_eligibility_transform(reference_ego_from_camera)
    if (
        transform.transform.source_frame.kind != "camera"
        or transform.transform.target_frame.kind != "ego"
    ):
        raise ValueError("bearing/depth transform must map camera into reference ego")
    jacobian_camera = bearing_depth_jacobian_camera(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
    )
    jacobian_xy = transform.transform.rotation[:2, :] @ jacobian_camera
    result = np.asarray(jacobian_xy, dtype=np.float64)
    return immutable_float64_copy(result)


def propagate_bearing_depth_covariance(
    *,
    bearing_rad: float,
    optical_depth_m: float,
    covariance: BearingDepthCovariance,
    reference_ego_from_camera: EligibilityTransform,
) -> PropagatedCovariance:
    """Propagate either actual or reported covariance without changing its role."""

    checked_covariance = _require_covariance(covariance)
    jacobian = bearing_depth_jacobian_reference_ego_bev(
        bearing_rad=bearing_rad,
        optical_depth_m=optical_depth_m,
        reference_ego_from_camera=reference_ego_from_camera,
    )
    propagated = jacobian @ checked_covariance.matrix @ jacobian.T
    symmetric = np.asarray((propagated + propagated.T) / 2.0, dtype=np.float64)
    if not np.all(np.isfinite(symmetric)):
        raise ValueError("propagated covariance is non-finite")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
    roundoff_scale = float(np.max(np.abs(symmetric)))
    if minimum_eigenvalue < -np.finfo(np.float64).eps * max(1.0, roundoff_scale) * 32.0:
        raise ValueError("propagated covariance is not positive semidefinite")
    return PropagatedCovariance(
        role=checked_covariance.role,
        matrix_xy_m2=symmetric,
        jacobian_xy=jacobian,
    )
