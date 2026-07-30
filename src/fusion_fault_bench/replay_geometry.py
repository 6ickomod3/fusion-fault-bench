"""Full-3D reconstruction geometry for the frozen M5 replay panels.

The module keeps physical proxy generation separate from metadata-based
reconstruction.  Points are column vectors and every transform implements
``p_target = R_target_source @ p_source + t_target_source``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.camera import PinholeCamera, project_point
from fusion_fault_bench.geometry.se3 import quaternion_wxyz_to_rotation

type FloatArray = npt.NDArray[np.float64]

_ROTATION_ATOL = 1e-10
_SYMMETRY_ATOL = 1e-12
_ZERO_Z_LIFT = np.asarray(
    ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
    dtype=np.float64,
)
_PROJECT_XY = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


class _PrivateRepr:
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _finite_array(
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


def _finite_scalar(value: float, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _rotation(value: npt.ArrayLike, *, field_name: str) -> FloatArray:
    result = _finite_array(value, shape=(3, 3), field_name=field_name)
    if not np.allclose(
        result.T @ result,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=_ROTATION_ATOL,
    ):
        raise ValueError(f"{field_name} must be orthogonal")
    if abs(float(np.linalg.det(result)) - 1.0) > _ROTATION_ATOL:
        raise ValueError(f"{field_name} must be right-handed")
    return result


def _covariance2(value: npt.ArrayLike, *, field_name: str) -> FloatArray:
    result = _finite_array(value, shape=(2, 2), field_name=field_name)
    if not np.allclose(result, result.T, rtol=0.0, atol=_SYMMETRY_ATOL):
        raise ValueError(f"{field_name} must be symmetric")
    symmetric = np.asarray((result + result.T) * 0.5, dtype=np.float64)
    if float(np.min(np.linalg.eigvalsh(symmetric))) <= 0.0:
        raise ValueError(f"{field_name} must be positive definite")
    return immutable_float64_copy(symmetric)


def _rotation2(angle_rad: float) -> FloatArray:
    angle = _finite_scalar(angle_rad, field_name="angle_rad")
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return immutable_float64_copy(np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class RigidTransform3(_PrivateRepr):
    """One anonymous rigid transform ``T_target<-source``."""

    rotation: FloatArray
    translation_m: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation",
            _rotation(self.rotation, field_name="rotation"),
        )
        object.__setattr__(
            self,
            "translation_m",
            _finite_array(
                self.translation_m,
                shape=(3,),
                field_name="translation_m",
            ),
        )

    @classmethod
    def identity(cls) -> RigidTransform3:
        """Construct an identity transform."""

        return cls(
            rotation=np.eye(3, dtype=np.float64),
            translation_m=np.zeros(3, dtype=np.float64),
        )

    @classmethod
    def from_quaternion_wxyz(
        cls,
        *,
        translation_m: npt.ArrayLike,
        quaternion_wxyz: npt.ArrayLike,
    ) -> RigidTransform3:
        """Construct from a scalar-first nuScenes quaternion."""

        return cls(
            rotation=quaternion_wxyz_to_rotation(np.asarray(quaternion_wxyz, dtype=np.float64)),
            translation_m=np.asarray(translation_m, dtype=np.float64),
        )

    def inverse(self) -> RigidTransform3:
        """Return ``T_source<-target``."""

        inverse_rotation = self.rotation.T
        return RigidTransform3(
            rotation=inverse_rotation,
            translation_m=-(inverse_rotation @ self.translation_m),
        )

    def compose(self, right: RigidTransform3) -> RigidTransform3:
        """Return ``self @ right``."""

        return RigidTransform3(
            rotation=self.rotation @ right.rotation,
            translation_m=self.rotation @ right.translation_m + self.translation_m,
        )

    def apply(self, point_m: npt.ArrayLike) -> FloatArray:
        """Transform one finite 3D column point."""

        point = _finite_array(point_m, shape=(3,), field_name="point_m")
        return immutable_float64_copy(self.rotation @ point + self.translation_m)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class NominalEligibility(_PrivateRepr):
    """Pre-fault support decision and the two full-3D nominal centers."""

    center_reference_ego_m: FloatArray
    center_camera_m: FloatArray
    roi_pass: bool
    camera_center_pass: bool
    lidar_points_pass: bool
    camera_estimator_available: bool
    lidar_estimator_available: bool
    eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_reference_ego_m",
            _finite_array(
                self.center_reference_ego_m,
                shape=(3,),
                field_name="center_reference_ego_m",
            ),
        )
        object.__setattr__(
            self,
            "center_camera_m",
            _finite_array(
                self.center_camera_m,
                shape=(3,),
                field_name="center_camera_m",
            ),
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class CameraProxyPoint(_PrivateRepr):
    """Physical camera-frame proxy generated with true state and metadata."""

    point_camera_m: FloatArray
    physical_point_global_m: FloatArray
    base_error_jacobian_camera: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_camera_m",
            _finite_array(
                self.point_camera_m,
                shape=(3,),
                field_name="point_camera_m",
            ),
        )
        object.__setattr__(
            self,
            "physical_point_global_m",
            _finite_array(
                self.physical_point_global_m,
                shape=(3,),
                field_name="physical_point_global_m",
            ),
        )
        object.__setattr__(
            self,
            "base_error_jacobian_camera",
            _finite_array(
                self.base_error_jacobian_camera,
                shape=(3, 2),
                field_name="base_error_jacobian_camera",
            ),
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class FullReconstruction(_PrivateRepr):
    """One full-3D current-ego reconstruction and exact base-error Jacobian."""

    point_current_ego_m: FloatArray
    base_error_jacobian: FloatArray
    reference_state_time_s: float
    reported_state_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_current_ego_m",
            _finite_array(
                self.point_current_ego_m,
                shape=(3,),
                field_name="point_current_ego_m",
            ),
        )
        object.__setattr__(
            self,
            "base_error_jacobian",
            _finite_array(
                self.base_error_jacobian,
                shape=(3, 2),
                field_name="base_error_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "reference_state_time_s",
            _finite_scalar(
                self.reference_state_time_s,
                field_name="reference_state_time_s",
            ),
        )
        object.__setattr__(
            self,
            "reported_state_time_s",
            _finite_scalar(
                self.reported_state_time_s,
                field_name="reported_state_time_s",
            ),
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ProjectedEstimate(_PrivateRepr):
    """One 2D projection with the exact projection Jacobian and covariance."""

    point_m: FloatArray
    jacobian: FloatArray
    reported_covariance_m2: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_m",
            _finite_array(self.point_m, shape=(2,), field_name="point_m"),
        )
        object.__setattr__(
            self,
            "jacobian",
            _finite_array(self.jacobian, shape=(2, 2), field_name="jacobian"),
        )
        object.__setattr__(
            self,
            "reported_covariance_m2",
            _covariance2(
                self.reported_covariance_m2,
                field_name="reported_covariance_m2",
            ),
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class SceneBevAnchor(_PrivateRepr):
    """First-reference-pose origin and yaw defining monitoring frame ``S``."""

    origin_global_xy_m: FloatArray
    yaw_global_from_scene_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "origin_global_xy_m",
            _finite_array(
                self.origin_global_xy_m,
                shape=(2,),
                field_name="origin_global_xy_m",
            ),
        )
        object.__setattr__(
            self,
            "yaw_global_from_scene_rad",
            _finite_scalar(
                self.yaw_global_from_scene_rad,
                field_name="yaw_global_from_scene_rad",
            ),
        )

    @classmethod
    def from_first_reference_pose(
        cls,
        global_from_reference_ego: RigidTransform3,
    ) -> SceneBevAnchor:
        """Anchor ``S`` at the first LiDAR-time ego pose."""

        yaw = math.atan2(
            float(global_from_reference_ego.rotation[1, 0]),
            float(global_from_reference_ego.rotation[0, 0]),
        )
        return cls(
            origin_global_xy_m=global_from_reference_ego.translation_m[:2],
            yaw_global_from_scene_rad=yaw,
        )


def calibration_perturbation(
    *,
    translation_camera_ego_m: npt.ArrayLike = (0.0, 0.0, 0.0),
    yaw_camera_ego_rad: float = 0.0,
) -> RigidTransform3:
    """Build the left perturbation ``DeltaT_Ec`` used by M5 calibration faults."""

    translation = _finite_array(
        translation_camera_ego_m,
        shape=(3,),
        field_name="translation_camera_ego_m",
    )
    yaw = _finite_scalar(yaw_camera_ego_rad, field_name="yaw_camera_ego_rad")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return RigidTransform3(
        rotation=np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        translation_m=translation,
    )


def reported_camera_extrinsic(
    *,
    true_camera_ego_from_camera: RigidTransform3,
    perturbation_camera_ego: RigidTransform3,
) -> RigidTransform3:
    """Apply ``T_reported_{Ec<-C}=DeltaT_Ec*T_true_{Ec<-C}``."""

    return perturbation_camera_ego.compose(true_camera_ego_from_camera)


def evaluate_nominal_eligibility(
    *,
    center_global_m: npt.ArrayLike,
    global_from_reference_ego: RigidTransform3,
    global_from_camera_ego: RigidTransform3,
    true_camera_ego_from_camera: RigidTransform3,
    camera: PinholeCamera,
    num_lidar_points: int,
    camera_estimator_available: bool = True,
    lidar_estimator_available: bool = True,
) -> NominalEligibility:
    """Apply the exact frozen pre-fault common-support rule."""

    if type(num_lidar_points) is not int or num_lidar_points < 0:
        raise ValueError("num_lidar_points must be a non-negative integer")
    if type(camera_estimator_available) is not bool:
        raise ValueError("camera_estimator_available must be boolean")
    if type(lidar_estimator_available) is not bool:
        raise ValueError("lidar_estimator_available must be boolean")
    center_global = _finite_array(
        center_global_m,
        shape=(3,),
        field_name="center_global_m",
    )
    center_reference = global_from_reference_ego.inverse().apply(center_global)
    global_from_camera = global_from_camera_ego.compose(true_camera_ego_from_camera)
    center_camera = global_from_camera.inverse().apply(center_global)
    projection = project_point(center_camera, camera)
    camera_pass = bool(
        projection.uv_px is not None
        and camera.contains_strict(
            projection.uv_px,
            depth_m=projection.depth_m,
            minimum_depth_m=0.1,
        )
    )
    x_m = float(center_reference[0])
    y_m = float(center_reference[1])
    roi_pass = bool(x_m > 0.0 and 5.0 <= x_m <= 60.0 and abs(y_m) <= 40.0)
    lidar_pass = num_lidar_points > 0
    eligible = bool(
        roi_pass
        and camera_pass
        and lidar_pass
        and camera_estimator_available
        and lidar_estimator_available
    )
    return NominalEligibility(
        center_reference_ego_m=center_reference,
        center_camera_m=center_camera,
        roi_pass=roi_pass,
        camera_center_pass=camera_pass,
        lidar_points_pass=lidar_pass,
        camera_estimator_available=camera_estimator_available,
        lidar_estimator_available=lidar_estimator_available,
        eligible=eligible,
    )


def generate_camera_proxy(
    *,
    truth_global_at_reference_m: npt.ArrayLike,
    velocity_global_mps: npt.ArrayLike,
    reference_time_s: float,
    camera_time_s: float,
    base_error_reference_bev_m: npt.ArrayLike,
    global_from_reference_ego: RigidTransform3,
    global_from_camera_ego: RigidTransform3,
    true_camera_ego_from_camera: RigidTransform3,
) -> CameraProxyPoint:
    """Generate the physical camera proxy using only true pose and extrinsic."""

    truth = _finite_array(
        truth_global_at_reference_m,
        shape=(3,),
        field_name="truth_global_at_reference_m",
    )
    velocity = _finite_array(
        velocity_global_mps,
        shape=(3,),
        field_name="velocity_global_mps",
    )
    error = _finite_array(
        base_error_reference_bev_m,
        shape=(2,),
        field_name="base_error_reference_bev_m",
    )
    reference_time = _finite_scalar(reference_time_s, field_name="reference_time_s")
    camera_time = _finite_scalar(camera_time_s, field_name="camera_time_s")
    base_error_global_jacobian = global_from_reference_ego.rotation @ _ZERO_Z_LIFT
    physical_global = (
        truth + velocity * (camera_time - reference_time) + base_error_global_jacobian @ error
    )
    global_from_camera = global_from_camera_ego.compose(true_camera_ego_from_camera)
    camera_from_global = global_from_camera.inverse()
    return CameraProxyPoint(
        point_camera_m=camera_from_global.apply(physical_global),
        physical_point_global_m=physical_global,
        base_error_jacobian_camera=(camera_from_global.rotation @ base_error_global_jacobian),
    )


def reconstruct_camera_proxy(
    proxy: CameraProxyPoint,
    *,
    velocity_global_mps: npt.ArrayLike,
    reference_time_s: float,
    camera_time_s: float,
    timestamp_fault_s: float,
    additive_bias_reference_bev_m: npt.ArrayLike,
    global_from_reference_ego: RigidTransform3,
    global_from_camera_ego: RigidTransform3,
    reported_camera_ego_from_camera: RigidTransform3,
) -> FullReconstruction:
    """Reconstruct with reported metadata and align to the true reference clock."""

    velocity = _finite_array(
        velocity_global_mps,
        shape=(3,),
        field_name="velocity_global_mps",
    )
    bias = _finite_array(
        additive_bias_reference_bev_m,
        shape=(2,),
        field_name="additive_bias_reference_bev_m",
    )
    reference_time = _finite_scalar(reference_time_s, field_name="reference_time_s")
    camera_time = _finite_scalar(camera_time_s, field_name="camera_time_s")
    timestamp_fault = _finite_scalar(
        timestamp_fault_s,
        field_name="timestamp_fault_s",
    )
    global_from_reported_camera = global_from_camera_ego.compose(reported_camera_ego_from_camera)
    reconstructed_global_at_camera = global_from_reported_camera.apply(proxy.point_camera_m)
    aligned_global = (
        reconstructed_global_at_camera
        + velocity * (reference_time - camera_time)
        - velocity * timestamp_fault
    )
    reference_from_global = global_from_reference_ego.inverse()
    point_reference = reference_from_global.apply(aligned_global) + _ZERO_Z_LIFT @ bias
    base_jacobian = (
        reference_from_global.rotation
        @ global_from_reported_camera.rotation
        @ proxy.base_error_jacobian_camera
    )
    return FullReconstruction(
        point_current_ego_m=point_reference,
        base_error_jacobian=base_jacobian,
        reference_state_time_s=reference_time,
        reported_state_time_s=reference_time + timestamp_fault,
    )


def reconstruct_camera(
    *,
    truth_global_at_reference_m: npt.ArrayLike,
    velocity_global_mps: npt.ArrayLike,
    reference_time_s: float,
    camera_time_s: float,
    base_error_reference_bev_m: npt.ArrayLike,
    timestamp_fault_s: float,
    additive_bias_reference_bev_m: npt.ArrayLike,
    global_from_reference_ego: RigidTransform3,
    global_from_camera_ego: RigidTransform3,
    true_camera_ego_from_camera: RigidTransform3,
    reported_camera_ego_from_camera: RigidTransform3,
) -> FullReconstruction:
    """Generate with true metadata, then reconstruct with reported metadata."""

    proxy = generate_camera_proxy(
        truth_global_at_reference_m=truth_global_at_reference_m,
        velocity_global_mps=velocity_global_mps,
        reference_time_s=reference_time_s,
        camera_time_s=camera_time_s,
        base_error_reference_bev_m=base_error_reference_bev_m,
        global_from_reference_ego=global_from_reference_ego,
        global_from_camera_ego=global_from_camera_ego,
        true_camera_ego_from_camera=true_camera_ego_from_camera,
    )
    return reconstruct_camera_proxy(
        proxy,
        velocity_global_mps=velocity_global_mps,
        reference_time_s=reference_time_s,
        camera_time_s=camera_time_s,
        timestamp_fault_s=timestamp_fault_s,
        additive_bias_reference_bev_m=additive_bias_reference_bev_m,
        global_from_reference_ego=global_from_reference_ego,
        global_from_camera_ego=global_from_camera_ego,
        reported_camera_ego_from_camera=reported_camera_ego_from_camera,
    )


def reconstruct_lidar(
    *,
    truth_global_at_reference_m: npt.ArrayLike,
    velocity_global_mps: npt.ArrayLike,
    reference_time_s: float,
    base_error_reference_bev_m: npt.ArrayLike,
    timestamp_fault_s: float,
    additive_bias_reference_bev_m: npt.ArrayLike,
    global_from_reference_ego: RigidTransform3,
) -> FullReconstruction:
    """Construct the LiDAR proxy in current ``E_k`` with zero-z base error."""

    truth = _finite_array(
        truth_global_at_reference_m,
        shape=(3,),
        field_name="truth_global_at_reference_m",
    )
    velocity = _finite_array(
        velocity_global_mps,
        shape=(3,),
        field_name="velocity_global_mps",
    )
    error = _finite_array(
        base_error_reference_bev_m,
        shape=(2,),
        field_name="base_error_reference_bev_m",
    )
    bias = _finite_array(
        additive_bias_reference_bev_m,
        shape=(2,),
        field_name="additive_bias_reference_bev_m",
    )
    reference_time = _finite_scalar(reference_time_s, field_name="reference_time_s")
    timestamp_fault = _finite_scalar(
        timestamp_fault_s,
        field_name="timestamp_fault_s",
    )
    reference_from_global = global_from_reference_ego.inverse()
    aligned_truth = truth - velocity * timestamp_fault
    point = reference_from_global.apply(aligned_truth) + _ZERO_Z_LIFT @ (error + bias)
    return FullReconstruction(
        point_current_ego_m=point,
        base_error_jacobian=_ZERO_Z_LIFT,
        reference_state_time_s=reference_time,
        reported_state_time_s=reference_time + timestamp_fault,
    )


def persistent_panel_projection(
    reconstruction: FullReconstruction,
    *,
    reported_base_covariance_m2: npt.ArrayLike,
) -> ProjectedEstimate:
    """Apply M5-A ``A=P_xy B`` and ``C_Ek=A C_base A^T``."""

    base_covariance = _covariance2(
        reported_base_covariance_m2,
        field_name="reported_base_covariance_m2",
    )
    jacobian = _PROJECT_XY @ reconstruction.base_error_jacobian
    covariance = jacobian @ base_covariance @ jacobian.T
    return ProjectedEstimate(
        point_m=_PROJECT_XY @ reconstruction.point_current_ego_m,
        jacobian=jacobian,
        reported_covariance_m2=covariance,
    )


def monitoring_scene_projection(
    reconstruction: FullReconstruction,
    *,
    reported_base_covariance_m2: npt.ArrayLike,
    global_from_current_ego: RigidTransform3,
    scene_anchor: SceneBevAnchor,
) -> ProjectedEstimate:
    """Apply the full-3D M5-B monitoring-only map and covariance pushforward."""

    base_covariance = _covariance2(
        reported_base_covariance_m2,
        field_name="reported_base_covariance_m2",
    )
    scene_from_global_xy = _rotation2(-scene_anchor.yaw_global_from_scene_rad)
    point_global = global_from_current_ego.apply(reconstruction.point_current_ego_m)
    point_scene = scene_from_global_xy @ (
        _PROJECT_XY @ point_global - scene_anchor.origin_global_xy_m
    )
    jacobian = (
        scene_from_global_xy
        @ _PROJECT_XY
        @ global_from_current_ego.rotation
        @ reconstruction.base_error_jacobian
    )
    covariance = jacobian @ base_covariance @ jacobian.T
    return ProjectedEstimate(
        point_m=point_scene,
        jacobian=jacobian,
        reported_covariance_m2=covariance,
    )
