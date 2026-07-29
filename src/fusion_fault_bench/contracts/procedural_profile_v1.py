"""Fail-closed contracts for the three preregistered M3 procedural profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, FiniteFloat, PositiveFloat, TypeAdapter, model_validator

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.io import load_json_object

type SplitId = Literal["train", "validation", "test"]
type ProfileId = Literal[
    "constant-velocity-front-roi-v1",
    "constant-velocity-fov-edge-v1",
    "constant-velocity-ci-smoke-v1",
]

MAIN_PROFILE_SHA256 = "4771a6e69d75b9af41f99ab794c0af1b51e6103e43474c8e0f07df3e6f3ca68c"
EDGE_PROFILE_SHA256 = "ca1544f69023847af7bdad9f1306ae3885f2e5d067d6afc026038f87ae36448d"
SMOKE_PROFILE_SHA256 = "7f2479c064e0f8104789dfc3ce704a78aabdd46c1be7a31fdd2e75dbe3b407ed"

EXPECTED_PROFILE_DIGESTS: dict[ProfileId, str] = {
    "constant-velocity-front-roi-v1": MAIN_PROFILE_SHA256,
    "constant-velocity-fov-edge-v1": EDGE_PROFILE_SHA256,
    "constant-velocity-ci-smoke-v1": SMOKE_PROFILE_SHA256,
}


class ProceduralSourceSpec(ContractModel):
    kind: Literal["seeded-constant-velocity-known-id-bev"]
    frame_count: Annotated[int, Field(ge=2, le=10_000)]
    frame_index_start: Literal[0]
    frame_period_s: PositiveFloat
    frame_time_equation: Literal["t_k_s=frame_index*frame_period_s"]
    object_count: Annotated[int, Field(ge=1, le=10_000)]
    object_id_format: Literal["object:{index:02d}"]
    truth_equation: Literal["p_xy(object,frame)=initial_xy_m(object)+velocity_xy_mps(object)*t_k_s"]


class EgoAxesSpec(ContractModel):
    x: Literal["forward"]
    y: Literal["left"]
    z: Literal["up"]


class CoordinateContractSpec(ContractModel):
    handedness: Literal["right-handed"]
    point_representation: Literal["column-vector"]
    transform_notation: Literal["T_{target<-source}"]
    scoring_frame: Literal["ego-bev-at-frame-time"]
    ego_axes: EgoAxesSpec
    length_unit: Literal["m"]
    time_unit: Literal["s"]
    angle_unit: Literal["rad"]
    floating_point: Literal["float64"]


class LatentDrawSpec(ContractModel):
    stream_name: Literal["latent"]
    distribution: Literal["Generator.random"]
    dtype: Literal["float64"]
    calls_per_sequence: Literal[1]
    shape: tuple[int, Literal[4]]
    component_order: tuple[str, str, str, str]


class ErrorDrawSpec(ContractModel):
    stream_name: Literal["camera", "lidar"]
    distribution: Literal["Generator.standard_normal"]
    dtype: Literal["float64"]
    calls_per_sequence: Literal[1]
    shape: tuple[Literal["eligible_object_frame_count"], Literal[2]]
    coordinate_order: tuple[Literal["x"], Literal["y"]]


class DropoutDrawSpec(ContractModel):
    stream_name: Literal["fault"]
    distribution: Literal["Generator.random"]
    dtype: Literal["float64"]
    calls_per_sequence: Literal[1]
    shape: tuple[int]
    draw_order: Literal["increasing-frame-index"]
    shared_across_objects_in_frame: Literal[True]
    drop_rule: Literal["drop-when-u-less-than-probability"]
    reuse_across_probability_grid: Literal[True]


class RngContractSpec(ContractModel):
    engine: Literal["numpy-pcg64dxsm-v1"]
    stream_derivation: Literal["sha256-name-and-sequence-v1"]
    seed_source: Literal["experiment-manifest"]
    latent_draw: LatentDrawSpec
    camera_error_draw: ErrorDrawSpec
    lidar_error_draw: ErrorDrawSpec
    dropout_draw: DropoutDrawSpec
    reuse_base_draws_across_fault_families_severities_directions_and_methods: Literal[True]

    @model_validator(mode="after")
    def require_sensor_streams(self) -> Self:
        if self.camera_error_draw.stream_name != "camera":
            raise ValueError("camera_error_draw must use the camera stream")
        if self.lidar_error_draw.stream_name != "lidar":
            raise ValueError("lidar_error_draw must use the lidar stream")
        return self


class DeterministicOrderSpec(ContractModel):
    sequences: Literal["ascending-zero-based-index"]
    frames: Literal["ascending-integer-frame-index"]
    objects: Literal["utf8-byte-order-of-object-id"]
    coordinates: tuple[Literal["x"], Literal["y"]]
    eligible_error_rows: Literal["frame-then-object"]
    filesystem_enumeration_used: Literal[False]


class EligibilitySpec(ContractModel):
    decision_stage: Literal["before-fault-injection"]
    reuse_same_object_frames_across_all_severities_directions_and_methods: Literal[True]
    frame: Literal["ego-bev"]
    x_min_m: FiniteFloat
    x_max_m: FiniteFloat
    abs_y_max_m: PositiveFloat
    positive_forward_required: Literal[True]
    camera_support: Literal["inclusive-symmetric-bearing-half-fov"]
    camera_half_fov_rad: Annotated[FiniteFloat, Field(gt=0.0, lt=1.5707963267948966)]
    camera_and_lidar_estimator_availability_required: Literal[True]
    lidar_support_available: Literal[True]

    @model_validator(mode="after")
    def require_frozen_roi(self) -> Self:
        if (
            self.x_min_m,
            self.x_max_m,
            self.abs_y_max_m,
            self.camera_half_fov_rad,
        ) != (5.0, 60.0, 40.0, 0.7):
            raise ValueError("M3 profiles must use the preregistered common ROI")
        return self


class CameraFrameAxesSpec(ContractModel):
    x: Literal["right"]
    y: Literal["down"]
    z: Literal["forward"]


class CameraExtrinsicSpec(ContractModel):
    notation: Literal["T_{ego<-camera}"]
    translation_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    quaternion_order: Literal["wxyz"]
    quaternion_wxyz: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def require_frozen_extrinsic(self) -> Self:
        if self.translation_m != (1.5, 0.0, 1.5):
            raise ValueError("camera translation differs from the M3 profile")
        if self.quaternion_wxyz != (0.5, -0.5, 0.5, -0.5):
            raise ValueError("camera quaternion differs from the M3 profile")
        return self


class RigSpec(ContractModel):
    rig_id: Literal["stationary-front-camera-lidar-v1"]
    ego_motion: Literal["stationary"]
    camera_channel: Literal["CAM_FRONT"]
    camera_frame_axes: CameraFrameAxesSpec
    camera_true_extrinsic: CameraExtrinsicSpec
    lidar_channel: Literal["LIDAR_TOP"]
    lidar_estimator_output_frame: Literal["ego-bev-at-frame-time"]


class ReconstructionSpec(ContractModel):
    camera_base_error_application: Literal[
        "add-manifest-scaled-ego-bev-error-before-true-pose-backprojection"
    ]
    camera_physical_proxy_observation_equation: Literal[
        "q_camera=T_{camera<-ego}^{true}*[p_truth_x+error_x,p_truth_y+error_y,0]"
    ]
    camera_reconstruction_equation: Literal["z_camera_ego=T_{ego<-camera}^{reported}*q_camera"]
    lidar_reconstruction_equation: Literal[
        "z_lidar_ego_xy=p_truth_xy+manifest-scaled-lidar-error_xy"
    ]
    calibration_generation_transform: Literal["true-only"]
    calibration_translation_fault_equation: Literal["T_reported=DeltaT_ego*T_true"]
    calibration_yaw_fault_equation: Literal["T_reported=DeltaR_ego_z*T_true"]
    calibration_fault_may_change_eligibility: Literal[False]
    reported_covariance_frame: Literal["reconstructed-ego-bev"]
    reported_covariance_rotation_rule: Literal["rotate-with-reported-transform"]
    isotropic_camera_covariance_keeps_xy_diagonal_under_yaw: Literal[True] = Field(
        alias="m3_isotropic-camera-covariance_keeps_xy_diagonal_under-yaw"
    )


class TimingSpec(ContractModel):
    reference_timestamp: Literal["frame-time"]
    camera_timestamp_true: Literal["frame-time"]
    lidar_timestamp_true: Literal["frame-time"]
    physical_camera_time_offset_s: FiniteFloat
    physical_lidar_time_offset_s: FiniteFloat
    timestamp_fault_sign: Literal["reported-minus-true"]
    timestamp_fault_equation: Literal["t_reported=t_true+signed-magnitude"]
    alignment_equation: Literal[
        "z_aligned_xy=z_raw_xy+latent_velocity_xy_mps*(t_reference-t_reported)"
    ]
    alignment_velocity_source: Literal["declared-latent-constant-velocity"]
    identity_equation: Literal["t_reference=t_true=t_reported"]
    timestamp_fault_may_change_eligibility: Literal[False]

    @model_validator(mode="after")
    def require_zero_physical_offsets(self) -> Self:
        if self.physical_camera_time_offset_s != 0.0:
            raise ValueError("physical camera time offset must be zero")
        if self.physical_lidar_time_offset_s != 0.0:
            raise ValueError("physical lidar time offset must be zero")
        return self


class FaultPairingSpec(ContractModel):
    persistent_bias_calibration_and_timestamp_faults: Literal["constant-within-sequence"]
    noise_faults: Literal["sequence-configuration-with-iid-object-frame-errors"]
    signed_direction_order: tuple[Literal["negative"], Literal["positive"]]
    identity_emitted_once: Literal[True]
    faults_never_accumulate_across_severity: Literal[True]


class ScalarEquationSupport(ContractModel):
    equation: str
    support: tuple[FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def require_ordered_support(self) -> Self:
        if self.support[1] < self.support[0]:
            raise ValueError("support bounds must be ordered")
        return self


class MainInitialYTrain(ContractModel):
    equation: Literal["lane_centers_m[object_index_mod_2]+0.25*(2*u1-1)"]
    lane_centers_m: tuple[FiniteFloat, FiniteFloat]
    jitter_half_width_m: FiniteFloat

    @model_validator(mode="after")
    def require_frozen_values(self) -> Self:
        if self.lane_centers_m != (-3.5, 3.5) or self.jitter_half_width_m != 0.25:
            raise ValueError("train lateral mapping differs from the frozen profile")
        return self


class MainTrainSplit(ContractModel):
    sequence_count: Literal[200]
    layout_family: Literal["near-two-lane-flow"]
    initial_x_m: ScalarEquationSupport
    initial_y_m: MainInitialYTrain
    velocity_x_mps: ScalarEquationSupport
    velocity_y_mps: ScalarEquationSupport

    @model_validator(mode="after")
    def require_equations(self) -> Self:
        expected = (
            (self.initial_x_m.equation, self.initial_x_m.support),
            (self.velocity_x_mps.equation, self.velocity_x_mps.support),
            (self.velocity_y_mps.equation, self.velocity_y_mps.support),
        )
        if expected != (
            ("10+18*u0", (10.0, 28.0)),
            ("-1+2*u2", (-1.0, 1.0)),
            ("0.1*(2*u3-1)", (-0.1, 0.1)),
        ):
            raise ValueError("train split equations differ from the frozen profile")
        return self


class SideParitySpec(ContractModel):
    even: Literal[-1, 1]
    odd: Literal[-1, 1]


class AbsoluteEquationSupport(ContractModel):
    equation: str
    absolute_support: tuple[FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def require_ordered_support(self) -> Self:
        if self.absolute_support[1] < self.absolute_support[0]:
            raise ValueError("absolute support bounds must be ordered")
        return self


class ValidationVelocityY(ContractModel):
    equation: Literal["-side*(1.5+1.5*u3)"]
    absolute_support: tuple[FiniteFloat, FiniteFloat]
    direction: Literal["toward-centerline-at-frame-zero"]

    @model_validator(mode="after")
    def require_frozen_support(self) -> Self:
        if self.absolute_support != (1.5, 3.0):
            raise ValueError("validation lateral velocity support differs from the profile")
        return self


class MainValidationSplit(ContractModel):
    sequence_count: Literal[200]
    layout_family: Literal["mid-lateral-crossing"]
    object_side_by_parity: SideParitySpec
    initial_x_m: ScalarEquationSupport
    initial_y_m: AbsoluteEquationSupport
    velocity_x_mps: ScalarEquationSupport
    velocity_y_mps: ValidationVelocityY

    @model_validator(mode="after")
    def require_equations(self) -> Self:
        if self.object_side_by_parity != SideParitySpec(even=-1, odd=1):
            raise ValueError("validation side parity differs from the frozen profile")
        if (
            self.initial_x_m.equation,
            self.initial_x_m.support,
            self.initial_y_m.equation,
            self.initial_y_m.absolute_support,
            self.velocity_x_mps.equation,
            self.velocity_x_mps.support,
        ) != (
            "30+10*u0",
            (30.0, 40.0),
            "side*(5+3*u1)",
            (5.0, 8.0),
            "-1+2*u2",
            (-1.0, 1.0),
        ):
            raise ValueError("validation equations differ from the frozen profile")
        return self


class MainInitialYTest(ContractModel):
    equation: Literal["lateral_centers_m[object_index]+0.25*(2*u1-1)"]
    lateral_centers_m: tuple[
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
        FiniteFloat,
    ]
    jitter_half_width_m: FiniteFloat

    @model_validator(mode="after")
    def require_frozen_values(self) -> Self:
        if self.lateral_centers_m != (-7.0, -4.0, -1.0, 1.0, 4.0, 7.0):
            raise ValueError("test lateral centers differ from the frozen profile")
        if self.jitter_half_width_m != 0.25:
            raise ValueError("test lateral jitter differs from the frozen profile")
        return self


class MainTestSplit(ContractModel):
    sequence_count: Literal[200]
    layout_family: Literal["far-fast-approach"]
    initial_x_m: ScalarEquationSupport
    initial_y_m: MainInitialYTest
    velocity_x_mps: ScalarEquationSupport
    velocity_y_mps: ScalarEquationSupport

    @model_validator(mode="after")
    def require_equations(self) -> Self:
        if (
            self.initial_x_m.equation,
            self.initial_x_m.support,
            self.velocity_x_mps.equation,
            self.velocity_x_mps.support,
            self.velocity_y_mps.equation,
            self.velocity_y_mps.support,
        ) != (
            "44+12*u0",
            (44.0, 56.0),
            "-(3+2*u2)",
            (-5.0, -3.0),
            "0.2*(2*u3-1)",
            (-0.2, 0.2),
        ):
            raise ValueError("test equations differ from the frozen profile")
        return self


class MainSplits(ContractModel):
    train: MainTrainSplit
    validation: MainValidationSplit
    test: MainTestSplit


class InitialPositionEquation(ContractModel):
    equation: Literal["initial_range_m*[cos(bearing_rad),sin(bearing_rad)]"]


class BearingSpec(ContractModel):
    equation: Literal["side*(0.7-(0.005+0.015*u1))"]
    absolute_support: tuple[FiniteFloat, FiniteFloat]
    inside_half_fov_margin_rad: tuple[FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def require_frozen_support(self) -> Self:
        if self.absolute_support != (0.68, 0.695):
            raise ValueError("edge bearing support differs from the frozen profile")
        if self.inside_half_fov_margin_rad != (0.005, 0.02):
            raise ValueError("edge FOV margin differs from the frozen profile")
        return self


class EdgeVelocitySpec(ContractModel):
    equation: Literal["radial_speed_mps*[cos(bearing_rad),sin(bearing_rad)]"]
    bearing_is_constant_without_noise: Literal[True]


class EdgeTestSplit(ContractModel):
    sequence_count: Literal[100]
    layout_family: Literal["fov-edge-radial-motion"]
    object_side_by_parity: SideParitySpec
    initial_range_m: ScalarEquationSupport
    bearing_rad: BearingSpec
    initial_position_m: InitialPositionEquation
    radial_speed_mps: ScalarEquationSupport
    velocity_xy_mps: EdgeVelocitySpec

    @model_validator(mode="after")
    def require_equations(self) -> Self:
        if self.object_side_by_parity != SideParitySpec(even=1, odd=-1):
            raise ValueError("edge side parity differs from the frozen profile")
        if (
            self.initial_range_m.equation,
            self.initial_range_m.support,
            self.radial_speed_mps.equation,
            self.radial_speed_mps.support,
        ) != (
            "20+20*u0",
            (20.0, 40.0),
            "-0.5+u2",
            (-0.5, 0.5),
        ):
            raise ValueError("edge equations differ from the frozen profile")
        return self


class EdgeSplits(ContractModel):
    test: EdgeTestSplit


class SmokeInitialY(ContractModel):
    equation: Literal["lateral_centers_m[object_index]+0.1*(2*u1-1)"]
    lateral_centers_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    jitter_half_width_m: FiniteFloat

    @model_validator(mode="after")
    def require_frozen_values(self) -> Self:
        if self.lateral_centers_m != (-2.0, 0.0, 2.0):
            raise ValueError("smoke lateral centers differ from the frozen profile")
        if self.jitter_half_width_m != 0.1:
            raise ValueError("smoke jitter differs from the frozen profile")
        return self


class SmokeTestSplit(ContractModel):
    sequence_count: Literal[4]
    layout_family: Literal["small-front-roi-smoke"]
    initial_x_m: ScalarEquationSupport
    initial_y_m: SmokeInitialY
    velocity_x_mps: ScalarEquationSupport
    velocity_y_mps: ScalarEquationSupport

    @model_validator(mode="after")
    def require_equations(self) -> Self:
        if (
            self.initial_x_m.equation,
            self.initial_x_m.support,
            self.velocity_x_mps.equation,
            self.velocity_x_mps.support,
            self.velocity_y_mps.equation,
            self.velocity_y_mps.support,
        ) != (
            "10+10*u0",
            (10.0, 20.0),
            "-0.5+u2",
            (-0.5, 0.5),
            "0.05*(2*u3-1)",
            (-0.05, 0.05),
        ):
            raise ValueError("smoke equations differ from the frozen profile")
        return self


class SmokeSplits(ContractModel):
    test: SmokeTestSplit


class _BaseProceduralProfile(ContractModel):
    schema_id: Literal["ffb.procedural-profile/v1"] = Field(alias="schema")
    source: ProceduralSourceSpec
    coordinate_contract: CoordinateContractSpec
    rng_contract: RngContractSpec
    deterministic_order: DeterministicOrderSpec
    eligibility: EligibilitySpec
    rig: RigSpec
    reconstruction: ReconstructionSpec
    timing: TimingSpec
    fault_pairing: FaultPairingSpec

    def _validate_shape(
        self,
        *,
        frame_count: int,
        object_count: int,
        components: tuple[str, str, str, str],
    ) -> None:
        if (
            self.source.frame_count != frame_count
            or self.source.object_count != object_count
            or self.source.frame_period_s != 0.1
        ):
            raise ValueError("source dimensions differ from the frozen profile")
        if self.rng_contract.latent_draw.shape != (object_count, 4):
            raise ValueError("latent draw shape disagrees with object_count")
        if self.rng_contract.latent_draw.component_order != components:
            raise ValueError("latent component order differs from the frozen profile")
        if self.rng_contract.dropout_draw.shape != (frame_count,):
            raise ValueError("dropout draw shape disagrees with frame_count")


class MainProceduralProfile(_BaseProceduralProfile):
    profile_id: Literal["constant-velocity-front-roi-v1"]
    splits: MainSplits

    @model_validator(mode="after")
    def require_main_shape(self) -> Self:
        self._validate_shape(
            frame_count=48,
            object_count=6,
            components=("u0", "u1", "u2", "u3"),
        )
        return self


class EdgeProceduralProfile(_BaseProceduralProfile):
    profile_id: Literal["constant-velocity-fov-edge-v1"]
    splits: EdgeSplits

    @model_validator(mode="after")
    def require_edge_shape(self) -> Self:
        self._validate_shape(
            frame_count=48,
            object_count=4,
            components=("u0", "u1", "u2", "u3-unused"),
        )
        return self


class SmokeProceduralProfile(_BaseProceduralProfile):
    profile_id: Literal["constant-velocity-ci-smoke-v1"]
    splits: SmokeSplits

    @model_validator(mode="after")
    def require_smoke_shape(self) -> Self:
        self._validate_shape(
            frame_count=8,
            object_count=3,
            components=("u0", "u1", "u2", "u3"),
        )
        return self


type ProceduralProfileV1 = Annotated[
    MainProceduralProfile | EdgeProceduralProfile | SmokeProceduralProfile,
    Field(discriminator="profile_id"),
]
PROCEDURAL_PROFILE_ADAPTER = TypeAdapter(ProceduralProfileV1)


def load_procedural_profile(path: Path) -> ProceduralProfileV1:
    """Strictly load one preregistered profile and verify its canonical digest."""

    raw = path.read_text(encoding="utf-8")
    profile = PROCEDURAL_PROFILE_ADAPTER.validate_json(raw)
    mapping = load_json_object(path)
    digest = sha256_digest(mapping)
    expected = EXPECTED_PROFILE_DIGESTS[profile.profile_id]
    if digest != expected:
        raise ValueError("procedural profile digest is not preregistered")
    return profile


def profile_sequence_count(profile: ProceduralProfileV1, split: SplitId) -> int:
    """Return the exact sequence count for a split or reject its absence."""

    if isinstance(profile, MainProceduralProfile):
        return getattr(profile.splits, split).sequence_count
    if split != "test":
        raise ValueError("this procedural profile defines only the test split")
    return profile.splits.test.sequence_count


def procedural_profile_json_schema() -> dict[str, Any]:
    """Return the complete discriminated profile schema."""

    return PROCEDURAL_PROFILE_ADAPTER.json_schema(by_alias=True)
