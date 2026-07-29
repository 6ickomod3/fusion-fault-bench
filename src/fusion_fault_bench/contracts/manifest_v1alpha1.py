"""Scientifically constrained schemas for immutable v0.1 experiment intent."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    Field,
    FiniteFloat,
    PositiveFloat,
    TypeAdapter,
    field_validator,
    model_validator,
)

from fusion_fault_bench.contracts.common import ContractModel

Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MagnitudeGrid = Annotated[tuple[FiniteFloat, ...], Field(min_length=2)]
ProbabilityGrid = Annotated[tuple[FiniteFloat, ...], Field(min_length=2)]
StdScaleGrid = Annotated[tuple[FiniteFloat, ...], Field(min_length=2)]
FixedToleranceM2 = Annotated[FiniteFloat, Field(ge=1e-12, le=1e-12)]

type Modality = Literal["camera", "lidar"]
type SeverityDirection = Literal["identity", "negative", "positive", "increase"]
type SeverityUnit = Literal["m", "rad", "s", "probability", "std-scale"]
type FaultAxis = Literal["x", "y", "xy", "yaw", "time", "availability"]
type FaultFamily = Literal[
    "additive-position-bias",
    "increased-noise-correctly-reported",
    "increased-noise-underreported",
    "calibration-translation",
    "calibration-yaw",
    "timestamp-offset",
    "dropout",
    "common-mode-position-bias",
]
type MethodId = Literal[
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    "fault-target-drop-policy",
    "performance-oracle",
]


def _reject_negative_zero(value: float, *, field_name: str) -> None:
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{field_name} must use canonical positive zero")


def _validate_magnitude_grid(values: tuple[float, ...], *, field_name: str) -> None:
    for value in values:
        _reject_negative_zero(value, field_name=field_name)
    if values[0] != 0.0:
        raise ValueError(f"{field_name} must begin with identity magnitude 0")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{field_name} must be non-negative")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field_name} must be strictly increasing")


def _validate_std_scale_grid(values: tuple[float, ...]) -> None:
    for value in values:
        _reject_negative_zero(value, field_name="std_scale_values")
    if values[0] != 1.0:
        raise ValueError("std_scale_values must begin with identity scale 1")
    if any(value <= 0.0 for value in values):
        raise ValueError("std_scale_values must be positive")
    if tuple(sorted(set(values))) != values:
        raise ValueError("std_scale_values must be strictly increasing")


class RngSpec(ContractModel):
    """Named streams whose seeds do not depend on draw or call order."""

    engine: Literal["numpy-pcg64dxsm-v1"]
    stream_derivation: Literal["sha256-name-and-sequence-v1"]
    data_master_seed: Annotated[int, Field(ge=0, lt=2**128)]
    bootstrap_seed: Annotated[int, Field(ge=0, lt=2**128)]
    data_streams: tuple[
        Literal["latent"],
        Literal["camera"],
        Literal["lidar"],
        Literal["fault"],
    ]

    @model_validator(mode="after")
    def require_distinct_seeds(self) -> RngSpec:
        if self.data_master_seed == self.bootstrap_seed:
            raise ValueError("data_master_seed and bootstrap_seed must differ")
        return self


class AnalyticSource(ContractModel):
    """Closed-form one-object, one-frame Gaussian verification case."""

    kind: Literal["analytic"]
    case_id: Literal["one-object-static-local-error-gaussian-v1"]
    sequence_count: Annotated[int, Field(ge=2)]


class AnalyticValidationSpec(ContractModel):
    """Fixed independent-oracle acceptance thresholds for analytic runs."""

    population_mse_abs_tolerance_m2: FixedToleranceM2
    grid_crossover_abs_tolerance: FixedToleranceM2
    monte_carlo_standard_error_multiplier: Annotated[
        FiniteFloat,
        Field(ge=6.0, le=6.0),
    ]
    continuous_root_reporting: Literal["separate-from-grid-estimand"]


class ProceduralSource(ContractModel):
    """Reference to a committed, content-addressed generator profile."""

    kind: Literal["procedural"]
    split: Literal["train", "validation", "test"]
    profile_id: Slug
    profile_sha256: Digest
    sequence_count: Annotated[int, Field(ge=2)]


class NuScenesMiniSource(ContractModel):
    """Local nuScenes-mini latent-scene selection without a dataset path."""

    kind: Literal["nuscenes-mini"]
    adapter_profile: Literal["nuscenes-mini-matched-centers-v1"]
    scene_names: Annotated[
        tuple[Slug, ...],
        Field(min_length=2, json_schema_extra={"uniqueItems": True}),
    ]
    camera_channel: Literal["CAM_FRONT"]

    @field_validator("scene_names")
    @classmethod
    def require_unique_scenes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("scene_names must be unique")
        return value


type GeometrySource = Annotated[
    ProceduralSource | NuScenesMiniSource,
    Field(discriminator="kind"),
]


class CommonRoiSpec(ContractModel):
    """Common front-camera/LiDAR support in the ego BEV frame."""

    frame: Literal["ego-bev"]
    x_min_m: FiniteFloat
    x_max_m: FiniteFloat
    abs_y_max_m: PositiveFloat
    camera_half_fov_rad: Annotated[FiniteFloat, Field(gt=0.0, lt=math.pi / 2.0)]

    @field_validator("x_min_m", "x_max_m")
    @classmethod
    def reject_negative_zero(cls, value: float) -> float:
        _reject_negative_zero(value, field_name="ROI x bound")
        return value

    @model_validator(mode="after")
    def require_ordered_range(self) -> CommonRoiSpec:
        if self.x_min_m < 0.0:
            raise ValueError("x_min_m must be non-negative")
        if self.x_max_m <= self.x_min_m:
            raise ValueError("x_max_m must be greater than x_min_m")
        return self


class CartesianGaussianSensorSpec(ContractModel):
    """A fully declared v0.1 Cartesian estimator-output model."""

    output_space: Literal["ego-bev-xy"]
    true_error_distribution: Literal["gaussian"]
    true_error_mean_xy_m: Annotated[
        tuple[FiniteFloat, FiniteFloat],
        Field(json_schema_extra={"prefixItems": [{"const": 0.0}, {"const": 0.0}]}),
    ]
    true_covariance_structure: Literal["diagonal"]
    actual_std_xy_m: tuple[PositiveFloat, PositiveFloat]
    reported_covariance_structure: Literal["diagonal"]
    reported_std_xy_m: tuple[PositiveFloat, PositiveFloat]
    temporal_dependence: Literal["iid"]
    range_dependence: Literal["constant"]

    @field_validator("true_error_mean_xy_m")
    @classmethod
    def reject_noncanonical_zero(cls, value: tuple[float, float]) -> tuple[float, float]:
        for component in value:
            _reject_negative_zero(component, field_name="true_error_mean_xy_m")
            if component != 0.0:
                raise ValueError("true_error_mean_xy_m must be exactly zero")
        return value


class ObservationModelsSpec(ContractModel):
    """Matched camera and LiDAR estimator-output contracts."""

    camera: CartesianGaussianSensorSpec
    lidar: CartesianGaussianSensorSpec
    base_stochastic_sensor_errors: Literal["independent"]

    @model_validator(mode="after")
    def require_calibrated_identity_uncertainty(self) -> ObservationModelsSpec:
        for name, sensor in (("camera", self.camera), ("lidar", self.lidar)):
            if sensor.actual_std_xy_m != sensor.reported_std_xy_m:
                raise ValueError(f"{name} actual and reported base std must match at identity")
        return self


class AdditivePositionBiasFault(ContractModel):
    """z_target <- z_target + direction * magnitude * axis_vector."""

    kind: Literal["additive-position-bias"]
    target: Modality
    axis: Literal["x", "y"]
    unit: Literal["m"]
    injection_site: Literal["estimator-output"]
    direction_policy: Literal["symmetric-paired"]
    persistence: Literal["sequence"]
    magnitude_values_m: MagnitudeGrid

    @field_validator("magnitude_values_m")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_magnitude_grid(value, field_name="magnitude_values_m")
        return value


class CorrectlyReportedNoiseFault(ContractModel):
    """actual_std and reported_std <- nominal_std * std_scale."""

    kind: Literal["increased-noise-correctly-reported"]
    target: Modality
    axis: Literal["xy"]
    unit: Literal["std-scale"]
    injection_site: Literal["true-error-model"]
    reported_uncertainty: Literal["tracks-actual"]
    persistence: Literal["sequence-configuration"]
    std_scale_values: StdScaleGrid

    @field_validator("std_scale_values")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_std_scale_grid(value)
        return value


class UnderreportedNoiseFault(ContractModel):
    """actual_std <- nominal_std * std_scale; reported_std stays nominal."""

    kind: Literal["increased-noise-underreported"]
    target: Modality
    axis: Literal["xy"]
    unit: Literal["std-scale"]
    injection_site: Literal["true-error-model"]
    reported_uncertainty: Literal["nominal"]
    persistence: Literal["sequence-configuration"]
    std_scale_values: StdScaleGrid

    @field_validator("std_scale_values")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_std_scale_grid(value)
        return value


class CalibrationTranslationFault(ContractModel):
    """T_used <- Delta_T_ego(direction * magnitude) T_true."""

    kind: Literal["calibration-translation"]
    target: Literal["camera"]
    axis: Literal["x", "y"]
    unit: Literal["m"]
    injection_site: Literal["calibration-metadata"]
    perturbation_frame: Literal["ego"]
    direction_policy: Literal["symmetric-paired"]
    persistence: Literal["sequence"]
    magnitude_values_m: MagnitudeGrid

    @field_validator("magnitude_values_m")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_magnitude_grid(value, field_name="magnitude_values_m")
        return value


class CalibrationYawFault(ContractModel):
    """T_used <- Delta_R_ego_z(direction * magnitude) T_true."""

    kind: Literal["calibration-yaw"]
    target: Literal["camera"]
    axis: Literal["yaw"]
    unit: Literal["rad"]
    injection_site: Literal["calibration-metadata"]
    perturbation_frame: Literal["ego"]
    direction_policy: Literal["symmetric-paired"]
    persistence: Literal["sequence"]
    magnitude_values_rad: MagnitudeGrid

    @field_validator("magnitude_values_rad")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_magnitude_grid(value, field_name="magnitude_values_rad")
        return value


class TimestampOffsetFault(ContractModel):
    """t_reported <- t_true + direction * magnitude."""

    kind: Literal["timestamp-offset"]
    target: Modality
    axis: Literal["time"]
    unit: Literal["s"]
    injection_site: Literal["timestamp-metadata"]
    timestamp_convention: Literal["reported-minus-true"]
    direction_policy: Literal["symmetric-paired"]
    persistence: Literal["sequence"]
    magnitude_values_s: MagnitudeGrid

    @field_validator("magnitude_values_s")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_magnitude_grid(value, field_name="magnitude_values_s")
        return value


class CommonModePositionBiasFault(ContractModel):
    """Apply the same Cartesian bias to both modality outputs."""

    kind: Literal["common-mode-position-bias"]
    target: Literal["both"]
    axis: Literal["x", "y"]
    unit: Literal["m"]
    injection_site: Literal["estimator-output"]
    direction_policy: Literal["symmetric-paired"]
    persistence: Literal["sequence"]
    magnitude_values_m: MagnitudeGrid

    @field_validator("magnitude_values_m")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_magnitude_grid(value, field_name="magnitude_values_m")
        return value


class DropoutFault(ContractModel):
    """Independently remove target outputs with probability p per frame."""

    kind: Literal["dropout"]
    target: Modality
    axis: Literal["availability"]
    unit: Literal["probability"]
    injection_site: Literal["availability"]
    process: Literal["shared-target-modality-frame-bernoulli"]
    probability_values: ProbabilityGrid

    @field_validator("probability_values")
    @classmethod
    def validate_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _validate_magnitude_grid(value, field_name="probability_values")
        if any(probability > 1.0 for probability in value):
            raise ValueError("probability_values must be in [0, 1]")
        return value


type AnalyticSingleSensorFault = Annotated[
    AdditivePositionBiasFault | CorrectlyReportedNoiseFault | UnderreportedNoiseFault,
    Field(discriminator="kind"),
]
type GeometrySingleSensorFault = Annotated[
    AdditivePositionBiasFault
    | CorrectlyReportedNoiseFault
    | UnderreportedNoiseFault
    | CalibrationTranslationFault
    | CalibrationYawFault
    | TimestampOffsetFault,
    Field(discriminator="kind"),
]


class BootstrapSpec(ContractModel):
    """Complete-sequence paired percentile bootstrap."""

    method: Literal["percentile"]
    unit: Literal["sequence"]
    resampling: Literal["paired-indices-across-severities-and-methods"]
    interval_scope: Literal["pointwise"]
    replicates: Annotated[int, Field(ge=200, multiple_of=40)]
    confidence_level: Annotated[FiniteFloat, Field(ge=0.95, le=0.95)]


class CrossoverSpec(ContractModel):
    """Operational first-zero rule, including censored bootstrap replicates."""

    fit: Literal["nondecreasing-isotonic-pava"]
    fit_weights: Literal["equal-severity"]
    interpolation: Literal["linear-first-zero"]
    zero_tolerance_m2: FixedToleranceM2
    no_crossing_handling: Literal["right-censored-above-tested-maximum"]
    status_rule: Literal["two-sided-bootstrap-crossing-fraction"]


class SingleSensorCrossoverEvaluation(ContractModel):
    """Primary fused-minus-healthy crossover estimand."""

    mode: Literal["primary-single-sensor-crossover"]
    primary_loss: Literal["matched-center-mse"]
    loss_unit: Literal["m^2"]
    aggregation: Literal["object-frame-mean-then-sequence-mean"]
    primary_contrast: Literal["fused-minus-healthy"]
    bootstrap: BootstrapSpec
    crossover: CrossoverSpec
    target_drop_identity_action: Literal["fixed-fusion"]
    performance_oracle_selection_unit: Literal["sequence"]
    performance_oracle_candidates: tuple[
        Literal["camera-only"],
        Literal["lidar-only"],
        Literal["fixed-fusion"],
    ]
    oracle_recovery_denominator_tolerance_m2: FixedToleranceM2


class CommonModeControlEvaluation(ContractModel):
    """Absolute-error blind-spot control without a healthy reference."""

    mode: Literal["common-mode-blind-spot-control"]
    primary_loss: Literal["matched-center-mse"]
    loss_unit: Literal["m^2"]
    aggregation: Literal["object-frame-mean-then-sequence-mean"]
    primary_contrast: Literal["none"]
    bootstrap: BootstrapSpec
    crossover: Literal["not-applicable"]


class AvailabilityControlEvaluation(ContractModel):
    """Dropout evaluation that never hides missing outputs."""

    mode: Literal["availability-control"]
    metrics: tuple[
        Literal["coverage"],
        Literal["conditional-matched-center-mse"],
        Literal["undefined-output-rate"],
    ]
    missing_output_policy: Literal["undefined-no-localization-penalty"]
    rate_aggregation: Literal["count-ratio-with-sequence-bootstrap"]
    conditional_loss_aggregation: Literal["valid-object-frame-ratio-with-sequence-bootstrap"]
    undefined_bootstrap_replicate_action: Literal["exclude-and-require-two-sided-support"]
    unimodal_missing_input_action: Literal["undefined"]
    fixed_fusion_missing_input_action: Literal["undefined"]
    target_drop_identity_action: Literal["fixed-fusion"]
    target_drop_nonidentity_action: Literal["use-nontarget-modality"]
    bootstrap: BootstrapSpec
    crossover: Literal["not-applicable"]


type PrimaryMethods = tuple[
    Literal["camera-only"],
    Literal["lidar-only"],
    Literal["fixed-fusion"],
    Literal["fault-target-drop-policy"],
    Literal["performance-oracle"],
]
type ControlMethods = tuple[
    Literal["camera-only"],
    Literal["lidar-only"],
    Literal["fixed-fusion"],
]
type AvailabilityMethods = tuple[
    Literal["camera-only"],
    Literal["lidar-only"],
    Literal["fixed-fusion"],
    Literal["fault-target-drop-policy"],
]


class AnalyticCrossoverManifest(ContractModel):
    """Closed-form-compatible single-sensor crossover experiment."""

    schema_id: Literal["ffb.manifest/v1alpha1"] = Field(alias="schema")
    kind: Literal["analytic-crossover"]
    experiment: Slug
    rng: RngSpec
    source: AnalyticSource
    analytic_validation: AnalyticValidationSpec
    observations: ObservationModelsSpec
    fault_sweep: AnalyticSingleSensorFault
    methods: PrimaryMethods
    evaluation: SingleSensorCrossoverEvaluation

    @property
    def healthy_modality(self) -> Modality:
        return "lidar" if self.fault_sweep.target == "camera" else "camera"


class GeometryCrossoverManifest(ContractModel):
    """Geometry-capable temporal single-sensor crossover experiment."""

    schema_id: Literal["ffb.manifest/v1alpha1"] = Field(alias="schema")
    kind: Literal["geometry-crossover"]
    experiment: Slug
    rng: RngSpec
    source: GeometrySource
    roi: CommonRoiSpec
    observations: ObservationModelsSpec
    fault_sweep: GeometrySingleSensorFault
    methods: PrimaryMethods
    evaluation: SingleSensorCrossoverEvaluation

    @property
    def healthy_modality(self) -> Modality:
        return "lidar" if self.fault_sweep.target == "camera" else "camera"


class CommonModeControlManifest(ContractModel):
    """Procedural common-mode blind-spot control."""

    schema_id: Literal["ffb.manifest/v1alpha1"] = Field(alias="schema")
    kind: Literal["common-mode-control"]
    experiment: Slug
    rng: RngSpec
    source: ProceduralSource
    roi: CommonRoiSpec
    observations: ObservationModelsSpec
    fault_sweep: CommonModePositionBiasFault
    methods: ControlMethods
    evaluation: CommonModeControlEvaluation


class AvailabilityControlManifest(ContractModel):
    """Procedural dropout evaluation with explicit missingness metrics."""

    schema_id: Literal["ffb.manifest/v1alpha1"] = Field(alias="schema")
    kind: Literal["availability-control"]
    experiment: Slug
    rng: RngSpec
    source: ProceduralSource
    roi: CommonRoiSpec
    observations: ObservationModelsSpec
    fault_sweep: DropoutFault
    methods: AvailabilityMethods
    evaluation: AvailabilityControlEvaluation


type ExperimentManifestV1Alpha1 = Annotated[
    AnalyticCrossoverManifest
    | GeometryCrossoverManifest
    | CommonModeControlManifest
    | AvailabilityControlManifest,
    Field(discriminator="kind"),
]
EXPERIMENT_MANIFEST_ADAPTER = TypeAdapter(ExperimentManifestV1Alpha1)


def manifest_json_schema() -> dict[str, object]:
    """Return the JSON Schema artifact for the complete manifest union."""

    return EXPERIMENT_MANIFEST_ADAPTER.json_schema(by_alias=True)
