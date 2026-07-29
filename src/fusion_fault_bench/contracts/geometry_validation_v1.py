"""Strict contracts for the frozen M2 geometry-validation artifact."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, TypeAdapter, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, Identifier

GEOMETRY_MANIFEST_FILE = "manifest.json"
GEOMETRY_VALIDATION_FILE = "geometry-validation.json"
GEOMETRY_PAYLOAD_INDEX_FILE = "payload-index.json"
GEOMETRY_RUN_FILE = "run.json"
GEOMETRY_SUCCESS_FILE = "_SUCCESS"

GEOMETRY_ARTIFACT_CONTRACT = "ffb.geometry-validation-payload/v1"
GEOMETRY_MANIFEST_SCHEMA = "ffb.geometry-validation-manifest/v1"
GEOMETRY_VALIDATION_SCHEMA = "ffb.geometry-validation/v1"
GEOMETRY_PAYLOAD_INDEX_SCHEMA = "ffb.geometry-payload-index/v1"
GEOMETRY_MANIFEST_SHA256 = "7bfb5427c1ea5450a795bedef327457e5959860316bcd23f930e6eaa5917a068"
FROZEN_GEOMETRY_MANIFEST_SHA256 = GEOMETRY_MANIFEST_SHA256

GEOMETRY_INDEXED_PAYLOAD_PATHS = (
    GEOMETRY_MANIFEST_FILE,
    GEOMETRY_VALIDATION_FILE,
)
GEOMETRY_ARTIFACT_PATHS = (
    *GEOMETRY_INDEXED_PAYLOAD_PATHS,
    GEOMETRY_PAYLOAD_INDEX_FILE,
    GEOMETRY_RUN_FILE,
    GEOMETRY_SUCCESS_FILE,
)
GEOMETRY_MEMBER_BYTE_CAP = 1_048_576
GEOMETRY_TREE_BYTE_CAP = 5_242_880
GEOMETRY_LOGICAL_COMMAND = (
    "ffb",
    "geometry",
    "validate",
    "examples/validation/m2-geometry-v1.json",
    "--dataset-root-env",
    "NUSCENES_ROOT",
    "--output-dir",
    "reports/generated/m2-geometry",
)

EXPECTED_KEYFRAME_BLOB_CHECK_COUNT = 808
EXPECTED_MONTE_CARLO_SAMPLE_COUNT = 200_000
SYNTHETIC_FIXTURE_ID = "m2-nuscenes-convention-independent-v1"
SYNTHETIC_FIXTURE_SHA256 = "0676993f48e5a40034dfe497df7165b33f2d2f96dad234afd62af8e461beb252"

ROTATION_MAX_ABS_TOLERANCE = 1e-12
TRANSLATION_MAX_ABS_TOLERANCE_M = 1e-10
POINT_ROUND_TRIP_MAX_ABS_TOLERANCE_M = 1e-10
QUATERNION_SIGN_MAX_ABS_TOLERANCE = 1e-12
SYNTHETIC_PROJECTION_MAX_ABS_TOLERANCE_PX = 1e-9
SYNTHETIC_DEPTH_MAX_ABS_TOLERANCE_M = 1e-12
BOX_CORNER_MAX_ABS_TOLERANCE_M = 1e-12
FINITE_DIFFERENCE_MAX_ABS_TOLERANCE = 1e-7
COVARIANCE_DERIVATION_ABS_TOLERANCE = 1e-12

type GeometryIndexedPayloadPath = Literal[
    "manifest.json",
    "geometry-validation.json",
]
type NonNegativeFiniteFloat = Annotated[FiniteFloat, Field(ge=0.0)]
type PositiveFiniteFloat = Annotated[FiniteFloat, Field(gt=0.0)]
type CovarianceEntryName = Literal["xx", "xy", "yy"]
type _ExactFloatZero = Annotated[FiniteFloat, Field(ge=0.0, le=0.0)]
type _ExactFloatPointZeroFour = Annotated[FiniteFloat, Field(ge=0.04, le=0.04)]
type _ExactFloatPointZeroSixTwoFive = Annotated[
    FiniteFloat,
    Field(ge=0.0625, le=0.0625),
]
type _ExactFloatPointOne = Annotated[FiniteFloat, Field(ge=0.1, le=0.1)]
type _ExactFloatPointTwo = Annotated[FiniteFloat, Field(ge=0.2, le=0.2)]
type _ExactFloatPointFive = Annotated[FiniteFloat, Field(ge=0.5, le=0.5)]
type _ExactFloatOne = Annotated[FiniteFloat, Field(ge=1.0, le=1.0)]
type _ExactFloatSix = Annotated[FiniteFloat, Field(ge=6.0, le=6.0)]
type _ExactFloatTwentyFive = Annotated[FiniteFloat, Field(ge=25.0, le=25.0)]
type _ExactFloat1EMinus4 = Annotated[FiniteFloat, Field(ge=1e-4, le=1e-4)]
type _ExactFloat1EMinus6 = Annotated[FiniteFloat, Field(ge=1e-6, le=1e-6)]
type _ExactFloat1EMinus7 = Annotated[FiniteFloat, Field(ge=1e-7, le=1e-7)]
type _ExactFloat1EMinus8 = Annotated[FiniteFloat, Field(ge=1e-8, le=1e-8)]
type _ExactFloat1EMinus9 = Annotated[FiniteFloat, Field(ge=1e-9, le=1e-9)]
type _ExactFloat1EMinus10 = Annotated[FiniteFloat, Field(ge=1e-10, le=1e-10)]
type _ExactFloat1EMinus12 = Annotated[FiniteFloat, Field(ge=1e-12, le=1e-12)]
type _ExactFloat2Point25EMinus8 = Annotated[
    FiniteFloat,
    Field(ge=2.25e-8, le=2.25e-8),
]
type _ExactFloat5EMinus6 = Annotated[FiniteFloat, Field(ge=5e-6, le=5e-6)]
type _ExactFloatNegative3Point75EMinus6 = Annotated[
    FiniteFloat,
    Field(ge=-3.75e-6, le=-3.75e-6),
]

type RequiredDatasetTables = tuple[
    Literal["attribute"],
    Literal["calibrated_sensor"],
    Literal["category"],
    Literal["ego_pose"],
    Literal["instance"],
    Literal["log"],
    Literal["sample"],
    Literal["sample_annotation"],
    Literal["sample_data"],
    Literal["scene"],
    Literal["sensor"],
    Literal["visibility"],
]
type PropertyDrawOrder = tuple[
    Literal["left-quaternion-standard-normal-256x4-normalize"],
    Literal["right-quaternion-standard-normal-256x4-normalize"],
    Literal["left-translation-uniform-minus100-plus100-m-256x3"],
    Literal["right-translation-uniform-minus100-plus100-m-256x3"],
    Literal["point-standard-normal-times25-m-256x3"],
]
type GeometryArtifactFiles = tuple[
    Literal["manifest.json"],
    Literal["geometry-validation.json"],
    Literal["payload-index.json"],
    Literal["run.json"],
    Literal["_SUCCESS"],
]
type GeometryIndexedFiles = tuple[
    Literal["manifest.json"],
    Literal["geometry-validation.json"],
]
type GeometryLogicalCommand = tuple[
    Literal["ffb"],
    Literal["geometry"],
    Literal["validate"],
    Literal["examples/validation/m2-geometry-v1.json"],
    Literal["--dataset-root-env"],
    Literal["NUSCENES_ROOT"],
    Literal["--output-dir"],
    Literal["reports/generated/m2-geometry"],
]


class GeometryHeadlineCountsV1(ContractModel):
    """The three public counts fixed by the official nuScenes-mini profile."""

    scene: Literal[10]
    sample: Literal[404]
    sample_annotation: Literal[18_538]


class GeometryDiagnosticSelectorV1(ContractModel):
    """Deterministic local-only diagnostic selection without selected IDs."""

    scene: Literal["lexicographically-first-scene-name"]
    sample: Literal["declared-first-sample-link"]
    annotations: Literal["all-sample-annotations-sorted-by-utf8-token"]


class GeometryDiagnosticNonvacuityV1(ContractModel):
    """Predeclared minimum content for the local diagnostic."""

    minimum_annotation_count: Literal[1]
    minimum_finite_positive_depth_center_count: Literal[1] = Field(
        alias="minimum_finite_positive-depth-center_count"
    )


class GeometryDatasetSpecV1(ContractModel):
    """Frozen local dataset profile without a dataset identity or path."""

    profile: Literal["official-nuscenes-v1.0-mini"]
    version_directory: Literal["v1.0-mini"]
    required_tables: RequiredDatasetTables
    camera_channel: Literal["CAM_FRONT"]
    reference_lidar_channel: Literal["LIDAR_TOP"]
    canonical_bev_frame: Literal["reference-lidar-keyframe-ego"]
    camera_visibility_frame: Literal["front-camera-keyframe-ego"]
    annotation_semantics: Literal["recorded-sample-level-center-treated-as-benchmark-truth"]
    blob_check_scope: Literal["camera-and-reference-lidar-keyframes-only"]
    expected_keyframe_blob_check_count: Literal[808]
    expected_headline_counts: GeometryHeadlineCountsV1
    diagnostic_selector: GeometryDiagnosticSelectorV1
    diagnostic_nonvacuity: GeometryDiagnosticNonvacuityV1
    scalar_reference: Literal["independent-stdlib-json-and-scalar-algebra-v1"]
    public_local_projection_detail: Literal["pass-fail-only"]
    dataset_authentication: Literal["summary-does-not-authenticate-dataset-bytes"]


class DatasetTermsV1(ContractModel):
    """Exact public nuScenes attribution, terms, and non-endorsement notice."""

    source: Literal["nuScenes v1.0-mini, Motional"]
    license: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]
    terms_url: Literal["https://www.nuscenes.org/terms-of-use"]
    attribution: Literal[
        "nuScenes: A multimodal dataset for autonomous driving, Caesar et al., 2020"
    ]
    non_endorsement: Literal["Motional does not sponsor, approve, or endorse Fusion Fault Bench"]


class GeometryConventionSpecV1(ContractModel):
    """Frozen transform, projection, box, and ROI conventions."""

    transform_notation: Literal["T_target_from_source-column-vectors"]
    frames: Literal["right-handed"]
    quaternion_order: Literal["wxyz"]
    quaternion_unit_norm_tolerance: _ExactFloat1EMinus6
    rotation_orthogonality_tolerance: _ExactFloat1EMinus10
    rotation_determinant_tolerance: _ExactFloat1EMinus10
    projection_valid_min_depth_m: _ExactFloatZero
    center_roi_min_depth_m: _ExactFloatPointOne
    image_bounds: Literal["strict-open"]
    procedural_half_fov_bounds: Literal["inclusive-closed"]
    box_visibility: Literal["nuscenes-devkit-any"]
    box_visible_corner_min_depth_m: _ExactFloatOne
    box_all_corners_min_depth_m: _ExactFloatPointOne
    box_size_order: Literal["width-length-height"]
    roi_longitudinal_bounds: Literal["inclusive-closed"]
    roi_lateral_bounds: Literal["inclusive-closed"]


class SyntheticFixtureSpecV1(ContractModel):
    """Independent repository-owned fixture identity."""

    path: Literal["tests/fixtures/m2_geometry_reference_v1.json"]
    file_sha256: Literal["0676993f48e5a40034dfe497df7165b33f2d2f96dad234afd62af8e461beb252"]
    fixture_id: Literal["m2-nuscenes-convention-independent-v1"]
    official_devkit_revision: Literal["d9de17a73bdc06ce97a02f77ae7edb9b0406e851"]


class GeometryPropertyValidationSpecV1(ContractModel):
    """Frozen property-test stream and analytic-oracle tolerances."""

    rng_engine: Literal["numpy-pcg64dxsm-v1"]
    seed: Literal[81_985_529_216_486_895]
    transform_count: Literal[256]
    draw_order: PropertyDrawOrder
    rotation_identity_composition_max_abs_tolerance: _ExactFloat1EMinus12
    translation_inverse_composition_max_abs_tolerance_m: _ExactFloat1EMinus10
    point_round_trip_max_abs_tolerance_m: _ExactFloat1EMinus10
    quaternion_sign_rotation_max_abs_tolerance: _ExactFloat1EMinus12
    synthetic_projection_max_abs_tolerance_px: _ExactFloat1EMinus9
    synthetic_depth_max_abs_tolerance_m: _ExactFloat1EMinus12
    local_scalar_projection_max_abs_tolerance_px: _ExactFloat1EMinus9
    local_scalar_depth_max_abs_tolerance_m: _ExactFloat1EMinus10
    box_corner_max_abs_tolerance_m: _ExactFloat1EMinus12


class ActualSamplingCovarianceSpecV1(ContractModel):
    """Frozen actual bearing/depth sampling covariance."""

    role: Literal["actual"]
    matrix_rad2_rad_m_m2: tuple[
        tuple[_ExactFloat1EMinus8, _ExactFloat5EMinus6],
        tuple[_ExactFloat5EMinus6, _ExactFloatPointZeroFour],
    ]


class ReportedEstimatorCovarianceSpecV1(ContractModel):
    """Frozen reported estimator covariance, separate from actual sampling."""

    role: Literal["reported"]
    matrix_rad2_rad_m_m2: tuple[
        tuple[_ExactFloat2Point25EMinus8, _ExactFloatNegative3Point75EMinus6],
        tuple[_ExactFloatNegative3Point75EMinus6, _ExactFloatPointZeroSixTwoFive],
    ]


class CovarianceMonteCarloSpecV1(ContractModel):
    """Frozen nonlinear covariance Monte Carlo draw contract."""

    rng_engine: Literal["numpy-pcg64dxsm-v1"]
    seed: Literal[13_464_654_573_299_691_533]
    sample_count: Literal[200_000]
    mean_bearing_rad_depth_m: tuple[_ExactFloatPointTwo, _ExactFloatTwentyFive]
    standard_normal_draw: Literal["single-call-size-200000x2-float64"]
    covariance_factor: Literal["numpy-linalg-cholesky-lower"]
    sample_construction: Literal["mean-plus-lower-factor-times-column-standard-normal"]
    nonpositive_depth_action: Literal["fail-no-clip-condition-or-redraw"]
    sample_covariance_ddof: Literal[1]
    checked_symmetric_entries: tuple[
        Literal["xx"],
        Literal["xy"],
        Literal["yy"],
    ]
    standard_error_multiplier: _ExactFloatSix
    nonlinear_roundoff_allowance_m2: _ExactFloat1EMinus8


class CovarianceFiniteDifferenceSpecV1(ContractModel):
    """Frozen central finite-difference Jacobian check."""

    scheme: Literal["central"]
    bearing_step_rad: _ExactFloat1EMinus6
    depth_step_m: _ExactFloat1EMinus4
    max_abs_tolerance: _ExactFloat1EMinus7


class BearingDepthCovarianceValidationSpecV1(ContractModel):
    """Frozen bearing/depth covariance roles, propagation, and checks."""

    parameter_order: tuple[Literal["bearing_rad"], Literal["optical_depth_m"]]
    camera_vertical_coordinate_m: _ExactFloatPointFive
    translation_treatment: Literal["constant-no-covariance-contribution"]
    output_frame: Literal["reference-lidar-keyframe-ego-bev-xy"]
    actual_sampling_covariance: ActualSamplingCovarianceSpecV1
    reported_estimator_covariance: ReportedEstimatorCovarianceSpecV1
    monte_carlo: CovarianceMonteCarloSpecV1
    finite_difference: CovarianceFiniteDifferenceSpecV1
    full_covariance_action: Literal["retain-symmetric-2x2-no-diagonalization"]
    m1_fusion_integration: Literal["not-part-of-m2"]


class GeometryArtifactSpecV1(ContractModel):
    """Frozen five-file layout and dataset-path-free logical command."""

    artifact_contract: Literal["ffb.geometry-validation-payload/v1"]
    payload_index_schema: Literal["ffb.geometry-payload-index/v1"]
    success_schema: Literal["ffb.success/v1alpha1"]
    ordered_files: GeometryArtifactFiles
    indexed_payload_files: GeometryIndexedFiles
    canonical_file_format: Literal["key-sorted-compact-json-plus-lf"]
    member_byte_cap: Literal[1_048_576]
    tree_byte_cap: Literal[5_242_880]
    logical_command: GeometryLogicalCommand


class PublicSummaryAllowlistV1(ContractModel):
    """Exact, ordered public result and run-record field allowlists."""

    geometry_validation_fields: tuple[
        Literal["schema"],
        Literal["run_id"],
        Literal["manifest_sha256"],
        Literal["dataset_terms"],
        Literal["dataset_validation"],
        Literal["synthetic_geometry_validation"],
        Literal["covariance_validation"],
        Literal["all_checks_passed"],
    ]
    dataset_terms_fields: tuple[
        Literal["source"],
        Literal["license"],
        Literal["terms_url"],
        Literal["attribution"],
        Literal["non_endorsement"],
    ]
    dataset_validation_fields: tuple[
        Literal["profile"],
        Literal["expected_headline_counts"],
        Literal["headline_profile_passed_attested"],
        Literal["structural_integrity_passed_attested"],
        Literal["keyframe_blob_check_count"],
        Literal["keyframe_blob_validation_passed_attested"],
        Literal["local_projection_crosscheck_passed_attested"],
        Literal["diagnostic_svg_generated_attested"],
        Literal["dataset_authentication"],
        Literal["all_checks_passed"],
    ]
    synthetic_geometry_validation_fields: tuple[
        Literal["fixture_id"],
        Literal["fixture_file_sha256"],
        Literal["rotation_max_abs_error"],
        Literal["translation_max_abs_error_m"],
        Literal["point_round_trip_max_abs_error_m"],
        Literal["quaternion_sign_max_abs_error"],
        Literal["projection_max_abs_error_px"],
        Literal["depth_max_abs_error_m"],
        Literal["box_corner_max_abs_error_m"],
        Literal["all_checks_passed"],
    ]
    covariance_validation_fields: tuple[
        Literal["finite_difference_max_abs_error"],
        Literal["monte_carlo_sample_count"],
        Literal["covariance_entries"],
        Literal["covariance_entry_max_abs_error_m2"],
        Literal["covariance_entry_max_allowed_error_m2"],
        Literal["covariance_entry_max_gate_ratio"],
        Literal["actual_sampling_gate_passed"],
        Literal["reported_role_separation_passed_attested"],
        Literal["all_checks_passed"],
    ]
    covariance_entry_fields: tuple[
        Literal["entry"],
        Literal["absolute_error_m2"],
        Literal["allowed_error_m2"],
        Literal["gate_ratio"],
        Literal["passed"],
    ]
    run_fields: tuple[
        Literal["schema"],
        Literal["run_id"],
        Literal["manifest_sha256"],
        Literal["package_version"],
        Literal["git_revision"],
        Literal["source_dirty"],
        Literal["lockfile_sha256"],
        Literal["command"],
        Literal["environment"],
        Literal["started_at"],
        Literal["ended_at"],
        Literal["status"],
        Literal["artifact_sha256"],
    ]


class GeometryValidationManifestV1(ContractModel):
    """Complete frozen M2 geometry-validation preregistration."""

    schema_id: Literal["ffb.geometry-validation-manifest/v1"] = Field(alias="schema")
    validation_id: Literal["m2-geometry-and-nuscenes-mini-grounding-v1"]
    dataset: GeometryDatasetSpecV1
    public_dataset_terms: DatasetTermsV1
    geometry: GeometryConventionSpecV1
    synthetic_fixture: SyntheticFixtureSpecV1
    property_validation: GeometryPropertyValidationSpecV1
    bearing_depth_covariance_validation: BearingDepthCovarianceValidationSpecV1
    artifact: GeometryArtifactSpecV1
    public_summary_allowlist: PublicSummaryAllowlistV1
    prohibited_public_dataset_fields: tuple[
        Literal["observed-nonheadline-table-counts"],
        Literal["archive-or-table-sha256"],
        Literal["dataset-root"],
        Literal["scene-sample-annotation-or-instance-identifiers"],
        Literal["timestamps"],
        Literal["filenames"],
        Literal["calibration-pose-or-box-values"],
        Literal["selected-sample-counts"],
        Literal["local-projection-residuals-pixels-or-depths"],
        Literal["diagnostic-svg"],
    ]


class DatasetValidationV1(ContractModel):
    """Sanitized local-data attestations and their recomputed conjunction."""

    profile: Literal["official-nuscenes-v1.0-mini"]
    expected_headline_counts: GeometryHeadlineCountsV1
    headline_profile_passed_attested: bool
    structural_integrity_passed_attested: bool
    keyframe_blob_check_count: Literal[808]
    keyframe_blob_validation_passed_attested: bool
    local_projection_crosscheck_passed_attested: bool
    diagnostic_svg_generated_attested: bool
    dataset_authentication: Literal["summary-does-not-authenticate-dataset-bytes"]
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_attestation_conjunction(self) -> Self:
        expected = (
            self.headline_profile_passed_attested
            and self.structural_integrity_passed_attested
            and self.keyframe_blob_check_count == EXPECTED_KEYFRAME_BLOB_CHECK_COUNT
            and self.keyframe_blob_validation_passed_attested
            and self.local_projection_crosscheck_passed_attested
            and self.diagnostic_svg_generated_attested
        )
        if self.all_checks_passed != expected:
            raise ValueError(
                "dataset all_checks_passed must be the conjunction of the five "
                "attestations and exact key-frame count"
            )
        return self


class SyntheticGeometryValidationV1(ContractModel):
    """Repository-owned synthetic errors and their tolerance decision."""

    fixture_id: Literal["m2-nuscenes-convention-independent-v1"]
    fixture_file_sha256: Literal["0676993f48e5a40034dfe497df7165b33f2d2f96dad234afd62af8e461beb252"]
    rotation_max_abs_error: NonNegativeFiniteFloat
    translation_max_abs_error_m: NonNegativeFiniteFloat
    point_round_trip_max_abs_error_m: NonNegativeFiniteFloat
    quaternion_sign_max_abs_error: NonNegativeFiniteFloat
    projection_max_abs_error_px: NonNegativeFiniteFloat
    depth_max_abs_error_m: NonNegativeFiniteFloat
    box_corner_max_abs_error_m: NonNegativeFiniteFloat
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_tolerance_conjunction(self) -> Self:
        expected = (
            self.fixture_id == SYNTHETIC_FIXTURE_ID
            and self.fixture_file_sha256 == SYNTHETIC_FIXTURE_SHA256
            and self.rotation_max_abs_error <= ROTATION_MAX_ABS_TOLERANCE
            and self.translation_max_abs_error_m <= TRANSLATION_MAX_ABS_TOLERANCE_M
            and self.point_round_trip_max_abs_error_m <= POINT_ROUND_TRIP_MAX_ABS_TOLERANCE_M
            and self.quaternion_sign_max_abs_error <= QUATERNION_SIGN_MAX_ABS_TOLERANCE
            and self.projection_max_abs_error_px <= SYNTHETIC_PROJECTION_MAX_ABS_TOLERANCE_PX
            and self.depth_max_abs_error_m <= SYNTHETIC_DEPTH_MAX_ABS_TOLERANCE_M
            and self.box_corner_max_abs_error_m <= BOX_CORNER_MAX_ABS_TOLERANCE_M
        )
        if self.all_checks_passed != expected:
            raise ValueError(
                "synthetic all_checks_passed must match fixture identity and "
                "the frozen error tolerances"
            )
        return self


class CovarianceEntryV1(ContractModel):
    """One ordered nonlinear covariance-entry sampling gate."""

    entry: CovarianceEntryName
    absolute_error_m2: NonNegativeFiniteFloat
    allowed_error_m2: PositiveFiniteFloat
    gate_ratio: NonNegativeFiniteFloat
    passed: bool

    @model_validator(mode="after")
    def require_derived_ratio_and_flag(self) -> Self:
        expected_ratio = self.absolute_error_m2 / self.allowed_error_m2
        if not math.isclose(
            self.gate_ratio,
            expected_ratio,
            rel_tol=0.0,
            abs_tol=COVARIANCE_DERIVATION_ABS_TOLERANCE,
        ):
            raise ValueError(
                "covariance gate_ratio must equal absolute_error_m2 / allowed_error_m2"
            )
        if self.passed != (self.gate_ratio <= 1.0):
            raise ValueError("covariance passed must equal gate_ratio <= 1")
        return self


class CovarianceValidationV1(ContractModel):
    """Sanitized Jacobian and covariance evidence with derived decisions."""

    finite_difference_max_abs_error: NonNegativeFiniteFloat
    monte_carlo_sample_count: Literal[200_000]
    covariance_entries: tuple[
        CovarianceEntryV1,
        CovarianceEntryV1,
        CovarianceEntryV1,
    ]
    covariance_entry_max_abs_error_m2: NonNegativeFiniteFloat
    covariance_entry_max_allowed_error_m2: PositiveFiniteFloat
    covariance_entry_max_gate_ratio: NonNegativeFiniteFloat
    actual_sampling_gate_passed: bool
    reported_role_separation_passed_attested: bool
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_order_maxima_and_conjunctions(self) -> Self:
        if tuple(record.entry for record in self.covariance_entries) != (
            "xx",
            "xy",
            "yy",
        ):
            raise ValueError("covariance entries must use the exact xx, xy, yy order")

        expected_maxima = (
            max(record.absolute_error_m2 for record in self.covariance_entries),
            max(record.allowed_error_m2 for record in self.covariance_entries),
            max(record.gate_ratio for record in self.covariance_entries),
        )
        observed_maxima = (
            self.covariance_entry_max_abs_error_m2,
            self.covariance_entry_max_allowed_error_m2,
            self.covariance_entry_max_gate_ratio,
        )
        if any(
            not math.isclose(
                observed,
                expected,
                rel_tol=0.0,
                abs_tol=COVARIANCE_DERIVATION_ABS_TOLERANCE,
            )
            for observed, expected in zip(observed_maxima, expected_maxima, strict=True)
        ):
            raise ValueError("covariance maximum fields must equal the corresponding entry maxima")

        expected_sampling_pass = all(record.passed for record in self.covariance_entries)
        if self.actual_sampling_gate_passed != expected_sampling_pass:
            raise ValueError("actual_sampling_gate_passed must be the covariance-entry conjunction")

        expected_all = (
            self.finite_difference_max_abs_error <= FINITE_DIFFERENCE_MAX_ABS_TOLERANCE
            and self.monte_carlo_sample_count == EXPECTED_MONTE_CARLO_SAMPLE_COUNT
            and self.actual_sampling_gate_passed
            and self.reported_role_separation_passed_attested
        )
        if self.all_checks_passed != expected_all:
            raise ValueError(
                "covariance all_checks_passed must be the finite-difference, sample "
                "count, sampling-gate, and role-attestation conjunction"
            )
        return self


class GeometryValidationV1(ContractModel):
    """Complete sanitized M2 public validation result."""

    schema_id: Literal["ffb.geometry-validation/v1"] = Field(alias="schema")
    run_id: Identifier
    manifest_sha256: Digest
    dataset_terms: DatasetTermsV1
    dataset_validation: DatasetValidationV1
    synthetic_geometry_validation: SyntheticGeometryValidationV1
    covariance_validation: CovarianceValidationV1
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_component_conjunction(self) -> Self:
        expected = (
            self.dataset_validation.all_checks_passed
            and self.synthetic_geometry_validation.all_checks_passed
            and self.covariance_validation.all_checks_passed
        )
        if self.all_checks_passed != expected:
            raise ValueError("top-level all_checks_passed must be the component conjunction")
        return self


class PayloadFileEntryV1(ContractModel):
    """One exact raw payload member committed by the M2 index."""

    path: GeometryIndexedPayloadPath
    byte_length: Annotated[int, Field(ge=1, le=GEOMETRY_MEMBER_BYTE_CAP)]
    sha256: Digest


class GeometryPayloadIndexV1(ContractModel):
    """Deterministic envelope over the two M2 payload members."""

    schema_id: Literal["ffb.geometry-payload-index/v1"] = Field(alias="schema")
    artifact_contract: Literal["ffb.geometry-validation-payload/v1"]
    run_id: Identifier
    manifest_sha256: Digest
    files: Annotated[
        tuple[PayloadFileEntryV1, ...],
        Field(min_length=2, max_length=2),
    ]

    @model_validator(mode="after")
    def require_exact_file_order(self) -> Self:
        if tuple(entry.path for entry in self.files) != GEOMETRY_INDEXED_PAYLOAD_PATHS:
            raise ValueError("payload index files must use the fixed two-member order")
        return self


def validate_geometry_result_against_manifest(
    manifest: GeometryValidationManifestV1,
    result: GeometryValidationV1,
) -> None:
    """Cross-check sanitized evidence against its frozen intent and release gate."""

    if result.manifest_sha256 != GEOMETRY_MANIFEST_SHA256:
        raise ValueError("geometry result does not identify the frozen manifest")
    if result.dataset_terms != manifest.public_dataset_terms:
        raise ValueError("geometry result dataset terms disagree with the manifest")

    dataset = result.dataset_validation
    if dataset.profile != manifest.dataset.profile:
        raise ValueError("geometry result dataset profile disagrees with the manifest")
    if dataset.expected_headline_counts != manifest.dataset.expected_headline_counts:
        raise ValueError("geometry result headline counts disagree with the manifest")
    if dataset.keyframe_blob_check_count != manifest.dataset.expected_keyframe_blob_check_count:
        raise ValueError("geometry result key-frame check count disagrees with manifest")
    if dataset.dataset_authentication != manifest.dataset.dataset_authentication:
        raise ValueError("geometry result authentication notice disagrees with manifest")

    synthetic = result.synthetic_geometry_validation
    if synthetic.fixture_id != manifest.synthetic_fixture.fixture_id:
        raise ValueError("geometry result fixture ID disagrees with the manifest")
    if synthetic.fixture_file_sha256 != manifest.synthetic_fixture.file_sha256:
        raise ValueError("geometry result fixture digest disagrees with the manifest")

    synthetic_expected = (
        synthetic.rotation_max_abs_error
        <= manifest.property_validation.rotation_identity_composition_max_abs_tolerance
        and synthetic.translation_max_abs_error_m
        <= manifest.property_validation.translation_inverse_composition_max_abs_tolerance_m
        and synthetic.point_round_trip_max_abs_error_m
        <= manifest.property_validation.point_round_trip_max_abs_tolerance_m
        and synthetic.quaternion_sign_max_abs_error
        <= manifest.property_validation.quaternion_sign_rotation_max_abs_tolerance
        and synthetic.projection_max_abs_error_px
        <= manifest.property_validation.synthetic_projection_max_abs_tolerance_px
        and synthetic.depth_max_abs_error_m
        <= manifest.property_validation.synthetic_depth_max_abs_tolerance_m
        and synthetic.box_corner_max_abs_error_m
        <= manifest.property_validation.box_corner_max_abs_tolerance_m
    )
    if synthetic.all_checks_passed != synthetic_expected:
        raise ValueError("geometry result synthetic decision disagrees with the manifest")

    covariance = result.covariance_validation
    covariance_spec = manifest.bearing_depth_covariance_validation
    covariance_expected = (
        covariance.finite_difference_max_abs_error
        <= covariance_spec.finite_difference.max_abs_tolerance
        and covariance.monte_carlo_sample_count == covariance_spec.monte_carlo.sample_count
        and covariance.actual_sampling_gate_passed
        and covariance.reported_role_separation_passed_attested
    )
    if covariance.all_checks_passed != covariance_expected:
        raise ValueError("geometry result covariance decision disagrees with manifest")
    if not result.all_checks_passed:
        raise ValueError("public geometry-validation release requires all checks to pass")


GEOMETRY_VALIDATION_MANIFEST_ADAPTER = TypeAdapter(GeometryValidationManifestV1)
GEOMETRY_VALIDATION_ADAPTER = TypeAdapter(GeometryValidationV1)
GEOMETRY_PAYLOAD_INDEX_ADAPTER = TypeAdapter(GeometryPayloadIndexV1)
