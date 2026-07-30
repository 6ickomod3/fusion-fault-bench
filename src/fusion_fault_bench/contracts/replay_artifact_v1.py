"""Strict aggregate-only contracts for the frozen M5 curated replay artifact."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, field_validator, model_validator

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_HEALTH_PANEL_ID,
    M5_PERSISTENT_EXPERIMENT_IDS,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_PERSISTENT_PANEL_ID,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
    ReplayExperimentIdentityV1,
    expected_replay_identities,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, Identifier
from fusion_fault_bench.inference import bootstrap_crossover_status

REPLAY_CURATED_ARTIFACT_CONTRACT = "ffb.replay-curated-payload/v1"
M5_REPLAY_RELEASE_ID = "m5-nuscenes-replay-v0.1.0"
M5_HEALTH_FIT_RUN_SHA256 = "0311aec90df031bd0d1720d5fa15aae91e2e5c3dfea923dacc2eefe518134fcd"
M5_TRACKED_AGGREGATE_TERMS = "CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"
M5_SCIENTIFIC_SOURCE_ROLES = (
    "descriptor-aggregates",
    "health-population-metrics",
    "health-sequence-contrasts",
    "health-sequence-events",
    "health-sequence-results",
    "persistent-crossovers",
    "persistent-population-metrics",
    "persistent-scene-evaluations",
)

REPLAY_INTENT_FILE = "intent.json"
REPLAY_PROFILE_SUMMARY_FILE = "replay-profile-summary.json"
REPLAY_DESCRIPTOR_AGGREGATES_FILE = "descriptor-aggregates.ndjson"
REPLAY_PERSISTENT_AGGREGATES_FILE = "persistent-panel-aggregates.ndjson"
REPLAY_PERSISTENT_CROSSOVERS_FILE = "persistent-panel-crossovers.ndjson"
REPLAY_HEALTH_AGGREGATES_FILE = "health-panel-aggregates.ndjson"
REPLAY_CLUSTER_SENSITIVITY_FILE = "leave-one-cluster-sensitivity.ndjson"
REPLAY_VALIDATION_FILE = "validation.json"
REPLAY_REPEAT_VERIFICATION_FILE = "repeat-verification.json"
REPLAY_FIGURE_RECORDS_FILE = "figure-records.ndjson"
REPLAY_SOURCE_COMMITMENTS_FILE = "source-member-commitments.ndjson"
REPLAY_RELEASE_INDEX_FILE = "release-index.json"
REPLAY_RUN_FILE = "run.json"
REPLAY_SUCCESS_FILE = "_SUCCESS"

REPLAY_INDEXED_PATHS = (
    REPLAY_INTENT_FILE,
    REPLAY_PROFILE_SUMMARY_FILE,
    REPLAY_DESCRIPTOR_AGGREGATES_FILE,
    REPLAY_PERSISTENT_AGGREGATES_FILE,
    REPLAY_PERSISTENT_CROSSOVERS_FILE,
    REPLAY_HEALTH_AGGREGATES_FILE,
    REPLAY_CLUSTER_SENSITIVITY_FILE,
    REPLAY_VALIDATION_FILE,
    REPLAY_REPEAT_VERIFICATION_FILE,
    REPLAY_FIGURE_RECORDS_FILE,
    REPLAY_SOURCE_COMMITMENTS_FILE,
)
REPLAY_ARTIFACT_PATHS = (
    *REPLAY_INDEXED_PATHS,
    REPLAY_RELEASE_INDEX_FILE,
    REPLAY_RUN_FILE,
    REPLAY_SUCCESS_FILE,
)

REPLAY_MAX_RECORD_BYTES = 1024 * 1024
REPLAY_MAX_MEMBER_BYTES = 32 * 1024 * 1024
REPLAY_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
REPLAY_MAX_NDJSON_RECORDS = 500_000

M5_RELEASE_VALIDATION_CHECK_IDS = (
    "intent-freeze",
    "fixed-scene-population",
    "base-support",
    "health-schedules",
    "transform-and-timing-oracles",
    "eligibility-and-fault-causality",
    "health-feature-leakage",
    "persistent-panel-completeness",
    "health-panel-completeness",
    "scene-bootstrap-and-cluster-sensitivity",
    "repeat-scientific-members",
    "cpu-and-memory-caps",
    "no-raw-payload-reads",
    "privacy-and-dataset-license",
    "implementation-review",
    "results-and-claims-review",
    "software-verification",
)

type ReplayIndexedPath = Literal[
    "intent.json",
    "replay-profile-summary.json",
    "descriptor-aggregates.ndjson",
    "persistent-panel-aggregates.ndjson",
    "persistent-panel-crossovers.ndjson",
    "health-panel-aggregates.ndjson",
    "leave-one-cluster-sensitivity.ndjson",
    "validation.json",
    "repeat-verification.json",
    "figure-records.ndjson",
    "source-member-commitments.ndjson",
]
type ReplayResultStatus = Literal["ok", "undefined", "not-applicable"]
type ReplayResultUnit = Literal[
    "m^2",
    "fraction",
    "count",
    "frames",
    "observation-step",
    "s",
    "m",
    "rad",
    "m/s",
    "m/s^2",
    "unitless",
]
type ReplayWindow = Literal["full", "score", "event", "recovery"]
type ReplayExpectedDirection = Literal["positive", "negative", "nonpositive", "none"]
type ReplayPersistenceLabel = Literal[
    "robustly-persistent",
    "directionally-consistent",
    "non-persistent",
    "undefined",
    "not-applicable",
]
type ReplayInferenceRole = Literal[
    "primary-directional",
    "nonpositive-control",
    "diagnostic",
    "descriptive",
]
type ReplayHypothesisId = Literal[
    "h5-a1",
    "h5-a2",
    "h5-a3",
    "h5-a4",
    "h5-a5",
    "h5-a6",
    "h5-b1",
    "h5-b2",
    "h5-b3",
    "h5-b4",
]
type ReplayFigureId = Literal[
    "m5-persistent-panel-summary",
    "m5-crossovers",
    "m5-health-transfer",
    "m5-descriptor-comparison",
    "m5-cluster-sensitivity",
]
type ReplayFigureKind = Literal[
    "panel-summary",
    "crossover",
    "health-transfer",
    "descriptor-comparison",
    "cluster-sensitivity",
]
type ReplayFigureSourceKind = Literal[
    "descriptor-aggregate",
    "persistent-aggregate",
    "persistent-crossover",
    "health-aggregate",
    "cluster-sensitivity",
]
type ReplayFigureSvgPath = Literal[
    "figures/m5-persistent-panel-summary.svg",
    "figures/m5-crossovers.svg",
    "figures/m5-health-transfer.svg",
    "figures/m5-descriptor-comparison.svg",
    "figures/m5-cluster-sensitivity.svg",
]

M5_REPLAY_FIGURE_DEFINITIONS: tuple[
    tuple[ReplayFigureId, ReplayFigureKind, ReplayFigureSvgPath],
    ...,
] = (
    (
        "m5-persistent-panel-summary",
        "panel-summary",
        "figures/m5-persistent-panel-summary.svg",
    ),
    ("m5-crossovers", "crossover", "figures/m5-crossovers.svg"),
    ("m5-health-transfer", "health-transfer", "figures/m5-health-transfer.svg"),
    (
        "m5-descriptor-comparison",
        "descriptor-comparison",
        "figures/m5-descriptor-comparison.svg",
    ),
    (
        "m5-cluster-sensitivity",
        "cluster-sensitivity",
        "figures/m5-cluster-sensitivity.svg",
    ),
)
M5_REPLAY_FIGURE_IDS = tuple(definition[0] for definition in M5_REPLAY_FIGURE_DEFINITIONS)
_M5_REPLAY_FIGURE_DEFINITION_BY_ID = {
    definition[0]: definition for definition in M5_REPLAY_FIGURE_DEFINITIONS
}
_M5_REPLAY_FIGURE_SOURCE_KINDS: dict[
    ReplayFigureId,
    frozenset[ReplayFigureSourceKind],
] = {
    "m5-persistent-panel-summary": frozenset({"persistent-aggregate"}),
    "m5-crossovers": frozenset({"persistent-crossover"}),
    "m5-health-transfer": frozenset({"health-aggregate"}),
    "m5-descriptor-comparison": frozenset({"descriptor-aggregate"}),
    "m5-cluster-sensitivity": frozenset(
        {
            "persistent-aggregate",
            "health-aggregate",
            "cluster-sensitivity",
        }
    ),
}
ReplayConditionSelector = Annotated[
    str,
    Field(
        pattern=(
            r"^replay-[a-z0-9][a-z0-9-]*:"
            r"(?:0|[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))$"
        )
    ),
]

M5_PERSISTENT_CONDITION_SELECTOR_COUNT = 71
M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256 = (
    "9bc53e38bb910e50e4111c7a72103edf980568d52739f64c58e13a5fcd735cac"
)
M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT = 464
M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256 = (
    "b36286e8c4a9fa5f1980e297b7f42f3f7fb26465aa21df67fd70e8fd0f83e09a"
)
M5_PERSISTENT_HYPOTHESIS_COORDINATES = (
    *(
        (
            hypothesis_id,
            selector,
            "fixed-fusion",
            "fused-minus-healthy",
            "full",
            "m^2",
            "equal-scene-mean",
            "primary-directional",
            direction,
        )
        for hypothesis_id, selector, direction in (
            ("h5-a1", "replay-lidar-y-bias:0", "negative"),
            ("h5-a1", "replay-camera-noise-correctly-reported:1", "negative"),
            ("h5-a1", "replay-camera-noise-underreported:1", "negative"),
            ("h5-a1", "replay-camera-calibration-x:0", "negative"),
            ("h5-a1", "replay-camera-calibration-yaw:0", "negative"),
            ("h5-a1", "replay-camera-timestamp-offset:0", "negative"),
            ("h5-a2", "replay-lidar-y-bias:-4", "positive"),
            ("h5-a2", "replay-lidar-y-bias:+4", "positive"),
            ("h5-a2", "replay-camera-calibration-x:-4", "positive"),
            ("h5-a2", "replay-camera-calibration-x:+4", "positive"),
            ("h5-a2", "replay-camera-calibration-yaw:-0.08", "positive"),
            ("h5-a2", "replay-camera-calibration-yaw:+0.08", "positive"),
            ("h5-a2", "replay-camera-timestamp-offset:-0.8", "positive"),
            ("h5-a2", "replay-camera-timestamp-offset:+0.8", "positive"),
            ("h5-a3", "replay-camera-noise-underreported:4", "positive"),
            ("h5-a4", "replay-camera-noise-correctly-reported:4", "negative"),
        )
    ),
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "fixed-fusion",
        "coverage",
        "full",
        "fraction",
        "pooled-valid-eligible-count-ratio",
        "diagnostic",
        "none",
    ),
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "fixed-fusion",
        "conditional-matched-center-mse",
        "full",
        "m^2",
        "pooled-valid-loss",
        "diagnostic",
        "none",
    ),
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "fault-target-drop-policy",
        "coverage",
        "full",
        "fraction",
        "pooled-valid-eligible-count-ratio",
        "diagnostic",
        "none",
    ),
    (
        "h5-a5",
        "replay-camera-dropout:1",
        "lidar-only",
        "coverage",
        "full",
        "fraction",
        "pooled-valid-eligible-count-ratio",
        "diagnostic",
        "none",
    ),
    *(
        (
            "h5-a6",
            selector,
            "fixed-fusion",
            "matched-center-mse",
            "full",
            "m^2",
            "equal-scene-mean",
            "diagnostic",
            "none",
        )
        for selector in ("replay-common-mode-x:-4", "replay-common-mode-x:+4")
    ),
    *(
        (
            "h5-a6",
            selector,
            "camera-lidar-pair",
            "camera-lidar-disagreement-mse",
            "full",
            "m^2",
            "equal-scene-mean",
            "diagnostic",
            "none",
        )
        for selector in (
            "replay-common-mode-x:0",
            "replay-common-mode-x:-0.25",
            "replay-common-mode-x:+0.25",
            "replay-common-mode-x:-0.5",
            "replay-common-mode-x:+0.5",
            "replay-common-mode-x:-1",
            "replay-common-mode-x:+1",
            "replay-common-mode-x:-2",
            "replay-common-mode-x:+2",
            "replay-common-mode-x:-4",
            "replay-common-mode-x:+4",
        )
    ),
)
M5_PERSISTENT_HYPOTHESIS_COORDINATE_COUNT = 33
M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256 = (
    "22c9e55602c9a8faefa6fea7e0f4c1fe8185f8b4160729072b091801ede36ab3"
)
M5_HEALTH_CONDITION_SELECTOR_COUNT = 43
M5_HEALTH_CONDITION_SELECTOR_SET_SHA256 = (
    "4b09fbbcff924ab94a17c94984e67bd06d6a0e56df43a95094dec64d60d3376c"
)
M5_HEALTH_AGGREGATE_COORDINATE_COUNT = 14_988
M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256 = (
    "33f1327d12a938f64857b4a85de6ed23e9e0f7e1c064f3e80ae4a0acf50bed2c"
)
M5_HEALTH_HYPOTHESIS_COORDINATES = (
    (
        "h5-b1",
        "replay-lidar-output-y-bias:+3",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "positive",
    ),
    (
        "h5-b1",
        "replay-lidar-timestamp-offset:+0.6",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "positive",
    ),
    (
        "h5-b1",
        "replay-camera-noise-underreported:3",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "positive",
    ),
    (
        "h5-b1",
        "replay-camera-timestamp-offset:+0.6",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "positive",
    ),
    (
        "h5-b1",
        "replay-camera-calibration-x:+3",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "positive",
    ),
    (
        "h5-b1",
        "replay-camera-output-y-bias:+3",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "positive",
    ),
    (
        "h5-b1",
        "replay-camera-calibration-yaw:+0.06",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "positive",
    ),
    (
        "h5-b2",
        "replay-lidar-noise-underreported:3",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "primary-directional",
        "negative",
    ),
    (
        "h5-b3",
        "replay-camera-noise-correctly-reported:3",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "nonpositive-control",
        "nonpositive",
    ),
    (
        "h5-b3",
        "replay-lidar-noise-correctly-reported:3",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "nonpositive-control",
        "nonpositive",
    ),
    (
        "h5-b4",
        "replay-common-mode-x:+4",
        "combined-health-gate",
        "policy-gain-vs-fixed",
        "event",
        "m^2",
        "diagnostic",
        "none",
    ),
)
M5_HEALTH_HYPOTHESIS_COORDINATE_COUNT = 11
M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256 = (
    "6d73453c8cef65e90bc6d4cb1fe972bce976a43a924046dcaafdbdc57b7f7cb8"
)

_PRIVATE_LABEL_MARKERS = (
    "nuscenes:scene-",
    "sample_token",
    "annotation_token",
    "instance_token",
    "calibrated_sensor_token",
    "ego_pose_token",
    "log_token",
    "filename",
    "filepath",
)


def _require_public_label(value: str) -> str:
    lowered = value.casefold()
    if "/" in value or "\\" in value or any(marker in lowered for marker in _PRIVATE_LABEL_MARKERS):
        raise ValueError("public replay label contains private source identity")
    return value


class ReplayIdentitySetV1(ContractModel):
    """The exact 22 replay identities in frozen panel and execution order."""

    schema_id: Literal["ffb.replay-identity-set/v1"] = Field(alias="schema")
    replay_intent_sha256: Digest
    identities: Annotated[
        tuple[ReplayExperimentIdentityV1, ...],
        Field(min_length=22, max_length=22),
    ]

    @model_validator(mode="after")
    def require_frozen_identity_order(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.identities != expected_replay_identities()
        ):
            raise ValueError("replay identity set does not match the frozen M5 execution order")
        return self


M5_REPLAY_IDENTITY_SET = ReplayIdentitySetV1(
    schema="ffb.replay-identity-set/v1",
    replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
    identities=expected_replay_identities(),
)
M5_REPLAY_IDENTITY_SET_SHA256 = sha256_digest(M5_REPLAY_IDENTITY_SET)


class _ReplayGlobalBinding(ContractModel):
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest

    @model_validator(mode="after")
    def require_global_binding(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
        ):
            raise ValueError("record does not bind the frozen M5 identity set")
        return self


class _ReplayIdentityBinding(_ReplayGlobalBinding):
    identity: ReplayExperimentIdentityV1
    replay_identity_sha256: Digest

    @model_validator(mode="after")
    def require_identity_digest(self) -> Self:
        if self.replay_identity_sha256 != replay_experiment_identity_sha256(self.identity):
            raise ValueError("record replay identity digest is invalid")
        return self


class ReplayExecutionResourceEvidenceV1(_ReplayGlobalBinding):
    """One external, raw-log-committed complete M5 replay measurement."""

    schema_id: Literal["ffb.replay-execution-resource-evidence/v1"] = Field(alias="schema")
    run_label: Literal["primary", "repeat"]
    local_artifact_sha256: Digest
    local_run_sha256: Digest
    environment_sha256: Digest
    logical_command_sha256: Digest
    persisted_internal_elapsed_seconds: Annotated[
        FiniteFloat,
        Field(gt=0.0, lt=1800.0),
    ]
    persisted_internal_peak_rss_bytes: Annotated[
        int,
        Field(gt=0, lt=1024 * 1024 * 1024),
    ]
    persisted_internal_measurement_scope: Literal[
        "metadata-through-canonical-scientific-members-before-publication"
    ]
    tool_path: Literal["/usr/bin/time"]
    tool_options: tuple[Literal["-l"]]
    parser_contract: Literal["ffb.darwin-time-l-strict/v1"]
    raw_log_sha256: Digest
    raw_log_byte_length: Annotated[int, Field(ge=1, le=65_536)]
    elapsed_seconds: Annotated[FiniteFloat, Field(gt=0.0, lt=1800.0)]
    peak_rss_bytes: Annotated[int, Field(gt=0, lt=1024 * 1024 * 1024)]
    exit_status: Literal[0]
    scientific_replay_worker_count: Literal[1]
    cpu_process_scope: Literal["one-scientific-replay-worker-no-benchmark-multiprocessing"]
    helper_process_policy: Literal["sequential-provenance-and-resource-measurement-helpers-only"]
    accelerator_requested: Literal[False]
    wall_time_cap_seconds: FiniteFloat
    peak_rss_cap_bytes: Literal[1_073_741_824]
    wall_time_within_cap: Literal[True]
    peak_rss_within_cap: Literal[True]
    measurement_scope: Literal[
        "operator-recorded-darwin-time-l-for-complete-replay-cli-lifetime;"
        "self-reported-not-independent-attestation"
    ]

    @model_validator(mode="after")
    def require_external_measurement_dominance(self) -> Self:
        if self.wall_time_cap_seconds != 1800.0:
            raise ValueError("wall-time cap is not the frozen M5 value")
        if self.elapsed_seconds < self.persisted_internal_elapsed_seconds:
            raise ValueError("external elapsed time does not cover the persisted diagnostic")
        if self.peak_rss_bytes < self.persisted_internal_peak_rss_bytes:
            raise ValueError("external peak RSS does not cover the persisted diagnostic")
        if self.wall_time_within_cap != (
            self.elapsed_seconds < self.wall_time_cap_seconds
        ) or self.peak_rss_within_cap != (self.peak_rss_bytes < self.peak_rss_cap_bytes):
            raise ValueError("external replay resource cap evidence is contradictory")
        return self


class ReplayProfileSummaryV1(_ReplayGlobalBinding):
    """Privacy-bounded public summary of the exact replay source profile."""

    schema_id: Literal["ffb.replay-profile-summary/v1"] = Field(alias="schema")
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    replay_intent_byte_sha256: Digest
    persistent_source_matrix_sha256: Digest
    health_fit_artifact_sha256: Digest
    health_fit_run_sha256: Digest
    dataset_profile: Literal["official-nuscenes-v1.0-mini"]
    adapter_profile: Literal["nuscenes-mini-matched-centers-v1"]
    scene_count: Literal[10]
    persistent_experiment_count: Literal[8]
    health_experiment_count: Literal[14]
    replay_experiment_count: Literal[22]
    distinct_log_group_count: Annotated[int, Field(ge=1, le=10)]
    all_scenes_have_base_support: Literal[True]
    all_health_schedules_valid: Literal[True]
    raw_sensor_payload_reads: Literal[0]
    scientific_replay_worker_count: Literal[1]
    gpu_used: Literal[False]
    torch_imported: Literal[False]
    cuda_used: Literal[False]
    resource_evidence: Annotated[
        tuple[ReplayExecutionResourceEvidenceV1, ...],
        Field(min_length=2, max_length=2),
    ]
    peak_rss_bytes: Annotated[int, Field(ge=1, lt=1024 * 1024 * 1024)]
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0.0, lt=1800.0)]
    dataset_root_serialized: Literal[False]
    dataset_bytes_authenticated: Literal[False]
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]
    attribution_and_non_endorsement_required: Literal[True]

    @model_validator(mode="after")
    def require_frozen_sources(self) -> Self:
        if (
            self.replay_intent_byte_sha256 != M5_REPLAY_INTENT_BYTE_SHA256
            or self.persistent_source_matrix_sha256 != M5_PERSISTENT_MATRIX_SHA256
            or self.health_fit_artifact_sha256 != M5_HEALTH_FIT_SHA256
            or self.health_fit_run_sha256 != M5_HEALTH_FIT_RUN_SHA256
        ):
            raise ValueError("replay profile summary does not bind frozen upstream sources")
        if tuple(record.run_label for record in self.resource_evidence) != (
            "primary",
            "repeat",
        ):
            raise ValueError("replay profile resource evidence is not in primary/repeat order")
        if any(
            record.run_id != self.run_id
            or record.replay_intent_sha256 != self.replay_intent_sha256
            or record.replay_identity_set_sha256 != self.replay_identity_set_sha256
            for record in self.resource_evidence
        ):
            raise ValueError("replay profile resource evidence has inconsistent global bindings")
        if self.elapsed_seconds != max(record.elapsed_seconds for record in self.resource_evidence):
            raise ValueError("public elapsed seconds is not the exact two-run maximum")
        if self.peak_rss_bytes != max(record.peak_rss_bytes for record in self.resource_evidence):
            raise ValueError("public peak RSS is not the exact two-run maximum")
        return self


class ReplayDescriptorAggregateV1(_ReplayGlobalBinding):
    """One across-scene or procedural-comparator descriptor aggregate."""

    schema_id: Literal["ffb.replay-descriptor-aggregate/v1"] = Field(alias="schema")
    descriptor_id: Identifier
    population: Literal["nuscenes-mini-replay", "m3-main-test-comparator"]
    population_count: Literal[10, 200]
    statistic: Literal[
        "count",
        "fraction",
        "minimum",
        "median",
        "maximum",
        "q0",
        "q25",
        "q50",
        "q75",
        "q100",
        "not-modeled",
    ]
    category_label: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    status: Literal["ok", "not-applicable"]
    value: FiniteFloat | None
    unit: ReplayResultUnit
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]

    @field_validator("descriptor_id", "category_label")
    @classmethod
    def reject_private_labels(cls, value: str | None) -> str | None:
        return None if value is None else _require_public_label(value)

    @model_validator(mode="after")
    def require_descriptor_status(self) -> Self:
        if self.population_count != (10 if self.population == "nuscenes-mini-replay" else 200):
            raise ValueError("descriptor population count is invalid")
        if self.status == "ok" and self.value is None:
            raise ValueError("defined descriptor aggregate requires a value")
        if self.status == "not-applicable" and (
            self.value is not None or self.statistic != "not-modeled"
        ):
            raise ValueError("not-applicable descriptor must use not-modeled with no value")
        return self


def replay_descriptor_source_id(record: ReplayDescriptorAggregateV1) -> str:
    """Derive the public ID for one exact descriptor coordinate."""

    return "replay-descriptor-" + sha256_digest(
        {
            "schema": "ffb.replay-descriptor-coordinate/v1",
            "descriptor_id": record.descriptor_id,
            "population": record.population,
            "statistic": record.statistic,
            "category_label": record.category_label,
        }
    )


class _ReplayPanelAggregate(_ReplayIdentityBinding):
    result_id: Identifier
    condition_id: Identifier
    condition_selector: ReplayConditionSelector
    hypothesis_id: ReplayHypothesisId | None = None
    method_id: Identifier
    metric_id: Identifier
    window: ReplayWindow
    inference_role: ReplayInferenceRole
    unit: ReplayResultUnit
    status: ReplayResultStatus
    estimate: FiniteFloat | None
    interval_lower: FiniteFloat | None
    interval_upper: FiniteFloat | None
    bootstrap_replicates: Literal[2000]
    defined_bootstrap_replicates: Annotated[int, Field(ge=0, le=2000)]
    confidence_level: Annotated[FiniteFloat, Field(ge=0.95, le=0.95)]
    interval_method: Literal["paired-scene-percentile-pointwise"]
    aggregation: Literal[
        "equal-scene-mean",
        "pooled-valid-eligible-count-ratio",
        "pooled-valid-loss",
        "conditional-observed-scene-mean",
        "unclipped-recovery-ratio",
    ]
    scene_count: Literal[10]
    positive_scene_count: Annotated[int, Field(ge=0, le=10)] | None = None
    zero_scene_count: Annotated[int, Field(ge=0, le=10)] | None = None
    negative_scene_count: Annotated[int, Field(ge=0, le=10)] | None = None
    undefined_scene_count: Annotated[int, Field(ge=0, le=10)] | None = None
    expected_direction: ReplayExpectedDirection
    persistence_label: ReplayPersistenceLabel
    nonpositive_control_supported: bool | None = None
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]

    @field_validator(
        "result_id",
        "condition_id",
        "condition_selector",
        "method_id",
        "metric_id",
    )
    @classmethod
    def reject_private_result_labels(cls, value: str) -> str:
        return _require_public_label(value)

    @model_validator(mode="after")
    def require_result_consistency(self) -> Self:
        if (
            self.condition_id != self.identity.experiment_id
            or not self.condition_selector.startswith(f"{self.condition_id}:")
        ):
            raise ValueError("aggregate condition selector does not bind its replay identity")
        values = (self.estimate, self.interval_lower, self.interval_upper)
        if self.status == "ok":
            if any(value is None for value in values):
                raise ValueError("defined replay aggregate requires estimate and interval")
            assert self.estimate is not None
            assert self.interval_lower is not None
            assert self.interval_upper is not None
            if self.interval_lower > self.interval_upper:
                raise ValueError("replay aggregate interval bounds are reversed")
            if self.defined_bootstrap_replicates <= 1950:
                raise ValueError("defined replay aggregate requires >97.5% defined replicates")
            if self.unit == "fraction" and not all(
                0.0 <= value <= 1.0
                for value in (
                    self.interval_lower,
                    self.estimate,
                    self.interval_upper,
                )
            ):
                raise ValueError("fraction replay aggregate must remain in [0, 1]")
        elif any(value is not None for value in values):
            raise ValueError("undefined replay aggregate cannot carry estimate or interval")
        if self.status == "not-applicable" and self.defined_bootstrap_replicates != 0:
            raise ValueError("not-applicable replay aggregate cannot carry bootstrap support")

        sign_counts = (
            self.positive_scene_count,
            self.zero_scene_count,
            self.negative_scene_count,
            self.undefined_scene_count,
        )
        if any(value is None for value in sign_counts) != all(
            value is None for value in sign_counts
        ):
            raise ValueError("scene sign counts must be all present or all absent")
        if all(value is not None for value in sign_counts):
            positive_count, zero_count, negative_count, undefined_count = sign_counts
            assert positive_count is not None
            assert zero_count is not None
            assert negative_count is not None
            assert undefined_count is not None
            if positive_count + zero_count + negative_count + undefined_count != 10:
                raise ValueError("scene sign counts must partition all ten scenes")
        if (
            self.status != "ok"
            or self.inference_role not in {"primary-directional", "nonpositive-control"}
        ) and any(value is not None for value in sign_counts):
            raise ValueError("only defined directional/control rows may carry scene sign counts")
        if self.status == "not-applicable" and (
            self.hypothesis_id is not None or self.inference_role != "descriptive"
        ):
            raise ValueError("not-applicable aggregates cannot carry an inference claim")

        requires_sign_counts = self.inference_role in {
            "primary-directional",
            "nonpositive-control",
        }
        if (
            requires_sign_counts
            and self.status == "ok"
            and any(value is None for value in sign_counts)
        ):
            raise ValueError("primary inference rows require all ten scene sign slots")
        if requires_sign_counts and self.status == "ok" and self.undefined_scene_count != 0:
            raise ValueError("defined all-ten-scene inference cannot omit a scene")

        if self.inference_role == "primary-directional":
            if self.expected_direction not in {"positive", "negative"}:
                raise ValueError("primary directional rows require a signed expectation")
            if self.nonpositive_control_supported is not None:
                raise ValueError("directional rows cannot carry a nonpositive-control result")
            expected_labels = (
                {"undefined"}
                if self.status != "ok"
                else {
                    "robustly-persistent",
                    "directionally-consistent",
                    "non-persistent",
                }
            )
            if self.persistence_label not in expected_labels:
                raise ValueError("directional persistence label has an invalid status shape")
        elif self.inference_role == "nonpositive-control":
            if self.expected_direction != "nonpositive":
                raise ValueError("nonpositive controls require the nonpositive expectation")
            expected_label = "undefined" if self.status != "ok" else "not-applicable"
            if self.persistence_label != expected_label:
                raise ValueError("nonpositive controls do not use persistence labels")
            if self.status == "ok":
                assert self.estimate is not None
                assert self.interval_upper is not None
                expected_support = self.estimate <= 0.0 and self.interval_upper <= 0.0
                if self.nonpositive_control_supported != expected_support:
                    raise ValueError("nonpositive control result contradicts point and interval")
            elif self.nonpositive_control_supported is not None:
                raise ValueError("undefined nonpositive controls cannot claim support")
        else:
            if self.expected_direction != "none":
                raise ValueError("diagnostic and descriptive rows are directionless")
            expected_label = "undefined" if self.status != "ok" else "not-applicable"
            if (
                self.persistence_label != expected_label
                or self.nonpositive_control_supported is not None
            ):
                raise ValueError("directionless rows cannot carry directional claims")
        return self


class ReplayPersistentAggregateV1(_ReplayPanelAggregate):
    """One aggregate-only M5-A panel result."""

    schema_id: Literal["ffb.replay-persistent-aggregate/v1"] = Field(alias="schema")

    @model_validator(mode="after")
    def require_persistent_panel(self) -> Self:
        if self.identity.panel_id != M5_PERSISTENT_PANEL_ID:
            raise ValueError("persistent aggregate must bind an M5-A identity")
        return self


class ReplayHealthAggregateV1(_ReplayPanelAggregate):
    """One aggregate-only M5-B apply-only health result."""

    schema_id: Literal["ffb.replay-health-aggregate/v1"] = Field(alias="schema")
    applicability_basis: Literal[
        "applicable",
        "structural-unavailable",
        "support-incompatible",
    ]
    recovery_support_compatible_scene_count: Annotated[int, Field(ge=0, le=10)] | None

    @model_validator(mode="after")
    def require_health_panel(self) -> Self:
        if self.identity.panel_id != M5_HEALTH_PANEL_ID:
            raise ValueError("health aggregate must bind an M5-B identity")
        structural = self.condition_id == "replay-common-mode-x" and self.metric_id in {
            "gap-vs-fault-target-drop",
            "gap-vs-frame-oracle",
            "frame-oracle-recoverable-loss-fraction",
        }
        recovery = self.metric_id == "frame-oracle-recoverable-loss-fraction"
        if structural:
            if (
                self.applicability_basis != "structural-unavailable"
                or self.recovery_support_compatible_scene_count is not None
                or self.status != "not-applicable"
            ):
                raise ValueError("structural health N/A evidence is inconsistent")
        elif recovery:
            count = self.recovery_support_compatible_scene_count
            if count is None:
                raise ValueError("recovery aggregate requires its ten-scene support count")
            expected_basis = "applicable" if count == 10 else "support-incompatible"
            if self.applicability_basis != expected_basis or (
                (self.status == "not-applicable") != (count < 10)
            ):
                raise ValueError("recovery status disagrees with all-scene support evidence")
        elif (
            self.applicability_basis != "applicable"
            or self.recovery_support_compatible_scene_count is not None
            or self.status == "not-applicable"
        ):
            raise ValueError("ordinary health aggregate has invalid applicability evidence")
        return self


class ReplayPersistentCrossoverV1(_ReplayIdentityBinding):
    """One M5-A crossover aggregate; dropout and common mode are excluded."""

    schema_id: Literal["ffb.replay-persistent-crossover/v1"] = Field(alias="schema")
    crossover_id: Identifier
    direction: Literal["negative", "positive", "increase"]
    severity_unit: Literal["m", "rad", "s", "std-scale"]
    tested_maximum: Annotated[FiniteFloat, Field(gt=0.0)]
    status: Literal["observed", "not-observed", "undetermined"]
    point_curve_crossed: bool
    point_estimate: FiniteFloat | None
    interval_lower: FiniteFloat | None
    interval_upper: FiniteFloat | Literal["positive-infinity"] | None
    censoring: Literal["none", "right-above-tested-maximum", "mixed-bootstrap"]
    bootstrap_replicates: Literal[2000]
    bootstrap_crossing_count: Annotated[int, Field(ge=0, le=2000)]
    bootstrap_crossing_fraction: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    confidence_level: Annotated[FiniteFloat, Field(ge=0.95, le=0.95)]
    interval_method: Literal["right-censored-paired-scene-percentile"]
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]

    @field_validator("crossover_id")
    @classmethod
    def reject_private_crossover_label(cls, value: str) -> str:
        return _require_public_label(value)

    @model_validator(mode="after")
    def require_crossover_consistency(self) -> Self:
        if (
            self.identity.panel_id != M5_PERSISTENT_PANEL_ID
            or self.identity.experiment_id not in M5_PERSISTENT_EXPERIMENT_IDS[:6]
        ):
            raise ValueError("crossover identity is not applicable in M5-A")
        expected_fraction = self.bootstrap_crossing_count / self.bootstrap_replicates
        if self.bootstrap_crossing_fraction != expected_fraction:
            raise ValueError("crossover crossing fraction disagrees with its exact count")
        expected_status = bootstrap_crossover_status(
            point_crossed=self.point_curve_crossed,
            crossing_count=self.bootstrap_crossing_count,
            bootstrap_replicates=self.bootstrap_replicates,
        )
        if self.status != expected_status:
            raise ValueError("crossover status disagrees with point and bootstrap support")
        if self.status == "observed":
            if (
                not self.point_curve_crossed
                or self.point_estimate is None
                or self.interval_lower is None
                or not isinstance(self.interval_upper, float)
                or self.censoring != "none"
            ):
                raise ValueError("observed replay row requires a finite point and interval")
            assert self.point_estimate is not None
            assert self.interval_lower is not None
            assert isinstance(self.interval_upper, float)
            if not (
                0.0 <= self.point_estimate <= self.tested_maximum
                and 0.0 <= self.interval_lower <= self.tested_maximum
                and 0.0 <= self.interval_upper <= self.tested_maximum
            ):
                raise ValueError("replay crossover point or interval lies outside the tested range")
            if self.interval_lower > self.interval_upper:
                raise ValueError("replay crossover interval bounds are reversed")
        elif self.status == "not-observed":
            if (
                self.point_curve_crossed
                or self.point_estimate is not None
                or self.interval_lower != self.tested_maximum
                or self.interval_upper != "positive-infinity"
                or self.censoring != "right-above-tested-maximum"
            ):
                raise ValueError("not-observed crossover requires exact right censoring")
        else:
            if (
                self.point_curve_crossed != (self.point_estimate is not None)
                or self.interval_lower is not None
                or self.interval_upper is not None
                or self.censoring != "mixed-bootstrap"
            ):
                raise ValueError("undetermined crossover has an invalid mixed-support shape")
            if self.point_estimate is not None and not (
                0.0 <= self.point_estimate <= self.tested_maximum
            ):
                raise ValueError("undetermined crossover point lies outside the tested range")
        return self


class ReplayClusterSensitivityV1(_ReplayIdentityBinding):
    """One public leave-one-cluster result with only opaque cluster ordinals."""

    schema_id: Literal["ffb.replay-cluster-sensitivity/v1"] = Field(alias="schema")
    sensitivity_id: Identifier
    source_result_id: Identifier
    source_record_sha256: Digest
    cluster_kind: Literal["leave-one-scene-out", "leave-one-log-group-out"]
    cluster_id: Annotated[str, Field(min_length=12, max_length=32)]
    status: Literal["ok", "undefined"]
    estimate: FiniteFloat | None
    unit: ReplayResultUnit
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]

    @field_validator("sensitivity_id", "source_result_id")
    @classmethod
    def reject_private_sensitivity_labels(cls, value: str) -> str:
        return _require_public_label(value)

    @model_validator(mode="after")
    def require_opaque_cluster(self) -> Self:
        prefix = "scene-ordinal:" if self.cluster_kind == "leave-one-scene-out" else "log-group:"
        suffix = self.cluster_id.removeprefix(prefix)
        if (
            not self.cluster_id.startswith(prefix)
            or len(suffix) != 2
            or not suffix.isascii()
            or not suffix.isdigit()
        ):
            raise ValueError("cluster sensitivity must use an opaque two-digit ordinal")
        if self.status == "ok" and self.estimate is None:
            raise ValueError("defined cluster sensitivity requires an estimate")
        if self.status == "undefined" and self.estimate is not None:
            raise ValueError("undefined cluster sensitivity cannot carry an estimate")
        return self


class ReplayValidationCheckV1(ContractModel):
    """One sanitized, content-addressed M5 release-gate result."""

    check_id: Identifier
    passed: bool
    evidence_sha256: Digest


class ReplayValidationV1(_ReplayGlobalBinding):
    """Exact ordered validation conjunction for a publishable M5 artifact."""

    schema_id: Literal["ffb.replay-validation/v1"] = Field(alias="schema")
    checks: Annotated[
        tuple[ReplayValidationCheckV1, ...],
        Field(
            min_length=len(M5_RELEASE_VALIDATION_CHECK_IDS),
            max_length=len(M5_RELEASE_VALIDATION_CHECK_IDS),
        ),
    ]
    scene_count: Literal[10]
    replay_experiment_count: Literal[22]
    raw_sensor_payload_reads: Literal[0]
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_exact_conjunction(self) -> Self:
        if tuple(check.check_id for check in self.checks) != M5_RELEASE_VALIDATION_CHECK_IDS:
            raise ValueError("replay validation checks do not use the fixed order")
        if self.all_checks_passed != all(check.passed for check in self.checks):
            raise ValueError("replay validation result is not the exact conjunction")
        return self


class ReplaySourceMemberCommitmentV1(_ReplayGlobalBinding):
    """Exact primary/repeat commitment without a local source path or identifier."""

    schema_id: Literal["ffb.replay-source-member-commitment/v1"] = Field(alias="schema")
    relative_role: Identifier
    primary_byte_length: Annotated[int, Field(ge=1, le=REPLAY_MAX_ARTIFACT_BYTES)]
    repeat_byte_length: Annotated[int, Field(ge=1, le=REPLAY_MAX_ARTIFACT_BYTES)]
    primary_record_count: Annotated[int, Field(ge=0, le=REPLAY_MAX_NDJSON_RECORDS)]
    repeat_record_count: Annotated[int, Field(ge=0, le=REPLAY_MAX_NDJSON_RECORDS)]
    primary_sha256: Digest
    repeat_sha256: Digest
    equal: bool

    @field_validator("relative_role")
    @classmethod
    def reject_private_commitment_role(cls, value: str) -> str:
        return _require_public_label(value)

    @model_validator(mode="after")
    def require_commitment_equality(self) -> Self:
        expected = (
            self.primary_byte_length == self.repeat_byte_length
            and self.primary_record_count == self.repeat_record_count
            and self.primary_sha256 == self.repeat_sha256
        )
        if self.equal != expected:
            raise ValueError("source commitment equality is contradictory")
        return self


class ReplayRepeatVerificationV1(_ReplayGlobalBinding):
    """Two-run scientific-member equality and provenance summary."""

    schema_id: Literal["ffb.replay-repeat-verification/v1"] = Field(alias="schema")
    primary_local_artifact_sha256: Digest
    repeat_local_artifact_sha256: Digest
    primary_run_sha256: Digest
    repeat_run_sha256: Digest
    source_member_commitments_sha256: Digest
    scientific_member_count: Annotated[int, Field(ge=1, le=1024)]
    mismatch_count: Annotated[int, Field(ge=0, le=1024)]
    scientific_members_all_equal: bool
    run_records_distinct: bool
    source_paths_and_inodes_independent: bool
    same_named_cpu_environment: bool
    evidence_scope: Literal[
        "distinct-path-inode-run-and-member-consistency-not-cryptographic-proof"
    ]
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_repeat_conjunction(self) -> Self:
        if self.mismatch_count > self.scientific_member_count:
            raise ValueError("repeat mismatch count exceeds compared members")
        if self.scientific_members_all_equal != (self.mismatch_count == 0):
            raise ValueError("repeat equality disagrees with mismatch count")
        if self.run_records_distinct != (self.primary_run_sha256 != self.repeat_run_sha256):
            raise ValueError("run-record distinctness disagrees with its commitments")
        expected = (
            self.scientific_members_all_equal
            and self.run_records_distinct
            and self.source_paths_and_inodes_independent
            and self.same_named_cpu_environment
        )
        if self.all_checks_passed != expected:
            raise ValueError("repeat verification is not the exact conjunction")
        return self


class ReplayFigureRecordV1(_ReplayIdentityBinding):
    """Aggregate-to-figure binding without embedding a rendered dataset payload."""

    schema_id: Literal["ffb.replay-figure-record/v1"] = Field(alias="schema")
    figure_id: Identifier
    figure_kind: Literal[
        "panel-summary",
        "crossover",
        "health-transfer",
        "descriptor-comparison",
        "cluster-sensitivity",
    ]
    source_result_id: Identifier
    source_record_sha256: Digest
    figure_spec_sha256: Digest
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]

    @field_validator("figure_id", "source_result_id")
    @classmethod
    def reject_private_figure_labels(cls, value: str) -> str:
        return _require_public_label(value)


class ReplayFigureSourceBindingV1(_ReplayGlobalBinding):
    """One authenticated source mark in the frozen five-figure M5 release."""

    schema_id: Literal["ffb.replay-figure-source-binding/v1"] = Field(alias="schema")
    figure_id: ReplayFigureId
    figure_kind: ReplayFigureKind
    mark_ordinal: Annotated[int, Field(ge=0, lt=REPLAY_MAX_NDJSON_RECORDS)]
    source_kind: ReplayFigureSourceKind
    source_id: Identifier
    source_record_sha256: Digest
    identity: ReplayExperimentIdentityV1 | None = None
    replay_identity_sha256: Digest | None = None
    figure_spec_sha256: Digest
    rendered_svg_path: ReplayFigureSvgPath
    rendered_svg_sha256: Digest
    rendered_svg_byte_length: Annotated[int, Field(ge=1, le=REPLAY_MAX_MEMBER_BYTES)]
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]

    @field_validator("source_id")
    @classmethod
    def reject_private_source_label(cls, value: str) -> str:
        return _require_public_label(value)

    @model_validator(mode="after")
    def require_figure_and_source_shape(self) -> Self:
        expected_id, expected_kind, expected_path = _M5_REPLAY_FIGURE_DEFINITION_BY_ID[
            self.figure_id
        ]
        if (
            self.figure_id != expected_id
            or self.figure_kind != expected_kind
            or self.rendered_svg_path != expected_path
        ):
            raise ValueError("figure binding disagrees with the frozen public figure definition")
        if self.source_kind not in _M5_REPLAY_FIGURE_SOURCE_KINDS[self.figure_id]:
            raise ValueError("figure binding source kind is invalid for the public figure")

        descriptor = self.source_kind == "descriptor-aggregate"
        if descriptor:
            if self.identity is not None or self.replay_identity_sha256 is not None:
                raise ValueError("descriptor figure sources cannot carry a synthetic identity")
        else:
            if self.identity is None or self.replay_identity_sha256 is None:
                raise ValueError("non-descriptor figure sources require a replay identity")
            if self.replay_identity_sha256 != replay_experiment_identity_sha256(self.identity):
                raise ValueError("figure source replay identity digest is invalid")
        return self


class ReplayPayloadFileEntryV1(ContractModel):
    """One exact curated member committed by the M5 release index."""

    path: ReplayIndexedPath
    byte_length: Annotated[int, Field(ge=1, le=REPLAY_MAX_MEMBER_BYTES)]
    sha256: Digest
    record_count: Annotated[int, Field(ge=1, le=REPLAY_MAX_NDJSON_RECORDS)] | None
    replay_identity_set_sha256: Digest

    @model_validator(mode="after")
    def require_entry_shape(self) -> Self:
        if self.path.endswith(".ndjson") != (self.record_count is not None):
            raise ValueError("record_count must be present exactly for NDJSON")
        if self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256:
            raise ValueError("payload entry does not bind the frozen replay identity set")
        return self


class ReplayReleaseIndexV1(ContractModel):
    """Exact ordered aggregate-only member envelope for M5."""

    schema_id: Literal["ffb.replay-release-index/v1"] = Field(alias="schema")
    artifact_contract: Literal["ffb.replay-curated-payload/v1"]
    release_id: Literal["m5-nuscenes-replay-v0.1.0"]
    run_id: Identifier
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest
    identities: Annotated[
        tuple[ReplayExperimentIdentityV1, ...],
        Field(min_length=22, max_length=22),
    ]
    persistent_source_matrix_sha256: Digest
    persistent_condition_selector_count: Literal[71]
    persistent_condition_selector_set_sha256: Digest
    persistent_aggregate_coordinate_count: Literal[464]
    persistent_aggregate_coordinate_set_sha256: Digest
    persistent_hypothesis_coordinate_count: Literal[33]
    persistent_hypothesis_coordinate_set_sha256: Digest
    health_fit_artifact_sha256: Digest
    health_fit_run_sha256: Digest
    health_condition_selector_count: Literal[43]
    health_condition_selector_set_sha256: Digest
    health_aggregate_coordinate_count: Literal[14988]
    health_aggregate_coordinate_set_sha256: Digest
    health_hypothesis_coordinate_count: Literal[11]
    health_hypothesis_coordinate_set_sha256: Digest
    tracked_aggregate_terms: Literal["CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms"]
    files: Annotated[
        tuple[ReplayPayloadFileEntryV1, ...],
        Field(min_length=len(REPLAY_INDEXED_PATHS), max_length=len(REPLAY_INDEXED_PATHS)),
    ]

    @model_validator(mode="after")
    def require_exact_index(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
            or self.identities != expected_replay_identities()
            or self.persistent_source_matrix_sha256 != M5_PERSISTENT_MATRIX_SHA256
            or self.persistent_condition_selector_set_sha256
            != M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256
            or self.persistent_aggregate_coordinate_set_sha256
            != M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256
            or self.persistent_hypothesis_coordinate_set_sha256
            != M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256
            or self.health_fit_artifact_sha256 != M5_HEALTH_FIT_SHA256
            or self.health_fit_run_sha256 != M5_HEALTH_FIT_RUN_SHA256
            or self.health_condition_selector_set_sha256 != M5_HEALTH_CONDITION_SELECTOR_SET_SHA256
            or self.health_aggregate_coordinate_set_sha256
            != M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256
            or self.health_hypothesis_coordinate_set_sha256
            != M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256
        ):
            raise ValueError("release index does not bind the frozen M5 provenance")
        if tuple(entry.path for entry in self.files) != REPLAY_INDEXED_PATHS:
            raise ValueError("release index does not use the fixed member order")
        return self


class ReplaySuccessV1(ContractModel):
    """Final completion marker binding M5 identity, artifact, and run."""

    schema_id: Literal["ffb.replay-success/v1"] = Field(alias="schema")
    release_artifact_sha256: Digest
    run_sha256: Digest
    replay_intent_sha256: Digest
    replay_identity_set_sha256: Digest

    @model_validator(mode="after")
    def require_frozen_success_binding(self) -> Self:
        if (
            self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256
            or self.replay_identity_set_sha256 != M5_REPLAY_IDENTITY_SET_SHA256
        ):
            raise ValueError("success marker does not bind the frozen M5 identity set")
        return self


def replay_resource_evidence_sha256(
    records: tuple[ReplayExecutionResourceEvidenceV1, ...],
) -> str:
    """Digest the exact ordered primary/repeat resource records."""

    if tuple(record.run_label for record in records) != ("primary", "repeat"):
        raise ValueError("resource evidence digest requires exact primary/repeat order")
    return sha256_digest(
        {
            "schema": "ffb.replay-execution-resource-evidence-set/v1",
            "records": [record.model_dump(mode="json", by_alias=True) for record in records],
        }
    )


def replay_execution_resource_evidence_json_schema() -> dict[str, object]:
    """Return the strict external M5 execution-resource schema."""

    return ReplayExecutionResourceEvidenceV1.model_json_schema(by_alias=True)


def replay_release_index_json_schema() -> dict[str, object]:
    """Return the strict public M5 release-index schema."""

    return ReplayReleaseIndexV1.model_json_schema(by_alias=True)


def replay_identity_set_digest() -> str:
    """Return the canonical digest shared by every M5 curated member."""

    return M5_REPLAY_IDENTITY_SET_SHA256
