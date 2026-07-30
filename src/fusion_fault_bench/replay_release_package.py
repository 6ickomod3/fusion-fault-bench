"""Self-contained semantic validation for the complete M5 release package.

The outer package loader proves the exact 41-file allowlist and its two
domain-separated digests.  This module closes the remaining semantic links:
the strict machine artifact, the immutable review candidate, review
attestations, public claims, deterministic figures, and all seventeen
validation-evidence slots.  No function in this module opens a dataset or
uses the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Never, cast

from pydantic import BaseModel, ValidationError

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_TRACKED_AGGREGATE_TERMS,
    REPLAY_ARTIFACT_PATHS,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_FIGURE_PATHS,
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_DESTINATION_PATH,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
    ReplayFigureSpecV1,
    ReplayImplementationReviewAttestationV1,
    ReplayPrivacyLicenseAttestationV1,
    ReplayPublicClaimProjectionsV1,
    ReplayReviewCandidateIndexV1,
    ReplaySoftwareVerificationV1,
    ReplayValidationInputsV1,
)
from fusion_fault_bench.replay_artifacts import (
    LoadedReplayCuratedArtifact,
    canonical_replay_ndjson_bytes,
    load_replay_curated_artifact,
)
from fusion_fault_bench.replay_claims import (
    ReplayClaimEvidence,
    build_presentation_files,
    build_public_claim_projections,
    build_release_summary,
)
from fusion_fault_bench.replay_figures import (
    build_figure_bundle,
    canonical_figure_spec_files,
)
from fusion_fault_bench.replay_release import (
    LoadedReplayReleasePackage,
    load_release_package,
    validate_review_candidate_members,
)
from fusion_fault_bench.replay_release_software import M5_SOFTWARE_COMMAND_BY_CHECK
from fusion_fault_bench.replay_release_validation import (
    M5_ELIGIBILITY_CAUSALITY_TEST_IDS,
    M5_HEALTH_LEAKAGE_TEST_IDS,
    M5_LOCAL_READ_ACCOUNTING_TEST_IDS,
    M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK,
    M5_SOFTWARE_VERIFICATION_CHECK_IDS,
    M5_TRANSFORM_TIMING_ORACLE_TEST_IDS,
    ResultsReviewBindings,
    load_final_replay_validation,
    load_pre_review_validation_inputs,
    load_results_review_attestation,
    software_verification_test_subset_bytes,
)

_FROZEN_METHODOLOGY_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "evidence/release-pipeline-plan.md": (
            "2b1f7c29a3be92418b6ca1fecb5a9cb44822617081187aef824931fed5657f66"
        ),
        "evidence/release-pipeline-plan-review.md": (
            "929fce2f74256bee527979c77554f2ce41bb230bac60d9b7ffbb2098607545d3"
        ),
        "evidence/resource-scope-amendment.md": (
            "f7eb19e03661bec1663b1a2f6ce953e465e8a3a76d63277ece5bee4e59708aff"
        ),
    }
)
_LICENSE_SHA256 = "357180c8878699f39558b7012c1541ca6ffdffd6f5d4507f91e8e9a21d2a4539"
_DATA_AND_MODEL_TERMS_SHA256 = "cb86c6d663e1ccfb4c470d6727da4913afe65d443202a2815d3f9bc01fd97fcc"
_MAX_PUBLICATION_MEMBER_BYTES = 8 * 1024 * 1024

_CANDIDATE_TO_RELEASE: Mapping[str, str] = MappingProxyType(
    {
        "machine/intent.json": "artifact/intent.json",
        "machine/replay-profile-summary.json": "artifact/replay-profile-summary.json",
        "machine/descriptor-aggregates.ndjson": "artifact/descriptor-aggregates.ndjson",
        "machine/persistent-panel-aggregates.ndjson": (
            "artifact/persistent-panel-aggregates.ndjson"
        ),
        "machine/persistent-panel-crossovers.ndjson": (
            "artifact/persistent-panel-crossovers.ndjson"
        ),
        "machine/health-panel-aggregates.ndjson": "artifact/health-panel-aggregates.ndjson",
        "machine/leave-one-cluster-sensitivity.ndjson": (
            "artifact/leave-one-cluster-sensitivity.ndjson"
        ),
        "machine/repeat-verification.json": "artifact/repeat-verification.json",
        "machine/figure-records.ndjson": "artifact/figure-records.ndjson",
        "machine/source-member-commitments.ndjson": ("artifact/source-member-commitments.ndjson"),
        "evidence/release-pipeline-plan.md": "evidence/release-pipeline-plan.md",
        "evidence/release-pipeline-plan-review.md": ("evidence/release-pipeline-plan-review.md"),
        "evidence/resource-scope-amendment.md": "evidence/resource-scope-amendment.md",
        "evidence/implementation-review.md": "evidence/implementation-review.md",
        "evidence/validation-inputs.json": "evidence/validation-inputs.json",
        "evidence/implementation-review-attestation.json": (
            "evidence/implementation-review-attestation.json"
        ),
        "evidence/software-verification.json": "evidence/software-verification.json",
        "evidence/privacy-license-attestation.json": ("evidence/privacy-license-attestation.json"),
        **{path: path for path in M5_FIGURE_PATHS},
        "presentation/release-summary.json": "release-summary.json",
        "presentation/public-claim-projections.json": ("evidence/public-claim-projections.json"),
    }
)
_PRESENTATION_TO_RELEASE: Mapping[str, str] = MappingProxyType(
    {
        "presentation/README.md": "README.md",
        "presentation/claim-evidence.md": "claim-evidence.md",
        "presentation/verification.md": "verification.md",
    }
)

M5_PUBLICATION_DOCUMENT_PATHS = (
    "README.md",
    "docs/results.md",
    "docs/benchmark-card.md",
    "docs/limitations.md",
    "docs/reproducibility.md",
    "docs/project-plan.md",
    "docs/dataset-preparation.md",
    "docs/m5-technical-walkthrough.md",
)


class ReplayReleasePackageValidationError(ValueError):
    """The complete M5 package or repository projection failed closed."""


@dataclass(frozen=True, slots=True)
class ValidatedReplayReleasePackage:
    """A package whose outer, machine, review, claim, and figure links passed."""

    package: LoadedReplayReleasePackage
    artifact: LoadedReplayCuratedArtifact
    candidate_index: ReplayReviewCandidateIndexV1
    candidate_files: Mapping[str, bytes]
    public_claim_projections: ReplayPublicClaimProjectionsV1
    release_summary: Mapping[str, Any]
    validation_inputs: ReplayValidationInputsV1
    software_verification: ReplaySoftwareVerificationV1
    implementation_review_attestation: ReplayImplementationReviewAttestationV1
    privacy_license_attestation: ReplayPrivacyLicenseAttestationV1
    release_package_sha256: str
    claim_projection_sha256: str


def _fail(message: str) -> Never:
    raise ReplayReleasePackageValidationError(message) from None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_model[ModelT: BaseModel](
    value: bytes,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    try:
        observed = model.model_validate_json(value)
    except (ValueError, ValidationError) as error:
        raise ReplayReleasePackageValidationError(
            f"{label} violates its strict contract"
        ) from error
    if canonical_json_bytes(observed) != value:
        _fail(f"{label} is not canonical JSON")
    return observed


def _canonical_mapping(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = cast(object, json.loads(value))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayReleasePackageValidationError(f"{label} is not JSON") from error
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        _fail(f"{label} is not a canonical JSON object")
    mapping = cast(dict[str, Any], parsed)
    if canonical_json_bytes(mapping) != value:
        _fail(f"{label} is not a canonical JSON object")
    return mapping


def _record_count(path: str, value: bytes) -> int | None:
    if not path.endswith(".ndjson"):
        return None
    return len(value.splitlines())


def _substitution_pairs(
    *,
    machine_artifact_sha256: str,
    machine_run_sha256: str,
    results_review_attestation_sha256: str,
    machine_artifact_byte_length: int,
) -> tuple[tuple[bytes, bytes], ...]:
    replacements = (
        machine_artifact_sha256,
        machine_run_sha256,
        results_review_attestation_sha256,
        str(machine_artifact_byte_length),
    )
    return tuple(
        (placeholder.encode("ascii"), replacement.encode("ascii"))
        for placeholder, replacement in zip(
            M5_PRESENTATION_PLACEHOLDERS,
            replacements,
            strict=True,
        )
    )


def _reverse_presentation_substitutions(
    value: bytes,
    substitutions: Sequence[tuple[bytes, bytes]],
    *,
    label: str,
) -> bytes:
    candidate = value
    for placeholder, replacement in substitutions:
        placeholder_token = b"`" + placeholder + b"`"
        replacement_token = b"`" + replacement + b"`"
        if placeholder in candidate or candidate.count(replacement_token) != 1:
            _fail(f"{label} does not contain exactly the four allowed final substitutions")
        candidate = candidate.replace(replacement_token, placeholder_token, 1)
    regenerated = candidate
    for placeholder, replacement in substitutions:
        if regenerated.count(placeholder) != 1:
            _fail(f"{label} has an invalid reviewed placeholder shape")
        regenerated = regenerated.replace(placeholder, replacement, 1)
    if regenerated != value:
        _fail(f"{label} contains a presentation change beyond the four allowed substitutions")
    return candidate


def reconstruct_reviewed_candidate(
    package_files: Mapping[str, bytes],
    *,
    machine_artifact_sha256: str,
    machine_run_sha256: str,
    results_review_attestation_sha256: str,
    machine_artifact_byte_length: int,
) -> tuple[ReplayReviewCandidateIndexV1, Mapping[str, bytes]]:
    """Reconstruct and authenticate the reviewed 34-file candidate in memory."""

    try:
        index_bytes = package_files["evidence/review-candidate-index.json"]
    except KeyError as error:
        raise ReplayReleasePackageValidationError(
            "release package lacks the reviewed candidate index"
        ) from error
    index = _canonical_model(
        index_bytes,
        ReplayReviewCandidateIndexV1,
        label="review candidate index",
    )
    substitutions = _substitution_pairs(
        machine_artifact_sha256=machine_artifact_sha256,
        machine_run_sha256=machine_run_sha256,
        results_review_attestation_sha256=results_review_attestation_sha256,
        machine_artifact_byte_length=machine_artifact_byte_length,
    )
    reconstructed: dict[str, bytes] = {}
    for candidate_path in M5_REVIEW_CANDIDATE_INDEXED_PATHS:
        if candidate_path in _PRESENTATION_TO_RELEASE:
            release_path = _PRESENTATION_TO_RELEASE[candidate_path]
            try:
                final_value = package_files[release_path]
            except KeyError as error:
                raise ReplayReleasePackageValidationError(
                    "release package lacks a final presentation member"
                ) from error
            reconstructed[candidate_path] = _reverse_presentation_substitutions(
                final_value,
                substitutions,
                label=release_path,
            )
        else:
            try:
                release_path = _CANDIDATE_TO_RELEASE[candidate_path]
                reconstructed[candidate_path] = package_files[release_path]
            except KeyError as error:
                raise ReplayReleasePackageValidationError(
                    "release package cannot reconstruct the reviewed candidate"
                ) from error
    validate_review_candidate_members(reconstructed)
    for entry in index.files:
        value = reconstructed[entry.path]
        if (
            entry.byte_length != len(value)
            or entry.sha256 != _sha256(value)
            or entry.record_count != _record_count(entry.path, value)
        ):
            _fail("final package differs from an indexed reviewed candidate member")
    return index, MappingProxyType(reconstructed)


def _set_digest(paths: Sequence[str], files: Mapping[str, bytes], *, schema: str) -> str:
    return sha256_digest(
        {
            "schema": schema,
            "files": [
                {
                    "path": path,
                    "byte_length": len(files[path]),
                    "sha256": _sha256(files[path]),
                }
                for path in paths
            ],
        }
    )


def derive_results_review_bindings(
    index: ReplayReviewCandidateIndexV1,
    index_bytes: bytes,
    candidate_files: Mapping[str, bytes],
) -> ResultsReviewBindings:
    """Derive the seven exact candidate bindings named by the results review."""

    return ResultsReviewBindings(
        candidate_sha256=index.candidate_sha256,
        candidate_index_sha256=_sha256(index_bytes),
        scientific_member_set_sha256=_set_digest(
            M5_REVIEW_CANDIDATE_INDEXED_PATHS[:10],
            candidate_files,
            schema="ffb.m5-reviewed-scientific-member-set/v1",
        ),
        claim_projection_sha256=_sha256(
            candidate_files["presentation/public-claim-projections.json"]
        ),
        figure_spec_set_sha256=_set_digest(
            M5_FIGURE_PATHS[::2],
            candidate_files,
            schema="ffb.m5-reviewed-figure-spec-set/v1",
        ),
        rendered_figure_set_sha256=_set_digest(
            M5_FIGURE_PATHS[1::2],
            candidate_files,
            schema="ffb.m5-reviewed-rendered-figure-set/v1",
        ),
        presentation_template_set_sha256=_set_digest(
            (
                "presentation/README.md",
                "presentation/claim-evidence.md",
                "presentation/verification.md",
                "presentation/release-summary.json",
            ),
            candidate_files,
            schema="ffb.m5-reviewed-presentation-template-set/v1",
        ),
    )


def _projection_bytes(schema: str, **fields: object) -> bytes:
    return canonical_json_bytes({"schema": schema, **fields})


def _claim_coordinate_bytes(
    registry: ReplayPublicClaimProjectionsV1,
    group: str,
) -> bytes:
    rows = tuple(
        row
        for row in registry.projections
        if row.projection_group == group and row.hypothesis_id is not None
    )
    return _projection_bytes(
        "ffb.m5-validation-claim-coordinate-set/v1",
        projection_group=group,
        projections=[row.model_dump(mode="json", by_alias=True) for row in rows],
    )


def _bootstrap_fields(artifact: LoadedReplayCuratedArtifact) -> bytes:
    rows = (*artifact.persistent_aggregates, *artifact.health_aggregates)
    projected: list[dict[str, object]] = []
    for row in rows:
        projected.append(
            {
                "result_id": row.result_id,
                "status": row.status,
                "bootstrap_replicates": row.bootstrap_replicates,
                "defined_bootstrap_replicates": row.defined_bootstrap_replicates,
                "confidence_level": row.confidence_level,
                "interval_method": row.interval_method,
                "positive_scene_count": row.positive_scene_count,
                "zero_scene_count": row.zero_scene_count,
                "negative_scene_count": row.negative_scene_count,
                "undefined_scene_count": row.undefined_scene_count,
            }
        )
    return _projection_bytes("ffb.m5-validation-bootstrap-fields/v1", rows=projected)


def derive_release_validation_evidence(
    *,
    artifact: LoadedReplayCuratedArtifact,
    candidate_files: Mapping[str, bytes],
    software_verification: ReplaySoftwareVerificationV1,
    privacy_license_attestation: ReplayPrivacyLicenseAttestationV1,
    implementation_review_attestation: ReplayImplementationReviewAttestationV1,
    public_claim_projections: ReplayPublicClaimProjectionsV1,
    results_review_attestation_bytes: bytes | None,
) -> Mapping[str, Mapping[str, bytes]]:
    """Re-derive every named validation authority retained by the package.

    Passing ``None`` for the results-review bytes returns the sixteen
    pre-review check mappings.  Supplying the canonical attestation returns
    all seventeen mappings used by final ``validation.json``.
    """

    profile = artifact.profile_summary
    release_index = artifact.release_index
    commitments = artifact.source_commitments
    descriptor_commitment = commitments[0]
    support_commitments = commitments[1:]
    dropout_rows = tuple(
        row
        for row in public_claim_projections.projections
        if row.projection_group == "dropout-nesting"
    )
    if len(dropout_rows) != 1:
        _fail("public claims do not retain one dropout-nesting derivation")
    scan = _projection_bytes(
        "ffb.m5-role-aware-privacy-scan-result/v1",
        scan_contract=privacy_license_attestation.scan_contract,
        scan_scope=privacy_license_attestation.scan_scope,
        forbidden_match_count=privacy_license_attestation.forbidden_match_count,
    )
    fixed_terms = _projection_bytes(
        "ffb.m5-fixed-aggregate-terms/v1",
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )
    attribution = _projection_bytes(
        "ffb.m5-attribution-non-endorsement/v1",
        profile_attribution_and_non_endorsement_required=(
            profile.attribution_and_non_endorsement_required
        ),
        attribution_present=privacy_license_attestation.attribution_present,
        non_endorsement_present=privacy_license_attestation.non_endorsement_present,
    )
    evidence: dict[str, Mapping[str, bytes]] = {
        "intent-freeze": {
            "frozen-intent-bytes": artifact.intent_bytes,
            "frozen-intent-canonical-digest": _projection_bytes(
                "ffb.m5-frozen-intent-digest/v1",
                replay_intent_sha256=release_index.replay_intent_sha256,
            ),
            "release-pipeline-plan-bytes": candidate_files["evidence/release-pipeline-plan.md"],
            "release-pipeline-plan-review-bytes": candidate_files[
                "evidence/release-pipeline-plan-review.md"
            ],
            "resource-scope-amendment-bytes": candidate_files[
                "evidence/resource-scope-amendment.md"
            ],
        },
        "fixed-scene-population": {
            "profile-scene-and-experiment-counts": _projection_bytes(
                "ffb.m5-profile-population-counts/v1",
                scene_count=profile.scene_count,
                persistent_experiment_count=profile.persistent_experiment_count,
                health_experiment_count=profile.health_experiment_count,
                replay_experiment_count=profile.replay_experiment_count,
                distinct_log_group_count=profile.distinct_log_group_count,
            ),
            "replay-identity-set-commitment": _projection_bytes(
                "ffb.m5-replay-identity-set-commitment/v1",
                count=len(release_index.identities),
                sha256=release_index.replay_identity_set_sha256,
            ),
            "persistent-selector-set-commitment": _projection_bytes(
                "ffb.m5-persistent-selector-set-commitment/v1",
                count=release_index.persistent_condition_selector_count,
                sha256=release_index.persistent_condition_selector_set_sha256,
            ),
            "health-selector-set-commitment": _projection_bytes(
                "ffb.m5-health-selector-set-commitment/v1",
                count=release_index.health_condition_selector_count,
                sha256=release_index.health_condition_selector_set_sha256,
            ),
        },
        "base-support": {
            "profile-base-support-attestation": _projection_bytes(
                "ffb.m5-profile-base-support/v1",
                all_scenes_have_base_support=profile.all_scenes_have_base_support,
            ),
            "authenticated-descriptor-commitments": _projection_bytes(
                "ffb.m5-authenticated-descriptor-commitments/v1",
                descriptor_aggregate_sha256=_sha256(
                    candidate_files["machine/descriptor-aggregates.ndjson"]
                ),
                source_commitment=descriptor_commitment.model_dump(mode="json", by_alias=True),
            ),
            "authenticated-support-commitments": _projection_bytes(
                "ffb.m5-authenticated-support-commitments/v1",
                source_commitments=[
                    row.model_dump(mode="json", by_alias=True) for row in support_commitments
                ],
            ),
        },
        "health-schedules": {
            "profile-schedule-attestation": _projection_bytes(
                "ffb.m5-profile-health-schedules/v1",
                all_health_schedules_valid=profile.all_health_schedules_valid,
            ),
            "health-selector-set-commitment": _projection_bytes(
                "ffb.m5-health-selector-set-commitment/v1",
                count=release_index.health_condition_selector_count,
                sha256=release_index.health_condition_selector_set_sha256,
            ),
            "health-coordinate-set-commitment": _projection_bytes(
                "ffb.m5-health-coordinate-set-commitment/v1",
                count=release_index.health_aggregate_coordinate_count,
                sha256=release_index.health_aggregate_coordinate_set_sha256,
            ),
        },
        "transform-and-timing-oracles": {
            "software-transform-and-timing-oracle-entries": (
                software_verification_test_subset_bytes(
                    software_verification,
                    M5_TRANSFORM_TIMING_ORACLE_TEST_IDS,
                )
            ),
            "implementation-review-attestation-bytes": candidate_files[
                "evidence/implementation-review-attestation.json"
            ],
        },
        "eligibility-and-fault-causality": {
            "software-eligibility-and-causality-entries": (
                software_verification_test_subset_bytes(
                    software_verification,
                    M5_ELIGIBILITY_CAUSALITY_TEST_IDS,
                )
            ),
            "dropout-nesting-evidence": canonical_json_bytes(dropout_rows[0]),
        },
        "health-feature-leakage": {
            "software-health-leakage-entries": software_verification_test_subset_bytes(
                software_verification,
                M5_HEALTH_LEAKAGE_TEST_IDS,
            )
        },
        "persistent-panel-completeness": {
            "persistent-aggregate-bytes": candidate_files[
                "machine/persistent-panel-aggregates.ndjson"
            ],
            "persistent-71-selector-commitment": _projection_bytes(
                "ffb.m5-persistent-selector-set-commitment/v1",
                count=release_index.persistent_condition_selector_count,
                sha256=release_index.persistent_condition_selector_set_sha256,
            ),
            "persistent-464-coordinate-commitment": _projection_bytes(
                "ffb.m5-persistent-coordinate-set-commitment/v1",
                count=release_index.persistent_aggregate_coordinate_count,
                sha256=release_index.persistent_aggregate_coordinate_set_sha256,
            ),
            "persistent-33-claim-coordinate-commitment": _claim_coordinate_bytes(
                public_claim_projections,
                "persistent-panel",
            ),
        },
        "health-panel-completeness": {
            "health-aggregate-bytes": candidate_files["machine/health-panel-aggregates.ndjson"],
            "health-43-selector-commitment": _projection_bytes(
                "ffb.m5-health-selector-set-commitment/v1",
                count=release_index.health_condition_selector_count,
                sha256=release_index.health_condition_selector_set_sha256,
            ),
            "health-14988-coordinate-commitment": _projection_bytes(
                "ffb.m5-health-coordinate-set-commitment/v1",
                count=release_index.health_aggregate_coordinate_count,
                sha256=release_index.health_aggregate_coordinate_set_sha256,
            ),
            "health-11-claim-coordinate-commitment": _claim_coordinate_bytes(
                public_claim_projections,
                "health-transfer",
            ),
        },
        "scene-bootstrap-and-cluster-sensitivity": {
            "aggregate-bootstrap-fields": _bootstrap_fields(artifact),
            "complete-cluster-sensitivity-bytes": candidate_files[
                "machine/leave-one-cluster-sensitivity.ndjson"
            ],
        },
        "repeat-scientific-members": {
            "repeat-verification-bytes": candidate_files["machine/repeat-verification.json"],
            "ordered-source-member-commitments-bytes": candidate_files[
                "machine/source-member-commitments.ndjson"
            ],
        },
        "cpu-and-memory-caps": {
            "ordered-primary-repeat-resource-evidence": canonical_replay_ndjson_bytes(
                profile.resource_evidence
            )
        },
        "no-raw-payload-reads": {
            "profile-zero-read-field": _projection_bytes(
                "ffb.m5-profile-zero-raw-reads/v1",
                raw_sensor_payload_reads=profile.raw_sensor_payload_reads,
            ),
            "local-read-accounting-attestation": software_verification_test_subset_bytes(
                software_verification,
                M5_LOCAL_READ_ACCOUNTING_TEST_IDS,
            ),
            "privacy-license-attestation-bytes": candidate_files[
                "evidence/privacy-license-attestation.json"
            ],
        },
        "privacy-and-dataset-license": {
            "deterministic-candidate-scan": scan,
            "fixed-aggregate-terms": fixed_terms,
            "attribution-and-non-endorsement-fields": attribution,
            "privacy-license-attestation-bytes": candidate_files[
                "evidence/privacy-license-attestation.json"
            ],
        },
        "implementation-review": {
            "implementation-review-report-bytes": candidate_files[
                "evidence/implementation-review.md"
            ],
            "implementation-review-attestation-bytes": candidate_files[
                "evidence/implementation-review-attestation.json"
            ],
        },
        "software-verification": {
            "software-verification-bytes": candidate_files["evidence/software-verification.json"]
        },
    }
    if results_review_attestation_bytes is not None:
        evidence["results-and-claims-review"] = {
            "results-review-attestation-bytes": results_review_attestation_bytes
        }
    return MappingProxyType(
        {check_id: MappingProxyType(dict(parts)) for check_id, parts in evidence.items()}
    )


def _require_release_permitting_review(
    attestation: ReplayImplementationReviewAttestationV1,
) -> None:
    blocking = any(
        row.status == "unresolved" and row.severity in {"p0", "p1"} for row in attestation.findings
    )
    if attestation.disposition not in {"pass", "pass-with-nonblocking-findings"} or blocking:
        _fail("implementation review does not permit release")


def _require_software_verification_authority(
    software: ReplaySoftwareVerificationV1,
) -> None:
    if tuple(check.check_id for check in software.checks) != (M5_SOFTWARE_VERIFICATION_CHECK_IDS):
        _fail("software verification check IDs are incomplete or reordered")
    for check in software.checks:
        if check.required_test_ids != M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[check.check_id]:
            _fail("software verification reassigns a required-test authority")
        if check.command != M5_SOFTWARE_COMMAND_BY_CHECK[check.check_id]:
            _fail("software verification command differs from its frozen check authority")


def _claim_evidence(
    artifact: LoadedReplayCuratedArtifact,
    software: ReplaySoftwareVerificationV1,
) -> ReplayClaimEvidence:
    return ReplayClaimEvidence(
        profile_summary=artifact.profile_summary,
        descriptor_aggregates=artifact.descriptor_aggregates,
        persistent_aggregates=artifact.persistent_aggregates,
        persistent_crossovers=artifact.persistent_crossovers,
        health_aggregates=artifact.health_aggregates,
        cluster_sensitivity=artifact.cluster_sensitivity,
        repeat_verification=artifact.repeat_verification,
        software_verification=software,
    )


def _validate_methodology_and_authority(
    *,
    candidate_files: Mapping[str, bytes],
    candidate_index: ReplayReviewCandidateIndexV1,
    artifact: LoadedReplayCuratedArtifact,
) -> tuple[
    ReplayImplementationReviewAttestationV1,
    ReplaySoftwareVerificationV1,
    ReplayPrivacyLicenseAttestationV1,
]:
    for path, expected in _FROZEN_METHODOLOGY_SHA256.items():
        if _sha256(candidate_files[path]) != expected:
            _fail(f"frozen methodology bytes changed: {path}")
    implementation = _canonical_model(
        candidate_files["evidence/implementation-review-attestation.json"],
        ReplayImplementationReviewAttestationV1,
        label="implementation review attestation",
    )
    software = _canonical_model(
        candidate_files["evidence/software-verification.json"],
        ReplaySoftwareVerificationV1,
        label="software verification",
    )
    privacy = _canonical_model(
        candidate_files["evidence/privacy-license-attestation.json"],
        ReplayPrivacyLicenseAttestationV1,
        label="privacy/license attestation",
    )
    _require_release_permitting_review(implementation)
    _require_software_verification_authority(software)
    implementation_report = candidate_files["evidence/implementation-review.md"]
    if implementation.review_report_sha256 != _sha256(implementation_report):
        _fail("implementation review attestation does not bind its report")
    run = artifact.run
    if (
        candidate_index.scientific_git_revision != run.git_revision
        or candidate_index.lockfile_sha256 != run.lockfile_sha256
        or candidate_index.package_version != run.package_version
        or candidate_index.run_id != run.run_id
        or software.scientific_git_revision != run.git_revision
        or software.tooling_revision != run.git_revision
        or software.lockfile_sha256 != run.lockfile_sha256
        or software.package_version != run.package_version
        or software.implementation_snapshot_sha256 != implementation.implementation_snapshot_sha256
        or privacy.scientific_git_revision != run.git_revision
        or privacy.run_id != run.run_id
    ):
        _fail("candidate authority does not bind the exact scientific run revision")
    if (
        privacy.repository_license_sha256 != _LICENSE_SHA256
        or privacy.data_and_model_terms_sha256 != _DATA_AND_MODEL_TERMS_SHA256
    ):
        _fail("privacy/license attestation does not bind the frozen tracked terms")
    repeat = artifact.repeat_verification
    if (
        candidate_index.primary_local_artifact_sha256 != repeat.primary_local_artifact_sha256
        or candidate_index.repeat_local_artifact_sha256 != repeat.repeat_local_artifact_sha256
        or candidate_index.primary_local_run_sha256 != repeat.primary_run_sha256
        or candidate_index.repeat_local_run_sha256 != repeat.repeat_run_sha256
    ):
        _fail("reviewed candidate does not bind the authenticated primary/repeat inputs")
    return implementation, software, privacy


def _validate_claims_figures_and_presentations(
    *,
    package: LoadedReplayReleasePackage,
    artifact: LoadedReplayCuratedArtifact,
    candidate_files: Mapping[str, bytes],
    software: ReplaySoftwareVerificationV1,
    substitutions: Sequence[tuple[bytes, bytes]],
) -> tuple[ReplayPublicClaimProjectionsV1, Mapping[str, Any]]:
    registry_bytes = candidate_files["presentation/public-claim-projections.json"]
    registry = _canonical_model(
        registry_bytes,
        ReplayPublicClaimProjectionsV1,
        label="public claim projections",
    )
    evidence = _claim_evidence(artifact, software)
    expected_registry = build_public_claim_projections(evidence)
    if registry != expected_registry or canonical_json_bytes(expected_registry) != registry_bytes:
        _fail("public claim projections do not regenerate from machine evidence")
    expected_summary = build_release_summary(registry, evidence)
    summary_bytes = canonical_json_bytes(expected_summary)
    if candidate_files["presentation/release-summary.json"] != summary_bytes:
        _fail("release summary does not regenerate from reviewed public claims")
    summary = _canonical_mapping(summary_bytes, label="release summary")
    expected_templates = build_presentation_files(registry, summary)
    for candidate_path, expected in expected_templates.items():
        if candidate_files[candidate_path] != expected:
            _fail("reviewed presentation template does not regenerate from public claims")
        final = expected
        for placeholder, replacement in substitutions:
            final = final.replace(placeholder, replacement, 1)
        if package.files[_PRESENTATION_TO_RELEASE[candidate_path]] != final:
            _fail("final presentation differs beyond the four reviewed identity substitutions")

    bundle = build_figure_bundle(registry, evidence)
    spec_files = canonical_figure_spec_files(bundle.specs)
    for spec, spec_path in zip(bundle.specs, M5_FIGURE_PATHS[::2], strict=True):
        observed = _canonical_model(
            package.files[spec_path],
            ReplayFigureSpecV1,
            label=spec_path,
        )
        if observed != spec or spec_files[spec_path] != package.files[spec_path]:
            _fail("figure specification does not regenerate from reviewed claims")
    for svg_path in M5_FIGURE_PATHS[1::2]:
        if package.files[svg_path] != bundle.svgs[svg_path]:
            _fail("rendered figure does not regenerate byte-for-byte")
    if artifact.figures != bundle.bindings:
        _fail("machine figure bindings do not regenerate from specs and SVGs")
    return registry, MappingProxyType(summary)


def _validate_release_package(path: Path) -> ValidatedReplayReleasePackage:
    package = load_release_package(path)
    artifact = load_replay_curated_artifact(package.path / "artifact")
    machine_byte_length = sum(
        len(package.files[f"artifact/{member}"]) for member in REPLAY_ARTIFACT_PATHS
    )
    attestation_bytes = package.files["evidence/results-review-attestation.json"]
    attestation_sha256 = _sha256(attestation_bytes)
    if (
        package.index.machine_artifact_sha256 != artifact.artifact_sha256
        or package.index.machine_run_sha256 != artifact.run_sha256
        or package.index.machine_artifact_byte_length != machine_byte_length
        or package.index.results_review_attestation_sha256 != attestation_sha256
        or package.index.scientific_git_revision != artifact.run.git_revision
    ):
        _fail("outer package index disagrees with the strict machine artifact")
    candidate_index, candidate_files = reconstruct_reviewed_candidate(
        package.files,
        machine_artifact_sha256=artifact.artifact_sha256,
        machine_run_sha256=artifact.run_sha256,
        results_review_attestation_sha256=attestation_sha256,
        machine_artifact_byte_length=machine_byte_length,
    )
    if package.index.reviewed_candidate_sha256 != candidate_index.candidate_sha256:
        _fail("outer package does not bind the reconstructed reviewed candidate")
    implementation, software, privacy = _validate_methodology_and_authority(
        candidate_files=candidate_files,
        candidate_index=candidate_index,
        artifact=artifact,
    )
    bindings = derive_results_review_bindings(
        candidate_index,
        package.files["evidence/review-candidate-index.json"],
        candidate_files,
    )
    load_results_review_attestation(
        attestation_bytes,
        review_report=package.files["evidence/results-review.md"],
        scientific_git_revision=artifact.run.git_revision,
        bindings=bindings,
        require_release_permitting=True,
    )
    substitutions = _substitution_pairs(
        machine_artifact_sha256=artifact.artifact_sha256,
        machine_run_sha256=artifact.run_sha256,
        results_review_attestation_sha256=attestation_sha256,
        machine_artifact_byte_length=machine_byte_length,
    )
    registry, summary = _validate_claims_figures_and_presentations(
        package=package,
        artifact=artifact,
        candidate_files=candidate_files,
        software=software,
        substitutions=substitutions,
    )
    pre_review_evidence = derive_release_validation_evidence(
        artifact=artifact,
        candidate_files=candidate_files,
        software_verification=software,
        privacy_license_attestation=privacy,
        implementation_review_attestation=implementation,
        public_claim_projections=registry,
        results_review_attestation_bytes=None,
    )
    validation_inputs = load_pre_review_validation_inputs(
        candidate_files["evidence/validation-inputs.json"],
        run_id=artifact.run.run_id,
        evidence_by_check=pre_review_evidence,
    )
    final_evidence = derive_release_validation_evidence(
        artifact=artifact,
        candidate_files=candidate_files,
        software_verification=software,
        privacy_license_attestation=privacy,
        implementation_review_attestation=implementation,
        public_claim_projections=registry,
        results_review_attestation_bytes=attestation_bytes,
    )
    final_validation = load_final_replay_validation(
        package.files["artifact/validation.json"],
        run_id=artifact.run.run_id,
        evidence_by_check=final_evidence,
        pre_review_inputs=validation_inputs,
    )
    if final_validation != artifact.validation:
        _fail("strict artifact validation object changed during package validation")
    return ValidatedReplayReleasePackage(
        package=package,
        artifact=artifact,
        candidate_index=candidate_index,
        candidate_files=candidate_files,
        public_claim_projections=registry,
        release_summary=summary,
        validation_inputs=validation_inputs,
        software_verification=software,
        implementation_review_attestation=implementation,
        privacy_license_attestation=privacy,
        release_package_sha256=package.release_package_sha256,
        claim_projection_sha256=_sha256(
            candidate_files["presentation/public-claim-projections.json"]
        ),
    )


def validate_release_package(path: Path) -> ValidatedReplayReleasePackage:
    """Validate one exact M5 package without a dataset or network dependency."""

    try:
        return _validate_release_package(path)
    except ReplayReleasePackageValidationError:
        raise
    except (OSError, ValueError, ValidationError) as error:
        raise ReplayReleasePackageValidationError(
            "M5 release package semantic validation failed"
        ) from error


def _read_publication_file(source_root: Path, relative: str) -> bytes:
    root = source_root.resolve(strict=True)
    target = root.joinpath(*relative.split("/"))
    try:
        before = os.lstat(target)
    except OSError as error:
        raise ReplayReleasePackageValidationError(
            f"publication member is missing: {relative}"
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= _MAX_PUBLICATION_MEMBER_BYTES
    ):
        _fail(f"publication member is not a bounded regular file: {relative}")
    value = target.read_bytes()
    after = os.lstat(target)
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
    if len(value) != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in stable
    ):
        _fail(f"publication member changed while reading: {relative}")
    return value


def validate_publication(release: Path, source_root: Path) -> str:
    """Validate package plus its fixed public review and documentation projection."""

    root = source_root.resolve(strict=True)
    expected_release = root / M5_RELEASE_DESTINATION_PATH
    observed_release = (release if release.is_absolute() else root / release).resolve(strict=False)
    if observed_release != expected_release:
        _fail("publication release is not at the frozen tracked package path")
    validated = validate_release_package(observed_release)
    exact_public_copies = {
        "docs/m5-release-pipeline-plan.md": "evidence/release-pipeline-plan.md",
        "docs/reviews/m5-release-pipeline-plan-review.md": (
            "evidence/release-pipeline-plan-review.md"
        ),
        "docs/m5-resource-scope-amendment.md": "evidence/resource-scope-amendment.md",
        "docs/reviews/m5-release-implementation-review.md": ("evidence/implementation-review.md"),
        "docs/reviews/m5-release-implementation-review-attestation.json": (
            "evidence/implementation-review-attestation.json"
        ),
        "docs/reviews/m5-results-review.md": "evidence/results-review.md",
        "docs/reviews/m5-results-review-attestation.json": (
            "evidence/results-review-attestation.json"
        ),
    }
    for public_path, package_path in exact_public_copies.items():
        public_value = _read_publication_file(root, public_path)
        if public_value != validated.package.files[package_path]:
            _fail(f"public evidence copy differs from the release package: {public_path}")

    required_tokens = (
        b"m5-nuscenes-replay-v0.1.0",
        validated.release_package_sha256.encode("ascii"),
        validated.claim_projection_sha256.encode("ascii"),
    )
    for relative in M5_PUBLICATION_DOCUMENT_PATHS:
        value = _read_publication_file(root, relative)
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReplayReleasePackageValidationError(
                f"release documentation is not UTF-8: {relative}"
            ) from error
        if b"\x00" in value or b"\r" in value or not value.endswith(b"\n"):
            _fail(f"release documentation has noncanonical text framing: {relative}")
        if any(token not in value for token in required_tokens):
            _fail(f"release documentation lacks the exact package projection: {relative}")
    return validated.release_package_sha256


__all__ = [
    "M5_PUBLICATION_DOCUMENT_PATHS",
    "ReplayReleasePackageValidationError",
    "ValidatedReplayReleasePackage",
    "derive_release_validation_evidence",
    "derive_results_review_bindings",
    "reconstruct_reviewed_candidate",
    "validate_publication",
    "validate_release_package",
]
