"""Shared prereview/final evidence derivation bridge for the M5 package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import cast

from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_AGGREGATE_COORDINATE_COUNT,
    M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256,
    M5_HEALTH_CONDITION_SELECTOR_COUNT,
    M5_HEALTH_CONDITION_SELECTOR_SET_SHA256,
    M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT,
    M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256,
    M5_PERSISTENT_CONDITION_SELECTOR_COUNT,
    M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256,
    M5_REPLAY_IDENTITY_SET_SHA256,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayProfileSummaryV1,
    ReplaySourceMemberCommitmentV1,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    ReplayImplementationReviewAttestationV1,
    ReplayPrivacyLicenseAttestationV1,
    ReplayPublicClaimProjectionsV1,
    ReplaySoftwareVerificationV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_SHA256,
    expected_replay_identities,
)
from fusion_fault_bench.replay_artifacts import LoadedReplayCuratedArtifact
from fusion_fault_bench.replay_release_package import derive_release_validation_evidence


def derive_pre_review_evidence_from_parts(
    *,
    intent_bytes: bytes,
    candidate_files: Mapping[str, bytes],
    profile: ReplayProfileSummaryV1,
    persistent: Sequence[ReplayPersistentAggregateV1],
    health: Sequence[ReplayHealthAggregateV1],
    commitments: Sequence[ReplaySourceMemberCommitmentV1],
    software: ReplaySoftwareVerificationV1,
    privacy: ReplayPrivacyLicenseAttestationV1,
    implementation_review: ReplayImplementationReviewAttestationV1,
    registry: ReplayPublicClaimProjectionsV1,
) -> dict[str, dict[str, bytes]]:
    """Use the package validator's exact derivation before a release index exists."""

    release_index = SimpleNamespace(
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        identities=expected_replay_identities(),
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        persistent_condition_selector_count=M5_PERSISTENT_CONDITION_SELECTOR_COUNT,
        persistent_condition_selector_set_sha256=M5_PERSISTENT_CONDITION_SELECTOR_SET_SHA256,
        persistent_aggregate_coordinate_count=M5_PERSISTENT_AGGREGATE_COORDINATE_COUNT,
        persistent_aggregate_coordinate_set_sha256=M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256,
        health_condition_selector_count=M5_HEALTH_CONDITION_SELECTOR_COUNT,
        health_condition_selector_set_sha256=M5_HEALTH_CONDITION_SELECTOR_SET_SHA256,
        health_aggregate_coordinate_count=M5_HEALTH_AGGREGATE_COORDINATE_COUNT,
        health_aggregate_coordinate_set_sha256=M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256,
    )
    artifact = SimpleNamespace(
        profile_summary=profile,
        release_index=release_index,
        source_commitments=tuple(commitments),
        intent_bytes=intent_bytes,
        persistent_aggregates=tuple(persistent),
        health_aggregates=tuple(health),
    )
    evidence = derive_release_validation_evidence(
        artifact=cast(LoadedReplayCuratedArtifact, artifact),
        candidate_files=candidate_files,
        software_verification=software,
        privacy_license_attestation=privacy,
        implementation_review_attestation=implementation_review,
        public_claim_projections=registry,
        results_review_attestation_bytes=None,
    )
    return {check_id: dict(parts) for check_id, parts in evidence.items()}


__all__ = ["derive_pre_review_evidence_from_parts"]
