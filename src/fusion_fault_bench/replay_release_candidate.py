"""Deterministic construction and semantic validation of the M5 review candidate."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_REPLAY_IDENTITY_SET_SHA256,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    ReplayProfileSummaryV1,
    ReplayRepeatVerificationV1,
    ReplaySourceMemberCommitmentV1,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_FIGURE_PATHS,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
    ReplayFigureSourceBindingV1,
    ReplayFigureSpecV1,
    ReplayImplementationReviewAttestationV1,
    ReplayPublicClaimProjectionsV1,
    ReplaySoftwareVerificationV1,
    ReplayValidationInputsV1,
)
from fusion_fault_bench.replay_artifacts import (
    ReplayReviewMachineMembersRequest,
    prepare_replay_review_machine_members,
)
from fusion_fault_bench.replay_claims import (
    ReplayClaimEvidence,
    build_presentation_files,
    build_public_claim_projections,
    build_release_summary,
    validate_public_claim_projections,
)
from fusion_fault_bench.replay_figures import (
    M5_FIGURE_ORDER,
    build_figure_bundle,
    canonical_figure_spec_files,
    validate_figure_bundle,
)
from fusion_fault_bench.replay_release import (
    LoadedReplayReviewCandidate,
    ReplayReviewCandidateContent,
    load_review_candidate,
    publish_review_candidate,
)
from fusion_fault_bench.replay_release_authority import ImplementationSnapshot
from fusion_fault_bench.replay_release_evidence_bridge import (
    derive_pre_review_evidence_from_parts,
)
from fusion_fault_bench.replay_release_validation import (
    ResultsReviewBindings,
    build_privacy_license_attestation,
    derive_pre_review_validation_inputs,
    load_pre_review_validation_inputs,
    load_privacy_license_attestation,
    load_software_verification,
)
from fusion_fault_bench.replay_runner import (
    curate_replay_verified_repeat,
    verify_replay_repeat_artifacts,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

_PLAN_AUTHORITIES = {
    "evidence/release-pipeline-plan.md": Path("docs/m5-release-pipeline-plan.md"),
    "evidence/release-pipeline-plan-review.md": Path(
        "docs/reviews/m5-release-pipeline-plan-review.md"
    ),
    "evidence/resource-scope-amendment.md": Path("docs/m5-resource-scope-amendment.md"),
}


class ReplayCandidateWorkflowError(ValueError):
    """Candidate construction or semantic regeneration failed closed."""


def _model[ModelT: BaseModel](value: bytes, model_type: type[ModelT], *, label: str) -> ModelT:
    try:
        model = model_type.model_validate_json(value)
    except (ValueError, ValidationError) as error:
        raise ReplayCandidateWorkflowError(f"M5 candidate {label} violates its contract") from error
    if canonical_json_bytes(model) != value:
        raise ReplayCandidateWorkflowError(f"M5 candidate {label} is not canonical")
    return model


def _ndjson[ModelT: BaseModel](
    value: bytes, model_type: type[ModelT], *, label: str
) -> tuple[ModelT, ...]:
    if not value or not value.endswith(b"\n"):
        raise ReplayCandidateWorkflowError(f"M5 candidate {label} is not canonical NDJSON")
    output: list[ModelT] = []
    for line in value.splitlines(keepends=True):
        output.append(_model(line, model_type, label=label))
    return tuple(output)


def _results_bindings(candidate: LoadedReplayReviewCandidate) -> ResultsReviewBindings:
    def set_digest(paths: Sequence[str], schema: str) -> str:
        return sha256_digest(
            {
                "schema": schema,
                "files": [
                    {
                        "path": path,
                        "byte_length": len(candidate.files[path]),
                        "sha256": hashlib.sha256(candidate.files[path]).hexdigest(),
                    }
                    for path in paths
                ],
            }
        )

    return ResultsReviewBindings(
        candidate_sha256=candidate.candidate_sha256,
        candidate_index_sha256=candidate.candidate_index_sha256,
        scientific_member_set_sha256=set_digest(
            M5_REVIEW_CANDIDATE_INDEXED_PATHS[:10],
            "ffb.m5-reviewed-scientific-member-set/v1",
        ),
        claim_projection_sha256=hashlib.sha256(
            candidate.files["presentation/public-claim-projections.json"]
        ).hexdigest(),
        figure_spec_set_sha256=set_digest(
            M5_FIGURE_PATHS[::2],
            "ffb.m5-reviewed-figure-spec-set/v1",
        ),
        rendered_figure_set_sha256=set_digest(
            M5_FIGURE_PATHS[1::2],
            "ffb.m5-reviewed-rendered-figure-set/v1",
        ),
        presentation_template_set_sha256=set_digest(
            (
                "presentation/README.md",
                "presentation/claim-evidence.md",
                "presentation/verification.md",
                "presentation/release-summary.json",
            ),
            "ffb.m5-reviewed-presentation-template-set/v1",
        ),
    )


def _plan_files(source_root: Path) -> dict[str, bytes]:
    return {
        candidate_path: (source_root / path).read_bytes()
        for candidate_path, path in _PLAN_AUTHORITIES.items()
    }


def _claim_evidence(
    *,
    profile: ReplayProfileSummaryV1,
    descriptors: Sequence[ReplayDescriptorAggregateV1],
    persistent: Sequence[ReplayPersistentAggregateV1],
    crossovers: Sequence[ReplayPersistentCrossoverV1],
    health: Sequence[ReplayHealthAggregateV1],
    sensitivity: Sequence[ReplayClusterSensitivityV1],
    repeat: ReplayRepeatVerificationV1,
    software: ReplaySoftwareVerificationV1,
) -> ReplayClaimEvidence:
    return ReplayClaimEvidence(
        profile_summary=profile,
        descriptor_aggregates=tuple(descriptors),
        persistent_aggregates=tuple(persistent),
        persistent_crossovers=tuple(crossovers),
        health_aggregates=tuple(health),
        cluster_sensitivity=tuple(sensitivity),
        repeat_verification=repeat,
        software_verification=software,
    )


def _candidate_files(
    *,
    machine_files: Mapping[str, bytes],
    plan_files: Mapping[str, bytes],
    implementation_report: bytes,
    implementation_attestation: bytes,
    validation_inputs: ReplayValidationInputsV1,
    software_bytes: bytes,
    privacy_bytes: bytes,
    registry: ReplayPublicClaimProjectionsV1,
    summary: Mapping[str, Any],
    presentation: Mapping[str, bytes],
    specs: Mapping[str, bytes],
    svgs: Mapping[str, bytes],
) -> dict[str, bytes]:
    unordered = {
        **{f"machine/{path}": value for path, value in machine_files.items()},
        **plan_files,
        "evidence/implementation-review.md": implementation_report,
        "evidence/validation-inputs.json": canonical_json_bytes(validation_inputs),
        "evidence/implementation-review-attestation.json": implementation_attestation,
        "evidence/software-verification.json": software_bytes,
        "evidence/privacy-license-attestation.json": privacy_bytes,
        **specs,
        **svgs,
        **presentation,
        "presentation/release-summary.json": canonical_json_bytes(summary),
        "presentation/public-claim-projections.json": canonical_json_bytes(registry),
    }
    try:
        return {path: unordered[path] for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS}
    except KeyError as error:
        raise ReplayCandidateWorkflowError(
            "M5 candidate generation omitted a fixed member"
        ) from error


def prepare_review_candidate(
    *,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
    output_dir: Path,
    source_root: Path,
    clean_snapshot: Any,
    implementation_snapshot: ImplementationSnapshot,
    implementation_report: bytes,
    implementation_attestation: bytes,
) -> LoadedReplayReviewCandidate:
    """Regenerate and atomically publish the exact 34-file review candidate."""

    try:
        repeat_evidence = verify_replay_repeat_artifacts(
            primary_path=primary_artifact,
            repeat_path=repeat_artifact,
        )
        curated = curate_replay_verified_repeat(
            repeat_evidence,
            primary_log_path=primary_time_l,
            repeat_log_path=repeat_time_l,
        )
        software_bytes = software_verification.read_bytes()
        software = load_software_verification(
            software_bytes,
            snapshot=implementation_snapshot,
            lockfile_sha256=clean_snapshot.lockfile_sha256,
            package_version=clean_snapshot.package_version,
        )
        evidence = _claim_evidence(
            profile=curated.profile_summary,
            descriptors=curated.descriptor_aggregates,
            persistent=curated.persistent_aggregates,
            crossovers=curated.persistent_crossovers,
            health=curated.health_aggregates,
            sensitivity=curated.cluster_sensitivity,
            repeat=repeat_evidence.repeat_verification,
            software=software,
        )
        registry = build_public_claim_projections(evidence)
        summary = build_release_summary(registry, evidence)
        presentation = build_presentation_files(registry, summary)
        figures = build_figure_bundle(registry, evidence)
        machine = prepare_replay_review_machine_members(
            ReplayReviewMachineMembersRequest(
                profile_summary=curated.profile_summary,
                descriptor_aggregates=curated.descriptor_aggregates,
                persistent_aggregates=curated.persistent_aggregates,
                persistent_crossovers=curated.persistent_crossovers,
                health_aggregates=curated.health_aggregates,
                cluster_sensitivity=curated.cluster_sensitivity,
                repeat_verification=repeat_evidence.repeat_verification,
                figures=figures.bindings,
                source_commitments=repeat_evidence.source_commitments,
                run=curated.run,
            ),
            source_root=clean_snapshot.source_root,
        )
        privacy = build_privacy_license_attestation(
            snapshot=implementation_snapshot,
            run_id=curated.run.run_id,
        )
        privacy_bytes = canonical_json_bytes(privacy)
        plans = _plan_files(clean_snapshot.source_root)
        implementation = _model(
            implementation_attestation,
            ReplayImplementationReviewAttestationV1,
            label="implementation review attestation",
        )
        authority_files = {
            **{f"machine/{path}": value for path, value in machine.files.items()},
            **plans,
            "evidence/implementation-review.md": implementation_report,
            "evidence/implementation-review-attestation.json": implementation_attestation,
            "evidence/software-verification.json": software_bytes,
            "evidence/privacy-license-attestation.json": privacy_bytes,
        }
        evidence_by_check = derive_pre_review_evidence_from_parts(
            intent_bytes=machine.files["intent.json"],
            candidate_files=authority_files,
            profile=curated.profile_summary,
            persistent=curated.persistent_aggregates,
            health=curated.health_aggregates,
            commitments=repeat_evidence.source_commitments,
            software=software,
            privacy=privacy,
            implementation_review=implementation,
            registry=registry,
        )
        validation_inputs = derive_pre_review_validation_inputs(
            run_id=curated.run.run_id,
            evidence_by_check=evidence_by_check,
        )
        files = _candidate_files(
            machine_files=machine.files,
            plan_files=plans,
            implementation_report=implementation_report,
            implementation_attestation=implementation_attestation,
            validation_inputs=validation_inputs,
            software_bytes=software_bytes,
            privacy_bytes=privacy_bytes,
            registry=registry,
            summary=summary,
            presentation=presentation,
            specs=canonical_figure_spec_files(figures.specs),
            svgs=figures.svgs,
        )
        content = ReplayReviewCandidateContent(
            scientific_git_revision=clean_snapshot.git_revision,
            lockfile_sha256=clean_snapshot.lockfile_sha256,
            package_version=clean_snapshot.package_version,
            run_id=curated.run.run_id,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            primary_local_artifact_sha256=repeat_evidence.primary.artifact_sha256,
            repeat_local_artifact_sha256=repeat_evidence.repeat.artifact_sha256,
            primary_local_run_sha256=repeat_evidence.primary.run_sha256,
            repeat_local_run_sha256=repeat_evidence.repeat.run_sha256,
            files=files,
        )
        return publish_review_candidate(content, output_dir)
    except ReplayCandidateWorkflowError:
        raise
    except (OSError, RuntimeError, TypeError, ValidationError, ValueError) as error:
        raise ReplayCandidateWorkflowError("M5 review candidate preparation failed") from error


def _parse_candidate_machine(
    candidate: LoadedReplayReviewCandidate,
) -> tuple[
    ReplayProfileSummaryV1,
    tuple[ReplayDescriptorAggregateV1, ...],
    tuple[ReplayPersistentAggregateV1, ...],
    tuple[ReplayPersistentCrossoverV1, ...],
    tuple[ReplayHealthAggregateV1, ...],
    tuple[ReplayClusterSensitivityV1, ...],
    ReplayRepeatVerificationV1,
    tuple[ReplayFigureSourceBindingV1, ...],
    tuple[ReplaySourceMemberCommitmentV1, ...],
]:
    files = candidate.files
    return (
        _model(
            files["machine/replay-profile-summary.json"], ReplayProfileSummaryV1, label="profile"
        ),
        _ndjson(
            files["machine/descriptor-aggregates.ndjson"],
            ReplayDescriptorAggregateV1,
            label="descriptors",
        ),
        _ndjson(
            files["machine/persistent-panel-aggregates.ndjson"],
            ReplayPersistentAggregateV1,
            label="persistent aggregates",
        ),
        _ndjson(
            files["machine/persistent-panel-crossovers.ndjson"],
            ReplayPersistentCrossoverV1,
            label="crossovers",
        ),
        _ndjson(
            files["machine/health-panel-aggregates.ndjson"],
            ReplayHealthAggregateV1,
            label="health aggregates",
        ),
        _ndjson(
            files["machine/leave-one-cluster-sensitivity.ndjson"],
            ReplayClusterSensitivityV1,
            label="sensitivity",
        ),
        _model(
            files["machine/repeat-verification.json"],
            ReplayRepeatVerificationV1,
            label="repeat verification",
        ),
        _ndjson(
            files["machine/figure-records.ndjson"],
            ReplayFigureSourceBindingV1,
            label="figure bindings",
        ),
        _ndjson(
            files["machine/source-member-commitments.ndjson"],
            ReplaySourceMemberCommitmentV1,
            label="source commitments",
        ),
    )


def load_validated_review_candidate(
    *,
    path: Path,
    source_root: Path,
    clean_snapshot: Any,
    implementation_snapshot: ImplementationSnapshot,
    implementation_report: bytes,
    implementation_attestation: bytes,
) -> LoadedReplayReviewCandidate:
    """Strictly parse and regenerate every source-bound candidate member."""

    candidate = load_review_candidate(path)
    if (
        candidate.index.scientific_git_revision != clean_snapshot.git_revision
        or candidate.index.lockfile_sha256 != clean_snapshot.lockfile_sha256
        or candidate.index.package_version != clean_snapshot.package_version
    ):
        raise ReplayCandidateWorkflowError("M5 review candidate has stale source authority")
    plans = _plan_files(clean_snapshot.source_root)
    for candidate_path, value in plans.items():
        if candidate.files[candidate_path] != value:
            raise ReplayCandidateWorkflowError("M5 candidate methodology differs from source")
    if (
        candidate.files["evidence/implementation-review.md"] != implementation_report
        or candidate.files["evidence/implementation-review-attestation.json"]
        != implementation_attestation
    ):
        raise ReplayCandidateWorkflowError("M5 candidate implementation review is stale")
    (
        profile,
        descriptors,
        persistent,
        crossovers,
        health,
        sensitivity,
        repeat,
        bindings,
        commitments,
    ) = _parse_candidate_machine(candidate)
    if any(
        getattr(row, "run_id", candidate.index.run_id) != candidate.index.run_id
        for row in (
            profile,
            *descriptors,
            *persistent,
            *crossovers,
            *health,
            *sensitivity,
            repeat,
            *bindings,
            *commitments,
        )
    ):
        raise ReplayCandidateWorkflowError("M5 candidate record run bindings disagree")
    software_bytes = candidate.files["evidence/software-verification.json"]
    software = load_software_verification(
        software_bytes,
        snapshot=implementation_snapshot,
        lockfile_sha256=clean_snapshot.lockfile_sha256,
        package_version=clean_snapshot.package_version,
    )
    evidence = _claim_evidence(
        profile=profile,
        descriptors=descriptors,
        persistent=persistent,
        crossovers=crossovers,
        health=health,
        sensitivity=sensitivity,
        repeat=repeat,
        software=software,
    )
    registry = _model(
        candidate.files["presentation/public-claim-projections.json"],
        ReplayPublicClaimProjectionsV1,
        label="public claims",
    )
    validate_public_claim_projections(registry, evidence)
    summary = build_release_summary(registry, evidence)
    if canonical_json_bytes(summary) != candidate.files["presentation/release-summary.json"]:
        raise ReplayCandidateWorkflowError("M5 candidate release summary does not regenerate")
    presentation = build_presentation_files(registry, summary)
    if any(candidate.files[path] != value for path, value in presentation.items()):
        raise ReplayCandidateWorkflowError("M5 candidate presentation does not regenerate")
    specs = tuple(
        _model(
            candidate.files[f"figures/{figure_id}.spec.json"],
            ReplayFigureSpecV1,
            label=f"{figure_id} spec",
        )
        for figure_id in M5_FIGURE_ORDER
    )
    svgs = {spec.figure_file: candidate.files[spec.figure_file] for spec in specs}
    validate_figure_bundle(specs, svgs, bindings, evidence)
    privacy_bytes = candidate.files["evidence/privacy-license-attestation.json"]
    privacy = load_privacy_license_attestation(
        privacy_bytes,
        snapshot=implementation_snapshot,
        run_id=candidate.index.run_id,
    )
    implementation = _model(
        implementation_attestation,
        ReplayImplementationReviewAttestationV1,
        label="implementation review attestation",
    )
    evidence_by_check = derive_pre_review_evidence_from_parts(
        intent_bytes=candidate.files["machine/intent.json"],
        candidate_files=candidate.files,
        profile=profile,
        persistent=persistent,
        health=health,
        commitments=commitments,
        software=software,
        privacy=privacy,
        implementation_review=implementation,
        registry=registry,
    )
    load_pre_review_validation_inputs(
        candidate.files["evidence/validation-inputs.json"],
        run_id=candidate.index.run_id,
        evidence_by_check=evidence_by_check,
    )
    _results_bindings(candidate)
    return candidate


def candidate_results_review_bindings(
    candidate: LoadedReplayReviewCandidate,
) -> ResultsReviewBindings:
    """Return the exact digest bridge for a semantically validated candidate."""

    return _results_bindings(candidate)


__all__ = [
    "ReplayCandidateWorkflowError",
    "candidate_results_review_bindings",
    "load_validated_review_candidate",
    "prepare_review_candidate",
]
