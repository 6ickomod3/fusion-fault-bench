from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from test_m5_release_presentation import _evidence
from test_replay_artifacts import ROOT, _request

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_IMPLEMENTATION_REVIEW_AREAS,
    M5_SOFTWARE_VERIFICATION_CATEGORIES,
    ReplayImplementationReviewDecisionV1,
    ReplayResultsReviewDecisionV1,
    ReplaySoftwareVerificationCheckV1,
)
from fusion_fault_bench.replay_artifacts import (
    ReplayCuratedArtifactWriteRequest,
    ReplayReviewMachineMembersRequest,
    _ordered_records,
    prepare_replay_machine_payload,
    prepare_replay_review_machine_members,
)
from fusion_fault_bench.replay_claims import (
    build_presentation_files,
    build_public_claim_projections,
    build_release_summary,
)
from fusion_fault_bench.replay_figures import (
    build_figure_bundle,
    canonical_figure_spec_files,
)
from fusion_fault_bench.replay_release import (
    ReplayReleasePackageContent,
    ReplayReviewCandidateContent,
    load_release_package,
    publish_release_package,
    publish_review_candidate,
)
from fusion_fault_bench.replay_release_authority import implementation_snapshot_from_files
from fusion_fault_bench.replay_release_build import _sidecar_files
from fusion_fault_bench.replay_release_candidate import _candidate_files, _plan_files
from fusion_fault_bench.replay_release_evidence_bridge import (
    derive_pre_review_evidence_from_parts,
)
from fusion_fault_bench.replay_release_package import (
    derive_results_review_bindings,
    validate_release_package,
)
from fusion_fault_bench.replay_release_software import M5_SOFTWARE_COMMAND_BY_CHECK
from fusion_fault_bench.replay_release_validation import (
    M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK,
    M5_SOFTWARE_VERIFICATION_CHECK_IDS,
    build_implementation_review_attestation,
    build_privacy_license_attestation,
    build_results_review_attestation,
    build_software_verification,
    derive_final_replay_validation,
    derive_pre_review_validation_inputs,
)
from fusion_fault_bench.replay_resources import M5_PUBLIC_REPLAY_COMMAND


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_dataset_free_final_package_build_reload_and_semantic_validation(tmp_path: Path) -> None:
    """Exercise every real aggregate-only package layer without opening nuScenes."""

    request = _request()
    public_run = request.run.model_copy(update={"command": M5_PUBLIC_REPLAY_COMMAND})
    revision = request.run.git_revision
    implementation_snapshot = implementation_snapshot_from_files(
        {
            "LICENSE": (ROOT / "LICENSE").read_bytes(),
            "DATA_AND_MODEL_TERMS.md": (ROOT / "DATA_AND_MODEL_TERMS.md").read_bytes(),
        },
        scientific_git_revision=revision,
    )
    software_checks = tuple(
        ReplaySoftwareVerificationCheckV1(
            check_id=check_id,
            category=category,
            command=M5_SOFTWARE_COMMAND_BY_CHECK[check_id],
            required_test_ids=M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[check_id],
            exit_status=0,
            output_sha256=_digest(f"software-output:{check_id}"),
            output_normalization=("stable-command-output-with-runtime-paths-and-durations-removed"),
        )
        for check_id, category in zip(
            M5_SOFTWARE_VERIFICATION_CHECK_IDS,
            M5_SOFTWARE_VERIFICATION_CATEGORIES,
            strict=True,
        )
    )
    software = build_software_verification(
        software_checks,
        snapshot=implementation_snapshot,
        lockfile_sha256=request.run.lockfile_sha256,
        package_version=request.run.package_version,
    )

    primary_local_artifact_sha256 = _digest("primary-local-artifact")
    repeat_local_artifact_sha256 = _digest("repeat-local-artifact")
    primary_resource, repeat_resource = request.profile_summary.resource_evidence
    profile = request.profile_summary.model_copy(
        update={
            "resource_evidence": (
                primary_resource.model_copy(
                    update={"local_artifact_sha256": primary_local_artifact_sha256}
                ),
                repeat_resource.model_copy(
                    update={"local_artifact_sha256": repeat_local_artifact_sha256}
                ),
            )
        }
    )
    repeat_verification = request.repeat_verification.model_copy(
        update={
            "primary_local_artifact_sha256": primary_local_artifact_sha256,
            "repeat_local_artifact_sha256": repeat_local_artifact_sha256,
        }
    )
    base_evidence = _evidence()
    evidence = replace(
        base_evidence,
        profile_summary=profile,
        repeat_verification=repeat_verification,
        software_verification=software,
    )
    ordered = _ordered_records(
        ReplayReviewMachineMembersRequest(
            profile_summary=evidence.profile_summary,
            descriptor_aggregates=evidence.descriptor_aggregates,
            persistent_aggregates=evidence.persistent_aggregates,
            persistent_crossovers=evidence.persistent_crossovers,
            health_aggregates=evidence.health_aggregates,
            cluster_sensitivity=evidence.cluster_sensitivity,
            repeat_verification=evidence.repeat_verification,
            figures=(),
            source_commitments=request.source_commitments,
            run=public_run,
        )
    )
    evidence = replace(
        evidence,
        descriptor_aggregates=ordered[0],
        persistent_aggregates=ordered[1],
        persistent_crossovers=ordered[2],
        health_aggregates=ordered[3],
        cluster_sensitivity=ordered[4],
    )
    registry = build_public_claim_projections(evidence)
    summary = build_release_summary(registry, evidence)
    presentation = build_presentation_files(registry, summary)
    figures = build_figure_bundle(registry, evidence)

    review_machine = prepare_replay_review_machine_members(
        ReplayReviewMachineMembersRequest(
            profile_summary=evidence.profile_summary,
            descriptor_aggregates=evidence.descriptor_aggregates,
            persistent_aggregates=evidence.persistent_aggregates,
            persistent_crossovers=evidence.persistent_crossovers,
            health_aggregates=evidence.health_aggregates,
            cluster_sensitivity=evidence.cluster_sensitivity,
            repeat_verification=evidence.repeat_verification,
            figures=figures.bindings,
            source_commitments=request.source_commitments,
            run=public_run,
        ),
        source_root=ROOT,
    )

    implementation_report = (
        b"# M5 implementation review\n\nSynthetic aggregate package path passed.\n"
    )
    implementation_decision = ReplayImplementationReviewDecisionV1(
        schema="ffb.m5-implementation-review-decision/v1",
        reviewer_identity="dataset-free integration fixture",
        reviewed_areas=M5_IMPLEMENTATION_REVIEW_AREAS,
        findings=(),
        disposition="pass",
    )
    implementation = build_implementation_review_attestation(
        implementation_decision,
        review_report=implementation_report,
        snapshot=implementation_snapshot,
    )
    implementation_bytes = canonical_json_bytes(implementation)
    software_bytes = canonical_json_bytes(software)
    privacy = build_privacy_license_attestation(
        snapshot=implementation_snapshot,
        run_id=request.run.run_id,
    )
    privacy_bytes = canonical_json_bytes(privacy)
    plans = _plan_files(ROOT)
    authority_files = {
        **{f"machine/{path}": value for path, value in review_machine.files.items()},
        **plans,
        "evidence/implementation-review.md": implementation_report,
        "evidence/implementation-review-attestation.json": implementation_bytes,
        "evidence/software-verification.json": software_bytes,
        "evidence/privacy-license-attestation.json": privacy_bytes,
    }
    pre_review_evidence = derive_pre_review_evidence_from_parts(
        intent_bytes=review_machine.files["intent.json"],
        candidate_files=authority_files,
        profile=evidence.profile_summary,
        persistent=evidence.persistent_aggregates,
        health=evidence.health_aggregates,
        commitments=request.source_commitments,
        software=software,
        privacy=privacy,
        implementation_review=implementation,
        registry=registry,
    )
    validation_inputs = derive_pre_review_validation_inputs(
        run_id=request.run.run_id,
        evidence_by_check=pre_review_evidence,
    )
    candidate_files = _candidate_files(
        machine_files=review_machine.files,
        plan_files=plans,
        implementation_report=implementation_report,
        implementation_attestation=implementation_bytes,
        validation_inputs=validation_inputs,
        software_bytes=software_bytes,
        privacy_bytes=privacy_bytes,
        registry=registry,
        summary=summary,
        presentation=presentation,
        specs=canonical_figure_spec_files(figures.specs),
        svgs=figures.svgs,
    )
    for path, value in candidate_files.items():
        if path.endswith(".json") and path != "machine/intent.json":
            assert canonical_json_bytes(json.loads(value)) == value, path

    candidate = publish_review_candidate(
        ReplayReviewCandidateContent(
            scientific_git_revision=revision,
            lockfile_sha256=request.run.lockfile_sha256,
            package_version=request.run.package_version,
            run_id=request.run.run_id,
            replay_identity_set_sha256=request.profile_summary.replay_identity_set_sha256,
            primary_local_artifact_sha256=primary_local_artifact_sha256,
            repeat_local_artifact_sha256=repeat_local_artifact_sha256,
            primary_local_run_sha256=repeat_verification.primary_run_sha256,
            repeat_local_run_sha256=repeat_verification.repeat_run_sha256,
            files=candidate_files,
        ),
        tmp_path / "candidate",
    )

    results_report = b"# M5 results review\n\nSynthetic aggregate package path passed.\n"
    results_decision = ReplayResultsReviewDecisionV1(
        schema="ffb.m5-results-review-decision/v1",
        reviewer_identity="dataset-free integration fixture",
        findings=(),
        negative_and_undefined_results_reviewed_and_retained=True,
        limitations_reviewed_and_retained=True,
        disposition="pass",
    )
    results_attestation = build_results_review_attestation(
        results_decision,
        review_report=results_report,
        scientific_git_revision=revision,
        bindings=derive_results_review_bindings(
            candidate.index,
            candidate.index_bytes,
            candidate.files,
        ),
    )
    results_attestation_bytes = canonical_json_bytes(results_attestation)
    final_validation = derive_final_replay_validation(
        run_id=request.run.run_id,
        evidence_by_check={
            **pre_review_evidence,
            "results-and-claims-review": {
                "results-review-attestation-bytes": results_attestation_bytes
            },
        },
        pre_review_inputs=validation_inputs,
    )
    machine = prepare_replay_machine_payload(
        ReplayCuratedArtifactWriteRequest(
            profile_summary=evidence.profile_summary,
            descriptor_aggregates=evidence.descriptor_aggregates,
            persistent_aggregates=evidence.persistent_aggregates,
            persistent_crossovers=evidence.persistent_crossovers,
            health_aggregates=evidence.health_aggregates,
            cluster_sensitivity=evidence.cluster_sensitivity,
            validation=final_validation,
            repeat_verification=evidence.repeat_verification,
            figures=figures.bindings,
            source_commitments=request.source_commitments,
            run=public_run,
        ),
        source_root=ROOT,
    )
    machine_byte_length = sum(len(value) for value in machine.files.values())
    sidecars = _sidecar_files(
        candidate,
        results_review=results_report,
        results_attestation=results_attestation_bytes,
        machine_artifact_sha256=machine.artifact_sha256,
        machine_run_sha256=machine.run_sha256,
        machine_artifact_byte_length=machine_byte_length,
    )
    release_path = tmp_path / "release"
    published = publish_release_package(
        ReplayReleasePackageContent(
            reviewed_candidate_sha256=candidate.candidate_sha256,
            results_review_attestation_sha256=hashlib.sha256(results_attestation_bytes).hexdigest(),
            machine_artifact_sha256=machine.artifact_sha256,
            machine_run_sha256=machine.run_sha256,
            scientific_git_revision=revision,
            machine_files=machine.files,
            sidecar_files=sidecars,
        ),
        release_path,
    )

    reloaded = load_release_package(release_path)
    validated = validate_release_package(release_path)

    assert reloaded.release_package_sha256 == published.release_package_sha256
    assert validated.release_package_sha256 == published.release_package_sha256
    assert validated.artifact.validation == final_validation
    assert validated.candidate_index == candidate.index
    assert validated.public_claim_projections == registry
