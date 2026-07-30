"""Final, review-gated construction of the frozen M5 release package.

This module consumes only authenticated aggregate artifacts and review evidence.
It never opens the nuScenes dataset and it never chooses a review disposition.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_FIGURE_PATHS,
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_DESTINATION_PATH,
    M5_RELEASE_SIDECAR_INDEXED_PATHS,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
    ReplayFigureSourceBindingV1,
    ReplayPublicClaimProjectionsV1,
    ReplayValidationInputsV1,
)
from fusion_fault_bench.provenance import CleanSourceSnapshot
from fusion_fault_bench.replay_artifacts import (
    ReplayCuratedArtifactWriteRequest,
    prepare_replay_machine_payload,
)
from fusion_fault_bench.replay_release import (
    LoadedReplayReviewCandidate,
    ReplayReleasePackageContent,
    publish_release_package,
)
from fusion_fault_bench.replay_release_authority import ImplementationSnapshot
from fusion_fault_bench.replay_release_candidate import (
    load_validated_review_candidate,
    prepare_review_candidate,
)
from fusion_fault_bench.replay_release_evidence_bridge import (
    derive_pre_review_evidence_from_parts,
)
from fusion_fault_bench.replay_release_package import (
    ValidatedReplayReleasePackage,
    derive_results_review_bindings,
    validate_release_package,
)
from fusion_fault_bench.replay_release_validation import (
    derive_final_replay_validation,
    load_implementation_review_attestation,
    load_pre_review_validation_inputs,
    load_privacy_license_attestation,
    load_results_review_attestation,
    load_software_verification,
)
from fusion_fault_bench.replay_runner import (
    curate_replay_verified_repeat,
    verify_replay_repeat_artifacts,
)

M5_RELEASE_DESTINATION = Path(M5_RELEASE_DESTINATION_PATH)
_RESULTS_REVIEW_MAX_BYTES = 1024 * 1024
_RESULTS_ATTESTATION_MAX_BYTES = 4 * 1024 * 1024
_TEMPLATE_TO_RELEASE = {
    "presentation/README.md": "README.md",
    "presentation/claim-evidence.md": "claim-evidence.md",
    "presentation/verification.md": "verification.md",
}


class ReplayReleaseBuildError(ValueError):
    """Final release construction failed before release authority was established."""


def _require_matching_publication_digest(
    *,
    expected: str,
    published: str,
    validated: str,
) -> None:
    if published != expected or validated != expected:
        raise ReplayReleaseBuildError(
            "M5 final publication digest differs from the validated preflight"
        )


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_bounded_regular(path: Path, *, byte_cap: int, label: str) -> bytes:
    """Read one no-follow input and prove its path did not change around the read."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > byte_cap
        ):
            raise ReplayReleaseBuildError(f"M5 {label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ReplayReleaseBuildError(f"M5 {label} changed during reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ReplayReleaseBuildError(f"M5 {label} exceeds its declared size")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reopened = os.lstat(absolute)
    if _fingerprint(before) != _fingerprint(after) or _fingerprint(before) != _fingerprint(
        reopened
    ):
        raise ReplayReleaseBuildError(f"M5 {label} changed during reading")
    return b"".join(chunks)


def _require_source_authority(
    *,
    source_root: Path,
    clean_snapshot: CleanSourceSnapshot,
    implementation_snapshot: ImplementationSnapshot,
) -> Path:
    root = Path(os.path.abspath(os.fspath(source_root)))
    if (
        clean_snapshot.source_root != root
        or clean_snapshot.git_revision != implementation_snapshot.scientific_git_revision
    ):
        raise ReplayReleaseBuildError("M5 clean source authorities disagree")
    return root


def _require_release_destination(output_dir: Path, source_root: Path) -> None:
    expected_absolute = source_root / M5_RELEASE_DESTINATION
    observed_absolute = Path(os.path.abspath(os.fspath(output_dir)))
    if output_dir != M5_RELEASE_DESTINATION and observed_absolute != expected_absolute:
        raise ReplayReleaseBuildError("M5 release destination is not the frozen package path")


def _require_identical_candidate(
    reviewed: LoadedReplayReviewCandidate,
    regenerated: LoadedReplayReviewCandidate,
) -> None:
    if (
        reviewed.index != regenerated.index
        or reviewed.index_bytes != regenerated.index_bytes
        or dict(reviewed.files) != dict(regenerated.files)
        or reviewed.candidate_sha256 != regenerated.candidate_sha256
        or reviewed.candidate_index_sha256 != regenerated.candidate_index_sha256
    ):
        raise ReplayReleaseBuildError(
            "M5 original inputs do not regenerate the exact reviewed candidate"
        )


def _canonical_figure_bindings(value: bytes) -> tuple[ReplayFigureSourceBindingV1, ...]:
    if not value or not value.endswith(b"\n"):
        raise ReplayReleaseBuildError("M5 figure bindings are not canonical NDJSON")
    rows: list[ReplayFigureSourceBindingV1] = []
    try:
        for line in value.splitlines(keepends=True):
            row = ReplayFigureSourceBindingV1.model_validate_json(line)
            if canonical_json_bytes(row) != line:
                raise ReplayReleaseBuildError("M5 figure bindings are not canonical NDJSON")
            rows.append(row)
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseBuildError("M5 figure bindings violate their contract") from error
    return tuple(rows)


def _canonical_claim_registry(value: bytes) -> ReplayPublicClaimProjectionsV1:
    try:
        registry = ReplayPublicClaimProjectionsV1.model_validate_json(value)
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseBuildError("M5 public claim registry violates its contract") from error
    if canonical_json_bytes(registry) != value:
        raise ReplayReleaseBuildError("M5 public claim registry is not canonical")
    return registry


def _canonical_validation_inputs(value: bytes) -> ReplayValidationInputsV1:
    try:
        inputs = ReplayValidationInputsV1.model_validate_json(value)
    except (ValueError, ValidationError) as error:
        raise ReplayReleaseBuildError("M5 validation inputs violate their contract") from error
    if canonical_json_bytes(inputs) != value:
        raise ReplayReleaseBuildError("M5 validation inputs are not canonical")
    return inputs


def _substitute_reviewed_templates(
    candidate_files: Mapping[str, bytes],
    *,
    machine_artifact_sha256: str,
    machine_run_sha256: str,
    results_review_attestation_sha256: str,
    machine_artifact_byte_length: int,
) -> dict[str, bytes]:
    replacements = (
        machine_artifact_sha256,
        machine_run_sha256,
        results_review_attestation_sha256,
        str(machine_artifact_byte_length),
    )
    output: dict[str, bytes] = {}
    for candidate_path, release_path in _TEMPLATE_TO_RELEASE.items():
        final = candidate_files[candidate_path]
        for placeholder, replacement in zip(
            M5_PRESENTATION_PLACEHOLDERS,
            replacements,
            strict=True,
        ):
            token = placeholder.encode("ascii")
            if final.count(token) != 1:
                raise ReplayReleaseBuildError(
                    "M5 reviewed presentation does not contain the exact placeholders"
                )
            final = final.replace(token, replacement.encode("ascii"), 1)
        output[release_path] = final
    return output


def _sidecar_files(
    candidate: LoadedReplayReviewCandidate,
    *,
    results_review: bytes,
    results_attestation: bytes,
    machine_artifact_sha256: str,
    machine_run_sha256: str,
    machine_artifact_byte_length: int,
) -> dict[str, bytes]:
    attestation_sha256 = hashlib.sha256(results_attestation).hexdigest()
    unordered = {
        **_substitute_reviewed_templates(
            candidate.files,
            machine_artifact_sha256=machine_artifact_sha256,
            machine_run_sha256=machine_run_sha256,
            results_review_attestation_sha256=attestation_sha256,
            machine_artifact_byte_length=machine_artifact_byte_length,
        ),
        "release-summary.json": candidate.files["presentation/release-summary.json"],
        **{path: candidate.files[path] for path in M5_FIGURE_PATHS},
        "evidence/release-pipeline-plan.md": candidate.files["evidence/release-pipeline-plan.md"],
        "evidence/release-pipeline-plan-review.md": candidate.files[
            "evidence/release-pipeline-plan-review.md"
        ],
        "evidence/resource-scope-amendment.md": candidate.files[
            "evidence/resource-scope-amendment.md"
        ],
        "evidence/implementation-review.md": candidate.files["evidence/implementation-review.md"],
        "evidence/review-candidate-index.json": candidate.index_bytes,
        "evidence/validation-inputs.json": candidate.files["evidence/validation-inputs.json"],
        "evidence/implementation-review-attestation.json": candidate.files[
            "evidence/implementation-review-attestation.json"
        ],
        "evidence/software-verification.json": candidate.files[
            "evidence/software-verification.json"
        ],
        "evidence/privacy-license-attestation.json": candidate.files[
            "evidence/privacy-license-attestation.json"
        ],
        "evidence/public-claim-projections.json": candidate.files[
            "presentation/public-claim-projections.json"
        ],
        "evidence/results-review.md": results_review,
        "evidence/results-review-attestation.json": results_attestation,
    }
    try:
        return {path: unordered[path] for path in M5_RELEASE_SIDECAR_INDEXED_PATHS}
    except KeyError as error:
        raise ReplayReleaseBuildError("M5 final sidecar construction omitted a member") from error


def _build_reviewed_release(
    *,
    candidate: Path,
    results_review: Path,
    results_review_attestation: Path,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
    output_dir: Path,
    source_root: Path,
    clean_snapshot: CleanSourceSnapshot,
    implementation_snapshot: ImplementationSnapshot,
    implementation_report: bytes,
    implementation_attestation: bytes,
    prepublish_authority: Callable[[], None],
) -> ValidatedReplayReleasePackage:
    root = _require_source_authority(
        source_root=source_root,
        clean_snapshot=clean_snapshot,
        implementation_snapshot=implementation_snapshot,
    )
    _require_release_destination(output_dir, root)
    reviewed = load_validated_review_candidate(
        path=candidate,
        source_root=root,
        clean_snapshot=clean_snapshot,
        implementation_snapshot=implementation_snapshot,
        implementation_report=implementation_report,
        implementation_attestation=implementation_attestation,
    )

    with tempfile.TemporaryDirectory(prefix="ffb-m5-release-regeneration-") as temporary:
        temporary_root = Path(temporary)
        regenerated_path = temporary_root / "candidate"
        prepare_review_candidate(
            primary_artifact=primary_artifact,
            repeat_artifact=repeat_artifact,
            primary_time_l=primary_time_l,
            repeat_time_l=repeat_time_l,
            software_verification=software_verification,
            output_dir=regenerated_path,
            source_root=root,
            clean_snapshot=clean_snapshot,
            implementation_snapshot=implementation_snapshot,
            implementation_report=implementation_report,
            implementation_attestation=implementation_attestation,
        )
        regenerated = load_validated_review_candidate(
            path=regenerated_path,
            source_root=root,
            clean_snapshot=clean_snapshot,
            implementation_snapshot=implementation_snapshot,
            implementation_report=implementation_report,
            implementation_attestation=implementation_attestation,
        )
        _require_identical_candidate(reviewed, regenerated)

        report_bytes = _read_bounded_regular(
            results_review,
            byte_cap=_RESULTS_REVIEW_MAX_BYTES,
            label="results review report",
        )
        attestation_bytes = _read_bounded_regular(
            results_review_attestation,
            byte_cap=_RESULTS_ATTESTATION_MAX_BYTES,
            label="results review attestation",
        )
        bindings = derive_results_review_bindings(
            reviewed.index,
            reviewed.index_bytes,
            reviewed.files,
        )
        load_results_review_attestation(
            attestation_bytes,
            review_report=report_bytes,
            scientific_git_revision=clean_snapshot.git_revision,
            bindings=bindings,
            require_release_permitting=True,
        )

        repeat_evidence = verify_replay_repeat_artifacts(
            primary_path=primary_artifact,
            repeat_path=repeat_artifact,
        )
        curated = curate_replay_verified_repeat(
            repeat_evidence,
            primary_log_path=primary_time_l,
            repeat_log_path=repeat_time_l,
        )
        software = load_software_verification(
            reviewed.files["evidence/software-verification.json"],
            snapshot=implementation_snapshot,
            lockfile_sha256=clean_snapshot.lockfile_sha256,
            package_version=clean_snapshot.package_version,
        )
        implementation = load_implementation_review_attestation(
            implementation_attestation,
            review_report=implementation_report,
            snapshot=implementation_snapshot,
            require_release_permitting=True,
        )
        privacy = load_privacy_license_attestation(
            reviewed.files["evidence/privacy-license-attestation.json"],
            snapshot=implementation_snapshot,
            run_id=curated.run.run_id,
        )
        registry = _canonical_claim_registry(
            reviewed.files["presentation/public-claim-projections.json"]
        )
        pre_review_evidence = derive_pre_review_evidence_from_parts(
            intent_bytes=reviewed.files["machine/intent.json"],
            candidate_files=reviewed.files,
            profile=curated.profile_summary,
            persistent=curated.persistent_aggregates,
            health=curated.health_aggregates,
            commitments=repeat_evidence.source_commitments,
            software=software,
            privacy=privacy,
            implementation_review=implementation,
            registry=registry,
        )
        pre_review_inputs = load_pre_review_validation_inputs(
            reviewed.files["evidence/validation-inputs.json"],
            run_id=curated.run.run_id,
            evidence_by_check=pre_review_evidence,
        )
        if pre_review_inputs != _canonical_validation_inputs(
            reviewed.files["evidence/validation-inputs.json"]
        ):
            raise ReplayReleaseBuildError("M5 validation-input parsing is inconsistent")
        final_evidence = {
            **pre_review_evidence,
            "results-and-claims-review": {"results-review-attestation-bytes": attestation_bytes},
        }
        final_validation = derive_final_replay_validation(
            run_id=curated.run.run_id,
            evidence_by_check=final_evidence,
            pre_review_inputs=pre_review_inputs,
        )
        figures = _canonical_figure_bindings(reviewed.files["machine/figure-records.ndjson"])
        machine = prepare_replay_machine_payload(
            ReplayCuratedArtifactWriteRequest(
                profile_summary=curated.profile_summary,
                descriptor_aggregates=curated.descriptor_aggregates,
                persistent_aggregates=curated.persistent_aggregates,
                persistent_crossovers=curated.persistent_crossovers,
                health_aggregates=curated.health_aggregates,
                cluster_sensitivity=curated.cluster_sensitivity,
                validation=final_validation,
                repeat_verification=repeat_evidence.repeat_verification,
                figures=figures,
                source_commitments=repeat_evidence.source_commitments,
                run=curated.run,
            ),
            source_root=root,
        )
        for candidate_path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[:10]:
            machine_path = candidate_path.removeprefix("machine/")
            if machine.files[machine_path] != reviewed.files[candidate_path]:
                raise ReplayReleaseBuildError(
                    "M5 final machine artifact differs from reviewed scientific bytes"
                )

        machine_byte_length = sum(len(value) for value in machine.files.values())
        attestation_sha256 = hashlib.sha256(attestation_bytes).hexdigest()
        sidecars = _sidecar_files(
            reviewed,
            results_review=report_bytes,
            results_attestation=attestation_bytes,
            machine_artifact_sha256=machine.artifact_sha256,
            machine_run_sha256=machine.run_sha256,
            machine_artifact_byte_length=machine_byte_length,
        )
        content = ReplayReleasePackageContent(
            reviewed_candidate_sha256=reviewed.candidate_sha256,
            results_review_attestation_sha256=attestation_sha256,
            machine_artifact_sha256=machine.artifact_sha256,
            machine_run_sha256=machine.run_sha256,
            scientific_git_revision=clean_snapshot.git_revision,
            machine_files=machine.files,
            sidecar_files=sidecars,
        )

        preflight_path = temporary_root / "release"
        preflight = publish_release_package(content, preflight_path)
        preflight_validated = validate_release_package(preflight_path)
        if preflight_validated.release_package_sha256 != preflight.release_package_sha256:
            raise ReplayReleaseBuildError("M5 preflight package validation digest changed")
        expected_release_package_sha256 = preflight_validated.release_package_sha256

    prepublish_authority()
    published = publish_release_package(content, output_dir)
    validated = validate_release_package(output_dir)
    _require_matching_publication_digest(
        expected=expected_release_package_sha256,
        published=published.release_package_sha256,
        validated=validated.release_package_sha256,
    )
    return validated


def build_reviewed_release(
    *,
    candidate: Path,
    results_review: Path,
    results_review_attestation: Path,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
    output_dir: Path,
    source_root: Path,
    clean_snapshot: CleanSourceSnapshot,
    implementation_snapshot: ImplementationSnapshot,
    implementation_report: bytes,
    implementation_attestation: bytes,
    prepublish_authority: Callable[[], None],
) -> ValidatedReplayReleasePackage:
    """Rebuild, review-gate, atomically publish, and strictly validate M5."""

    try:
        return _build_reviewed_release(
            candidate=candidate,
            results_review=results_review,
            results_review_attestation=results_review_attestation,
            primary_artifact=primary_artifact,
            repeat_artifact=repeat_artifact,
            primary_time_l=primary_time_l,
            repeat_time_l=repeat_time_l,
            software_verification=software_verification,
            output_dir=output_dir,
            source_root=source_root,
            clean_snapshot=clean_snapshot,
            implementation_snapshot=implementation_snapshot,
            implementation_report=implementation_report,
            implementation_attestation=implementation_attestation,
            prepublish_authority=prepublish_authority,
        )
    except ReplayReleaseBuildError:
        raise
    except (OSError, RuntimeError, TypeError, ValidationError, ValueError) as error:
        raise ReplayReleaseBuildError("M5 reviewed release construction failed") from error


__all__ = [
    "M5_RELEASE_DESTINATION",
    "ReplayReleaseBuildError",
    "build_reviewed_release",
]
