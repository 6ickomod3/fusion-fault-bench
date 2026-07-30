from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m5_review_candidate import _candidate_content

import fusion_fault_bench.replay_release_package as release_package
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_DESTINATION_PATH,
    M5_SOFTWARE_VERIFICATION_CATEGORIES,
    ReplaySoftwareVerificationCheckV1,
    ReplaySoftwareVerificationV1,
)
from fusion_fault_bench.replay_release import build_review_candidate_files
from fusion_fault_bench.replay_release_package import (
    M5_PUBLICATION_DOCUMENT_PATHS,
    ReplayReleasePackageValidationError,
    derive_results_review_bindings,
    reconstruct_reviewed_candidate,
    validate_publication,
)
from fusion_fault_bench.replay_release_software import M5_SOFTWARE_COMMAND_BY_CHECK
from fusion_fault_bench.replay_release_validation import (
    M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK,
    M5_SOFTWARE_VERIFICATION_CHECK_IDS,
)

_ARTIFACT_SHA256 = "e" * 64
_RUN_SHA256 = "f" * 64
_ATTESTATION_SHA256 = "9" * 64
_MACHINE_BYTES = 12345

ROOT = Path(__file__).resolve().parents[1]


def _software_verification() -> ReplaySoftwareVerificationV1:
    revision = "1" * 40
    checks = tuple(
        ReplaySoftwareVerificationCheckV1(
            check_id=check_id,
            category=category,
            command=M5_SOFTWARE_COMMAND_BY_CHECK[check_id],
            required_test_ids=M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[check_id],
            exit_status=0,
            output_sha256=hashlib.sha256(check_id.encode()).hexdigest(),
            output_normalization=("stable-command-output-with-runtime-paths-and-durations-removed"),
        )
        for check_id, category in zip(
            M5_SOFTWARE_VERIFICATION_CHECK_IDS,
            M5_SOFTWARE_VERIFICATION_CATEGORIES,
            strict=True,
        )
    )
    return ReplaySoftwareVerificationV1(
        schema="ffb.m5-software-verification/v1",
        release_id="m5-nuscenes-replay-v0.1.0",
        scientific_git_revision=revision,
        lockfile_sha256="2" * 64,
        package_version="0.1.0",
        implementation_snapshot_sha256="3" * 64,
        tooling_revision=revision,
        checks=checks,
    )


def test_frozen_methodology_digests_match_exact_tracked_authorities() -> None:
    source_by_candidate_path = {
        "evidence/release-pipeline-plan.md": ROOT / "docs/m5-release-pipeline-plan.md",
        "evidence/release-pipeline-plan-review.md": (
            ROOT / "docs/reviews/m5-release-pipeline-plan-review.md"
        ),
        "evidence/resource-scope-amendment.md": (ROOT / "docs/m5-resource-scope-amendment.md"),
    }

    assert set(release_package._FROZEN_METHODOLOGY_SHA256) == set(source_by_candidate_path)
    for candidate_path, source in source_by_candidate_path.items():
        expected = release_package._FROZEN_METHODOLOGY_SHA256[candidate_path]
        assert len(expected) == 64
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


def test_package_software_authority_requires_exact_command_for_each_check() -> None:
    software = _software_verification()
    release_package._require_software_verification_authority(software)

    tampered_check = software.checks[0].model_copy(
        update={"command": (*software.checks[0].command[:-1], "not-the-repository-root")}
    )
    tampered = software.model_copy(update={"checks": (tampered_check, *software.checks[1:])})
    with pytest.raises(ReplayReleasePackageValidationError, match="command differs"):
        release_package._require_software_verification_authority(tampered)


def _final_package_projection() -> tuple[dict[str, bytes], dict[str, bytes]]:
    candidate_tree = build_review_candidate_files(_candidate_content())
    candidate_files = {
        path: value for path, value in candidate_tree.items() if path != "candidate-index.json"
    }
    package_files: dict[str, bytes] = {
        "evidence/review-candidate-index.json": candidate_tree["candidate-index.json"]
    }
    replacements = (
        _ARTIFACT_SHA256,
        _RUN_SHA256,
        _ATTESTATION_SHA256,
        str(_MACHINE_BYTES),
    )
    for candidate_path, value in candidate_files.items():
        if candidate_path in release_package._PRESENTATION_TO_RELEASE:
            final = value
            for placeholder, replacement in zip(
                M5_PRESENTATION_PLACEHOLDERS,
                replacements,
                strict=True,
            ):
                final = final.replace(placeholder.encode(), replacement.encode(), 1)
            package_files[release_package._PRESENTATION_TO_RELEASE[candidate_path]] = final
        else:
            package_files[release_package._CANDIDATE_TO_RELEASE[candidate_path]] = value
    return package_files, candidate_files


def test_reconstructs_exact_reviewed_candidate_and_bindings() -> None:
    package_files, expected_files = _final_package_projection()

    index, observed_files = reconstruct_reviewed_candidate(
        package_files,
        machine_artifact_sha256=_ARTIFACT_SHA256,
        machine_run_sha256=_RUN_SHA256,
        results_review_attestation_sha256=_ATTESTATION_SHA256,
        machine_artifact_byte_length=_MACHINE_BYTES,
    )

    assert dict(observed_files) == expected_files
    bindings = derive_results_review_bindings(
        index,
        package_files["evidence/review-candidate-index.json"],
        observed_files,
    )
    assert bindings.candidate_sha256 == index.candidate_sha256
    assert bindings.claim_projection_sha256 == next(
        entry.sha256
        for entry in index.files
        if entry.path == "presentation/public-claim-projections.json"
    )


def test_candidate_reconstruction_rejects_non_substitution_tamper() -> None:
    package_files, _ = _final_package_projection()
    package_files["README.md"] += b"Unreviewed outcome sentence.\n"

    with pytest.raises(
        ReplayReleasePackageValidationError,
        match="differs from an indexed reviewed candidate",
    ):
        reconstruct_reviewed_candidate(
            package_files,
            machine_artifact_sha256=_ARTIFACT_SHA256,
            machine_run_sha256=_RUN_SHA256,
            results_review_attestation_sha256=_ATTESTATION_SHA256,
            machine_artifact_byte_length=_MACHINE_BYTES,
        )


def test_candidate_reconstruction_rejects_missing_identity_substitution() -> None:
    package_files, _ = _final_package_projection()
    package_files["verification.md"] = package_files["verification.md"].replace(
        _RUN_SHA256.encode(),
        b"0" * 64,
        1,
    )

    with pytest.raises(ReplayReleasePackageValidationError, match="four allowed"):
        reconstruct_reviewed_candidate(
            package_files,
            machine_artifact_sha256=_ARTIFACT_SHA256,
            machine_run_sha256=_RUN_SHA256,
            results_review_attestation_sha256=_ATTESTATION_SHA256,
            machine_artifact_byte_length=_MACHINE_BYTES,
        )


def _write_publication_fixture(
    root: Path,
    *,
    package_sha256: str,
    claim_sha256: str,
) -> dict[str, bytes]:
    evidence = {
        "evidence/release-pipeline-plan.md": b"# Plan\n",
        "evidence/release-pipeline-plan-review.md": b"# Plan review\n",
        "evidence/resource-scope-amendment.md": b"# Scope\n",
        "evidence/implementation-review.md": b"# Implementation review\n",
        "evidence/implementation-review-attestation.json": b"{}\n",
        "evidence/results-review.md": b"# Results review\n",
        "evidence/results-review-attestation.json": b"{}\n",
    }
    copies = {
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
    for path, package_path in copies.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(evidence[package_path])
    projection = (
        "m5-nuscenes-replay-v0.1.0\n"
        f"release-package-sha256: {package_sha256}\n"
        f"claim-projection-sha256: {claim_sha256}\n"
    ).encode()
    for path in M5_PUBLICATION_DOCUMENT_PATHS:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(projection)
    return evidence


def test_publication_requires_exact_review_copies_and_package_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_sha256 = "a" * 64
    claim_sha256 = "b" * 64
    evidence = _write_publication_fixture(
        tmp_path,
        package_sha256=package_sha256,
        claim_sha256=claim_sha256,
    )
    validated = SimpleNamespace(
        package=SimpleNamespace(files=evidence),
        release_package_sha256=package_sha256,
        claim_projection_sha256=claim_sha256,
    )
    monkeypatch.setattr(release_package, "validate_release_package", lambda _path: validated)

    release = tmp_path / M5_RELEASE_DESTINATION_PATH
    assert validate_publication(release, tmp_path) == package_sha256

    (tmp_path / "docs/reviews/m5-results-review.md").write_bytes(b"# Changed review\n")
    with pytest.raises(ReplayReleasePackageValidationError, match="differs"):
        validate_publication(release, tmp_path)


def test_publication_rejects_document_not_bound_to_claim_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_sha256 = "c" * 64
    claim_sha256 = "d" * 64
    evidence = _write_publication_fixture(
        tmp_path,
        package_sha256=package_sha256,
        claim_sha256=claim_sha256,
    )
    validated = SimpleNamespace(
        package=SimpleNamespace(files=evidence),
        release_package_sha256=package_sha256,
        claim_projection_sha256=claim_sha256,
    )
    monkeypatch.setattr(release_package, "validate_release_package", lambda _path: validated)
    (tmp_path / "docs/results.md").write_bytes(b"m5-nuscenes-replay-v0.1.0\n")

    with pytest.raises(ReplayReleasePackageValidationError, match="exact package projection"):
        validate_publication(tmp_path / M5_RELEASE_DESTINATION_PATH, tmp_path)


def test_publication_rejects_package_outside_frozen_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_package,
        "validate_release_package",
        lambda _path: pytest.fail("semantic validation must not inspect an external package"),
    )

    with pytest.raises(ReplayReleasePackageValidationError, match="frozen tracked"):
        validate_publication(tmp_path / "external-release", tmp_path)
