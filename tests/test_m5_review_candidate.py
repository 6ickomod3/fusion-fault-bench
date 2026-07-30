from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_REPLAY_IDENTITY_SET_SHA256,
    REPLAY_ARTIFACT_PATHS,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_PACKAGE_PATHS,
    M5_RELEASE_SIDECAR_INDEXED_PATHS,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
    M5_REVIEW_CANDIDATE_PATHS,
    ReplayResultsReviewDecisionV1,
)
from fusion_fault_bench.replay_release import (
    LoadedReplayReleasePackage,
    LoadedReplayReviewCandidate,
    ReplayReleaseError,
    ReplayReleasePackageContent,
    ReplayReviewCandidateContent,
    attest_results_review,
    build_release_package_files,
    build_review_candidate_files,
    load_release_package,
    load_review_candidate,
    publish_release_package,
    publish_review_candidate,
    sync_reviewed_evidence,
)
from fusion_fault_bench.replay_review_sync import ReplayReviewSyncError

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64
_FROZEN_INTENT = (
    Path(__file__).resolve().parents[1] / "examples/replay/m5-nuscenes-mini-replay-v1.json"
).read_bytes()


def _json(path: str) -> bytes:
    return canonical_json_bytes({"path": path, "value": 1})


def _candidate_member(path: str) -> bytes:
    if path == "machine/intent.json":
        return _FROZEN_INTENT
    if path.endswith(".ndjson"):
        return canonical_json_bytes({"path": path, "row": 1})
    if path.endswith(".json"):
        return _json(path)
    if path.endswith(".svg"):
        return b'<svg xmlns="http://www.w3.org/2000/svg"><text>fixed figure</text></svg>\n'
    if path.startswith("presentation/"):
        placeholders = "\n".join(f"`{value}`" for value in M5_PRESENTATION_PLACEHOLDERS)
        return f"# Reviewed M5 template\n\n{placeholders}\n".encode()
    return b"# Frozen M5 evidence\n"


def _candidate_content() -> ReplayReviewCandidateContent:
    return ReplayReviewCandidateContent(
        scientific_git_revision="1" * 40,
        lockfile_sha256="2" * 64,
        package_version="0.1.0",
        run_id="run:m5-test",
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        primary_local_artifact_sha256=_DIGEST_A,
        repeat_local_artifact_sha256=_DIGEST_B,
        primary_local_run_sha256=_DIGEST_C,
        repeat_local_run_sha256=_DIGEST_D,
        files={path: _candidate_member(path) for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS},
    )


def _package_member(path: str) -> bytes:
    if path == "intent.json":
        return _FROZEN_INTENT
    if path.endswith(".ndjson"):
        return canonical_json_bytes({"path": path, "row": 1})
    if path.endswith(".json") or path == "_SUCCESS":
        return _json(path)
    if path.endswith(".svg"):
        return b'<svg xmlns="http://www.w3.org/2000/svg"><text>fixed figure</text></svg>\n'
    return b"# Reviewed M5 release evidence\n"


def _package_content() -> ReplayReleasePackageContent:
    return ReplayReleasePackageContent(
        reviewed_candidate_sha256=_DIGEST_A,
        results_review_attestation_sha256=_DIGEST_B,
        machine_artifact_sha256=_DIGEST_C,
        machine_run_sha256=_DIGEST_D,
        scientific_git_revision="1" * 40,
        machine_files={path: _package_member(path) for path in REPLAY_ARTIFACT_PATHS},
        sidecar_files={path: _package_member(path) for path in M5_RELEASE_SIDECAR_INDEXED_PATHS},
    )


def _sync_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LoadedReplayReleasePackage, Path, Path, Path]:
    destination = tmp_path / "release"
    loaded = publish_release_package(_package_content(), destination)
    reloaded = load_release_package(destination)

    class ValidatedPackage:
        package = reloaded

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_package.validate_release_package",
        lambda _path: ValidatedPackage(),
    )
    source_root = tmp_path / "source"
    (source_root / "docs/reviews").mkdir(parents=True)
    subprocess.run(("git", "init", "-q", source_root), check=True)
    return (
        loaded,
        source_root,
        Path("docs/reviews/m5-results-review.md"),
        Path("docs/reviews/m5-results-review-attestation.json"),
    )


def test_review_candidate_build_publish_and_reload(tmp_path: Path) -> None:
    content = _candidate_content()
    files = build_review_candidate_files(content)

    assert tuple(files) == M5_REVIEW_CANDIDATE_PATHS
    assert len(files) == 34
    destination = tmp_path / "candidate"
    loaded = publish_review_candidate(content, destination)
    reloaded = load_review_candidate(destination)

    assert loaded.candidate_sha256 == reloaded.candidate_sha256
    assert loaded.index.results_review_status == "pending"
    assert len(loaded.index.files) == 33
    assert tuple(entry.path for entry in loaded.index.files) == M5_REVIEW_CANDIDATE_INDEXED_PATHS
    with pytest.raises(FileExistsError):
        publish_review_candidate(content, destination)


def test_review_candidate_rejects_order_tampering_and_member_mutation(tmp_path: Path) -> None:
    content = _candidate_content()
    reversed_files = dict(reversed(tuple(content.files.items())))
    with pytest.raises(ReplayReleaseError, match="order"):
        build_review_candidate_files(replace(content, files=reversed_files))

    destination = tmp_path / "candidate"
    publish_review_candidate(content, destination)
    target = destination / "machine" / "replay-profile-summary.json"
    target.write_bytes(_json("changed"))
    with pytest.raises(ReplayReleaseError, match="changed"):
        load_review_candidate(destination)


def test_results_review_attestation_binds_exact_candidate(tmp_path: Path) -> None:
    candidate = publish_review_candidate(_candidate_content(), tmp_path / "candidate")
    decision = ReplayResultsReviewDecisionV1(
        schema="ffb.m5-results-review-decision/v1",
        reviewer_identity="independent-reviewer",
        findings=(),
        negative_and_undefined_results_reviewed_and_retained=True,
        limitations_reviewed_and_retained=True,
        disposition="pass",
    )

    attestation = attest_results_review(
        candidate,
        review_report=b"# Independent M5 results review\n",
        decision=decision,
    )

    assert attestation.candidate_sha256 == candidate.candidate_sha256
    assert attestation.candidate_index_sha256 == candidate.candidate_index_sha256
    assert attestation.p0_count == attestation.p1_count == attestation.p2_count == 0
    assert attestation.disposition == "pass"


@pytest.mark.parametrize(
    "field",
    (
        "negative_and_undefined_results_reviewed_and_retained",
        "limitations_reviewed_and_retained",
    ),
)
def test_results_review_attestation_never_overrides_retention_decision(field: str) -> None:
    candidate = cast(LoadedReplayReviewCandidate, object())
    decision = ReplayResultsReviewDecisionV1(
        schema="ffb.m5-results-review-decision/v1",
        reviewer_identity="independent-reviewer",
        findings=(),
        negative_and_undefined_results_reviewed_and_retained=True,
        limitations_reviewed_and_retained=True,
        disposition="pass",
    ).model_copy(update={field: False})

    with pytest.raises(ValueError):
        attest_results_review(
            candidate,
            review_report=b"# Independent M5 results review\n",
            decision=decision,
        )


def test_outer_release_package_build_publish_reload_and_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _package_content()
    files = build_release_package_files(content)

    assert tuple(files) == M5_RELEASE_PACKAGE_PATHS
    assert len(files) == 41
    destination = tmp_path / "release"
    loaded = publish_release_package(content, destination)
    reloaded = load_release_package(destination)

    assert loaded.release_package_sha256 == reloaded.release_package_sha256
    assert len(loaded.index.files) == 26
    assert loaded.index.indexed_sidecar_payload_byte_length == sum(
        entry.byte_length for entry in loaded.index.files
    )

    validated_paths: list[Path] = []

    def validate_semantically(path: Path) -> object:
        validated_paths.append(path)
        return type("ValidatedPackage", (), {"package": reloaded})()

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_package.validate_release_package",
        validate_semantically,
    )
    source_root = tmp_path / "source"
    (source_root / "docs/reviews").mkdir(parents=True)
    subprocess.run(("git", "init", "-q", source_root), check=True)
    report = Path("docs/reviews/m5-results-review.md")
    attestation = Path("docs/reviews/m5-results-review-attestation.json")
    sync_reviewed_evidence(
        loaded,
        report_output=report,
        attestation_output=attestation,
        source_root=source_root,
    )
    assert (source_root / report).read_bytes() == loaded.files["evidence/results-review.md"]
    assert (source_root / attestation).read_bytes() == loaded.files[
        "evidence/results-review-attestation.json"
    ]
    assert validated_paths == [destination]
    sync_reviewed_evidence(
        loaded,
        report_output=report,
        attestation_output=attestation,
        source_root=source_root,
    )
    assert validated_paths == [destination, destination]
    assert not (source_root / "docs/reviews/.ffb-m5-reviewed-evidence-staging").exists()


@pytest.mark.parametrize("existing_name", ("report", "attestation"))
def test_sync_resumes_an_exact_partial_public_pair(
    existing_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, source_root, report, attestation = _sync_fixture(tmp_path, monkeypatch)
    values = {
        "report": loaded.files["evidence/results-review.md"],
        "attestation": loaded.files["evidence/results-review-attestation.json"],
    }
    paths = {"report": report, "attestation": attestation}
    (source_root / paths[existing_name]).write_bytes(values[existing_name])

    sync_reviewed_evidence(
        loaded,
        report_output=report,
        attestation_output=attestation,
        source_root=source_root,
    )

    assert (source_root / report).read_bytes() == values["report"]
    assert (source_root / attestation).read_bytes() == values["attestation"]
    assert not (source_root / "docs/reviews/.ffb-m5-reviewed-evidence-staging").exists()


def test_sync_resumes_after_interruption_between_publications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fusion_fault_bench import replay_review_sync

    loaded, source_root, report, attestation = _sync_fixture(tmp_path, monkeypatch)
    original_rename = replay_review_sync.atomic_rename_directory_no_replace_at
    calls = 0

    def interrupt_second_publication(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption")
        original_rename(*args)

    monkeypatch.setattr(
        replay_review_sync,
        "atomic_rename_directory_no_replace_at",
        interrupt_second_publication,
    )
    with pytest.raises(OSError, match="simulated interruption"):
        sync_reviewed_evidence(
            loaded,
            report_output=report,
            attestation_output=attestation,
            source_root=source_root,
        )

    assert (source_root / report).read_bytes() == loaded.files["evidence/results-review.md"]
    assert not (source_root / attestation).exists()
    assert (source_root / "docs/reviews/.ffb-m5-reviewed-evidence-staging").is_dir()

    monkeypatch.setattr(
        replay_review_sync,
        "atomic_rename_directory_no_replace_at",
        original_rename,
    )
    sync_reviewed_evidence(
        loaded,
        report_output=report,
        attestation_output=attestation,
        source_root=source_root,
    )

    assert (source_root / attestation).read_bytes() == loaded.files[
        "evidence/results-review-attestation.json"
    ]
    assert not (source_root / "docs/reviews/.ffb-m5-reviewed-evidence-staging").exists()


def test_sync_rejects_mismatched_existing_public_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, source_root, report, attestation = _sync_fixture(tmp_path, monkeypatch)
    (source_root / report).write_bytes(b"not the reviewed package bytes\n")

    with pytest.raises(ReplayReviewSyncError, match="differs from the release package"):
        sync_reviewed_evidence(
            loaded,
            report_output=report,
            attestation_output=attestation,
            source_root=source_root,
        )

    assert not (source_root / attestation).exists()


def test_sync_rejects_unsafe_existing_public_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, source_root, report, attestation = _sync_fixture(tmp_path, monkeypatch)
    external = tmp_path / "external-review.md"
    external.write_bytes(loaded.files["evidence/results-review.md"])
    (source_root / report).symlink_to(external)

    with pytest.raises(ReplayReviewSyncError, match="unsafe"):
        sync_reviewed_evidence(
            loaded,
            report_output=report,
            attestation_output=attestation,
            source_root=source_root,
        )

    assert not (source_root / attestation).exists()


def test_sync_rejects_semantically_invalid_package_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "release"
    loaded = publish_release_package(_package_content(), destination)
    source_root = tmp_path / "source"
    review_root = source_root / "docs/reviews"
    review_root.mkdir(parents=True)
    subprocess.run(("git", "init", "-q", source_root), check=True)

    def reject_semantically(_path: Path) -> object:
        raise ValueError("semantic package validation failed")

    monkeypatch.setattr(
        "fusion_fault_bench.replay_release_package.validate_release_package",
        reject_semantically,
    )
    report = Path("docs/reviews/m5-results-review.md")
    attestation = Path("docs/reviews/m5-results-review-attestation.json")

    with pytest.raises(ValueError, match="semantic package validation failed"):
        sync_reviewed_evidence(
            loaded,
            report_output=report,
            attestation_output=attestation,
            source_root=source_root,
        )

    assert not (source_root / report).exists()
    assert not (source_root / attestation).exists()
