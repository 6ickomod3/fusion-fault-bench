from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fusion_fault_bench.replay_release_build as release_build
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_FIGURE_PATHS,
    M5_PRESENTATION_PLACEHOLDERS,
    M5_RELEASE_SIDECAR_INDEXED_PATHS,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
)
from fusion_fault_bench.replay_release import ReplayReleasePackageContent
from fusion_fault_bench.replay_release_build import (
    M5_RELEASE_DESTINATION,
    ReplayReleaseBuildError,
    build_reviewed_release,
)

_REVISION = "a" * 40
_ARTIFACT_SHA256 = "b" * 64
_RUN_SHA256 = "c" * 64
_CANDIDATE_SHA256 = "d" * 64
_CANDIDATE_INDEX_SHA256 = "e" * 64
_PACKAGE_SHA256 = "f" * 64


def _template(label: str) -> bytes:
    placeholders = "\n".join(f"`{value}`" for value in M5_PRESENTATION_PLACEHOLDERS)
    return f"# {label}\n{placeholders}\n".encode()


def _candidate_files() -> dict[str, bytes]:
    files = {path: f"{path}\n".encode() for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS}
    files.update(
        {
            "presentation/README.md": _template("README"),
            "presentation/claim-evidence.md": _template("Claims"),
            "presentation/verification.md": _template("Verification"),
        }
    )
    return files


def _candidate(*, files: dict[str, bytes] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        index=SimpleNamespace(candidate_sha256=_CANDIDATE_SHA256),
        index_bytes=b'{"candidate":"index"}\n',
        files=_candidate_files() if files is None else files,
        candidate_sha256=_CANDIDATE_SHA256,
        candidate_index_sha256=_CANDIDATE_INDEX_SHA256,
    )


def test_sidecars_are_exact_and_only_substitute_four_reviewed_tokens() -> None:
    candidate = _candidate()
    report = b"# Results review\n"
    attestation = b'{"schema":"attestation"}\n'
    machine_bytes = 9876

    sidecars = release_build._sidecar_files(
        candidate,
        results_review=report,
        results_attestation=attestation,
        machine_artifact_sha256=_ARTIFACT_SHA256,
        machine_run_sha256=_RUN_SHA256,
        machine_artifact_byte_length=machine_bytes,
    )

    assert tuple(sidecars) == M5_RELEASE_SIDECAR_INDEXED_PATHS
    assert len(sidecars) == 26
    replacements = (
        _ARTIFACT_SHA256,
        _RUN_SHA256,
        hashlib.sha256(attestation).hexdigest(),
        str(machine_bytes),
    )
    for path in ("README.md", "claim-evidence.md", "verification.md"):
        for placeholder, replacement in zip(
            M5_PRESENTATION_PLACEHOLDERS,
            replacements,
            strict=True,
        ):
            assert placeholder.encode() not in sidecars[path]
            assert sidecars[path].count(replacement.encode()) == 1
    assert sidecars["release-summary.json"] == candidate.files["presentation/release-summary.json"]
    assert sidecars["evidence/review-candidate-index.json"] == candidate.index_bytes
    assert sidecars["evidence/results-review.md"] == report
    assert sidecars["evidence/results-review-attestation.json"] == attestation
    assert all(sidecars[path] == candidate.files[path] for path in M5_FIGURE_PATHS)


def test_presentation_substitution_rejects_missing_reviewed_placeholder() -> None:
    files = _candidate_files()
    files["presentation/README.md"] = files["presentation/README.md"].replace(
        M5_PRESENTATION_PLACEHOLDERS[0].encode(),
        b"removed",
    )

    with pytest.raises(ReplayReleaseBuildError, match="exact placeholders"):
        release_build._substitute_reviewed_templates(
            files,
            machine_artifact_sha256=_ARTIFACT_SHA256,
            machine_run_sha256=_RUN_SHA256,
            results_review_attestation_sha256="1" * 64,
            machine_artifact_byte_length=1,
        )


def test_regeneration_requires_byte_identical_index_members_and_digests() -> None:
    reviewed = _candidate()
    changed_files = dict(reviewed.files)
    changed_files["machine/intent.json"] += b"changed\n"

    with pytest.raises(ReplayReleaseBuildError, match="exact reviewed candidate"):
        release_build._require_identical_candidate(
            reviewed,
            _candidate(files=changed_files),
        )


def test_build_authenticates_review_recurates_and_strictly_validates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reviewed = _candidate()
    report_path = tmp_path / "results-review.md"
    attestation_path = tmp_path / "results-review-attestation.json"
    report_bytes = b"# Results review\n"
    attestation_bytes = b'{"schema":"results-review"}\n'
    report_path.write_bytes(report_bytes)
    attestation_path.write_bytes(attestation_bytes)
    clean = SimpleNamespace(
        source_root=tmp_path,
        git_revision=_REVISION,
        lockfile_sha256="2" * 64,
        package_version="0.1.0",
    )
    implementation_snapshot = SimpleNamespace(scientific_git_revision=_REVISION)
    prepared_paths: list[Path] = []
    load_paths: list[Path] = []
    review_calls: list[dict[str, Any]] = []
    package_contents: list[ReplayReleasePackageContent] = []
    machine_requests: list[Any] = []
    publication_events: list[str] = []

    def prepare_candidate(**kwargs: Any) -> SimpleNamespace:
        prepared_paths.append(kwargs["output_dir"])
        return reviewed

    def load_candidate(**kwargs: Any) -> SimpleNamespace:
        load_paths.append(kwargs["path"])
        return reviewed

    def load_results(value: bytes, **kwargs: Any) -> object:
        review_calls.append({"value": value, **kwargs})
        return object()

    repeat = SimpleNamespace(
        source_commitments=(object(),),
        repeat_verification=object(),
    )
    curated = SimpleNamespace(
        profile_summary=object(),
        descriptor_aggregates=(object(),),
        persistent_aggregates=(object(),),
        persistent_crossovers=(object(),),
        health_aggregates=(object(),),
        cluster_sensitivity=(object(),),
        run=SimpleNamespace(run_id="run-id"),
    )
    final_validation = object()
    machine_files = {
        candidate_path.removeprefix("machine/"): reviewed.files[candidate_path]
        for candidate_path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[:10]
    }
    machine_files.update(
        {
            "validation.json": b"validation\n",
            "release-index.json": b"release-index\n",
            "run.json": b"run\n",
            "_SUCCESS": b"success\n",
        }
    )
    machine = SimpleNamespace(
        files=machine_files,
        artifact_sha256=_ARTIFACT_SHA256,
        run_sha256=_RUN_SHA256,
    )
    pre_inputs = object()

    monkeypatch.setattr(release_build, "prepare_review_candidate", prepare_candidate)
    monkeypatch.setattr(release_build, "load_validated_review_candidate", load_candidate)
    monkeypatch.setattr(release_build, "derive_results_review_bindings", lambda *_args: object())
    monkeypatch.setattr(release_build, "load_results_review_attestation", load_results)
    monkeypatch.setattr(release_build, "verify_replay_repeat_artifacts", lambda **_kwargs: repeat)
    monkeypatch.setattr(
        release_build,
        "curate_replay_verified_repeat",
        lambda *_args, **_kwargs: curated,
    )
    monkeypatch.setattr(release_build, "load_software_verification", lambda *_a, **_k: object())
    monkeypatch.setattr(
        release_build,
        "load_implementation_review_attestation",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        release_build,
        "load_privacy_license_attestation",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(release_build, "_canonical_claim_registry", lambda _value: object())
    monkeypatch.setattr(
        release_build,
        "derive_pre_review_evidence_from_parts",
        lambda **_kwargs: {"pre-review": {"part": b"value"}},
    )
    monkeypatch.setattr(
        release_build,
        "load_pre_review_validation_inputs",
        lambda *_a, **_k: pre_inputs,
    )
    monkeypatch.setattr(release_build, "_canonical_validation_inputs", lambda _value: pre_inputs)
    monkeypatch.setattr(
        release_build,
        "derive_final_replay_validation",
        lambda **_kwargs: final_validation,
    )
    monkeypatch.setattr(release_build, "_canonical_figure_bindings", lambda _value: (object(),))

    def prepare_machine(request: Any, **_kwargs: Any) -> SimpleNamespace:
        machine_requests.append(request)
        return machine

    monkeypatch.setattr(release_build, "prepare_replay_machine_payload", prepare_machine)

    def publish(content: ReplayReleasePackageContent, destination: Path) -> SimpleNamespace:
        package_contents.append(content)
        publication_events.append("final-publish" if destination == output else "preflight-publish")
        return SimpleNamespace(release_package_sha256=_PACKAGE_SHA256, path=destination)

    validated = SimpleNamespace(release_package_sha256=_PACKAGE_SHA256)
    monkeypatch.setattr(release_build, "publish_release_package", publish)

    def validate(path: Path) -> SimpleNamespace:
        publication_events.append("final-validate" if path == output else "preflight-validate")
        return validated

    monkeypatch.setattr(release_build, "validate_release_package", validate)

    output = tmp_path / M5_RELEASE_DESTINATION

    def prepublish_authority() -> None:
        publication_events.append("authority")

    observed = build_reviewed_release(
        candidate=tmp_path / "candidate",
        results_review=report_path,
        results_review_attestation=attestation_path,
        primary_artifact=tmp_path / "primary",
        repeat_artifact=tmp_path / "repeat",
        primary_time_l=tmp_path / "primary.time-l.txt",
        repeat_time_l=tmp_path / "repeat.time-l.txt",
        software_verification=tmp_path / "software.json",
        output_dir=output,
        source_root=tmp_path,
        clean_snapshot=clean,
        implementation_snapshot=implementation_snapshot,
        implementation_report=b"implementation report\n",
        implementation_attestation=b"implementation attestation\n",
        prepublish_authority=prepublish_authority,
    )

    assert observed is validated
    assert load_paths[0] == tmp_path / "candidate"
    assert load_paths[1] == prepared_paths[0]
    assert review_calls[0]["value"] == attestation_bytes
    assert review_calls[0]["review_report"] == report_bytes
    assert review_calls[0]["require_release_permitting"] is True
    assert machine_requests[0].validation is final_validation
    assert len(package_contents) == 2
    assert package_contents[0] == package_contents[1]
    assert tuple(package_contents[1].sidecar_files) == M5_RELEASE_SIDECAR_INDEXED_PATHS
    assert (
        package_contents[1].results_review_attestation_sha256
        == hashlib.sha256(attestation_bytes).hexdigest()
    )
    assert package_contents[1].reviewed_candidate_sha256 == _CANDIDATE_SHA256
    assert package_contents[1].machine_files is machine_files
    assert publication_events == [
        "preflight-publish",
        "preflight-validate",
        "authority",
        "final-publish",
        "final-validate",
    ]


def test_build_rejects_any_other_release_destination(tmp_path: Path) -> None:
    clean = SimpleNamespace(source_root=tmp_path, git_revision=_REVISION)
    implementation = SimpleNamespace(scientific_git_revision=_REVISION)

    with pytest.raises(ReplayReleaseBuildError, match="frozen package path"):
        build_reviewed_release(
            candidate=tmp_path / "candidate",
            results_review=tmp_path / "review",
            results_review_attestation=tmp_path / "attestation",
            primary_artifact=tmp_path / "primary",
            repeat_artifact=tmp_path / "repeat",
            primary_time_l=tmp_path / "primary-time",
            repeat_time_l=tmp_path / "repeat-time",
            software_verification=tmp_path / "software",
            output_dir=tmp_path / "wrong",
            source_root=tmp_path,
            clean_snapshot=clean,
            implementation_snapshot=implementation,
            implementation_report=b"report\n",
            implementation_attestation=b"attestation\n",
            prepublish_authority=lambda: None,
        )


@pytest.mark.parametrize(
    ("published", "validated"),
    (
        (
            ("0" * 64, _PACKAGE_SHA256),
            (_PACKAGE_SHA256, "0" * 64),
        )
    ),
)
def test_final_publication_digest_must_equal_preflight(
    published: str,
    validated: str,
) -> None:
    with pytest.raises(ReplayReleaseBuildError, match="validated preflight"):
        release_build._require_matching_publication_digest(
            expected=_PACKAGE_SHA256,
            published=published,
            validated=validated,
        )
