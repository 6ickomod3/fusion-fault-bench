from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fusion_fault_bench import replay_release_package, replay_release_software
from fusion_fault_bench import replay_release_workflow as workflow
from fusion_fault_bench.contracts.replay_release_v1 import M5_RELEASE_DESTINATION_PATH


def test_verify_software_delegates_with_reviewed_source_and_postflights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = SimpleNamespace(source_root=tmp_path)
    implementation = object()
    authority_calls: list[Path] = []
    review_calls: list[tuple[Path, object]] = []
    delegated: dict[str, object] = {}
    verification = object()

    def clean_authority(source_root: Path) -> tuple[object, object]:
        authority_calls.append(source_root)
        return clean, implementation

    def review_authority(source_root: Path, snapshot: object) -> tuple[bytes, bytes]:
        review_calls.append((source_root, snapshot))
        return b"report", b"attestation"

    def run_software_verification(**arguments: object) -> object:
        delegated.update(arguments)
        return verification

    monkeypatch.setattr(workflow, "_clean_authority", clean_authority)
    monkeypatch.setattr(workflow, "_read_review_authority", review_authority)
    monkeypatch.setattr(replay_release_software, "verify_software", run_software_verification)

    output = Path("reports/generated/m5-software-verification-a.json")
    assert workflow.verify_software(source_root=Path("."), output=output) is verification
    assert authority_calls == [Path("."), Path(".")]
    assert review_calls == [(tmp_path, implementation), (tmp_path, implementation)]
    assert delegated == {
        "source_root": tmp_path,
        "output": output,
        "clean_snapshot": clean,
        "implementation_snapshot": implementation,
    }


def test_verify_software_rejects_postflight_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = (SimpleNamespace(source_root=tmp_path), object())
    second = (SimpleNamespace(source_root=tmp_path), object())
    authorities = iter((first, second))
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: next(authorities))
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda _root, _snapshot: (b"report", b"attestation"),
    )
    monkeypatch.setattr(
        replay_release_software,
        "verify_software",
        lambda **_arguments: object(),
    )

    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="changed"):
        workflow.verify_software(source_root=Path("."), output=Path("output"))


def test_package_validator_returns_digest_without_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    observed: list[Path] = []

    def validate(path: Path) -> object:
        observed.append(path)
        return SimpleNamespace(release_package_sha256=digest)

    monkeypatch.setattr(replay_release_package, "validate_release_package", validate)
    monkeypatch.setattr(
        workflow,
        "_clean_authority",
        lambda _root: (_ for _ in ()).throw(AssertionError("unexpected source access")),
    )

    assert workflow.validate_release_package(path=Path("release")) == digest
    assert observed == [Path("release")]


@pytest.mark.parametrize("digest", ("A" * 64, "short", 3))
def test_package_validator_rejects_noncanonical_digest(
    digest: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        replay_release_package,
        "validate_release_package",
        lambda _path: SimpleNamespace(release_package_sha256=digest),
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="semantic digest"):
        workflow.validate_release_package(path=Path("release"))


def test_publication_validator_uses_clean_root_and_postflights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = SimpleNamespace(source_root=tmp_path)
    implementation = object()
    authority_calls: list[Path] = []
    observed: list[tuple[Path, Path]] = []
    digest = "b" * 64

    def clean_authority(source_root: Path) -> tuple[object, object]:
        authority_calls.append(source_root)
        return clean, implementation

    def validate(release: Path, source_root: Path) -> str:
        observed.append((release, source_root))
        return digest

    monkeypatch.setattr(workflow, "_clean_authority", clean_authority)
    monkeypatch.setattr(replay_release_package, "validate_publication", validate)

    release = Path(M5_RELEASE_DESTINATION_PATH)
    assert workflow.validate_publication(release=release, source_root=Path(".")) == digest
    assert authority_calls == [Path("."), Path(".")]
    assert observed == [(tmp_path / release, tmp_path)]


def test_publication_validator_rejects_postflight_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = (SimpleNamespace(source_root=tmp_path), object())
    second = (SimpleNamespace(source_root=tmp_path), object())
    authorities = iter((first, second))
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: next(authorities))
    monkeypatch.setattr(
        replay_release_package,
        "validate_publication",
        lambda _release, _source: "c" * 64,
    )

    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="changed"):
        workflow.validate_publication(
            release=Path(M5_RELEASE_DESTINATION_PATH),
            source_root=Path("."),
        )


def test_publication_validator_rejects_external_package_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = SimpleNamespace(source_root=tmp_path)
    implementation = object()
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: (clean, implementation))
    monkeypatch.setattr(
        replay_release_package,
        "validate_publication",
        lambda _release, _source: pytest.fail("external package reached semantic validation"),
    )

    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="frozen tracked"):
        workflow.validate_publication(release=tmp_path / "external", source_root=tmp_path)
