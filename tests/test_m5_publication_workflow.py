from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fusion_fault_bench import replay_publication_authority as publication_authority
from fusion_fault_bench import replay_release_package as release_package
from fusion_fault_bench import replay_release_workflow as workflow
from fusion_fault_bench.contracts.replay_release_v1 import M5_RELEASE_DESTINATION_PATH

REVISION = "1" * 40
DIGEST = "2" * 64


def _validated(*, digest: str = DIGEST) -> SimpleNamespace:
    return SimpleNamespace(
        artifact=SimpleNamespace(run=SimpleNamespace(git_revision=REVISION)),
        release_package_sha256=digest,
    )


def _common_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[SimpleNamespace, SimpleNamespace, list[object]]:
    clean = SimpleNamespace(source_root=tmp_path, git_revision="3" * 40)
    implementation = SimpleNamespace(
        scientific_git_revision=clean.git_revision,
        sha256="4" * 64,
    )
    review_calls: list[object] = []
    monkeypatch.setattr(workflow, "_read_review_authority", lambda *_args: (b"report", b"{}\n"))
    monkeypatch.setattr(
        publication_authority,
        "validate_current_implementation_review",
        lambda validated, **kwargs: review_calls.append((validated, kwargs)),
    )
    monkeypatch.setattr(
        release_package,
        "validate_publication",
        lambda _release, _root: DIGEST,
    )
    return clean, implementation, review_calls


def test_publication_workflow_accepts_clean_descendant_and_rechecks_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, implementation, review_calls = _common_mocks(monkeypatch, tmp_path)
    clean_calls: list[Path] = []
    ancestor_calls: list[tuple[Path, str]] = []
    package_calls: list[Path] = []
    monkeypatch.setattr(
        workflow,
        "_clean_authority",
        lambda root: clean_calls.append(root) or (clean, implementation),
    )
    monkeypatch.setattr(
        release_package,
        "validate_release_package",
        lambda path: package_calls.append(path) or _validated(),
    )
    monkeypatch.setattr(
        publication_authority,
        "require_scientific_revision_ancestor",
        lambda root, revision: ancestor_calls.append((root, revision)),
    )

    release = tmp_path / M5_RELEASE_DESTINATION_PATH
    assert workflow.validate_publication(release=release, source_root=tmp_path) == DIGEST
    assert clean_calls == [tmp_path, tmp_path]
    assert package_calls == [release, release]
    assert ancestor_calls == [(tmp_path, REVISION), (tmp_path, REVISION)]
    assert len(review_calls) == 2


def test_publication_workflow_accepts_only_exact_fingerprinted_pending_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clean, implementation, review_calls = _common_mocks(monkeypatch, tmp_path)
    implementation.scientific_git_revision = REVISION
    token = object()
    pending_calls: list[tuple[Path, str]] = []
    verified: list[object] = []

    def dirty(_root: Path) -> object:
        raise workflow.ReplayReleaseWorkflowError("not clean")

    monkeypatch.setattr(workflow, "_clean_authority", dirty)
    monkeypatch.setattr(workflow, "build_implementation_snapshot", lambda _root: implementation)
    monkeypatch.setattr(release_package, "validate_release_package", lambda _path: _validated())
    monkeypatch.setattr(
        publication_authority,
        "authenticate_pending_publication",
        lambda root, *, scientific_git_revision: (
            pending_calls.append((root, scientific_git_revision)) or token
        ),
    )
    monkeypatch.setattr(
        publication_authority,
        "verify_pending_publication_unchanged",
        lambda observed: verified.append(observed),
    )

    release = tmp_path / M5_RELEASE_DESTINATION_PATH
    assert workflow.validate_publication(release=release, source_root=tmp_path) == DIGEST
    assert pending_calls == [(tmp_path, REVISION)]
    assert verified == [token]
    assert len(review_calls) == 2


def test_publication_workflow_rejects_package_swap_during_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean, implementation, _review_calls = _common_mocks(monkeypatch, tmp_path)
    observed = iter((_validated(), _validated(digest="5" * 64)))
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: (clean, implementation))
    monkeypatch.setattr(release_package, "validate_release_package", lambda _path: next(observed))
    monkeypatch.setattr(
        publication_authority,
        "require_scientific_revision_ancestor",
        lambda *_args: None,
    )

    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="changed during validation"):
        workflow.validate_publication(
            release=tmp_path / M5_RELEASE_DESTINATION_PATH,
            source_root=tmp_path,
        )


def test_publication_workflow_rejects_nonfrozen_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        release_package,
        "validate_release_package",
        lambda _path: pytest.fail("package validator must not run"),
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="frozen tracked package path"):
        workflow.validate_publication(release=Path("other-release"), source_root=tmp_path)
