from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import fusion_fault_bench.replay_publication_authority as authority
from fusion_fault_bench.contracts.replay_release_v1 import M5_RELEASE_PACKAGE_PATHS
from fusion_fault_bench.replay_publication import M5_PUBLICATION_DOCUMENT_PATHS
from fusion_fault_bench.replay_publication_authority import (
    ReplayPublicationAuthorityError,
    authenticate_pending_publication,
    require_scientific_revision_ancestor,
    validate_current_implementation_review,
    verify_pending_publication_unchanged,
)


def _state(
    revision: str,
    *,
    modified: tuple[str, ...] = M5_PUBLICATION_DOCUMENT_PATHS,
    untracked: tuple[str, ...] | None = None,
) -> tuple[bytes, ...]:
    expected_untracked = (
        *tuple(
            f"reports/releases/m5-nuscenes-replay-v0.1.0/{path}"
            for path in M5_RELEASE_PACKAGE_PATHS
        ),
        "docs/reviews/m5-results-review.md",
        "docs/reviews/m5-results-review-attestation.json",
    )
    return (
        f"{revision}\n".encode(),
        b"",
        b"H README.md\x00",
        b"\x00".join(path.encode() for path in modified) + b"\x00",
        b"",
        b"\x00".join(path.encode() for path in (untracked or expected_untracked)) + b"\x00",
    )


def test_pending_publication_accepts_only_exact_unstaged_closeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "1" * 40
    observed = _state(revision)
    monkeypatch.setattr(authority, "_git_state", lambda _root: observed)
    monkeypatch.setattr(authority, "_pending_content_digest", lambda *_args: b"digest")

    token = authenticate_pending_publication(tmp_path, scientific_git_revision=revision)
    assert token.state_digest_material == (*observed, b"digest")
    verify_pending_publication_unchanged(token)


@pytest.mark.parametrize(
    ("state_index", "replacement", "message"),
    (
        (3, b"README.md\x00", "eight closeout"),
        (4, b"README.md\x00", "index differs"),
        (5, b"unexpected\x00", "untracked paths"),
    ),
)
def test_pending_publication_rejects_extra_missing_or_staged_paths(
    state_index: int,
    replacement: bytes,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "1" * 40
    state = list(_state(revision))
    state[state_index] = replacement
    monkeypatch.setattr(authority, "_git_state", lambda _root: tuple(state))
    monkeypatch.setattr(authority, "_pending_content_digest", lambda *_args: b"digest")

    with pytest.raises(ReplayPublicationAuthorityError, match=message):
        authenticate_pending_publication(tmp_path, scientific_git_revision=revision)


def test_pending_publication_postflight_detects_same_path_content_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "1" * 40
    observed = _state(revision)
    digests = iter((b"before", b"after"))
    monkeypatch.setattr(authority, "_git_state", lambda _root: observed)
    monkeypatch.setattr(
        authority,
        "_pending_content_digest",
        lambda *_args: next(digests),
    )

    token = authenticate_pending_publication(tmp_path, scientific_git_revision=revision)
    with pytest.raises(ReplayPublicationAuthorityError, match="changed during validation"):
        verify_pending_publication_unchanged(token)


def test_package_review_must_equal_current_tracked_snapshot_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = b"# Current implementation review\n"
    attestation = b'{"schema":"review"}\n'
    expected = object()
    validated = SimpleNamespace(
        package=SimpleNamespace(
            files={
                "evidence/implementation-review.md": report,
                "evidence/implementation-review-attestation.json": attestation,
            }
        ),
        implementation_review_attestation=expected,
    )
    snapshot = SimpleNamespace()
    calls: list[dict[str, object]] = []

    def load(value: bytes, **kwargs: object) -> object:
        calls.append({"value": value, **kwargs})
        return expected

    monkeypatch.setattr(authority, "load_implementation_review_attestation", load)
    validate_current_implementation_review(
        validated,
        snapshot=snapshot,  # type: ignore[arg-type]
        tracked_report=report,
        tracked_attestation=attestation,
    )
    assert calls == [
        {
            "value": attestation,
            "review_report": report,
            "snapshot": snapshot,
            "require_release_permitting": True,
        }
    ]

    with pytest.raises(ReplayPublicationAuthorityError, match="differs from current"):
        validate_current_implementation_review(
            validated,
            snapshot=snapshot,  # type: ignore[arg-type]
            tracked_report=b"# Different review\n",
            tracked_attestation=attestation,
        )


def test_clean_publication_requires_scientific_revision_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    require_scientific_revision_ancestor(tmp_path, "1" * 40)
    assert calls[0][-3:] == ("--is-ancestor", "1" * 40, "HEAD")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(ReplayPublicationAuthorityError, match="not an ancestor"):
        require_scientific_revision_ancestor(tmp_path, "1" * 40)
