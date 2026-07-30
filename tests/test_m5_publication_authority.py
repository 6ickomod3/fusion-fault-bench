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
        (3, b"README.md\x00", "nine closeout"),
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


def test_pending_content_fingerprint_binds_bytes_and_mode(tmp_path: Path) -> None:
    member = tmp_path / "member"
    member.write_bytes(b"first")
    member.chmod(0o644)
    first = authority._pending_content_digest(tmp_path, frozenset({b"member"}))

    member.write_bytes(b"second")
    member.chmod(0o644)
    second = authority._pending_content_digest(tmp_path, frozenset({b"member"}))
    assert first != second

    member.chmod(0o600)
    private_mode = authority._pending_content_digest(tmp_path, frozenset({b"member"}))
    assert private_mode != second

    member.chmod(0o640)
    with pytest.raises(ReplayPublicationAuthorityError, match="unavailable or unsafe"):
        authority._pending_content_digest(tmp_path, frozenset({b"member"}))


def test_pending_content_fingerprint_rejects_links(tmp_path: Path) -> None:
    member = tmp_path / "member"
    member.write_bytes(b"value")
    member.chmod(0o644)
    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(member)
    with pytest.raises(ReplayPublicationAuthorityError, match="unavailable or unsafe"):
        authority._pending_content_digest(tmp_path, frozenset({b"member"}))

    hardlink.unlink()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(member)
    with pytest.raises(ReplayPublicationAuthorityError, match="unavailable or unsafe"):
        authority._pending_content_digest(tmp_path, frozenset({b"symlink"}))


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


def _clean_state(
    head: str, *, worktree: bytes = b"", flags: bytes = b"H README.md\x00"
) -> tuple[bytes, ...]:
    return (f"{head}\n".encode(), b"", flags, worktree, b"", b"")


def _clean_delta_bytes(*, extra: tuple[bytes, bytes] | None = None) -> bytes:
    records = [
        *((b"M", path.encode()) for path in M5_PUBLICATION_DOCUMENT_PATHS),
        *((b"A", path) for path in authority._EXPECTED_ADDED_PATHS),
    ]
    if extra is not None:
        records.append(extra)
    return b"".join(status + b"\x00" + relative + b"\x00" for status, relative in records)


def _tree_bytes(entries: dict[bytes, tuple[bytes, bytes, bytes]]) -> bytes:
    return b"".join(
        mode + b" " + kind + b" " + object_id + b"\t" + relative + b"\x00"
        for relative, (mode, kind, object_id) in sorted(entries.items())
    )


def _clean_trees() -> tuple[
    dict[bytes, tuple[bytes, bytes, bytes]],
    dict[bytes, tuple[bytes, bytes, bytes]],
]:
    scientific = {
        b"unchanged.txt": (b"100644", b"blob", b"1" * 40),
        **{
            relative.encode(): (b"100644", b"blob", b"2" * 40)
            for relative in M5_PUBLICATION_DOCUMENT_PATHS
        },
    }
    release = dict(scientific)
    for relative in M5_PUBLICATION_DOCUMENT_PATHS:
        release[relative.encode()] = (b"100644", b"blob", b"3" * 40)
    for relative in authority._EXPECTED_ADDED_PATHS:
        release[relative] = (b"100644", b"blob", b"4" * 40)
    return scientific, release


def _mock_clean_queries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    delta: bytes | None = None,
    scientific: dict[bytes, tuple[bytes, bytes, bytes]] | None = None,
    release: dict[bytes, tuple[bytes, bytes, bytes]] | None = None,
    state: tuple[bytes, ...] | None = None,
    ancestor_returncode: int = 0,
) -> None:
    revision = "1" * 40
    head = "2" * 40
    default_scientific, default_release = _clean_trees()
    monkeypatch.setattr(authority, "_git_state", lambda _root: state or _clean_state(head))

    def git_bytes(_root: Path, *arguments: str) -> bytes:
        if arguments[0] == "diff":
            return delta if delta is not None else _clean_delta_bytes()
        if arguments[0] == "ls-tree":
            entries = scientific or default_scientific
            if arguments[-1] == "HEAD":
                entries = release or default_release
            return _tree_bytes(entries)
        raise AssertionError(arguments)

    monkeypatch.setattr(authority, "_git_bytes", git_bytes)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=ancestor_returncode),
    )
    require_scientific_revision_ancestor(tmp_path, revision)


def test_clean_publication_requires_exact_tree_mode_and_status_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_clean_queries(monkeypatch, tmp_path)


def test_clean_publication_rejects_every_extra_tree_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ReplayPublicationAuthorityError, match="exact reviewed release-tree"):
        _mock_clean_queries(
            monkeypatch,
            tmp_path,
            delta=_clean_delta_bytes(extra=(b"M", b"src/extra.py")),
        )


def test_clean_publication_rejects_wrong_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scientific, release = _clean_trees()
    release = dict(release)
    release[b"README.md"] = (b"100755", b"blob", b"3" * 40)
    with pytest.raises(ReplayPublicationAuthorityError, match="invalid modified document"):
        _mock_clean_queries(
            monkeypatch,
            tmp_path,
            scientific=scientific,
            release=release,
        )


def test_clean_publication_rejects_dirty_or_non_descendant_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ReplayPublicationAuthorityError, match="status is not empty"):
        _mock_clean_queries(
            monkeypatch,
            tmp_path,
            state=_clean_state("2" * 40, worktree=b"README.md\x00"),
        )

    with pytest.raises(ReplayPublicationAuthorityError, match="not an ancestor"):
        _mock_clean_queries(monkeypatch, tmp_path, ancestor_returncode=1)
