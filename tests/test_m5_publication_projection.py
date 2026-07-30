from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fusion_fault_bench.replay_publication as publication
from fusion_fault_bench.replay_publication import (
    M5_PUBLICATION_DOCUMENT_PATHS,
    M5_PUBLICATION_PROJECTION_PLACEHOLDER,
    ReplayPublicationProjectionError,
    expected_publication_document,
    render_publication_projection,
    validate_publication_documents,
)

_HYPOTHESES = (
    "h5-a1",
    "h5-a2",
    "h5-a3",
    "h5-a4",
    "h5-a5",
    "h5-a6",
    "h5-b1",
    "h5-b2",
    "h5-b3",
    "h5-b4",
)


def _validated() -> object:
    return SimpleNamespace(
        artifact=SimpleNamespace(run=SimpleNamespace(git_revision="1" * 40)),
        release_package_sha256="2" * 64,
        claim_projection_sha256="3" * 64,
        release_summary={
            "hypothesis_results": {
                hypothesis: {
                    "result": "directionally-consistent",
                    "role": "directional",
                }
                for hypothesis in _HYPOTHESES
            }
        },
    )


def test_every_closeout_document_contains_one_frozen_projection_placeholder() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in M5_PUBLICATION_DOCUMENT_PATHS:
        assert (root / relative).read_bytes().count(M5_PUBLICATION_PROJECTION_PLACEHOLDER) == 1


def test_closeout_projection_is_deterministic_and_reviewed_only() -> None:
    validated = _validated()
    first = render_publication_projection(validated)  # type: ignore[arg-type]
    second = render_publication_projection(validated)  # type: ignore[arg-type]

    assert first == second
    assert first.endswith(b"\n")
    assert b"reports/generated" not in first
    assert b"additional quantitative M5 selector" in first
    for hypothesis in _HYPOTHESES:
        assert first.count(hypothesis.upper().encode()) == 1


def test_closeout_projection_rejects_unsafe_or_reordered_results() -> None:
    validated = _validated()
    reordered = dict(reversed(tuple(validated.release_summary["hypothesis_results"].items())))
    validated.release_summary["hypothesis_results"] = reordered
    with pytest.raises(ReplayPublicationProjectionError, match="fixed hypothesis"):
        render_publication_projection(validated)  # type: ignore[arg-type]

    validated = _validated()
    validated.release_summary["hypothesis_results"]["h5-a1"]["result"] = "/Users/private"
    with pytest.raises(ReplayPublicationProjectionError, match="unsafe hypothesis"):
        render_publication_projection(validated)  # type: ignore[arg-type]


def test_expected_document_replaces_exactly_one_frozen_placeholder() -> None:
    projection = render_publication_projection(_validated())  # type: ignore[arg-type]
    base = b"# Existing reviewed documentation\n\n" + M5_PUBLICATION_PROJECTION_PLACEHOLDER
    expected = expected_publication_document(base, projection, relative="README.md")

    assert expected == b"# Existing reviewed documentation\n\n" + projection
    with pytest.raises(ReplayPublicationProjectionError, match="one frozen placeholder"):
        expected_publication_document(b"# Missing\n", projection, relative="README.md")


def test_publication_documents_require_exact_scientific_base_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validated = _validated()
    projection = render_publication_projection(validated)  # type: ignore[arg-type]
    bases = {
        relative: f"# {relative}\n\n".encode() + M5_PUBLICATION_PROJECTION_PLACEHOLDER
        for relative in M5_PUBLICATION_DOCUMENT_PATHS
    }
    current = {
        relative: expected_publication_document(bases[relative], projection, relative=relative)
        for relative in M5_PUBLICATION_DOCUMENT_PATHS
    }
    monkeypatch.setattr(
        publication,
        "_scientific_document_bytes",
        lambda _root, _revision, relative: bases[relative],
    )

    validate_publication_documents(  # type: ignore[arg-type]
        validated,
        tmp_path,
        read_current=lambda _root, relative: current[relative],
    )
    current["docs/results.md"] += b"Contradictory unreviewed claim: 999.\n"
    with pytest.raises(ReplayPublicationProjectionError, match="reviewed projection"):
        validate_publication_documents(  # type: ignore[arg-type]
            validated,
            tmp_path,
            read_current=lambda _root, relative: current[relative],
        )
