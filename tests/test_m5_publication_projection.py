from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fusion_fault_bench.replay_publication as publication
from fusion_fault_bench.replay_publication import (
    M5_DASHBOARD_DOCUMENT_PATH,
    M5_DASHBOARD_PROJECTION_END,
    M5_DASHBOARD_PROJECTION_START,
    M5_MARKDOWN_PUBLICATION_DOCUMENT_PATHS,
    M5_PUBLICATION_DOCUMENT_PATHS,
    M5_PUBLICATION_PROJECTION_PLACEHOLDER,
    ReplayPublicationProjectionError,
    expected_publication_document,
    render_dashboard_projection,
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


def test_every_closeout_document_contains_one_frozen_projection_region() -> None:
    root = Path(__file__).resolve().parents[1]
    assert len(M5_PUBLICATION_DOCUMENT_PATHS) == 9
    for relative in M5_MARKDOWN_PUBLICATION_DOCUMENT_PATHS:
        assert (root / relative).read_bytes().count(M5_PUBLICATION_PROJECTION_PLACEHOLDER) == 1
    dashboard = (root / M5_DASHBOARD_DOCUMENT_PATH).read_bytes()
    assert dashboard.count(M5_DASHBOARD_PROJECTION_START) == 1
    assert dashboard.count(M5_DASHBOARD_PROJECTION_END) == 1
    assert M5_PUBLICATION_PROJECTION_PLACEHOLDER not in dashboard


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


def test_dashboard_projection_is_deterministic_released_state_only() -> None:
    validated = _validated()
    first = render_dashboard_projection(validated)  # type: ignore[arg-type]
    second = render_dashboard_projection(validated)  # type: ignore[arg-type]

    assert first == second
    assert first.count(M5_DASHBOARD_PROJECTION_START) == 1
    assert first.count(M5_DASHBOARD_PROJECTION_END) == 1
    assert b"M1&ndash;M5 have reviewed release evidence" in first
    assert b"2" * 64 in first
    assert b"3" * 64 in first
    for identifier in (b"overview", b"architecture", b"evidence", b"m5", b"boundaries", b"next"):
        assert b'id="' + identifier + b'"' in first
    lowered = first.lower()
    for stale in (b"pre-outcome", b"not released", b">pending<", b">next<"):
        assert stale not in lowered


def test_actual_dashboard_closeout_removes_contradictory_preoutcome_language() -> None:
    root = Path(__file__).resolve().parents[1]
    base = (root / M5_DASHBOARD_DOCUMENT_PATH).read_bytes()
    projection = render_dashboard_projection(_validated())  # type: ignore[arg-type]
    final = expected_publication_document(
        base,
        projection,
        relative=M5_DASHBOARD_DOCUMENT_PATH,
    )

    assert b"Fusion Fault Bench reviewed evidence" in final
    for stale in (
        b"pre-outcome",
        b"not released",
        b"No M5 outcome",
        b"Whole-revision implementation review remains next",
        b">Next<",
        b">Pending<",
    ):
        assert stale not in final


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

    dashboard_projection = render_dashboard_projection(_validated())  # type: ignore[arg-type]
    dashboard_base = (
        b"<head>frozen</head>\n"
        + M5_DASHBOARD_PROJECTION_START
        + b"\n<main>pre-outcome; not released; pending</main>\n"
        + M5_DASHBOARD_PROJECTION_END
        + b"\n<footer>frozen</footer>\n"
    )
    dashboard = expected_publication_document(
        dashboard_base,
        dashboard_projection,
        relative=M5_DASHBOARD_DOCUMENT_PATH,
    )
    assert dashboard.startswith(b"<head>frozen</head>\n")
    assert dashboard.endswith(b"\n<footer>frozen</footer>\n")
    assert b"pre-outcome" not in dashboard
    with pytest.raises(ReplayPublicationProjectionError, match="one frozen projection region"):
        expected_publication_document(
            b"<main>missing markers</main>",
            dashboard_projection,
            relative=M5_DASHBOARD_DOCUMENT_PATH,
        )
    with pytest.raises(ReplayPublicationProjectionError, match="reversed frozen"):
        expected_publication_document(
            M5_DASHBOARD_PROJECTION_END + b"\n" + M5_DASHBOARD_PROJECTION_START,
            dashboard_projection,
            relative=M5_DASHBOARD_DOCUMENT_PATH,
        )
    with pytest.raises(ReplayPublicationProjectionError, match="invalid frozen marker shape"):
        expected_publication_document(
            dashboard_base,
            b"<main>unmarked projection</main>",
            relative=M5_DASHBOARD_DOCUMENT_PATH,
        )


def test_publication_documents_require_exact_scientific_base_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validated = _validated()
    projection = render_publication_projection(validated)  # type: ignore[arg-type]
    dashboard_projection = render_dashboard_projection(validated)  # type: ignore[arg-type]
    bases = {
        relative: (
            b"<head>frozen</head>\n"
            + M5_DASHBOARD_PROJECTION_START
            + b"\n<main>pre-outcome</main>\n"
            + M5_DASHBOARD_PROJECTION_END
            + b"\n<footer>frozen</footer>\n"
            if relative == M5_DASHBOARD_DOCUMENT_PATH
            else f"# {relative}\n\n".encode() + M5_PUBLICATION_PROJECTION_PLACEHOLDER
        )
        for relative in M5_PUBLICATION_DOCUMENT_PATHS
    }
    current = {
        relative: expected_publication_document(
            bases[relative],
            dashboard_projection if relative == M5_DASHBOARD_DOCUMENT_PATH else projection,
            relative=relative,
        )
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
    current[M5_DASHBOARD_DOCUMENT_PATH] = current[M5_DASHBOARD_DOCUMENT_PATH].replace(
        b"M5 is published", b"M5 is not released", 1
    )
    with pytest.raises(ReplayPublicationProjectionError, match="reviewed projection"):
        validate_publication_documents(  # type: ignore[arg-type]
            validated,
            tmp_path,
            read_current=lambda _root, relative: current[relative],
        )
