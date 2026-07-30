"""Deterministic repository-document projection for the reviewed M5 release."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Never, Protocol, cast

from fusion_fault_bench.contracts.replay_artifact_v1 import M5_REPLAY_RELEASE_ID

M5_MARKDOWN_PUBLICATION_DOCUMENT_PATHS = (
    "README.md",
    "docs/results.md",
    "docs/benchmark-card.md",
    "docs/limitations.md",
    "docs/reproducibility.md",
    "docs/project-plan.md",
    "docs/dataset-preparation.md",
    "docs/m5-technical-walkthrough.md",
)
M5_DASHBOARD_DOCUMENT_PATH = "docs/dashboard.html"
M5_PUBLICATION_DOCUMENT_PATHS = (
    *M5_MARKDOWN_PUBLICATION_DOCUMENT_PATHS,
    M5_DASHBOARD_DOCUMENT_PATH,
)

M5_PUBLICATION_PROJECTION_PLACEHOLDER = (
    b"<!-- FFB-M5-RELEASE-PROJECTION-V1:START -->\n"
    b"M5 reviewed release projection pending.\n"
    b"<!-- FFB-M5-RELEASE-PROJECTION-V1:END -->\n"
)
M5_DASHBOARD_PROJECTION_START = b"<!-- FFB-M5-DASHBOARD-PROJECTION-V1:START -->"
M5_DASHBOARD_PROJECTION_END = b"<!-- FFB-M5-DASHBOARD-PROJECTION-V1:END -->"

_HYPOTHESIS_ORDER = (
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
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RESULT_PATTERN = re.compile(r"^[a-z0-9-]+$")
_PRIVATE_PATTERN = re.compile(
    rb"(?:file:(?://)?|/(?:Users|home|private|tmp|Volumes)/|"
    rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|reports/generated/|interview/)",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    rb"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    rb"(?:[\"'\s]*[:=][\"'\s]*|_)[A-Za-z0-9+/=_-]{8,}",
    re.IGNORECASE,
)
_DATASET_PAYLOAD_PATTERN = re.compile(
    rb"(?:samples|sweeps|maps|v1\.0-mini)/|"
    rb"[A-Za-z0-9_.+-]+\.(?:jpg|jpeg|png|pcd|las|laz|bin|tar|tgz|zip)",
    re.IGNORECASE,
)


class ReplayPublicationProjectionError(ValueError):
    """A closeout document differs from its reviewed deterministic projection."""


class _Run(Protocol):
    @property
    def git_revision(self) -> str: ...


class _Artifact(Protocol):
    @property
    def run(self) -> _Run: ...


class ValidatedPublication(Protocol):
    @property
    def artifact(self) -> _Artifact: ...

    @property
    def release_package_sha256(self) -> str: ...

    @property
    def claim_projection_sha256(self) -> str: ...

    @property
    def release_summary(self) -> Mapping[str, Any]: ...


def _fail(message: str) -> Never:
    raise ReplayPublicationProjectionError(message) from None


def _reviewed_projection_values(
    validated: ValidatedPublication,
) -> tuple[str, str, tuple[tuple[str, str, str], ...]]:
    package_sha256 = validated.release_package_sha256
    claim_sha256 = validated.claim_projection_sha256
    if (
        _DIGEST_PATTERN.fullmatch(package_sha256) is None
        or _DIGEST_PATTERN.fullmatch(claim_sha256) is None
    ):
        _fail("publication projection lacks canonical package digests")
    raw_results_value = validated.release_summary.get("hypothesis_results")
    if not isinstance(raw_results_value, Mapping):
        _fail("publication projection lacks the fixed hypothesis result order")
    raw_results = cast(Mapping[str, object], raw_results_value)
    if tuple(raw_results) != _HYPOTHESIS_ORDER:
        _fail("publication projection lacks the fixed hypothesis result order")

    results: list[tuple[str, str, str]] = []
    for hypothesis_id in _HYPOTHESIS_ORDER:
        raw_value = raw_results[hypothesis_id]
        if not isinstance(raw_value, Mapping):
            _fail("publication projection contains a malformed hypothesis result")
        raw = cast(Mapping[str, object], raw_value)
        result = raw.get("result")
        role = raw.get("role")
        if (
            not isinstance(result, str)
            or not isinstance(role, str)
            or _SAFE_RESULT_PATTERN.fullmatch(result) is None
            or _SAFE_RESULT_PATTERN.fullmatch(role) is None
        ):
            _fail("publication projection contains an unsafe hypothesis result")
        results.append((hypothesis_id, result, role))
    return package_sha256, claim_sha256, tuple(results)


def _require_public_projection_safe(value: bytes) -> None:
    if (
        _PRIVATE_PATTERN.search(value)
        or _SECRET_PATTERN.search(value)
        or _DATASET_PAYLOAD_PATTERN.search(value)
    ):
        _fail("publication projection contains private or raw source material")


def render_publication_projection(validated: ValidatedPublication) -> bytes:
    """Render the only allowed Markdown closeout block from reviewed package values."""

    package_sha256, claim_sha256, results = _reviewed_projection_values(validated)
    lines = [
        "<!-- FFB-M5-RELEASE-PROJECTION-V1:START -->",
        "## M5 reviewed release projection",
        "",
        f"Release `{M5_REPLAY_RELEASE_ID}` is published from the immutable reviewed package.",
        "",
        f"- Release package SHA-256: `{package_sha256}`",
        f"- Public claim projection SHA-256: `{claim_sha256}`",
        "- Reviewed preregistered hypothesis outcomes:",
    ]
    for hypothesis_id, result, role in results:
        lines.append(f"  - `{hypothesis_id.upper()}`: `{result}` (`{role}`)")
    lines.extend(
        (
            "",
            "Authoritative quantitative evidence, undefined states, controls, limitations, and "
            "verification records remain in "
            f"`reports/releases/{M5_REPLAY_RELEASE_ID}/`.",
            "This repository document introduces no additional quantitative M5 selector or claim.",
            "",
            "<!-- FFB-M5-RELEASE-PROJECTION-V1:END -->",
            "",
        )
    )
    rendered = "\n".join(lines).encode("utf-8")
    _require_public_projection_safe(rendered)
    return rendered


def render_dashboard_projection(validated: ValidatedPublication) -> bytes:
    """Render the fixed released-state dashboard body from reviewed package values."""

    package_sha256, claim_sha256, results = _reviewed_projection_values(validated)
    lines = [
        M5_DASHBOARD_PROJECTION_START.decode("ascii"),
        '  <main id="main-content">',
        '    <section class="hero" id="overview" aria-labelledby="overview-title">',
        '      <div class="shell">',
        '        <p class="eyebrow">Deterministic estimator-output evaluation</p>',
        '        <h1 id="overview-title">Fusion Fault Bench reviewed evidence</h1>',
        "        <p>M1&ndash;M5 have reviewed release evidence. "
        "M5 is published from the immutable package.</p>",
        "      </div>",
        "    </section>",
        '    <section id="architecture" aria-labelledby="architecture-title">',
        '      <div class="shell">',
        '        <h2 id="architecture-title">Auditable from frozen intent to release</h2>',
        "        <p>The release preserves the estimator-output boundary, paired execution, "
        "aggregate-only publication, and content-addressed validation.</p>",
        "      </div>",
        "    </section>",
        '    <section id="evidence" aria-labelledby="evidence-title">',
        '      <div class="shell">',
        '        <h2 id="evidence-title">Reviewed preregistered M5 outcomes</h2>',
        '        <ul class="metric-list">',
    ]
    for hypothesis_id, result, role in results:
        lines.append(
            f"          <li><span>{hypothesis_id.upper()}</span>"
            f"<strong>{result} ({role})</strong></li>"
        )
    lines.extend(
        (
            "        </ul>",
            "      </div>",
            "    </section>",
            '    <section class="m5" id="m5" aria-labelledby="m5-title">',
            '      <div class="shell">',
            '        <p class="eyebrow">Reviewed release</p>',
            '        <h2 id="m5-title">M5 nuScenes replay v0.1.0</h2>',
            f"        <p>Release package SHA-256: <code>{package_sha256}</code></p>",
            f"        <p>Public claim projection SHA-256: <code>{claim_sha256}</code></p>",
            f'        <p><a href="../reports/releases/{M5_REPLAY_RELEASE_ID}/README.md">'
            "Open the authoritative release package</a></p>",
            "      </div>",
            "    </section>",
            '    <section id="boundaries" aria-labelledby="boundaries-title">',
            '      <div class="shell">',
            '        <h2 id="boundaries-title">Claim boundary</h2>',
            "        <p>The reviewed evidence concerns deterministic estimator-output replay. "
            "It does not establish detector quality, natural fault rates, planning performance, "
            "certification, or production safety.</p>",
            "      </div>",
            "    </section>",
            '    <section class="next-steps" id="next" aria-labelledby="next-title">',
            '      <div class="shell">',
            '        <h2 id="next-title">Release verification</h2>',
            "        <p>The immutable package, independent reviews, deterministic document "
            "projection, and offline validation are the authority for this closeout.</p>",
            '        <p><a href="results.md">Read the results ledger</a> · '
            '<a href="m5-technical-walkthrough.md">Read the technical walkthrough</a> · '
            '<a href="../tools/m5_release.py">Inspect the release tool</a></p>',
            "      </div>",
            "    </section>",
            "  </main>",
            M5_DASHBOARD_PROJECTION_END.decode("ascii"),
            "",
        )
    )
    rendered = "\n".join(lines).encode("utf-8")
    _require_public_projection_safe(rendered)
    return rendered


def expected_publication_document(
    base: bytes,
    projection: bytes,
    *,
    relative: str,
) -> bytes:
    """Apply exactly one frozen closeout transformation to a scientific blob."""

    if relative not in M5_PUBLICATION_DOCUMENT_PATHS:
        _fail("publication document path is not allowlisted")
    if relative == M5_DASHBOARD_DOCUMENT_PATH:
        if (
            base.count(M5_DASHBOARD_PROJECTION_START) != 1
            or base.count(M5_DASHBOARD_PROJECTION_END) != 1
        ):
            _fail("scientific dashboard lacks one frozen projection region")
        if (
            projection.count(M5_DASHBOARD_PROJECTION_START) != 1
            or projection.count(M5_DASHBOARD_PROJECTION_END) != 1
            or not projection.startswith(M5_DASHBOARD_PROJECTION_START)
            or not projection.rstrip(b"\n").endswith(M5_DASHBOARD_PROJECTION_END)
        ):
            _fail("dashboard projection has an invalid frozen marker shape")
        start = base.index(M5_DASHBOARD_PROJECTION_START)
        try:
            end = base.index(M5_DASHBOARD_PROJECTION_END, start) + len(M5_DASHBOARD_PROJECTION_END)
        except ValueError:
            _fail("scientific dashboard has reversed frozen projection markers")
        return base[:start] + projection.rstrip(b"\n") + base[end:]
    if base.count(M5_PUBLICATION_PROJECTION_PLACEHOLDER) != 1:
        _fail(f"scientific publication document lacks one frozen placeholder: {relative}")
    return base.replace(M5_PUBLICATION_PROJECTION_PLACEHOLDER, projection, 1)


def _scientific_document_bytes(source_root: Path, revision: str, relative: str) -> bytes:
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or relative not in M5_PUBLICATION_DOCUMENT_PATHS
    ):
        _fail("publication scientific source identity is invalid")
    result = subprocess.run(
        ("git", "-C", os.fspath(source_root), "show", f"{revision}:{relative}"),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _fail(f"scientific publication document is unavailable: {relative}")
    return result.stdout


def validate_publication_documents(
    validated: ValidatedPublication,
    source_root: Path,
    *,
    read_current: Callable[[Path, str], bytes],
) -> None:
    """Require every closeout document to be the exact reviewed projection."""

    markdown_projection = render_publication_projection(validated)
    dashboard_projection = render_dashboard_projection(validated)
    revision = validated.artifact.run.git_revision
    for relative in M5_PUBLICATION_DOCUMENT_PATHS:
        base = _scientific_document_bytes(source_root, revision, relative)
        projection = (
            dashboard_projection if relative == M5_DASHBOARD_DOCUMENT_PATH else markdown_projection
        )
        expected = expected_publication_document(base, projection, relative=relative)
        if read_current(source_root, relative) != expected:
            _fail(f"release documentation differs from the reviewed projection: {relative}")


__all__ = [
    "M5_DASHBOARD_DOCUMENT_PATH",
    "M5_DASHBOARD_PROJECTION_END",
    "M5_DASHBOARD_PROJECTION_START",
    "M5_MARKDOWN_PUBLICATION_DOCUMENT_PATHS",
    "M5_PUBLICATION_DOCUMENT_PATHS",
    "M5_PUBLICATION_PROJECTION_PLACEHOLDER",
    "ReplayPublicationProjectionError",
    "expected_publication_document",
    "render_dashboard_projection",
    "render_publication_projection",
    "validate_publication_documents",
]
