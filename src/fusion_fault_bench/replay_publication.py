"""Deterministic repository-document projection for the reviewed M5 release."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Never, Protocol, cast

from fusion_fault_bench.contracts.replay_artifact_v1 import M5_REPLAY_RELEASE_ID

M5_PUBLICATION_DOCUMENT_PATHS = (
    "README.md",
    "docs/results.md",
    "docs/benchmark-card.md",
    "docs/limitations.md",
    "docs/reproducibility.md",
    "docs/project-plan.md",
    "docs/dataset-preparation.md",
    "docs/m5-technical-walkthrough.md",
)

M5_PUBLICATION_PROJECTION_PLACEHOLDER = (
    b"<!-- FFB-M5-RELEASE-PROJECTION-V1:START -->\n"
    b"M5 reviewed release projection pending.\n"
    b"<!-- FFB-M5-RELEASE-PROJECTION-V1:END -->\n"
)

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


def render_publication_projection(validated: ValidatedPublication) -> bytes:
    """Render the only allowed M5 closeout block from reviewed package values."""

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
    value = "\n".join(lines).encode("utf-8")
    if (
        _PRIVATE_PATTERN.search(value)
        or _SECRET_PATTERN.search(value)
        or _DATASET_PAYLOAD_PATTERN.search(value)
    ):
        _fail("publication projection contains private or raw source material")
    return value


def expected_publication_document(
    base: bytes,
    projection: bytes,
    *,
    relative: str,
) -> bytes:
    """Replace exactly one pre-outcome placeholder in a scientific-revision blob."""

    if relative not in M5_PUBLICATION_DOCUMENT_PATHS:
        _fail("publication document path is not allowlisted")
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

    projection = render_publication_projection(validated)
    revision = validated.artifact.run.git_revision
    for relative in M5_PUBLICATION_DOCUMENT_PATHS:
        base = _scientific_document_bytes(source_root, revision, relative)
        expected = expected_publication_document(base, projection, relative=relative)
        if read_current(source_root, relative) != expected:
            _fail(f"release documentation differs from the reviewed projection: {relative}")


__all__ = [
    "M5_PUBLICATION_DOCUMENT_PATHS",
    "M5_PUBLICATION_PROJECTION_PLACEHOLDER",
    "ReplayPublicationProjectionError",
    "expected_publication_document",
    "render_publication_projection",
    "validate_publication_documents",
]
