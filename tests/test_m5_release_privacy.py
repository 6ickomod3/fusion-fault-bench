from __future__ import annotations

from pathlib import Path

import pytest

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_PRESENTATION_PLACEHOLDERS,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
)
from fusion_fault_bench.replay_release import (
    ReplayReleaseError,
    validate_review_candidate_members,
)


def _member(path: str) -> bytes:
    if path.endswith(".ndjson"):
        return canonical_json_bytes({"row": 1})
    if path.endswith(".json"):
        return canonical_json_bytes({"value": 1})
    if path.endswith(".svg"):
        return b'<svg xmlns="http://www.w3.org/2000/svg"><text>fixed</text></svg>\n'
    if path.startswith("presentation/"):
        return (
            "# Reviewed template\n\n"
            + "\n".join(f"`{placeholder}`" for placeholder in M5_PRESENTATION_PLACEHOLDERS)
            + "\n"
        ).encode()
    return b"# Fixed evidence\n"


def _members(source_root: Path) -> dict[str, bytes]:
    files = {path: _member(path) for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS}
    files["machine/intent.json"] = (
        source_root / "examples/replay/m5-nuscenes-mini-replay-v1.json"
    ).read_bytes()
    files["evidence/release-pipeline-plan.md"] = (
        source_root / "docs/m5-release-pipeline-plan.md"
    ).read_bytes()
    files["evidence/release-pipeline-plan-review.md"] = (
        source_root / "docs/reviews/m5-release-pipeline-plan-review.md"
    ).read_bytes()
    files["evidence/resource-scope-amendment.md"] = (
        source_root / "docs/m5-resource-scope-amendment.md"
    ).read_bytes()
    files["evidence/implementation-review.md"] = (
        source_root / "docs/reviews/m5-implementation-review.md"
    ).read_bytes()
    return files


def test_exact_frozen_methodology_passes_role_aware_scan() -> None:
    source_root = Path(__file__).resolve().parents[1]
    validate_review_candidate_members(_members(source_root))


@pytest.mark.parametrize(
    "leak",
    (
        b"\nRealized root: /Users/alice/datasets/nuScenes\n",
        b"\nPrivate cache: /private/var/folders/aa/run\n",
        b"\nPayload: samples/CAM_FRONT/frame.jpg\n",
    ),
)
def test_methodology_tampering_cannot_use_frozen_privacy_exception(leak: bytes) -> None:
    source_root = Path(__file__).resolve().parents[1]
    files = _members(source_root)
    files["evidence/release-pipeline-plan.md"] += leak

    with pytest.raises(ReplayReleaseError, match="privacy scan"):
        validate_review_candidate_members(files)


def test_every_presentation_template_requires_all_four_placeholders() -> None:
    source_root = Path(__file__).resolve().parents[1]
    files = _members(source_root)
    files["presentation/README.md"] = files["presentation/README.md"].replace(
        M5_PRESENTATION_PLACEHOLDERS[0].encode(),
        b"missing",
    )

    with pytest.raises(ReplayReleaseError, match="placeholder count"):
        validate_review_candidate_members(files)
