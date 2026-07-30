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


def _safe_member(path: str) -> bytes:
    if path.endswith(".ndjson"):
        return canonical_json_bytes({"row": 1})
    if path.endswith(".json"):
        return canonical_json_bytes({"value": 1})
    if path.endswith(".svg"):
        return b'<svg xmlns="http://www.w3.org/2000/svg"><text>fixed</text></svg>\n'
    if path.startswith("presentation/"):
        return ("\n".join(M5_PRESENTATION_PLACEHOLDERS) + "\n").encode()
    return b"# Safe review\n"


def _members(source_root: Path) -> dict[str, bytes]:
    files = {path: _safe_member(path) for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS}
    files["machine/intent.json"] = (
        source_root / "examples/replay/m5-nuscenes-mini-replay-v1.json"
    ).read_bytes()
    authorities = {
        "evidence/release-pipeline-plan.md": "docs/m5-release-pipeline-plan.md",
        "evidence/release-pipeline-plan-review.md": (
            "docs/reviews/m5-release-pipeline-plan-review.md"
        ),
        "evidence/resource-scope-amendment.md": "docs/m5-resource-scope-amendment.md",
    }
    for candidate_path, source_path in authorities.items():
        files[candidate_path] = (source_root / source_path).read_bytes()
    return files


@pytest.mark.parametrize(
    "unsafe_placeholder",
    (
        b"# Review\n\nUnsafe: </Users/alice/private/nuScenes>\n",
        b"# Review\n\nUnsafe: <api_key=abcdefghijk>\n",
        b"# Review\n\nUnsafe: <samples/CAM_FRONT/frame.jpg>\n",
    ),
)
def test_methodology_placeholders_cannot_hide_private_values(
    unsafe_placeholder: bytes,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    files = _members(source_root)
    files["evidence/implementation-review.md"] = unsafe_placeholder

    with pytest.raises(ReplayReleaseError, match="privacy scan"):
        validate_review_candidate_members(files)


def test_abstract_relative_methodology_placeholder_is_allowed() -> None:
    source_root = Path(__file__).resolve().parents[1]
    files = _members(source_root)
    files["evidence/implementation-review.md"] = (
        b"# Review\n\nChecked reports/generated/candidate-<scientific-revision>.\n"
    )

    validate_review_candidate_members(files)
