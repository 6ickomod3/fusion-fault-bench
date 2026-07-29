from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fusion_fault_bench.artifacts import canonical_json_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
m2_release = importlib.import_module("tools.m2_release")
RELEASE_ROOT = REPOSITORY_ROOT / "reports" / "releases" / m2_release.RELEASE_ID
RELEASE_TOOL = REPOSITORY_ROOT / "tools" / "m2_release.py"


def _copy_release(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = tmp_path / m2_release.RELEASE_ID
    shutil.copytree(RELEASE_ROOT, candidate)
    return candidate


def _source_bundle_from_release(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for source_name, destination_name in m2_release.CURATED_SOURCE_FILES:
        shutil.copyfile(
            RELEASE_ROOT / m2_release.RECORD_DIRECTORY / destination_name,
            bundle / source_name,
        )
    return bundle


def test_curated_m2_release_validates() -> None:
    index = m2_release.validate_release(RELEASE_ROOT)

    assert index["release_id"] == m2_release.RELEASE_ID
    assert index["artifact_sha256"] == m2_release.RELEASE_ARTIFACT_SHA256
    assert index["verification"]["adversarial_results_review_passed"] is True
    assert not (RELEASE_ROOT / "m2-geometry-diagnostic.svg").exists()
    assert not any(path.name == "m2-geometry-diagnostic.svg" for path in RELEASE_ROOT.rglob("*"))


def test_m2_release_cli_validates_without_dataset_environment() -> None:
    completed = subprocess.run(
        [sys.executable, str(RELEASE_TOOL), "validate", str(RELEASE_ROOT)],
        cwd=REPOSITORY_ROOT,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert f"valid {m2_release.RELEASE_ID}" in completed.stdout
    assert "NUSCENES_ROOT" not in completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("README.md"),
        Path("figures/geometry-validation-summary.svg"),
        m2_release.RECORD_DIRECTORY / "geometry-validation.json",
        Path("release-index.json"),
    ],
)
def test_m2_release_rejects_modified_members(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    candidate = _copy_release(tmp_path)
    path = candidate / relative_path
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(m2_release.ReleaseValidationError):
        m2_release.validate_release(candidate)


def test_m2_release_rejects_noncanonical_index_even_if_semantically_equal(
    tmp_path: Path,
) -> None:
    candidate = _copy_release(tmp_path)
    path = candidate / "release-index.json"
    parsed = json.loads(path.read_bytes())
    path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    with pytest.raises(m2_release.ReleaseValidationError, match="canonical"):
        m2_release.validate_release(candidate)


def test_m2_release_rejects_extra_missing_and_symlink_members(tmp_path: Path) -> None:
    extra = _copy_release(tmp_path / "extra")
    (extra / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(m2_release.ReleaseValidationError, match="allowlist"):
        m2_release.validate_release(extra)

    missing = _copy_release(tmp_path / "missing")
    (missing / "claim-evidence.md").unlink()
    with pytest.raises(m2_release.ReleaseValidationError, match="allowlist"):
        m2_release.validate_release(missing)

    linked = _copy_release(tmp_path / "linked")
    target = linked / "README.md"
    target.unlink()
    target.symlink_to(RELEASE_ROOT / "README.md")
    with pytest.raises(m2_release.ReleaseValidationError, match="symlink"):
        m2_release.validate_release(linked)


def test_m2_release_rejects_terms_mutation_after_full_rechain(tmp_path: Path) -> None:
    candidate = _copy_release(tmp_path)
    result_path = candidate / m2_release.RECORD_DIRECTORY / m2_release.GEOMETRY_VALIDATION_FILE
    result = json.loads(result_path.read_bytes())
    result["dataset_terms"]["source"] = "changed"
    result_path.write_bytes(canonical_json_bytes(result))

    with pytest.raises(m2_release.ReleaseValidationError):
        m2_release.validate_release(candidate)


def test_m2_release_build_refuses_overwrite_before_loading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        m2_release.build_release(
            primary_dir=tmp_path / "missing-primary",
            repeat_dir=tmp_path / "missing-repeat",
            primary_diagnostic=tmp_path / "missing-primary.svg",
            repeat_diagnostic=tmp_path / "missing-repeat.svg",
            documents_root=tmp_path / "missing-documents",
            output_dir=output,
        )


def test_m2_release_build_rejects_primary_reused_as_repeat(tmp_path: Path) -> None:
    bundle = _source_bundle_from_release(tmp_path)

    with pytest.raises(m2_release.ReleaseValidationError):
        m2_release.build_release(
            primary_dir=bundle,
            repeat_dir=bundle,
            primary_diagnostic=tmp_path / "unused-primary.svg",
            repeat_diagnostic=tmp_path / "unused-repeat.svg",
            documents_root=tmp_path / "unused-documents",
            output_dir=tmp_path / "new-release",
        )
