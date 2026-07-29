"""Build and validate the curated Fusion Fault Bench M2 geometry release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    publish_directory_no_replace,
)
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    BOX_CORNER_MAX_ABS_TOLERANCE_M,
    FINITE_DIFFERENCE_MAX_ABS_TOLERANCE,
    GEOMETRY_ARTIFACT_CONTRACT,
    GEOMETRY_LOGICAL_COMMAND,
    GEOMETRY_VALIDATION_FILE,
    POINT_ROUND_TRIP_MAX_ABS_TOLERANCE_M,
    QUATERNION_SIGN_MAX_ABS_TOLERANCE,
    ROTATION_MAX_ABS_TOLERANCE,
    SYNTHETIC_DEPTH_MAX_ABS_TOLERANCE_M,
    SYNTHETIC_FIXTURE_ID,
    SYNTHETIC_FIXTURE_SHA256,
    SYNTHETIC_PROJECTION_MAX_ABS_TOLERANCE_PX,
    TRANSLATION_MAX_ABS_TOLERANCE_M,
)
from fusion_fault_bench.geometry_artifacts import load_geometry_validation_artifact

RELEASE_ID = "m2-geometry-v0.1.0"
RELEASE_SCHEMA = "ffb.m2-release/v1"
RELEASE_RELATIVE_PATH = f"reports/releases/{RELEASE_ID}"
RELEASE_SOURCE_REVISION = "cd9ce423d296a90dcf7c993c1c08b078dcfd4bd4"
RELEASE_LOCKFILE_SHA256 = "ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f"
RELEASE_PACKAGE_VERSION = "0.1.0"
RELEASE_MANIFEST_SHA256 = "7bfb5427c1ea5450a795bedef327457e5959860316bcd23f930e6eaa5917a068"
RELEASE_ARTIFACT_SHA256 = "09159042ca063b50762bf4150fb275b8a1760e4317ab74da8b0d24c133f42c90"
RELEASE_RUN_ID = "run:697f42275ec0c2bffd91718bf6806c4e8900318e597e7f5da727643200a88ff6"
RELEASE_PRIMARY_RUN_SHA256 = "462e15bb9da0b8caa43b6040b7f147fa64d84c7cdf51264acd02a1bd2eadc50f"
RELEASE_REPEAT_RUN_SHA256 = "5931411b25a169533a08849593af4416dd4a4d930054b93067f53199f0d27449"
RELEASE_ENVIRONMENT: dict[str, Any] = {
    "python_version": "3.12.13",
    "os_name": "Darwin",
    "os_release": "24.5.0",
    "machine": "arm64",
    "cpu_model": "Apple M3 Pro",
    "logical_cpu_count": 11,
    "memory_bytes": 19_327_352_832,
}

RECORD_DIRECTORY = Path("records/m2-geometry")
CURATED_SOURCE_FILES = (
    ("manifest.json", "manifest.json"),
    (GEOMETRY_VALIDATION_FILE, GEOMETRY_VALIDATION_FILE),
    ("run.json", "run.json"),
    ("payload-index.json", "source-payload-index.json"),
    ("_SUCCESS", "source-success.json"),
)
STABLE_SOURCE_FILES = (
    "manifest.json",
    GEOMETRY_VALIDATION_FILE,
    "payload-index.json",
)
DOCUMENT_PATHS = ("README.md", "claim-evidence.md", "verification.md")
FIGURE_PATH = Path("figures/geometry-validation-summary.svg")
FIGURE_TITLE = "M2 geometry validation: frozen gates and local profile"
OFFICIAL_DEVKIT_REVISION = "d9de17a73bdc06ce97a02f77ae7edb9b0406e851"
EXPECTED_DOCUMENT_SHA256 = {
    "README.md": "bfe5dfd10ed2133b481ea0c0299d9517d33b9dd4c49c2dd93af23793aa809567",
    "claim-evidence.md": "739a75c09a7a4ccf31112c964d35984a783b028ff73eb44f7d65a90b1bc0c3b7",
    "verification.md": "bbee8a6d2f9aacc01fb8e8207c3a754e63044c277f4908d49b237193257a0f4a",
}
DATASET_TERMS = {
    "source": "nuScenes v1.0-mini, Motional",
    "license": "CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms",
    "terms_url": "https://www.nuscenes.org/terms-of-use",
    "attribution": "nuScenes: A multimodal dataset for autonomous driving, Caesar et al., 2020",
    "non_endorsement": "Motional does not sponsor, approve, or endorse Fusion Fault Bench",
}
PUBLIC_CI = {
    "run_id": 30_437_837_817,
    "url": ("https://github.com/6ickomod3/fusion-fault-bench/actions/runs/30437837817"),
    "source_revision": RELEASE_SOURCE_REVISION,
    "conclusion": "success",
    "dataset_access": False,
}
CURATION_COMMAND = (
    "uv",
    "run",
    "python",
    "tools/m2_release.py",
    "build",
    "--primary-dir",
    "reports/generated/m2-geometry",
    "--repeat-dir",
    "<retained-repeat-dir>",
    "--primary-diagnostic",
    "reports/generated/m2-geometry-diagnostic.svg",
    "--repeat-diagnostic",
    "<retained-repeat-diagnostic>",
    "--documents-root",
    "<documents-root>",
    "--output-dir",
    RELEASE_RELATIVE_PATH,
)
VALIDATION_COMMAND = (
    "uv",
    "run",
    "python",
    "tools/m2_release.py",
    "validate",
    RELEASE_RELATIVE_PATH,
)


class ReleaseValidationError(ValueError):
    """Curated M2 evidence is incomplete, unsafe, or contradictory."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, *, cap: int = 16 * 1024 * 1024) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ReleaseValidationError("release member is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReleaseValidationError("release member must be a regular file")
    if metadata.st_size > cap:
        raise ReleaseValidationError("release member exceeds its byte cap")
    value = path.read_bytes()
    if len(value) != metadata.st_size:
        raise ReleaseValidationError("release member changed while reading")
    return value


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _file_entry(root: Path, relative: Path, *, source_name: str | None = None) -> dict[str, Any]:
    value = _read_regular(root / relative)
    entry: dict[str, Any] = {
        "path": relative.as_posix(),
        "byte_length": len(value),
        "sha256": _sha256(value),
    }
    if source_name is not None:
        entry["source_name"] = source_name
    return entry


def _verify_official_artifact(path: Path, *, expected_run_sha256: str) -> Any:
    try:
        artifact = load_geometry_validation_artifact(path)
    except (ArtifactValidationError, OSError, ValueError) as error:
        raise ReleaseValidationError("M2 source artifact failed strict validation") from error
    run = artifact.run
    if (
        artifact.artifact_sha256 != RELEASE_ARTIFACT_SHA256
        or artifact.run_sha256 != expected_run_sha256
        or run.run_id != RELEASE_RUN_ID
        or run.manifest_sha256 != RELEASE_MANIFEST_SHA256
        or run.git_revision != RELEASE_SOURCE_REVISION
        or run.lockfile_sha256 != RELEASE_LOCKFILE_SHA256
        or run.package_version != RELEASE_PACKAGE_VERSION
        or run.environment.model_dump(mode="json", by_alias=True) != RELEASE_ENVIRONMENT
        or run.source_dirty
        or run.status != "succeeded"
        or not artifact.validation.all_checks_passed
        or artifact.validation.dataset_terms.model_dump(mode="json", by_alias=True) != DATASET_TERMS
    ):
        raise ReleaseValidationError("M2 source artifact identity or evidence changed")
    return artifact


def _verify_repeat(primary_dir: Path, repeat_dir: Path) -> int:
    primary = _verify_official_artifact(
        primary_dir,
        expected_run_sha256=RELEASE_PRIMARY_RUN_SHA256,
    )
    repeat = _verify_official_artifact(
        repeat_dir,
        expected_run_sha256=RELEASE_REPEAT_RUN_SHA256,
    )
    if primary.run.run_id != repeat.run.run_id:
        raise ReleaseValidationError("M2 repeat run identity changed")
    comparisons = 0
    for name in STABLE_SOURCE_FILES:
        if _read_regular(primary_dir / name) != _read_regular(repeat_dir / name):
            raise ReleaseValidationError("M2 repeat scientific bytes changed")
        comparisons += 1
    return comparisons


def _verify_diagnostics(primary: Path, repeat: Path) -> None:
    first = _read_regular(primary, cap=5 * 1024 * 1024)
    second = _read_regular(repeat, cap=5 * 1024 * 1024)
    if first != second:
        raise ReleaseValidationError("M2 repeat diagnostic bytes changed")
    if (
        not first.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\n<svg ')
        or not first.endswith(b"</svg>\n")
        or b"<image" in first
        or b"data:" in first
    ):
        raise ReleaseValidationError("M2 diagnostic is not the expected vector-only SVG")


def _copy_documents(staging: Path, documents_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in DOCUMENT_PATHS:
        value = _read_regular(documents_root / name, cap=2 * 1024 * 1024)
        if _sha256(value) != EXPECTED_DOCUMENT_SHA256[name]:
            raise ReleaseValidationError(f"official M2 release document changed: {name}")
        if not value.startswith(b"# ") or not value.endswith(b"\n"):
            raise ReleaseValidationError(f"M2 release document framing is invalid: {name}")
        relative = Path(name)
        _write_exclusive(staging / relative, value)
        entries.append(_file_entry(staging, relative))
    return entries


def _copy_records(staging: Path, primary_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source_name, destination_name in CURATED_SOURCE_FILES:
        relative = RECORD_DIRECTORY / destination_name
        _write_exclusive(staging / relative, _read_regular(primary_dir / source_name))
        entries.append(_file_entry(staging, relative, source_name=source_name))
    return entries


def _render_summary_svg(artifact: Any) -> bytes:
    validation = artifact.validation
    synthetic = validation.synthetic_geometry_validation
    covariance = validation.covariance_validation
    ratios = [
        (
            "Rotation composition",
            synthetic.rotation_max_abs_error / ROTATION_MAX_ABS_TOLERANCE,
        ),
        (
            "Translation inverse",
            synthetic.translation_max_abs_error_m / TRANSLATION_MAX_ABS_TOLERANCE_M,
        ),
        (
            "Point round trip",
            synthetic.point_round_trip_max_abs_error_m / POINT_ROUND_TRIP_MAX_ABS_TOLERANCE_M,
        ),
        (
            "Quaternion sign",
            synthetic.quaternion_sign_max_abs_error / QUATERNION_SIGN_MAX_ABS_TOLERANCE,
        ),
        (
            "Synthetic projection",
            synthetic.projection_max_abs_error_px / SYNTHETIC_PROJECTION_MAX_ABS_TOLERANCE_PX,
        ),
        (
            "Synthetic depth",
            synthetic.depth_max_abs_error_m / SYNTHETIC_DEPTH_MAX_ABS_TOLERANCE_M,
        ),
        (
            "Synthetic box corners",
            synthetic.box_corner_max_abs_error_m / BOX_CORNER_MAX_ABS_TOLERANCE_M,
        ),
        (
            "Covariance finite difference",
            covariance.finite_difference_max_abs_error / FINITE_DIFFERENCE_MAX_ABS_TOLERANCE,
        ),
        *(
            (f"Covariance Monte Carlo {entry.entry}", entry.gate_ratio)
            for entry in covariance.covariance_entries
        ),
    ]
    width = 1200
    height = 880
    label_x = 64
    bar_x = 360
    bar_width = 560
    value_x = 948
    first_y = 132
    row_height = 39
    floor = 1e-6

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{FIGURE_TITLE}</title>",
        (
            "<desc>Preregistered synthetic numeric gate utilization and sanitized "
            "nuScenes-mini profile aggregates. Lower numeric gate utilization is better.</desc>"
        ),
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#182230}",
        ".muted{fill:#5b677a}.grid{stroke:#dce3ec;stroke-width:1}",
        ".gate{stroke:#bf3b3b;stroke-width:2}.bar{fill:#1f8a70}.track{fill:#edf2f6}",
        ".card{fill:#f7f9fc;stroke:#dce3ec;stroke-width:1}",
        "</style>",
        f'<rect width="1200" height="{height}" fill="#ffffff"/>',
        f'<text x="64" y="44" font-size="27" font-weight="700">{FIGURE_TITLE}</text>',
        (
            '<text class="muted" x="64" y="73" font-size="15">'
            "Synthetic error ÷ frozen tolerance on a log scale; values ≤ 1 pass. "
            "Local counts are allowlisted aggregates.</text>"
        ),
        f'<text x="{label_x}" y="105" font-size="14" font-weight="650">Numeric check</text>',
        f'<text x="{bar_x}" y="105" font-size="13" class="muted">10⁻⁶</text>',
        (
            f'<text x="{bar_x + bar_width / 3:.1f}" y="105" font-size="13" '
            'class="muted" text-anchor="middle">10⁻⁴</text>'
        ),
        (
            f'<text x="{bar_x + 2 * bar_width / 3:.1f}" y="105" font-size="13" '
            'class="muted" text-anchor="middle">10⁻²</text>'
        ),
        (
            f'<text x="{bar_x + bar_width}" y="105" font-size="13" '
            'fill="#a32f2f" text-anchor="end">1 (gate)</text>'
        ),
    ]
    for fraction in (0.0, 1 / 3, 2 / 3):
        x = bar_x + fraction * bar_width
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="114" x2="{x:.2f}" y2="554"/>')
    lines.append(
        f'<line class="gate" x1="{bar_x + bar_width}" y1="114" x2="{bar_x + bar_width}" y2="554"/>'
    )
    for index, (label, ratio) in enumerate(ratios):
        y = first_y + index * row_height
        bounded = max(floor, min(1.0, ratio))
        fraction = (math.log10(bounded) - math.log10(floor)) / -math.log10(floor)
        observed_width = max(2.0, fraction * bar_width) if ratio > 0.0 else 0.0
        display = "0" if ratio == 0.0 else f"{ratio:.4g}"
        lines.extend(
            [
                f'<text x="{label_x}" y="{y + 6}" font-size="14">{label}</text>',
                (
                    f'<rect class="track" x="{bar_x}" y="{y - 9}" width="{bar_width}" '
                    'height="18" rx="4"/>'
                ),
                (
                    f'<rect class="bar" x="{bar_x}" y="{y - 9}" '
                    f'width="{observed_width:.3f}" height="18" rx="4"/>'
                ),
                (
                    f'<text x="{value_x}" y="{y + 6}" font-size="14" '
                    f'font-variant-numeric="tabular-nums">gate ratio {display}</text>'
                ),
                (
                    f'<text x="1120" y="{y + 6}" font-size="13" '
                    'fill="#16745f" text-anchor="end">PASS</text>'
                ),
            ]
        )

    counts = validation.dataset_validation.expected_headline_counts
    cards = (
        ("Scenes", counts.scene),
        ("Samples", counts.sample),
        ("Sample annotations", counts.sample_annotation),
        ("Key-frame references checked", validation.dataset_validation.keyframe_blob_check_count),
    )
    card_y = 594
    card_width = 252
    gap = 18
    lines.extend(
        [
            '<text x="64" y="580" font-size="17" font-weight="700">'
            "Sanitized local nuScenes-mini profile</text>",
        ]
    )
    for index, (label, value) in enumerate(cards):
        x = 64 + index * (card_width + gap)
        lines.extend(
            [
                (
                    f'<rect class="card" x="{x}" y="{card_y}" width="{card_width}" '
                    'height="103" rx="9"/>'
                ),
                (
                    f'<text x="{x + 18}" y="{card_y + 34}" font-size="14" '
                    f'class="muted">{label}</text>'
                ),
                (
                    f'<text x="{x + 18}" y="{card_y + 75}" font-size="29" '
                    f'font-weight="700">{value:,}</text>'
                ),
                (
                    f'<text x="{x + card_width - 18}" y="{card_y + 75}" '
                    'font-size="13" fill="#16745f" text-anchor="end">PASS</text>'
                ),
            ]
        )
    lines.extend(
        [
            '<text class="muted" x="64" y="731" font-size="14">'
            "Counts are expected-profile values whose local match is attested; "
            "808 checks referenced-file existence, not file contents.</text>",
            '<text class="muted" x="64" y="756" font-size="14">'
            "Public summary does not authenticate dataset bytes. No per-scene, "
            "per-frame, token, path, residual, or diagnostic geometry is shown.</text>",
            '<text class="muted" x="64" y="792" font-size="12">'
            "Source: nuScenes v1.0-mini, Motional.</text>",
            '<text class="muted" x="64" y="812" font-size="12">'
            "Attribution: nuScenes: A multimodal dataset for autonomous driving, "
            "Caesar et al., 2020.</text>",
            '<text class="muted" x="64" y="832" font-size="12">'
            "License: CC-BY-NC-SA-4.0-plus-Motional-Dataset-Terms. "
            "Terms: https://www.nuscenes.org/terms-of-use</text>",
            '<text class="muted" x="64" y="852" font-size="12">'
            "Motional does not sponsor, approve, or endorse Fusion Fault Bench.</text>",
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _figure_entry(root: Path) -> dict[str, Any]:
    entry = _file_entry(root, FIGURE_PATH)
    entry.update(
        {
            "title": FIGURE_TITLE,
            "width_px": 1200,
            "height_px": 880,
            "source_records": [
                f"{RECORD_DIRECTORY.as_posix()}/manifest.json",
                f"{RECORD_DIRECTORY.as_posix()}/{GEOMETRY_VALIDATION_FILE}",
            ],
            "license": DATASET_TERMS["license"],
        }
    )
    return entry


def _expected_index(
    root: Path,
    *,
    curated_files: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    figures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if curated_files is None:
        curated_files = [
            _file_entry(
                root,
                RECORD_DIRECTORY / destination_name,
                source_name=source_name,
            )
            for source_name, destination_name in CURATED_SOURCE_FILES
        ]
    if documents is None:
        documents = [_file_entry(root, Path(name)) for name in DOCUMENT_PATHS]
    if figures is None:
        figures = [_figure_entry(root)]
    return {
        "schema": RELEASE_SCHEMA,
        "release_id": RELEASE_ID,
        "scope": "cpu-only-geometry-validation-and-local-nuscenes-mini-grounding",
        "source_revision": RELEASE_SOURCE_REVISION,
        "lockfile_sha256": RELEASE_LOCKFILE_SHA256,
        "package_version": RELEASE_PACKAGE_VERSION,
        "environment": RELEASE_ENVIRONMENT,
        "artifact_contract": GEOMETRY_ARTIFACT_CONTRACT,
        "manifest_sha256": RELEASE_MANIFEST_SHA256,
        "artifact_sha256": RELEASE_ARTIFACT_SHA256,
        "run_id": RELEASE_RUN_ID,
        "logical_command": list(GEOMETRY_LOGICAL_COMMAND),
        "synthetic_fixture": {
            "fixture_id": SYNTHETIC_FIXTURE_ID,
            "file_sha256": SYNTHETIC_FIXTURE_SHA256,
            "official_devkit_revision": OFFICIAL_DEVKIT_REVISION,
        },
        "primary_run_sha256": RELEASE_PRIMARY_RUN_SHA256,
        "repeat_run_sha256": RELEASE_REPEAT_RUN_SHA256,
        "record_directory": RECORD_DIRECTORY.as_posix(),
        "curated_files": curated_files,
        "documents": documents,
        "figures": figures,
        "dataset_terms": DATASET_TERMS,
        "dataset_authentication": "summary-does-not-authenticate-dataset-bytes",
        "verification": {
            "strict_bundle_validations": 2,
            "stable_file_comparisons": 3,
            "stable_files": list(STABLE_SOURCE_FILES),
            "diagnostic_bytes_identical_attested": True,
            "diagnostic_visual_inspection_attested": True,
            "adversarial_implementation_reviews_passed": 2,
            "adversarial_results_review_passed": True,
            "public_ci": PUBLIC_CI,
        },
        "curation_command": list(CURATION_COMMAND),
        "validation_command": list(VALIDATION_COMMAND),
    }


def build_release(
    *,
    primary_dir: Path,
    repeat_dir: Path,
    primary_diagnostic: Path,
    repeat_diagnostic: Path,
    documents_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the exact no-overwrite M2 release from retained local evidence."""

    output = output_dir.absolute()
    if os.path.lexists(output):
        raise FileExistsError("M2 release destination already exists")
    stable_comparisons = _verify_repeat(primary_dir, repeat_dir)
    if stable_comparisons != len(STABLE_SOURCE_FILES):
        raise ReleaseValidationError("M2 stable comparison count changed")
    _verify_diagnostics(primary_diagnostic, repeat_diagnostic)
    primary_artifact = _verify_official_artifact(
        primary_dir,
        expected_run_sha256=RELEASE_PRIMARY_RUN_SHA256,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".m2-release-", dir=output.parent))
    published = False
    try:
        curated_files = _copy_records(staging, primary_dir)
        documents = _copy_documents(staging, documents_root)
        _write_exclusive(staging / FIGURE_PATH, _render_summary_svg(primary_artifact))
        figures = [_figure_entry(staging)]
        index = _expected_index(
            staging,
            curated_files=curated_files,
            documents=documents,
            figures=figures,
        )
        _write_exclusive(staging / "release-index.json", canonical_json_bytes(index))
        validate_release(staging)
        publish_directory_no_replace(staging, output)
        published = True
        validate_release(output)
        return index
    except BaseException:
        if not published and staging.exists():
            shutil.rmtree(staging)
        raise


def _release_file_allowlist() -> set[Path]:
    return {
        Path("release-index.json"),
        FIGURE_PATH,
        *(Path(name) for name in DOCUMENT_PATHS),
        *(RECORD_DIRECTORY / destination_name for _, destination_name in CURATED_SOURCE_FILES),
    }


def _scan_release(root: Path) -> set[Path]:
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise ReleaseValidationError("M2 release directory is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ReleaseValidationError("M2 release root must be a real directory")
    files: set[Path] = set()
    for path in root.rglob("*"):
        metadata = os.lstat(path)
        relative = path.relative_to(root)
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseValidationError("M2 release contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseValidationError("M2 release contains a non-file member")
        files.add(relative)
    if files != _release_file_allowlist():
        raise ReleaseValidationError("M2 release file allowlist mismatch")
    return files


def _load_canonical_index(path: Path) -> dict[str, Any]:
    value = _read_regular(path)
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseValidationError("M2 release index is invalid JSON") from error
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise ReleaseValidationError("M2 release index must be an object")
    typed = cast(dict[str, Any], parsed)
    if canonical_json_bytes(typed) != value:
        raise ReleaseValidationError("M2 release index is not canonical JSON")
    return typed


def _validate_document_hashes(root: Path) -> None:
    for name in DOCUMENT_PATHS:
        if _sha256(_read_regular(root / name)) != EXPECTED_DOCUMENT_SHA256[name]:
            raise ReleaseValidationError(f"official M2 document changed: {name}")


def _reconstruct_bundle(root: Path) -> Any:
    temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix="ffb-m2-release-validate-",
        dir=temporary_parent,
    ) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for source_name, destination_name in CURATED_SOURCE_FILES:
            _write_exclusive(
                bundle / source_name,
                _read_regular(root / RECORD_DIRECTORY / destination_name),
            )
        return _verify_official_artifact(
            bundle,
            expected_run_sha256=RELEASE_PRIMARY_RUN_SHA256,
        )


def validate_release(root: Path) -> dict[str, Any]:
    """Validate the exact curated M2 release and return its canonical index."""

    release_root = root.absolute()
    _scan_release(release_root)
    _validate_document_hashes(release_root)
    artifact = _reconstruct_bundle(release_root)
    if _read_regular(release_root / FIGURE_PATH) != _render_summary_svg(artifact):
        raise ReleaseValidationError("M2 summary figure does not match curated evidence")
    if artifact.validation.dataset_validation.dataset_authentication != (
        "summary-does-not-authenticate-dataset-bytes"
    ):
        raise ReleaseValidationError("M2 dataset-authentication boundary changed")
    observed = _load_canonical_index(release_root / "release-index.json")
    expected = _expected_index(release_root)
    if observed != expected:
        raise ReleaseValidationError("M2 release index is incomplete or contradictory")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the no-overwrite M2 release")
    build.add_argument("--primary-dir", type=Path, required=True)
    build.add_argument("--repeat-dir", type=Path, required=True)
    build.add_argument("--primary-diagnostic", type=Path, required=True)
    build.add_argument("--repeat-diagnostic", type=Path, required=True)
    build.add_argument("--documents-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate curated M2 evidence")
    validate.add_argument("release_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the M2 release builder or validator."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build":
            index = build_release(
                primary_dir=arguments.primary_dir,
                repeat_dir=arguments.repeat_dir,
                primary_diagnostic=arguments.primary_diagnostic,
                repeat_diagnostic=arguments.repeat_diagnostic,
                documents_root=arguments.documents_root,
                output_dir=arguments.output_dir,
            )
            print(f"built {index['release_id']} artifact_sha256={index['artifact_sha256']}")
        else:
            index = validate_release(arguments.release_dir)
            print(f"valid {index['release_id']} artifact_sha256={index['artifact_sha256']}")
    except (ArtifactValidationError, ReleaseValidationError, FileExistsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
