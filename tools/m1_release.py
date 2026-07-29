"""Build and validate the curated Fusion Fault Bench M1 analytic release."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from fusion_fault_bench.artifacts import (
    AGGREGATE_METRICS_FILE,
    ANALYTIC_VALIDATION_FILE,
    CROSSOVERS_FILE,
    MANIFEST_FILE,
    PAYLOAD_INDEX_FILE,
    RUN_FILE,
    SEQUENCE_METRICS_FILE,
    SUCCESS_FILE,
    ArtifactValidationError,
    LoadedArtifact,
    canonical_json_bytes,
    compute_artifact_digest,
    compute_run_record_digest,
    derive_run_id,
    load_artifact,
    publish_directory_no_replace,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import (
    AnalyticValidationV1Alpha1,
    PayloadIndexV1Alpha1,
    SuccessMarkerV1Alpha1,
)
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.io import validate_manifest_mapping
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    RunRecordV1Alpha1,
)
from fusion_fault_bench.inference import pava_non_decreasing

RELEASE_ID = "m1-analytic-v0.1.0"
RELEASE_SCHEMA = "ffb.m1-release/v1alpha1"
RELEASE_RELATIVE_PATH = f"reports/releases/{RELEASE_ID}"
WITHHELD_SOURCE_REVISION = "1649dec5de387dd8b408a14678fa0acad0818735"
RELEASE_SOURCE_REVISION = "524c8f70ece3eca2e61796165b23ffe51baadfbc"
RELEASE_LOCKFILE_SHA256 = "ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f"
RELEASE_PACKAGE_VERSION = "0.1.0"
RELEASE_ENVIRONMENT: dict[str, Any] = {
    "python_version": "3.12.13",
    "os_name": "Darwin",
    "os_release": "24.5.0",
    "machine": "arm64",
    "cpu_model": "Apple M3 Pro",
    "logical_cpu_count": 11,
    "memory_bytes": 19_327_352_832,
}

EXPECTED_EXPERIMENTS = (
    (
        "analytic-camera-x-bias",
        "analytic-camera-x-bias-a603d090f77a",
        "a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e",
    ),
    (
        "analytic-camera-noise-correctly-reported",
        "analytic-camera-noise-correctly-reported-3ea7ffc2949c",
        "3ea7ffc2949cf99f20d20ec18844f0b8dc3b3ebb81e13e926f7440b7c5084176",
    ),
    (
        "analytic-camera-noise-underreported",
        "analytic-camera-noise-underreported-9d26e1b33f1f",
        "9d26e1b33f1fd2e35b0de90703a960d2eba6bb26bd2219bce6f0bb82480f4ac4",
    ),
)

EXPECTED_RELEASE_IDENTITIES: dict[str, dict[str, str]] = {
    "analytic-camera-x-bias": {
        "artifact_sha256": "3717c2b3fdce9e9f2bc43463434fde28a4d24dd9ca1d72e451b3d9d5273c2959",
        "run_sha256": "6cc2cefdb0a37aad5c58da8b55462729d294d5e99bb707b866c9d54b2e8ad63c",
        "repeat_run_sha256": ("bb37c2f21acdc8a1e542c06a19f776bbd92cab1682e7355d41437e011bd816f4"),
        "run_id": "run:cc67cf35ac9b74116cbb2f39c934bbfddb041036c58194722090979549460687",
    },
    "analytic-camera-noise-correctly-reported": {
        "artifact_sha256": "51abb5043ddffd633c0fa81ea5d69dc6c1246d092185f22261f183915f911467",
        "run_sha256": "085d928b68a785d59b41f1879d4d262030a855ef0677771866d2d91efe6268e3",
        "repeat_run_sha256": ("5bda8ece678016346b57857d55a46db8eed7d1eabfb24c6cbb9b078273841585"),
        "run_id": "run:4d601cfe04c83839a25088a061ab9a6d4b3c29ffcec71ea4f6ce64c3343f4340",
    },
    "analytic-camera-noise-underreported": {
        "artifact_sha256": "8a3c2179e49cdc2ae994d9c791185a546788252710a0e80fca73cf39305165e7",
        "run_sha256": "7f1a2fa90eeb2e83267c76c1863eb23c707ecc893604cb025166a2f00da7f594",
        "repeat_run_sha256": ("dd54fb25500c7f47fc7e9c573b9aba8da976cc4cc4afbd77d9382189240503f3"),
        "run_id": "run:a58e65ece1915d49c485f36ee478f97fce35c18763f7fa4cec2ffd44bd90b234",
    },
}

EXPECTED_MANIFEST_PATHS = {
    "analytic-camera-x-bias": "examples/manifests/analytic-bias-v1alpha1.json",
    "analytic-camera-noise-correctly-reported": (
        "examples/manifests/analytic-noise-correct-v1alpha1.json"
    ),
    "analytic-camera-noise-underreported": (
        "examples/manifests/analytic-noise-underreported-v1alpha1.json"
    ),
}

STABLE_SOURCE_FILES = (
    MANIFEST_FILE,
    SEQUENCE_METRICS_FILE,
    AGGREGATE_METRICS_FILE,
    CROSSOVERS_FILE,
    ANALYTIC_VALIDATION_FILE,
    PAYLOAD_INDEX_FILE,
)
INDEXED_SCIENTIFIC_FILES = STABLE_SOURCE_FILES[:-1]

CURATED_SOURCE_FILES = (
    (MANIFEST_FILE, MANIFEST_FILE),
    (AGGREGATE_METRICS_FILE, AGGREGATE_METRICS_FILE),
    (CROSSOVERS_FILE, CROSSOVERS_FILE),
    (ANALYTIC_VALIDATION_FILE, ANALYTIC_VALIDATION_FILE),
    (RUN_FILE, RUN_FILE),
    (PAYLOAD_INDEX_FILE, "source-payload-index.json"),
    (SUCCESS_FILE, "source-success.json"),
)

DOCUMENT_PATHS = ("README.md", "claim-evidence.md", "verification.md")
EXPECTED_DOCUMENT_SHA256 = {
    "README.md": "97d515473f1192ba0f77a70405587470af27e6741842a311cd27569388f80327",
    "claim-evidence.md": "45c6a70b8e0778008f56be8d7496cbd8cbc6a8a7aa10adaf2f1e6272a45694bb",
    "verification.md": "21ce9019f22e9f224da932e72196a458e837ee91117cfde5d5eab14c9a0258a5",
}
ARTIFACT_CONTRACT = "ffb.scientific-payload/v1"
CURATION_COMMAND = (
    "uv",
    "run",
    "python",
    "tools/m1_release.py",
    "build",
    "--primary-root",
    "reports/generated",
    "--repeat-root",
    "<repeat-root>",
    "--withheld-root",
    "<withheld-root>",
    "--documents-root",
    "<documents-root>",
    "--output-dir",
    RELEASE_RELATIVE_PATH,
)
VALIDATION_COMMAND = (
    "uv",
    "run",
    "python",
    "tools/m1_release.py",
    "validate",
    RELEASE_RELATIVE_PATH,
)
OMISSION_POLICY = (
    "Synthetic sequence-metrics.ndjson rows remain reproducible from the "
    "frozen manifests and source revision; curated Git evidence retains "
    "their exact byte length, SHA-256, and record count."
)
GENERIC_CPU_MODELS = frozenset(
    {
        "aarch64",
        "amd64",
        "arm",
        "arm64",
        "i386",
        "i686",
        "unknown",
        "unknown-cpu",
        "x86",
        "x86_64",
    }
)


class ReleaseValidationError(ValueError):
    """Curated release evidence is incomplete, noncanonical, or contradictory."""


@dataclass(frozen=True, slots=True)
class FigurePoint:
    """One aggregate signed-contrast estimate and pointwise interval."""

    magnitude: float
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class FigureSeries:
    """One raw and PAVA-fitted curve."""

    identifier: str
    label: str
    color: str
    points: tuple[FigurePoint, ...]
    fitted: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ValidatedCuratedExperiment:
    """Cross-bound curated records used to reconstruct the release index."""

    allowed_files: frozenset[Path]
    aggregates: tuple[AggregateMetricRecordV1Alpha1, ...]
    expected_entry: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ReleaseValidationError(f"{label} must be a JSON object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseValidationError(f"{label} must be a JSON array")
    return value


def _safe_relative_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ReleaseValidationError(f"{label} must be a nonempty POSIX path")
    parts = raw.split("/")
    if raw.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ReleaseValidationError(f"{label} must be repository-relative")
    return Path(*parts)


def _specific_cpu_model(value: str) -> bool:
    normalized = value.strip()
    return bool(normalized) and normalized.casefold() not in GENERIC_CPU_MODELS


def _read_bytes(path: Path, *, cap: int = 512 * 1024 * 1024) -> bytes:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReleaseValidationError(f"release member must be a regular file: {path}")
    if metadata.st_size > cap:
        raise ReleaseValidationError(f"release member exceeds its byte cap: {path}")
    value = path.read_bytes()
    if len(value) != metadata.st_size:
        raise ReleaseValidationError(f"release member changed while reading: {path}")
    return value


def _file_entry(root: Path, relative: Path, *, source_name: str | None = None) -> dict[str, Any]:
    value = _read_bytes(root / relative)
    entry: dict[str, Any] = {
        "path": relative.as_posix(),
        "byte_length": len(value),
        "sha256": _sha256_bytes(value),
    }
    if source_name is not None:
        entry["source_name"] = source_name
    return entry


def _load_expected(root: Path) -> dict[str, LoadedArtifact]:
    loaded: dict[str, LoadedArtifact] = {}
    for experiment, directory, expected_manifest_digest in EXPECTED_EXPERIMENTS:
        artifact = load_artifact(root / directory)
        if artifact.manifest.experiment != experiment:
            raise ReleaseValidationError(f"unexpected experiment in {directory}")
        if sha256_digest(artifact.manifest) != expected_manifest_digest:
            raise ReleaseValidationError(f"frozen manifest digest changed for {experiment}")
        loaded[experiment] = artifact
    return loaded


def _common_provenance(artifacts: Mapping[str, LoadedArtifact]) -> dict[str, Any]:
    revisions = {artifact.run.git_revision for artifact in artifacts.values()}
    lock_digests = {artifact.run.lockfile_sha256 for artifact in artifacts.values()}
    package_versions = {artifact.run.package_version for artifact in artifacts.values()}
    environments = {
        canonical_json_bytes(artifact.run.environment).decode("utf-8")
        for artifact in artifacts.values()
    }
    if len(revisions) != 1 or len(lock_digests) != 1 or len(package_versions) != 1:
        raise ReleaseValidationError("M1 artifacts do not share one source and lock provenance")
    if len(environments) != 1:
        raise ReleaseValidationError("M1 artifacts do not share one runtime environment")
    if any(artifact.run.source_dirty for artifact in artifacts.values()):
        raise ReleaseValidationError("release artifacts must record source_dirty=false")
    first = next(iter(artifacts.values()))
    if not _specific_cpu_model(first.run.environment.cpu_model):
        raise ReleaseValidationError("release artifacts require a named CPU model")
    return {
        "source_revision": next(iter(revisions)),
        "lockfile_sha256": next(iter(lock_digests)),
        "package_version": next(iter(package_versions)),
        "environment": first.run.environment.model_dump(mode="json", by_alias=True),
    }


def _verify_repeat(
    primary: Mapping[str, LoadedArtifact],
    repeat: Mapping[str, LoadedArtifact],
) -> tuple[int, list[dict[str, Any]]]:
    comparisons = 0
    run_records: list[dict[str, Any]] = []
    for experiment, directory, _ in EXPECTED_EXPERIMENTS:
        first = primary[experiment]
        second = repeat[experiment]
        expected_identity = EXPECTED_RELEASE_IDENTITIES[experiment]
        if first.run.run_id != second.run.run_id:
            raise ReleaseValidationError(f"repeat run_id changed for {experiment}")
        if first.run.environment != second.run.environment:
            raise ReleaseValidationError(f"repeat environment changed for {experiment}")
        if first.artifact_sha256 != second.artifact_sha256:
            raise ReleaseValidationError(f"repeat artifact digest changed for {experiment}")
        if first.run_sha256 == second.run_sha256:
            raise ReleaseValidationError("repeat run digest did not capture volatile provenance")
        if (
            first.artifact_sha256 != expected_identity["artifact_sha256"]
            or first.run_sha256 != expected_identity["run_sha256"]
            or first.run.run_id != expected_identity["run_id"]
            or second.run_sha256 != expected_identity["repeat_run_sha256"]
        ):
            raise ReleaseValidationError(f"official release identity changed for {experiment}")
        for path in STABLE_SOURCE_FILES:
            if (first.path / path).read_bytes() != (second.path / path).read_bytes():
                raise ReleaseValidationError(
                    f"repeat scientific bytes changed for {experiment}: {path}"
                )
            comparisons += 1
        run_records.append(
            {
                "experiment": experiment,
                "directory": directory,
                "primary_run_sha256": first.run_sha256,
                "repeat_run_sha256": second.run_sha256,
            }
        )
    return comparisons, run_records


def _without_run_id(value: object) -> object:
    if isinstance(value, BaseModel):
        return _without_run_id(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value).items()
        return {str(key): _without_run_id(item) for key, item in items if str(key) != "run_id"}
    if isinstance(value, (list, tuple)):
        return [_without_run_id(item) for item in cast(Sequence[object], value)]
    return value


def _scientific_views(artifact: LoadedArtifact) -> tuple[tuple[str, object, int], ...]:
    return (
        ("manifest.json", artifact.manifest, 1),
        ("sequence-metrics.ndjson", artifact.metrics, len(artifact.metrics)),
        ("aggregate-metrics.ndjson", artifact.aggregates, len(artifact.aggregates)),
        ("crossovers.ndjson", artifact.crossovers, len(artifact.crossovers)),
        ("analytic-validation.json", artifact.analytic_validation, 1),
    )


def _verify_withheld_equivalence(
    primary: Mapping[str, LoadedArtifact],
    withheld_root: Path,
) -> tuple[int, int]:
    withheld = _load_expected(withheld_root)
    if {artifact.run.git_revision for artifact in withheld.values()} != {WITHHELD_SOURCE_REVISION}:
        raise ReleaseValidationError("withheld artifacts do not come from the amended revision")
    file_comparisons = 0
    record_count = 0
    for experiment, _, _ in EXPECTED_EXPERIMENTS:
        current_views = _scientific_views(primary[experiment])
        withheld_views = _scientific_views(withheld[experiment])
        for (current_name, current_value, current_count), (
            withheld_name,
            withheld_value,
            withheld_count,
        ) in zip(current_views, withheld_views, strict=True):
            if current_name != withheld_name or current_count != withheld_count:
                raise ReleaseValidationError(f"withheld record layout changed for {experiment}")
            if _without_run_id(current_value) != _without_run_id(withheld_value):
                raise ReleaseValidationError(
                    f"scientific values changed from withheld execution: "
                    f"{experiment}/{current_name}"
                )
            file_comparisons += 1
            record_count += current_count
    return file_comparisons, record_count


def _contrast_series(
    aggregates: Sequence[AggregateMetricRecordV1Alpha1],
    *,
    direction: str,
    identifier: str,
    label: str,
    color: str,
) -> FigureSeries:
    rows = [
        record
        for record in aggregates
        if record.metric_name == "fused-minus-healthy"
        and record.severity.direction in {"identity", direction}
    ]
    if len(rows) < 2:
        raise ReleaseValidationError(f"figure series {identifier} has insufficient points")
    points: list[FigurePoint] = []
    for row in rows:
        if row.estimate is None or row.interval_lower is None or row.interval_upper is None:
            raise ReleaseValidationError(f"figure series {identifier} contains undefined values")
        points.append(
            FigurePoint(
                magnitude=row.severity.magnitude,
                estimate=row.estimate,
                lower=row.interval_lower,
                upper=row.interval_upper,
            )
        )
    fitted = tuple(
        float(value) for value in pava_non_decreasing([point.estimate for point in points])
    )
    return FigureSeries(
        identifier=identifier,
        label=label,
        color=color,
        points=tuple(points),
        fitted=fitted,
    )


def _format_tick(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1.0:
        return f"{value:.3g}"
    if magnitude >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _render_svg(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    series: Sequence[FigureSeries],
) -> bytes:
    width = 1200
    height = 700
    left = 105
    right = 45
    top = 115
    bottom = 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for item in series for point in item.points]
    x_values = sorted({point.magnitude for point in all_points})
    if len(x_values) < 2:
        raise ReleaseValidationError("figure requires at least two severity values")
    x_min = min(x_values)
    x_max = max(x_values)
    y_values = [0.0]
    for item in series:
        y_values.extend(point.lower for point in item.points)
        y_values.extend(point.upper for point in item.points)
        y_values.extend(item.fitted)
    y_min = min(y_values)
    y_max = max(y_values)
    y_span = y_max - y_min
    y_padding = max(0.001, 0.12 * y_span)
    y_min -= y_padding
    y_max += y_padding

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)}</desc>",
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
        ".grid{stroke:#dce2ea;stroke-width:1}",
        ".axis{stroke:#687386;stroke-width:1.4}",
        ".zero{stroke:#111827;stroke-width:1.8;stroke-dasharray:8 6}",
        ".error{stroke-width:1.6}",
        ".pava{fill:none;stroke-width:3.2;stroke-linejoin:round;stroke-linecap:round}",
        ".point{stroke:#fff;stroke-width:1.8}",
        "</style>",
        '<rect width="1200" height="700" fill="#ffffff"/>',
        f'<text x="{left}" y="42" font-size="27" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left}" y="72" font-size="15" fill="#526078">{html.escape(subtitle)}</text>',
    ]

    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        y = y_position(value)
        lines.append(
            f'<line class="grid" x1="{left}" y1="{y:.3f}" x2="{left + plot_width}" y2="{y:.3f}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{y + 5:.3f}" font-size="13" '
            f'text-anchor="end">{html.escape(_format_tick(value))}</text>'
        )
    for value in x_values:
        x = x_position(value)
        lines.append(
            f'<line class="grid" x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + plot_height}"/>'
        )
        lines.append(
            f'<text x="{x:.3f}" y="{top + plot_height + 27}" font-size="13" '
            f'text-anchor="middle">{html.escape(_format_tick(value))}</text>'
        )

    zero_y = y_position(0.0)
    lines.extend(
        [
            (
                f'<line id="zero-line" class="zero" x1="{left}" y1="{zero_y:.3f}" '
                f'x2="{left + plot_width}" y2="{zero_y:.3f}"/>'
            ),
            (f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>'),
            (
                f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
                f'x2="{left + plot_width}" y2="{top + plot_height}"/>'
            ),
        ]
    )

    for item in series:
        polyline = " ".join(
            f"{x_position(point.magnitude):.3f},{y_position(fitted):.3f}"
            for point, fitted in zip(item.points, item.fitted, strict=True)
        )
        lines.append(
            f'<g id="series-{html.escape(item.identifier)}" data-point-count="{len(item.points)}">'
        )
        lines.append(
            f'<polyline class="pava pava-curve" stroke="{item.color}" points="{polyline}"/>'
        )
        for point in item.points:
            x = x_position(point.magnitude)
            estimate_y = y_position(point.estimate)
            lower_y = y_position(point.lower)
            upper_y = y_position(point.upper)
            tooltip = (
                f"{item.label}: severity={point.magnitude:g}, "
                f"D_H={point.estimate:.8g}, 95% interval="
                f"[{point.lower:.8g}, {point.upper:.8g}]"
            )
            lines.extend(
                [
                    (
                        f'<g class="raw-point" data-severity="{point.magnitude:.17g}" '
                        f'data-estimate="{point.estimate:.17g}">'
                    ),
                    f"<title>{html.escape(tooltip)}</title>",
                    (
                        f'<line class="error" stroke="{item.color}" x1="{x:.3f}" '
                        f'y1="{upper_y:.3f}" x2="{x:.3f}" y2="{lower_y:.3f}"/>'
                    ),
                    (
                        f'<line class="error" stroke="{item.color}" x1="{x - 6:.3f}" '
                        f'y1="{upper_y:.3f}" x2="{x + 6:.3f}" y2="{upper_y:.3f}"/>'
                    ),
                    (
                        f'<line class="error" stroke="{item.color}" x1="{x - 6:.3f}" '
                        f'y1="{lower_y:.3f}" x2="{x + 6:.3f}" y2="{lower_y:.3f}"/>'
                    ),
                    (
                        f'<circle class="point" fill="{item.color}" cx="{x:.3f}" '
                        f'cy="{estimate_y:.3f}" r="5.5"/>'
                    ),
                    "</g>",
                ]
            )
        lines.append("</g>")

    legend_x = left + 18
    legend_y = top + 24
    for index, item in enumerate(series):
        y = legend_y + 27 * index
        lines.extend(
            [
                (
                    f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" '
                    f'stroke="{item.color}" stroke-width="3.2"/>'
                ),
                (
                    f'<circle cx="{legend_x + 15}" cy="{y}" r="4.5" '
                    f'fill="{item.color}" stroke="#fff" stroke-width="1.5"/>'
                ),
                (
                    f'<text x="{legend_x + 40}" y="{y + 5}" font-size="14">'
                    f"{html.escape(item.label)}</text>"
                ),
            ]
        )

    lines.extend(
        [
            (
                f'<text x="{left + plot_width / 2:.3f}" y="{height - 34}" '
                f'font-size="15" text-anchor="middle">{html.escape(x_label)}</text>'
            ),
            (
                f'<text x="25" y="{top + plot_height / 2:.3f}" font-size="15" '
                f'text-anchor="middle" transform="rotate(-90 25 '
                f'{top + plot_height / 2:.3f})">D_H = fusion loss - healthy loss (m²)</text>'
            ),
            (
                f'<text x="{left + plot_width}" y="{height - 13}" font-size="12" '
                f'text-anchor="end" fill="#526078">'
                "Points: raw means and 95% pointwise intervals; lines: PAVA fits. "
                "Below zero favors fusion.</text>"
            ),
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_documents(staging: Path, documents_root: Path) -> list[dict[str, Any]]:
    root = documents_root.absolute()
    metadata = os.lstat(root)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ReleaseValidationError("release document root must be a real directory")
    entries: list[dict[str, Any]] = []
    for name in DOCUMENT_PATHS:
        value = _read_bytes(root / name, cap=2 * 1024 * 1024)
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleaseValidationError(f"release document is not UTF-8: {name}") from error
        if not text.startswith("# ") or not text.endswith("\n"):
            raise ReleaseValidationError(
                f"release document must have a heading and trailing newline: {name}"
            )
        if _sha256_bytes(value) != EXPECTED_DOCUMENT_SHA256[name]:
            raise ReleaseValidationError(f"official release document changed: {name}")
        relative = Path(name)
        _write_exclusive(staging / relative, value)
        entries.append(_file_entry(staging, relative))
    return entries


def _copy_curated_records(
    staging: Path,
    artifacts: Mapping[str, LoadedArtifact],
) -> list[dict[str, Any]]:
    experiment_entries: list[dict[str, Any]] = []
    for experiment, directory, manifest_digest in EXPECTED_EXPERIMENTS:
        artifact = artifacts[experiment]
        record_directory = Path("records") / directory
        curated_entries: list[dict[str, Any]] = []
        for source_name, destination_name in CURATED_SOURCE_FILES:
            destination = record_directory / destination_name
            _write_exclusive(staging / destination, (artifact.path / source_name).read_bytes())
            curated_entries.append(_file_entry(staging, destination, source_name=source_name))
        sequence_entry = next(
            entry for entry in artifact.payload_index.files if entry.path == SEQUENCE_METRICS_FILE
        )
        run_value = artifact.run.model_dump(mode="json", by_alias=True)
        duration_seconds = (
            artifact.run.ended_at - artifact.run.started_at
            if artifact.run.ended_at is not None
            else None
        )
        if duration_seconds is None:
            raise ReleaseValidationError(f"run is missing ended_at for {experiment}")
        experiment_entries.append(
            {
                "experiment": experiment,
                "record_directory": record_directory.as_posix(),
                "manifest_sha256": manifest_digest,
                "artifact_sha256": artifact.artifact_sha256,
                "run_sha256": artifact.run_sha256,
                "run_id": artifact.run.run_id,
                "logical_command": list(artifact.run.command),
                "started_at": run_value["started_at"],
                "ended_at": run_value["ended_at"],
                "evaluation_duration_seconds": duration_seconds.total_seconds(),
                "curated_files": curated_entries,
                "omitted_sequence_metrics": {
                    "source_name": SEQUENCE_METRICS_FILE,
                    "byte_length": sequence_entry.byte_length,
                    "sha256": sequence_entry.sha256,
                    "record_count": len(artifact.metrics),
                    "reason": "Reproducible synthetic sequence rows are omitted from Git history.",
                },
            }
        )
    return experiment_entries


def _figure_payloads(
    aggregates: Mapping[str, Sequence[AggregateMetricRecordV1Alpha1]],
) -> tuple[tuple[Path, str, bytes, int, int], ...]:
    bias = aggregates["analytic-camera-x-bias"]
    correct = aggregates["analytic-camera-noise-correctly-reported"]
    underreported = aggregates["analytic-camera-noise-underreported"]
    figures = (
        (
            Path("figures/bias-fused-minus-healthy.svg"),
            "Signed camera x-bias: fixed fusion relative to healthy LiDAR",
            "N=200 sequences; B=2,000 paired bootstrap replicates; both configured directions.",
            "Configured camera x-bias magnitude (m)",
            (
                _contrast_series(
                    bias,
                    direction="negative",
                    identifier="negative",
                    label="Negative x direction",
                    color="#2563eb",
                ),
                _contrast_series(
                    bias,
                    direction="positive",
                    identifier="positive",
                    label="Positive x direction",
                    color="#e76f51",
                ),
            ),
        ),
        (
            Path("figures/noise-reporting-fused-minus-healthy.svg"),
            "Camera noise: correct covariance reporting versus underreporting",
            "Identical axes; N=200 sequences; B=2,000 paired bootstrap replicates.",
            "Configured camera standard-deviation scale",
            (
                _contrast_series(
                    correct,
                    direction="increase",
                    identifier="correctly-reported",
                    label="Correctly reported covariance",
                    color="#2563eb",
                ),
                _contrast_series(
                    underreported,
                    direction="increase",
                    identifier="underreported",
                    label="Underreported covariance",
                    color="#d1495b",
                ),
            ),
        ),
    )
    payloads: list[tuple[Path, str, bytes, int, int]] = []
    for path, title, subtitle, x_label, series in figures:
        value = _render_svg(
            title=title,
            subtitle=subtitle,
            x_label=x_label,
            series=series,
        )
        payloads.append(
            (
                path,
                title,
                value,
                sum(len(item.points) for item in series),
                len(series),
            )
        )
    return tuple(payloads)


def _write_figures(
    staging: Path,
    artifacts: Mapping[str, LoadedArtifact],
) -> list[dict[str, Any]]:
    aggregates = {experiment: artifact.aggregates for experiment, artifact in artifacts.items()}
    entries: list[dict[str, Any]] = []
    for path, title, value, raw_point_count, pava_curve_count in _figure_payloads(aggregates):
        _write_exclusive(staging / path, value)
        entry = _file_entry(staging, path)
        entry.update(
            {
                "title": title,
                "width_px": 1200,
                "height_px": 700,
                "raw_point_count": raw_point_count,
                "pava_curve_count": pava_curve_count,
            }
        )
        entries.append(entry)
    return entries


def build_release(
    *,
    primary_root: Path,
    repeat_root: Path,
    documents_root: Path,
    output_dir: Path,
    withheld_root: Path,
) -> dict[str, Any]:
    """Build one no-overwrite curated M1 release from strict source artifacts."""

    output = output_dir.absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"release destination already exists: {output}")
    primary = _load_expected(primary_root)
    repeat = _load_expected(repeat_root)
    provenance = _common_provenance(primary)
    if _common_provenance(repeat) != provenance:
        raise ReleaseValidationError("repeat artifacts have different common provenance")
    expected_provenance = {
        "source_revision": RELEASE_SOURCE_REVISION,
        "lockfile_sha256": RELEASE_LOCKFILE_SHA256,
        "package_version": RELEASE_PACKAGE_VERSION,
        "environment": RELEASE_ENVIRONMENT,
    }
    if provenance != expected_provenance:
        raise ReleaseValidationError("official M1 release provenance changed")
    stable_comparisons, repeat_runs = _verify_repeat(primary, repeat)
    withheld_file_comparisons, withheld_record_count = _verify_withheld_equivalence(
        primary,
        withheld_root,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".m1-release-", dir=output.parent))
    published = False
    try:
        experiment_entries = _copy_curated_records(staging, primary)
        repeat_by_experiment = {entry["experiment"]: entry for entry in repeat_runs}
        for entry in experiment_entries:
            repeat_entry = repeat_by_experiment[entry["experiment"]]
            entry["repeat_run_sha256"] = repeat_entry["repeat_run_sha256"]
        figure_entries = _write_figures(staging, primary)
        document_entries = _copy_documents(staging, documents_root)
        index: dict[str, Any] = {
            "schema": RELEASE_SCHEMA,
            "release_id": RELEASE_ID,
            "scope": "cpu-only-analytic-estimator-output",
            **provenance,
            "artifact_contract": ARTIFACT_CONTRACT,
            "experiments": experiment_entries,
            "figures": figure_entries,
            "documents": document_entries,
            "verification": {
                "replacement_source_revision": provenance["source_revision"],
                "withheld_source_revision": WITHHELD_SOURCE_REVISION,
                "replacement_bundle_validations": 6,
                "withheld_comparison_bundle_validations": 3,
                "stable_file_comparisons": stable_comparisons,
                "stable_files_per_experiment": list(STABLE_SOURCE_FILES),
                "withheld_normalized_file_comparisons": withheld_file_comparisons,
                "withheld_normalized_record_count": withheld_record_count,
                "run_id_exclusion_only": True,
            },
            "curation_command": list(CURATION_COMMAND),
            "validation_command": list(VALIDATION_COMMAND),
            "omission_policy": OMISSION_POLICY,
        }
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


def _load_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    value = _read_bytes(path, cap=16 * 1024 * 1024)
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseValidationError(f"invalid JSON in {path}") from error
    mapping = _mapping(parsed, label=path.name)
    if canonical_json_bytes(mapping) != value:
        raise ReleaseValidationError(f"noncanonical JSON in {path}")
    return mapping, value


def _parse_canonical_model[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
) -> ModelT:
    value = _read_bytes(path, cap=16 * 1024 * 1024)
    parsed = model.model_validate_json(value)
    if canonical_json_bytes(parsed) != value:
        raise ReleaseValidationError(f"noncanonical model file: {path}")
    return parsed


def _parse_canonical_ndjson[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
) -> tuple[ModelT, ...]:
    value = _read_bytes(path)
    if not value or not value.endswith(b"\n"):
        raise ReleaseValidationError(f"invalid NDJSON framing: {path}")
    records: list[ModelT] = []
    for line in value.splitlines(keepends=True):
        parsed = model.model_validate_json(line)
        if canonical_json_bytes(parsed) != line:
            raise ReleaseValidationError(f"noncanonical NDJSON record: {path}")
        records.append(parsed)
    return tuple(records)


def _verify_indexed_file(root: Path, raw_entry: object) -> Path:
    entry = _mapping(raw_entry, label="file entry")
    relative = _safe_relative_path(entry.get("path"), label="file entry path")
    value = _read_bytes(root / relative)
    if entry.get("byte_length") != len(value) or entry.get("sha256") != _sha256_bytes(value):
        raise ReleaseValidationError(f"release file hash or length mismatch: {relative}")
    return relative


def _validate_curated_experiment(
    root: Path,
    raw_entry: object,
    *,
    expected_experiment: str,
    expected_directory: str,
    expected_manifest_digest: str,
    expected_source_revision: str,
    expected_lockfile_sha256: str,
    expected_package_version: str,
    expected_environment: Mapping[str, Any],
) -> ValidatedCuratedExperiment:
    entry = _mapping(raw_entry, label="experiment entry")
    if entry.get("experiment") != expected_experiment:
        raise ReleaseValidationError("release experiments are out of contractual order")
    record_directory = Path("records") / expected_directory
    if entry.get("record_directory") != record_directory.as_posix():
        raise ReleaseValidationError(f"record directory is invalid for {expected_experiment}")
    if entry.get("manifest_sha256") != expected_manifest_digest:
        raise ReleaseValidationError(f"manifest digest is invalid for {expected_experiment}")
    raw_files = _list(entry.get("curated_files"), label="curated_files")
    if len(raw_files) != len(CURATED_SOURCE_FILES):
        raise ReleaseValidationError(f"curated file count is invalid for {expected_experiment}")
    files_by_source: dict[str, Path] = {}
    expected_curated_files: list[dict[str, Any]] = []
    allowed: set[Path] = set()
    for raw_file, (expected_source_name, destination_name) in zip(
        raw_files,
        CURATED_SOURCE_FILES,
        strict=True,
    ):
        file_entry = _mapping(raw_file, label="curated file")
        source_name = file_entry.get("source_name")
        if source_name != expected_source_name:
            raise ReleaseValidationError(
                f"curated source file order changed for {expected_experiment}"
            )
        relative = _verify_indexed_file(root, file_entry)
        expected_relative = record_directory / destination_name
        if relative != expected_relative:
            raise ReleaseValidationError(
                f"curated destination changed for {expected_experiment}/{source_name}"
            )
        files_by_source[expected_source_name] = relative
        expected_curated_files.append(
            _file_entry(root, expected_relative, source_name=expected_source_name)
        )
        allowed.add(relative)
    expected_source_names = {source for source, _ in CURATED_SOURCE_FILES}
    if set(files_by_source) != expected_source_names:
        raise ReleaseValidationError(
            f"curated source file allowlist changed for {expected_experiment}"
        )

    manifest_mapping, manifest_bytes = _load_canonical_json(root / files_by_source[MANIFEST_FILE])
    manifest = validate_manifest_mapping(manifest_mapping)
    if sha256_digest(manifest) != expected_manifest_digest:
        raise ReleaseValidationError(f"curated manifest is invalid for {expected_experiment}")
    run = _parse_canonical_model(
        root / files_by_source[RUN_FILE],
        RunRecordV1Alpha1,
    )
    payload_index = _parse_canonical_model(
        root / files_by_source[PAYLOAD_INDEX_FILE],
        PayloadIndexV1Alpha1,
    )
    marker = _parse_canonical_model(
        root / files_by_source[SUCCESS_FILE],
        SuccessMarkerV1Alpha1,
    )
    analytic = _parse_canonical_model(
        root / files_by_source[ANALYTIC_VALIDATION_FILE],
        AnalyticValidationV1Alpha1,
    )
    aggregates = _parse_canonical_ndjson(
        root / files_by_source[AGGREGATE_METRICS_FILE],
        AggregateMetricRecordV1Alpha1,
    )
    crossovers = _parse_canonical_ndjson(
        root / files_by_source[CROSSOVERS_FILE],
        CrossoverRecordV1Alpha1,
    )
    assert isinstance(run, RunRecordV1Alpha1)
    assert isinstance(payload_index, PayloadIndexV1Alpha1)
    assert isinstance(marker, SuccessMarkerV1Alpha1)
    assert isinstance(analytic, AnalyticValidationV1Alpha1)
    if (
        run.git_revision != expected_source_revision
        or run.lockfile_sha256 != expected_lockfile_sha256
        or run.package_version != expected_package_version
        or run.source_dirty
    ):
        raise ReleaseValidationError(f"source provenance is invalid for {expected_experiment}")
    if run.environment.model_dump(mode="json", by_alias=True) != dict(expected_environment):
        raise ReleaseValidationError(f"hardware provenance is invalid for {expected_experiment}")
    if not _specific_cpu_model(run.environment.cpu_model):
        raise ReleaseValidationError(f"CPU model is generic for {expected_experiment}")
    if run.manifest_sha256 != expected_manifest_digest:
        raise ReleaseValidationError(f"run manifest digest is invalid for {expected_experiment}")
    expected_identity = EXPECTED_RELEASE_IDENTITIES[expected_experiment]
    expected_run_id = derive_run_id(
        manifest_sha256=expected_manifest_digest,
        git_revision=expected_source_revision,
        lockfile_sha256=expected_lockfile_sha256,
        package_version=expected_package_version,
    )
    expected_command = (
        "ffb",
        "run",
        EXPECTED_MANIFEST_PATHS[expected_experiment],
        "--output-dir",
        f"reports/generated/{expected_directory}",
    )
    if (
        entry.get("run_id") != run.run_id
        or run.run_id != expected_identity["run_id"]
        or run.run_id != expected_run_id
        or run.command != expected_command
    ):
        raise ReleaseValidationError(f"release run_id is invalid for {expected_experiment}")
    if (
        payload_index.run_id != run.run_id
        or payload_index.manifest_sha256 != expected_manifest_digest
        or payload_index.artifact_contract != ARTIFACT_CONTRACT
    ):
        raise ReleaseValidationError(
            f"source payload provenance is invalid for {expected_experiment}"
        )
    if analytic.run_id != run.run_id or analytic.manifest_sha256 != expected_manifest_digest:
        raise ReleaseValidationError(f"analytic provenance is invalid for {expected_experiment}")
    for record in (*aggregates, *crossovers):
        if record.run_id != run.run_id or record.manifest_sha256 != expected_manifest_digest:
            raise ReleaseValidationError(f"record provenance is invalid for {expected_experiment}")

    payload_index_bytes = _read_bytes(root / files_by_source[PAYLOAD_INDEX_FILE])
    artifact_digest = compute_artifact_digest(payload_index_bytes)
    run_bytes = _read_bytes(root / files_by_source[RUN_FILE])
    run_digest = compute_run_record_digest(run_bytes)
    if (
        entry.get("artifact_sha256") != artifact_digest
        or run.artifact_sha256 != artifact_digest
        or marker.artifact_sha256 != artifact_digest
        or artifact_digest != expected_identity["artifact_sha256"]
    ):
        raise ReleaseValidationError(f"artifact digest is invalid for {expected_experiment}")
    if (
        entry.get("run_sha256") != run_digest
        or marker.run_sha256 != run_digest
        or run_digest != expected_identity["run_sha256"]
    ):
        raise ReleaseValidationError(f"run digest is invalid for {expected_experiment}")
    if entry.get("repeat_run_sha256") != expected_identity["repeat_run_sha256"]:
        raise ReleaseValidationError(f"repeat run digest is invalid for {expected_experiment}")

    source_entries = {item.path: item for item in payload_index.files}
    if set(source_entries) != set(INDEXED_SCIENTIFIC_FILES):
        raise ReleaseValidationError(
            f"source payload index allowlist changed for {expected_experiment}"
        )
    present_source_bytes = {
        MANIFEST_FILE: manifest_bytes,
        AGGREGATE_METRICS_FILE: _read_bytes(root / files_by_source[AGGREGATE_METRICS_FILE]),
        CROSSOVERS_FILE: _read_bytes(root / files_by_source[CROSSOVERS_FILE]),
        ANALYTIC_VALIDATION_FILE: _read_bytes(root / files_by_source[ANALYTIC_VALIDATION_FILE]),
    }
    for source_name, value in present_source_bytes.items():
        source_entry = source_entries[source_name]
        if source_entry.byte_length != len(value) or source_entry.sha256 != _sha256_bytes(value):
            raise ReleaseValidationError(
                f"curated bytes disagree with source payload index: {expected_experiment}"
            )
    omitted = _mapping(entry.get("omitted_sequence_metrics"), label="omitted sequence entry")
    sequence_entry = source_entries[SEQUENCE_METRICS_FILE]
    expected_sequence_record_count = (
        len(expected_conditions(manifest))
        * len(expected_sequence_ids(manifest))
        * len(manifest.methods)
    )
    expected_omitted = {
        "source_name": SEQUENCE_METRICS_FILE,
        "byte_length": sequence_entry.byte_length,
        "sha256": sequence_entry.sha256,
        "record_count": expected_sequence_record_count,
        "reason": "Reproducible synthetic sequence rows are omitted from Git history.",
    }
    if omitted != expected_omitted or expected_sequence_record_count <= 0:
        raise ReleaseValidationError(
            f"omitted sequence evidence is invalid for {expected_experiment}"
        )
    run_value = run.model_dump(mode="json", by_alias=True)
    if run.ended_at is None:
        raise ReleaseValidationError(f"run is missing ended_at for {expected_experiment}")
    duration_seconds = (run.ended_at - run.started_at).total_seconds()
    expected_entry: dict[str, Any] = {
        "experiment": expected_experiment,
        "record_directory": record_directory.as_posix(),
        "manifest_sha256": expected_manifest_digest,
        "artifact_sha256": expected_identity["artifact_sha256"],
        "run_sha256": expected_identity["run_sha256"],
        "run_id": expected_identity["run_id"],
        "logical_command": list(run.command),
        "started_at": run_value["started_at"],
        "ended_at": run_value["ended_at"],
        "evaluation_duration_seconds": duration_seconds,
        "curated_files": expected_curated_files,
        "omitted_sequence_metrics": expected_omitted,
        "repeat_run_sha256": expected_identity["repeat_run_sha256"],
    }
    if entry != expected_entry:
        raise ReleaseValidationError(
            f"release experiment index is not cross-bound for {expected_experiment}"
        )
    return ValidatedCuratedExperiment(
        allowed_files=frozenset(allowed),
        aggregates=aggregates,
        expected_entry=expected_entry,
    )


def validate_release(root: Path) -> dict[str, Any]:
    """Validate one complete curated M1 release and every indexed byte."""

    release_root = root.absolute()
    root_stat = os.lstat(release_root)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ReleaseValidationError("release root must be a real directory")
    index, _ = _load_canonical_json(release_root / "release-index.json")
    if (
        index.get("schema") != RELEASE_SCHEMA
        or index.get("release_id") != RELEASE_ID
        or index.get("scope") != "cpu-only-analytic-estimator-output"
        or index.get("source_revision") != RELEASE_SOURCE_REVISION
        or index.get("lockfile_sha256") != RELEASE_LOCKFILE_SHA256
        or index.get("package_version") != RELEASE_PACKAGE_VERSION
        or index.get("environment") != RELEASE_ENVIRONMENT
    ):
        raise ReleaseValidationError("release identity or common provenance is invalid")

    raw_experiments = _list(index.get("experiments"), label="experiments")
    if len(raw_experiments) != len(EXPECTED_EXPERIMENTS):
        raise ReleaseValidationError("release experiment count is invalid")
    allowed_files: set[Path] = {Path("release-index.json")}
    expected_experiment_entries: list[dict[str, Any]] = []
    aggregates_by_experiment: dict[str, tuple[AggregateMetricRecordV1Alpha1, ...]] = {}
    for raw_entry, (
        expected_experiment,
        expected_directory,
        expected_manifest_digest,
    ) in zip(raw_experiments, EXPECTED_EXPERIMENTS, strict=True):
        validated = _validate_curated_experiment(
            release_root,
            raw_entry,
            expected_experiment=expected_experiment,
            expected_directory=expected_directory,
            expected_manifest_digest=expected_manifest_digest,
            expected_source_revision=RELEASE_SOURCE_REVISION,
            expected_lockfile_sha256=RELEASE_LOCKFILE_SHA256,
            expected_package_version=RELEASE_PACKAGE_VERSION,
            expected_environment=RELEASE_ENVIRONMENT,
        )
        allowed_files.update(validated.allowed_files)
        expected_experiment_entries.append(validated.expected_entry)
        aggregates_by_experiment[expected_experiment] = validated.aggregates

    raw_figures = _list(index.get("figures"), label="figures")
    expected_figure_payloads = _figure_payloads(aggregates_by_experiment)
    if len(raw_figures) != len(expected_figure_payloads):
        raise ReleaseValidationError("release must contain exactly two figures")
    expected_figure_entries: list[dict[str, Any]] = []
    for raw_figure, (
        expected_path,
        expected_title,
        expected_value,
        expected_raw_count,
        expected_curve_count,
    ) in zip(raw_figures, expected_figure_payloads, strict=True):
        figure = _mapping(raw_figure, label="figure")
        relative = _verify_indexed_file(release_root, figure)
        if relative != expected_path:
            raise ReleaseValidationError("release figure order or path is invalid")
        allowed_files.add(relative)
        actual_value = _read_bytes(release_root / relative, cap=4 * 1024 * 1024)
        if actual_value != expected_value:
            raise ReleaseValidationError(
                f"figure bytes disagree with curated aggregates: {relative}"
            )
        expected_figure = _file_entry(release_root, expected_path)
        expected_figure.update(
            {
                "title": expected_title,
                "width_px": 1200,
                "height_px": 700,
                "raw_point_count": expected_raw_count,
                "pava_curve_count": expected_curve_count,
            }
        )
        if figure != expected_figure:
            raise ReleaseValidationError(f"figure index contract is invalid: {relative}")
        expected_figure_entries.append(expected_figure)

    raw_documents = _list(index.get("documents"), label="documents")
    if len(raw_documents) != len(DOCUMENT_PATHS):
        raise ReleaseValidationError("release document count is invalid")
    expected_document_entries: list[dict[str, Any]] = []
    for raw_document, name in zip(raw_documents, DOCUMENT_PATHS, strict=True):
        document = _mapping(raw_document, label="document")
        relative = _verify_indexed_file(release_root, document)
        if relative != Path(name):
            raise ReleaseValidationError("release document order or path is invalid")
        value = _read_bytes(release_root / relative, cap=2 * 1024 * 1024)
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleaseValidationError(f"release document is not UTF-8: {relative}") from error
        if not text.startswith("# ") or not text.endswith("\n"):
            raise ReleaseValidationError(
                f"release document must have a heading and trailing newline: {relative}"
            )
        if _sha256_bytes(value) != EXPECTED_DOCUMENT_SHA256[name]:
            raise ReleaseValidationError(f"official release document changed: {relative}")
        expected_document = _file_entry(release_root, relative)
        if document != expected_document:
            raise ReleaseValidationError(f"document index contract is invalid: {relative}")
        expected_document_entries.append(expected_document)
        allowed_files.add(relative)

    expected_verification: dict[str, Any] = {
        "replacement_source_revision": RELEASE_SOURCE_REVISION,
        "withheld_source_revision": WITHHELD_SOURCE_REVISION,
        "replacement_bundle_validations": 6,
        "withheld_comparison_bundle_validations": 3,
        "stable_file_comparisons": 18,
        "stable_files_per_experiment": list(STABLE_SOURCE_FILES),
        "withheld_normalized_file_comparisons": 15,
        "withheld_normalized_record_count": 23148,
        "run_id_exclusion_only": True,
    }
    expected_index: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "release_id": RELEASE_ID,
        "scope": "cpu-only-analytic-estimator-output",
        "source_revision": RELEASE_SOURCE_REVISION,
        "lockfile_sha256": RELEASE_LOCKFILE_SHA256,
        "package_version": RELEASE_PACKAGE_VERSION,
        "environment": RELEASE_ENVIRONMENT,
        "artifact_contract": ARTIFACT_CONTRACT,
        "experiments": expected_experiment_entries,
        "figures": expected_figure_entries,
        "documents": expected_document_entries,
        "verification": expected_verification,
        "curation_command": list(CURATION_COMMAND),
        "validation_command": list(VALIDATION_COMMAND),
        "omission_policy": OMISSION_POLICY,
    }
    if index != expected_index:
        raise ReleaseValidationError("release index is not the exhaustive regenerated contract")

    actual_files: set[Path] = set()
    for path in release_root.rglob("*"):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseValidationError(f"release contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            actual_files.add(path.relative_to(release_root))
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseValidationError(f"release contains a nonstandard entry: {path}")
    unexpected = actual_files - allowed_files
    if unexpected:
        raise ReleaseValidationError(
            f"release contains unexpected files: {sorted(map(str, unexpected))}"
        )
    missing = allowed_files - actual_files
    if missing:
        raise ReleaseValidationError(
            f"release is missing indexed files: {sorted(map(str, missing))}"
        )
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build the no-overwrite curated M1 release.")
    build.add_argument("--primary-root", required=True, type=Path)
    build.add_argument("--repeat-root", required=True, type=Path)
    build.add_argument("--withheld-root", required=True, type=Path)
    build.add_argument("--documents-root", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    validate = commands.add_parser("validate", help="Validate curated M1 release evidence.")
    validate.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository-local M1 release tool."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            index = build_release(
                primary_root=args.primary_root,
                repeat_root=args.repeat_root,
                withheld_root=args.withheld_root,
                documents_root=args.documents_root,
                output_dir=args.output_dir,
            )
            print(
                f"built {args.output_dir} source_revision={index['source_revision']} "
                f"experiments={len(index['experiments'])}"
            )
        else:
            index = validate_release(args.path)
            print(
                f"valid {index['schema']} release_id={index['release_id']} "
                f"source_revision={index['source_revision']}"
            )
    except (OSError, ValueError, ArtifactValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
