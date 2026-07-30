"""Outcome-independent source selection and deterministic SVGs for M5."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from types import MappingProxyType
from typing import Any, Literal, cast

from pydantic import BaseModel

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_HYPOTHESIS_COORDINATES,
    M5_PERSISTENT_HYPOTHESIS_COORDINATES,
    M5_REPLAY_FIGURE_DEFINITIONS,
    M5_TRACKED_AGGREGATE_TERMS,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayFigureSourceBindingV1,
    ReplayFigureSvgPath,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    replay_descriptor_source_id,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_EXPERIMENT_IDS,
    M5_PERSISTENT_EXPERIMENT_IDS,
    ReplayExperimentIdentityV1,
)
from fusion_fault_bench.replay_curation import ReplayCuratedAggregateEvidence
from fusion_fault_bench.replay_plan import LoadedReplayPlan

type ReplayFigureId = Literal[
    "m5-persistent-panel-summary",
    "m5-crossovers",
    "m5-health-transfer",
    "m5-descriptor-comparison",
    "m5-cluster-sensitivity",
]
type ReplayFigureKind = Literal[
    "panel-summary",
    "crossover",
    "health-transfer",
    "descriptor-comparison",
    "cluster-sensitivity",
]
type ReplayFigureSourceKind = Literal[
    "descriptor-aggregate",
    "persistent-aggregate",
    "persistent-crossover",
    "health-aggregate",
    "cluster-sensitivity",
]
type JsonScalar = str | int | float | bool | None
type ReplayFigureSourceRecord = (
    ReplayDescriptorAggregateV1
    | ReplayPersistentAggregateV1
    | ReplayPersistentCrossoverV1
    | ReplayHealthAggregateV1
    | ReplayClusterSensitivityV1
)

M5_REPLAY_FIGURE_SPEC_SCHEMA = "ffb.m5-figure-spec/v1"
M5_REPLAY_FIGURE_RENDERER = "fusion-fault-bench-deterministic-svg/v1"
M5_REPLAY_FIGURE_WIDTH = 1600
M5_REPLAY_FIGURE_ROW_HEIGHT = 24

_FIGURE_TITLES: Mapping[ReplayFigureId, str] = MappingProxyType(
    {
        "m5-persistent-panel-summary": "M5-A persistent replay response",
        "m5-crossovers": "M5-A predeclared crossover status",
        "m5-health-transfer": "M5-B apply-only health transfer",
        "m5-descriptor-comparison": "nuScenes-mini replay profile context",
        "m5-cluster-sensitivity": "Primary/control cluster sensitivity",
    }
)
_FIGURE_SUBTITLES: Mapping[ReplayFigureId, str] = MappingProxyType(
    {
        "m5-persistent-panel-summary": (
            "All 100 predeclared rows; intervals are pointwise across complete scenes"
        ),
        "m5-crossovers": (
            "All 10 physical-axis rules; units remain separate and censoring is explicit"
        ),
        "m5-health-transfer": (
            "All 43 combined-gate event-gain rows; 11 hypotheses remain marked in context"
        ),
        "m5-descriptor-comparison": (
            "67 fixed descriptive rows; no inferential cross-population comparison"
        ),
        "m5-cluster-sensitivity": (
            "All 26 primary/control sources and every scene/log-group leave-out"
        ),
    }
)
_COLORS = MappingProxyType(
    {
        "background": "#f8fafc",
        "row_even": "#ffffff",
        "row_odd": "#f1f5f9",
        "ink": "#172033",
        "muted": "#5b677a",
        "grid": "#cbd5e1",
        "ok": "#0f766e",
        "undefined": "#7c3aed",
        "not-applicable": "#64748b",
        "observed": "#0f766e",
        "not-observed": "#b45309",
        "undetermined": "#7c3aed",
        "interval": "#0369a1",
        "point": "#0c4a6e",
    }
)
_SHARED_DESCRIPTOR_IDS = (
    "sample-count",
    "eligible-object-frame-count",
    "eligible-track-length-q50",
    "ego-range-q50",
    "ego-bearing-q50",
    "finite-difference-speed-q50",
    "reference-time-delta-q50",
)
_REPLAY_ONLY_DESCRIPTOR_IDS = (
    "support-all-annotations",
    "support-roi-pass",
    "support-camera-center-pass",
    "support-lidar-points-positive",
    "support-final-eligible",
    "unique-eligible-track-count",
    "zero-order-hold-velocity-fraction",
    "camera-minus-lidar-acquisition-offset-q50",
)
_DESCRIPTOR_STATISTICS = ("minimum", "median", "maximum")
_PERSISTENT_CROSSOVER_COORDINATES = (
    ("replay-lidar-y-bias", "negative", "m", 4.0),
    ("replay-lidar-y-bias", "positive", "m", 4.0),
    ("replay-camera-noise-correctly-reported", "increase", "std-scale", 4.0),
    ("replay-camera-noise-underreported", "increase", "std-scale", 4.0),
    ("replay-camera-calibration-x", "negative", "m", 4.0),
    ("replay-camera-calibration-x", "positive", "m", 4.0),
    ("replay-camera-calibration-yaw", "negative", "rad", 0.08),
    ("replay-camera-calibration-yaw", "positive", "rad", 0.08),
    ("replay-camera-timestamp-offset", "negative", "s", 0.8),
    ("replay-camera-timestamp-offset", "positive", "s", 0.8),
)
_PROJECTION_FIELDS: Mapping[ReplayFigureSourceKind, tuple[str, ...]] = MappingProxyType(
    {
        "descriptor-aggregate": (
            "descriptor_id",
            "population",
            "population_count",
            "statistic",
            "category_label",
            "status",
            "value",
            "unit",
        ),
        "persistent-aggregate": (
            "condition_id",
            "condition_selector",
            "hypothesis_id",
            "method_id",
            "metric_id",
            "window",
            "inference_role",
            "unit",
            "status",
            "estimate",
            "interval_lower",
            "interval_upper",
            "defined_bootstrap_replicates",
            "positive_scene_count",
            "zero_scene_count",
            "negative_scene_count",
            "undefined_scene_count",
            "expected_direction",
            "persistence_label",
            "aggregation",
        ),
        "persistent-crossover": (
            "direction",
            "severity_unit",
            "tested_maximum",
            "status",
            "point_curve_crossed",
            "point_estimate",
            "interval_lower",
            "interval_upper",
            "censoring",
            "bootstrap_crossing_count",
            "bootstrap_crossing_fraction",
        ),
        "health-aggregate": (
            "condition_id",
            "condition_selector",
            "hypothesis_id",
            "method_id",
            "metric_id",
            "window",
            "inference_role",
            "unit",
            "status",
            "estimate",
            "interval_lower",
            "interval_upper",
            "defined_bootstrap_replicates",
            "positive_scene_count",
            "zero_scene_count",
            "negative_scene_count",
            "undefined_scene_count",
            "expected_direction",
            "persistence_label",
            "nonpositive_control_supported",
            "aggregation",
            "applicability_basis",
        ),
        "cluster-sensitivity": (
            "source_result_id",
            "cluster_kind",
            "cluster_id",
            "status",
            "estimate",
            "unit",
        ),
    }
)


class ReplayFigureError(ValueError):
    """A curated aggregate set cannot produce the frozen M5 figure bundle."""


@dataclass(frozen=True, slots=True)
class ReplayFigureSourceSelection:
    """One ordered source projection before spec/render hashes are attached."""

    figure_id: ReplayFigureId
    figure_kind: ReplayFigureKind
    source_kind: ReplayFigureSourceKind
    source_id: str
    source_record_sha256: str
    identity: ReplayExperimentIdentityV1 | None
    replay_identity_sha256: str | None
    projection_id: str
    projected_fields: tuple[tuple[str, JsonScalar], ...]

    def projection_mapping(self) -> dict[str, JsonScalar]:
        """Return the ordered fields as a JSON-safe mapping."""

        return dict(self.projected_fields)


@dataclass(frozen=True, slots=True)
class RenderedReplayFigure:
    """One exact canonical spec and its deterministic SVG bytes."""

    figure_id: ReplayFigureId
    figure_kind: ReplayFigureKind
    relative_spec_path: str
    relative_svg_path: ReplayFigureSvgPath
    sources: tuple[ReplayFigureSourceSelection, ...]
    spec_bytes: bytes
    svg_bytes: bytes
    spec_sha256: str
    svg_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayFigureBundle:
    """The complete exact five-figure M5 candidate sidecar set."""

    figures: tuple[RenderedReplayFigure, ...]
    bindings: tuple[ReplayFigureSourceBindingV1, ...]

    def files(self) -> dict[str, bytes]:
        """Return exact spec/SVG paths and bytes in frozen figure order."""

        output: dict[str, bytes] = {}
        for figure in self.figures:
            output[figure.relative_spec_path] = figure.spec_bytes
            output[figure.relative_svg_path] = figure.svg_bytes
        return output


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_projection(
    record: ReplayFigureSourceRecord,
    *,
    source_kind: ReplayFigureSourceKind,
) -> tuple[tuple[str, JsonScalar], ...]:
    values = record.model_dump(mode="json", by_alias=True)
    output: list[tuple[str, JsonScalar]] = []
    for field in _PROJECTION_FIELDS[source_kind]:
        value = cast(JsonScalar, values.get(field))
        if isinstance(value, float) and value == 0.0:
            value = 0.0
        output.append((field, value))
    return tuple(output)


def _selection(
    *,
    figure_id: ReplayFigureId,
    figure_kind: ReplayFigureKind,
    source_kind: ReplayFigureSourceKind,
    record: ReplayFigureSourceRecord,
    source_id: str,
) -> ReplayFigureSourceSelection:
    identity = getattr(record, "identity", None)
    replay_identity_sha256 = getattr(record, "replay_identity_sha256", None)
    coordinate = {
        "schema": "ffb.m5-figure-projection-coordinate/v1",
        "figure_id": figure_id,
        "source_kind": source_kind,
        "source_id": source_id,
        "source_record_sha256": sha256_digest(record),
    }
    return ReplayFigureSourceSelection(
        figure_id=figure_id,
        figure_kind=figure_kind,
        source_kind=source_kind,
        source_id=source_id,
        source_record_sha256=sha256_digest(record),
        identity=cast(ReplayExperimentIdentityV1 | None, identity),
        replay_identity_sha256=cast(str | None, replay_identity_sha256),
        projection_id=f"m5-projection-{sha256_digest(coordinate)}",
        projected_fields=_model_projection(record, source_kind=source_kind),
    )


def _unique_index[ReplayFigureModelT: BaseModel](
    records: Sequence[ReplayFigureModelT],
    *,
    key: Callable[[ReplayFigureModelT], object],
    label: str,
) -> dict[object, ReplayFigureModelT]:
    output: dict[object, ReplayFigureModelT] = {}
    for record in records:
        coordinate = key(record)
        if coordinate in output:
            raise ReplayFigureError(f"{label} contains a duplicate coordinate")
        output[coordinate] = record
    return output


def _required_record[ReplayFigureModelT: BaseModel](
    records: Mapping[object, ReplayFigureModelT],
    coordinate: object,
    *,
    label: str,
) -> ReplayFigureModelT:
    record = records.get(coordinate)
    if record is None:
        raise ReplayFigureError(f"{label} is missing a predeclared source")
    return record


def _persistent_sources(
    evidence: ReplayCuratedAggregateEvidence,
    plan: LoadedReplayPlan,
) -> tuple[ReplayFigureSourceSelection, ...]:
    figure_id: ReplayFigureId = "m5-persistent-panel-summary"
    figure_kind: ReplayFigureKind = "panel-summary"
    records = _unique_index(
        evidence.persistent_aggregates,
        key=lambda row: (
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.aggregation,
        ),
        label="persistent figure sources",
    )
    selected: list[ReplayPersistentAggregateV1] = []
    for case in plan.persistent_cases:
        if case.identity.experiment_id not in M5_PERSISTENT_EXPERIMENT_IDS[:6]:
            continue
        coordinate = (
            case.fault_condition.selector,
            "fixed-fusion",
            "fused-minus-healthy",
            "full",
            "m^2",
            "equal-scene-mean",
        )
        selected.append(
            _required_record(records, coordinate, label="persistent ordinary-fault figure grid")
        )
    if len(selected) != 54:
        raise ReplayFigureError("persistent ordinary-fault figure grid is not exactly 54 rows")

    dropout_coordinates = (
        ("fixed-fusion", "coverage", "fraction", "pooled-valid-eligible-count-ratio"),
        ("fixed-fusion", "conditional-matched-center-mse", "m^2", "pooled-valid-loss"),
        (
            "fault-target-drop-policy",
            "coverage",
            "fraction",
            "pooled-valid-eligible-count-ratio",
        ),
        ("lidar-only", "coverage", "fraction", "pooled-valid-eligible-count-ratio"),
    )
    dropout_cases = tuple(
        case
        for case in plan.persistent_cases
        if case.identity.experiment_id == "replay-camera-dropout"
    )
    for method_id, metric_id, unit, aggregation in dropout_coordinates:
        for case in dropout_cases:
            coordinate = (
                case.fault_condition.selector,
                method_id,
                metric_id,
                "full",
                unit,
                aggregation,
            )
            selected.append(
                _required_record(records, coordinate, label="persistent dropout figure grid")
            )
    if len(selected) != 78:
        raise ReplayFigureError("persistent dropout figure grid is not exactly 24 rows")

    common_cases = tuple(
        case
        for case in plan.persistent_cases
        if case.identity.experiment_id == "replay-common-mode-x"
    )
    for method_id, metric_id in (
        ("fixed-fusion", "matched-center-mse"),
        ("camera-lidar-pair", "camera-lidar-disagreement-mse"),
    ):
        for case in common_cases:
            coordinate = (
                case.fault_condition.selector,
                method_id,
                metric_id,
                "full",
                "m^2",
                "equal-scene-mean",
            )
            selected.append(
                _required_record(records, coordinate, label="persistent common-mode figure grid")
            )
    if len(selected) != 100:
        raise ReplayFigureError("persistent common-mode figure grid is not exactly 22 rows")
    return tuple(
        _selection(
            figure_id=figure_id,
            figure_kind=figure_kind,
            source_kind="persistent-aggregate",
            record=record,
            source_id=record.result_id,
        )
        for record in selected
    )


def _crossover_sources(
    evidence: ReplayCuratedAggregateEvidence,
) -> tuple[ReplayFigureSourceSelection, ...]:
    records = _unique_index(
        evidence.persistent_crossovers,
        key=lambda row: (
            row.identity.experiment_id,
            row.direction,
            row.severity_unit,
            row.tested_maximum,
        ),
        label="crossover figure sources",
    )
    selected = tuple(
        _required_record(records, coordinate, label="crossover figure grid")
        for coordinate in _PERSISTENT_CROSSOVER_COORDINATES
    )
    if len(selected) != 10:
        raise ReplayFigureError("crossover figure grid is not exactly 10 rows")
    return tuple(
        _selection(
            figure_id="m5-crossovers",
            figure_kind="crossover",
            source_kind="persistent-crossover",
            record=record,
            source_id=record.crossover_id,
        )
        for record in selected
    )


def _health_sources(
    evidence: ReplayCuratedAggregateEvidence,
    plan: LoadedReplayPlan,
) -> tuple[ReplayFigureSourceSelection, ...]:
    records = _unique_index(
        evidence.health_aggregates,
        key=lambda row: (
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.aggregation,
        ),
        label="health figure sources",
    )
    selected = tuple(
        _required_record(
            records,
            (
                case.selector,
                "combined-health-gate",
                "policy-gain-vs-fixed",
                "event",
                "m^2",
                "equal-scene-mean",
            ),
            label="health-transfer figure grid",
        )
        for case in plan.health_cases
    )
    if len(selected) != 43:
        raise ReplayFigureError("health-transfer figure grid is not exactly 43 rows")
    if tuple(dict.fromkeys(row.condition_id for row in selected)) != M5_HEALTH_EXPERIMENT_IDS:
        raise ReplayFigureError("health-transfer figure identity order is invalid")
    return tuple(
        _selection(
            figure_id="m5-health-transfer",
            figure_kind="health-transfer",
            source_kind="health-aggregate",
            record=record,
            source_id=record.result_id,
        )
        for record in selected
    )


def _descriptor_sources(
    evidence: ReplayCuratedAggregateEvidence,
) -> tuple[ReplayFigureSourceSelection, ...]:
    records = _unique_index(
        evidence.descriptor_aggregates,
        key=lambda row: (
            row.descriptor_id,
            row.population,
            row.statistic,
            row.category_label,
        ),
        label="descriptor figure sources",
    )
    selected: list[ReplayDescriptorAggregateV1] = []
    for descriptor_id in _SHARED_DESCRIPTOR_IDS:
        for statistic in _DESCRIPTOR_STATISTICS:
            for population in ("nuscenes-mini-replay", "m3-main-test-comparator"):
                coordinate = (descriptor_id, population, statistic, None)
                selected.append(
                    _required_record(records, coordinate, label="descriptor comparison figure grid")
                )
    for descriptor_id in _REPLAY_ONLY_DESCRIPTOR_IDS:
        for statistic in _DESCRIPTOR_STATISTICS:
            coordinate = (descriptor_id, "nuscenes-mini-replay", statistic, None)
            selected.append(
                _required_record(records, coordinate, label="descriptor comparison figure grid")
            )
    selected.append(
        _required_record(
            records,
            (
                "distinct-log-group-count",
                "nuscenes-mini-replay",
                "count",
                None,
            ),
            label="descriptor comparison figure grid",
        )
    )
    if len(selected) != 67:
        raise ReplayFigureError("descriptor-comparison figure grid is not exactly 67 rows")
    return tuple(
        _selection(
            figure_id="m5-descriptor-comparison",
            figure_kind="descriptor-comparison",
            source_kind="descriptor-aggregate",
            record=record,
            source_id=replay_descriptor_source_id(record),
        )
        for record in selected
    )


def _claim_source_records(
    evidence: ReplayCuratedAggregateEvidence,
) -> tuple[ReplayPersistentAggregateV1 | ReplayHealthAggregateV1, ...]:
    persistent = _unique_index(
        evidence.persistent_aggregates,
        key=lambda row: (
            row.hypothesis_id,
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.aggregation,
            row.inference_role,
            row.expected_direction,
        ),
        label="persistent claim sources",
    )
    health = _unique_index(
        evidence.health_aggregates,
        key=lambda row: (
            row.hypothesis_id,
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.inference_role,
            row.expected_direction,
        ),
        label="health claim sources",
    )
    selected: list[ReplayPersistentAggregateV1 | ReplayHealthAggregateV1] = []
    for coordinate in M5_PERSISTENT_HYPOTHESIS_COORDINATES:
        if coordinate[7] != "primary-directional":
            continue
        selected.append(
            _required_record(persistent, coordinate, label="persistent cluster source grid")
        )
    for coordinate in M5_HEALTH_HYPOTHESIS_COORDINATES:
        if coordinate[6] not in {"primary-directional", "nonpositive-control"}:
            continue
        selected.append(
            _required_record(
                health,
                coordinate,
                label="health cluster source grid",
            )
        )
    if len(selected) != 26:
        raise ReplayFigureError("cluster source grid is not exactly 26 primary/control rows")
    return tuple(selected)


def _cluster_sources(
    evidence: ReplayCuratedAggregateEvidence,
) -> tuple[ReplayFigureSourceSelection, ...]:
    selected: list[ReplayFigureSourceSelection] = []
    sensitivity_by_source: dict[str, list[ReplayClusterSensitivityV1]] = {}
    for record in evidence.cluster_sensitivity:
        sensitivity_by_source.setdefault(record.source_result_id, []).append(record)
    expected_per_source = 10 + evidence.profile_summary.distinct_log_group_count
    for source in _claim_source_records(evidence):
        source_kind: ReplayFigureSourceKind = (
            "persistent-aggregate"
            if isinstance(source, ReplayPersistentAggregateV1)
            else "health-aggregate"
        )
        selected.append(
            _selection(
                figure_id="m5-cluster-sensitivity",
                figure_kind="cluster-sensitivity",
                source_kind=source_kind,
                record=source,
                source_id=source.result_id,
            )
        )
        rows = tuple(
            sorted(
                sensitivity_by_source.get(source.result_id, ()),
                key=lambda row: (
                    0 if row.cluster_kind == "leave-one-scene-out" else 1,
                    row.cluster_id,
                ),
            )
        )
        if len(rows) != expected_per_source:
            raise ReplayFigureError("cluster sensitivity coverage is incomplete")
        selected.extend(
            _selection(
                figure_id="m5-cluster-sensitivity",
                figure_kind="cluster-sensitivity",
                source_kind="cluster-sensitivity",
                record=record,
                source_id=record.sensitivity_id,
            )
            for record in rows
        )
    expected_count = 26 + 26 * expected_per_source
    if len(selected) != expected_count:
        raise ReplayFigureError("cluster-sensitivity figure grid has an invalid row count")
    return tuple(selected)


def select_replay_figure_sources(
    evidence: ReplayCuratedAggregateEvidence,
    *,
    plan: LoadedReplayPlan,
) -> tuple[tuple[ReplayFigureSourceSelection, ...], ...]:
    """Select every predeclared source without inspecting sign or favorability."""

    groups = (
        _persistent_sources(evidence, plan),
        _crossover_sources(evidence),
        _health_sources(evidence, plan),
        _descriptor_sources(evidence),
        _cluster_sources(evidence),
    )
    if tuple(group[0].figure_id for group in groups) != tuple(
        definition[0] for definition in M5_REPLAY_FIGURE_DEFINITIONS
    ):
        raise ReplayFigureError("figure source groups are not in frozen public order")
    return groups


def _axis_value(projection: Mapping[str, JsonScalar]) -> float | None:
    for field in ("estimate", "point_estimate", "value"):
        value = projection.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _axis_unit(projection: Mapping[str, JsonScalar]) -> str:
    value = projection.get("unit", projection.get("severity_unit", "unitless"))
    return value if isinstance(value, str) else "unitless"


def _axis_domains(
    sources: Sequence[ReplayFigureSourceSelection],
) -> tuple[dict[str, JsonScalar], ...]:
    values_by_unit: dict[str, list[float]] = {}
    for source in sources:
        projection = source.projection_mapping()
        unit = _axis_unit(projection)
        for field in ("estimate", "point_estimate", "value", "interval_lower", "interval_upper"):
            value = projection.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_unit.setdefault(unit, []).append(float(value))
    domains: list[dict[str, JsonScalar]] = []
    for unit in sorted(values_by_unit):
        values = values_by_unit[unit]
        lower = min(0.0, min(values))
        upper = max(0.0, max(values))
        if lower == upper:
            lower, upper = (lower - 1.0, upper + 1.0)
        domains.append({"unit": unit, "lower": lower, "upper": upper})
    return tuple(domains)


def _figure_spec(
    *,
    figure_id: ReplayFigureId,
    figure_kind: ReplayFigureKind,
    relative_svg_path: str,
    sources: Sequence[ReplayFigureSourceSelection],
) -> dict[str, Any]:
    height = 170 + M5_REPLAY_FIGURE_ROW_HEIGHT * len(sources)
    return {
        "schema": M5_REPLAY_FIGURE_SPEC_SCHEMA,
        "figure_id": figure_id,
        "figure_kind": figure_kind,
        "relative_svg_path": relative_svg_path,
        "renderer": M5_REPLAY_FIGURE_RENDERER,
        "dimensions": {
            "width": M5_REPLAY_FIGURE_WIDTH,
            "height": height,
            "row_height": M5_REPLAY_FIGURE_ROW_HEIGHT,
        },
        "title": _FIGURE_TITLES[figure_id],
        "subtitle": _FIGURE_SUBTITLES[figure_id],
        "font_family": "monospace",
        "colors": dict(_COLORS),
        "axis_rule": "per-native-unit-zero-inclusive-observed-range",
        "axis_domains": _axis_domains(sources),
        "undefined_presentation": "explicit-status-label-no-invented-value",
        "censoring_presentation": "explicit-status-and-censoring-label",
        "caption_boundary": (
            "matched-center-estimator-output-proxy-only;"
            "no-raw-sensor-detector-fleet-planning-or-safety-claim"
        ),
        "marks": [
            {
                "mark_ordinal": ordinal,
                "projection_id": source.projection_id,
                "source_kind": source.source_kind,
                "source_id": source.source_id,
                "source_record_sha256": source.source_record_sha256,
                "projected_fields": source.projection_mapping(),
                "replay_identity_sha256": source.replay_identity_sha256,
            }
            for ordinal, source in enumerate(sources)
        ],
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
        "attribution": "nuScenes / Motional; aggregate replay evidence only",
        "non_endorsement": True,
    }


def _format_value(value: JsonScalar) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return format(value, ".6g")
    return str(value)


def _mark_label(mark: Mapping[str, Any]) -> str:
    projection = cast("Mapping[str, JsonScalar]", mark["projected_fields"])
    if mark["source_kind"] == "descriptor-aggregate":
        label = (
            f"{projection['population']} | {projection['descriptor_id']} | "
            f"{projection['statistic']}"
        )
    elif mark["source_kind"] == "persistent-crossover":
        label = (
            f"{mark['source_id'][:24]} | {projection['direction']} | {projection['severity_unit']}"
        )
    elif mark["source_kind"] == "cluster-sensitivity":
        label = (
            f"{projection['cluster_kind']} | {projection['cluster_id']} | "
            f"{str(projection['source_result_id'])[:20]}"
        )
    else:
        label = (
            f"{projection['condition_selector']} | {projection['method_id']} | "
            f"{projection['metric_id']} | {projection['window']}"
        )
    return label if len(label) <= 108 else f"{label[:105]}..."


def _status_color(status: str, colors: Mapping[str, str]) -> str:
    return colors.get(status, colors["muted"])


def _x_position(value: float, *, lower: float, upper: float) -> float:
    if not math.isfinite(value) or not lower < upper:
        return 1100.0
    return 1030.0 + 460.0 * (value - lower) / (upper - lower)


def render_replay_figure_svg(spec: Mapping[str, Any]) -> bytes:
    """Render exact SVG bytes from one canonical M5 figure specification."""

    if (
        spec.get("schema") != M5_REPLAY_FIGURE_SPEC_SCHEMA
        or spec.get("renderer") != M5_REPLAY_FIGURE_RENDERER
    ):
        raise ReplayFigureError("figure specification schema or renderer is invalid")
    dimensions = cast("Mapping[str, int]", spec["dimensions"])
    width = dimensions["width"]
    height = dimensions["height"]
    row_height = dimensions["row_height"]
    colors = cast("Mapping[str, str]", spec["colors"])
    domains = {
        cast(str, row["unit"]): (
            float(cast(float, row["lower"])),
            float(cast(float, row["upper"])),
        )
        for row in cast("Sequence[Mapping[str, JsonScalar]]", spec["axis_domains"])
    }
    marks = cast("Sequence[Mapping[str, Any]]", spec["marks"])
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title description">'
        ),
        f'<rect width="{width}" height="{height}" fill="{colors["background"]}"/>',
        (
            f'<text id="title" x="42" y="45" fill="{colors["ink"]}" '
            f'font-family="monospace" font-size="24" font-weight="700">'
            f"{escape(cast(str, spec['title']))}</text>"
        ),
        (
            f'<text id="description" x="42" y="73" fill="{colors["muted"]}" '
            f'font-family="monospace" font-size="13">'
            f"{escape(cast(str, spec['subtitle']))}</text>"
        ),
        (
            f'<text x="42" y="105" fill="{colors["muted"]}" font-family="monospace" '
            f'font-size="11">source / coordinate</text>'
        ),
        (
            f'<text x="760" y="105" fill="{colors["muted"]}" font-family="monospace" '
            f'font-size="11">status</text>'
        ),
        (
            f'<text x="870" y="105" fill="{colors["muted"]}" font-family="monospace" '
            f'font-size="11">value [native unit]</text>'
        ),
    ]
    for ordinal, mark in enumerate(marks):
        projection = cast("Mapping[str, JsonScalar]", mark["projected_fields"])
        y = 126 + ordinal * row_height
        background = colors["row_even"] if ordinal % 2 == 0 else colors["row_odd"]
        status = str(projection.get("status", "ok"))
        color = _status_color(status, colors)
        unit = _axis_unit(projection)
        value = _axis_value(projection)
        value_text = _format_value(value)
        lines.extend(
            (
                f'<rect x="30" y="{y - 16}" width="1540" height="{row_height}" '
                f'fill="{background}"/>',
                (
                    f'<text x="42" y="{y}" fill="{colors["ink"]}" font-family="monospace" '
                    f'font-size="11">{escape(_mark_label(mark))}</text>'
                ),
                f'<circle cx="746" cy="{y - 4}" r="4" fill="{color}"/>',
                (
                    f'<text x="760" y="{y}" fill="{color}" font-family="monospace" '
                    f'font-size="11">{escape(status)}</text>'
                ),
                (
                    f'<text x="870" y="{y}" fill="{colors["ink"]}" font-family="monospace" '
                    f'font-size="11">{escape(value_text)} [{escape(unit)}]</text>'
                ),
            )
        )
        domain = domains.get(unit)
        if value is not None and domain is not None:
            lower, upper = domain
            x = _x_position(value, lower=lower, upper=upper)
            interval_lower = projection.get("interval_lower")
            interval_upper = projection.get("interval_upper")
            lines.append(
                f'<line x1="1030" y1="{y - 4}" x2="1490" y2="{y - 4}" '
                f'stroke="{colors["grid"]}" stroke-width="1"/>'
            )
            if isinstance(interval_lower, (int, float)) and isinstance(
                interval_upper, (int, float)
            ):
                x1 = _x_position(float(interval_lower), lower=lower, upper=upper)
                x2 = _x_position(float(interval_upper), lower=lower, upper=upper)
                lines.append(
                    f'<line x1="{x1:.3f}" y1="{y - 4}" x2="{x2:.3f}" y2="{y - 4}" '
                    f'stroke="{colors["interval"]}" stroke-width="3"/>'
                )
            lines.append(f'<circle cx="{x:.3f}" cy="{y - 4}" r="4" fill="{colors["point"]}"/>')
    footer_y = height - 28
    lines.extend(
        (
            (
                f'<text x="42" y="{footer_y}" fill="{colors["muted"]}" '
                f'font-family="monospace" font-size="10">'
                "CC BY-NC-SA 4.0 plus Motional Dataset Terms; attribution required; "
                "no endorsement.</text>"
            ),
            "</svg>",
        )
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_rendered_figures(
    source_groups: Sequence[Sequence[ReplayFigureSourceSelection]],
) -> tuple[RenderedReplayFigure, ...]:
    figures: list[RenderedReplayFigure] = []
    for definition, sources in zip(
        M5_REPLAY_FIGURE_DEFINITIONS,
        source_groups,
        strict=True,
    ):
        figure_id = definition[0]
        figure_kind = definition[1]
        relative_svg_path = definition[2]
        relative_spec_path = relative_svg_path.removesuffix(".svg") + ".spec.json"
        source_tuple = tuple(sources)
        if not source_tuple or any(
            source.figure_id != figure_id or source.figure_kind != figure_kind
            for source in source_tuple
        ):
            raise ReplayFigureError("figure source group disagrees with its definition")
        spec = _figure_spec(
            figure_id=figure_id,
            figure_kind=figure_kind,
            relative_svg_path=relative_svg_path,
            sources=source_tuple,
        )
        spec_bytes = canonical_json_bytes(spec)
        svg_bytes = render_replay_figure_svg(spec)
        figures.append(
            RenderedReplayFigure(
                figure_id=figure_id,
                figure_kind=figure_kind,
                relative_spec_path=relative_spec_path,
                relative_svg_path=relative_svg_path,
                sources=source_tuple,
                spec_bytes=spec_bytes,
                svg_bytes=svg_bytes,
                spec_sha256=_sha256_bytes(spec_bytes),
                svg_sha256=_sha256_bytes(svg_bytes),
            )
        )
    return tuple(figures)


def _build_bindings(
    *,
    figures: Sequence[RenderedReplayFigure],
    evidence: ReplayCuratedAggregateEvidence,
) -> tuple[ReplayFigureSourceBindingV1, ...]:
    bindings: list[ReplayFigureSourceBindingV1] = []
    for figure in figures:
        for ordinal, source in enumerate(figure.sources):
            bindings.append(
                ReplayFigureSourceBindingV1(
                    schema="ffb.replay-figure-source-binding/v1",
                    run_id=evidence.run.run_id,
                    replay_intent_sha256=evidence.profile_summary.replay_intent_sha256,
                    replay_identity_set_sha256=(
                        evidence.profile_summary.replay_identity_set_sha256
                    ),
                    figure_id=figure.figure_id,
                    figure_kind=figure.figure_kind,
                    mark_ordinal=ordinal,
                    source_kind=source.source_kind,
                    source_id=source.source_id,
                    source_record_sha256=source.source_record_sha256,
                    identity=source.identity,
                    replay_identity_sha256=source.replay_identity_sha256,
                    figure_spec_sha256=figure.spec_sha256,
                    rendered_svg_path=figure.relative_svg_path,
                    rendered_svg_sha256=figure.svg_sha256,
                    rendered_svg_byte_length=len(figure.svg_bytes),
                    tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
                )
            )
    return tuple(bindings)


def build_replay_figure_bundle(
    evidence: ReplayCuratedAggregateEvidence,
    *,
    plan: LoadedReplayPlan,
) -> ReplayFigureBundle:
    """Build the exact five specs, SVGs, and machine source bindings."""

    source_groups = select_replay_figure_sources(evidence, plan=plan)
    figures = _build_rendered_figures(source_groups)
    bindings = _build_bindings(figures=figures, evidence=evidence)
    return ReplayFigureBundle(figures=figures, bindings=bindings)


def validate_replay_figure_bundle(
    bundle: ReplayFigureBundle,
    *,
    evidence: ReplayCuratedAggregateEvidence,
    plan: LoadedReplayPlan,
) -> None:
    """Regenerate every source, spec, SVG, and binding byte-for-byte."""

    expected = build_replay_figure_bundle(evidence, plan=plan)
    if bundle != expected:
        raise ReplayFigureError("figure bundle differs from deterministic regeneration")
    for figure in bundle.figures:
        try:
            value = cast(object, json.loads(figure.spec_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayFigureError("figure spec is not valid JSON") from error
        if (
            not isinstance(value, dict)
            or canonical_json_bytes(cast("Mapping[str, Any]", value)) != figure.spec_bytes
        ):
            raise ReplayFigureError("figure spec is not canonical JSON")
        if render_replay_figure_svg(cast("Mapping[str, Any]", value)) != figure.svg_bytes:
            raise ReplayFigureError("rendered figure bytes do not match the canonical spec")
        if (
            _sha256_bytes(figure.spec_bytes) != figure.spec_sha256
            or _sha256_bytes(figure.svg_bytes) != figure.svg_sha256
        ):
            raise ReplayFigureError("figure digest differs from exact sidecar bytes")
