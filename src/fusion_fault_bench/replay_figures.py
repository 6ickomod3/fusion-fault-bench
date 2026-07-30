"""Deterministic M5 figure specifications, SVG rendering, and source bindings."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from types import MappingProxyType
from typing import cast

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_TRACKED_AGGREGATE_TERMS,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    FigureId,
    FigureKind,
    FigureSourceKind,
    ReplayFigureMarkV1,
    ReplayFigureSourceBindingV1,
    ReplayFigureSpecV1,
    ReplayPublicClaimProjectionsV1,
    ReplayPublicClaimProjectionV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_EXPERIMENT_IDS,
    M5_REPLAY_INTENT_SHA256,
    ReplayExperimentIdentityV1,
    expected_replay_identities,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.replay_claims import (
    ReplayClaimEvidence,
    build_public_claim_projections,
    descriptor_source_identifier,
    figure_projections,
    public_projection_by_id,
    validate_public_claim_projections,
)

M5_FIGURE_ORDER: tuple[FigureId, ...] = (
    "m5-persistent-panel-summary",
    "m5-crossovers",
    "m5-health-transfer",
    "m5-descriptor-comparison",
    "m5-cluster-sensitivity",
)

_COLORS = (
    "#0F172A",
    "#F8FAFC",
    "#2563EB",
    "#D97706",
    "#DC2626",
    "#64748B",
    "#059669",
)
_FOOTER = "CC BY-NC-SA 4.0 plus Motional Dataset Terms; attribution required; no endorsement."


@dataclass(frozen=True, slots=True)
class _FigureConfiguration:
    kind: FigureKind
    width: int
    height: int
    title: str
    units: tuple[str, ...]
    facets: tuple[str, ...]


_FIGURE_CONFIG: Mapping[FigureId, _FigureConfiguration] = MappingProxyType(
    {
        "m5-persistent-panel-summary": _FigureConfiguration(
            kind="persistent-panel-summary",
            width=1600,
            height=1800,
            title="M5 persistent-panel fixed projection",
            units=("m^2", "fraction"),
            facets=("ordinary-fault", "dropout", "common-mode"),
        ),
        "m5-crossovers": _FigureConfiguration(
            kind="crossovers",
            width=1400,
            height=900,
            title="M5 predeclared physical-axis crossovers",
            units=("m", "rad", "s", "std-scale"),
            facets=("m", "rad", "s", "std-scale"),
        ),
        "m5-health-transfer": _FigureConfiguration(
            kind="health-transfer",
            width=1600,
            height=1800,
            title="M5 apply-only health-transfer fixed projection",
            units=("m^2",),
            facets=M5_HEALTH_EXPERIMENT_IDS,
        ),
        "m5-descriptor-comparison": _FigureConfiguration(
            kind="descriptor-comparison",
            width=1500,
            height=1600,
            title="M5 preregistered descriptor comparison",
            units=("count", "frames", "m", "rad", "m/s", "s", "fraction"),
            facets=("shared-populations", "replay-only", "log-groups"),
        ),
        "m5-cluster-sensitivity": _FigureConfiguration(
            kind="cluster-sensitivity",
            width=1600,
            height=1800,
            title="M5 complete leave-one-cluster sensitivity",
            units=("m^2",),
            facets=("leave-one-scene-out", "leave-one-log-group-out"),
        ),
    }
)

type FigureSourceRecord = (
    ReplayPersistentAggregateV1
    | ReplayPersistentCrossoverV1
    | ReplayHealthAggregateV1
    | ReplayClusterSensitivityV1
    | ReplayDescriptorAggregateV1
)


def _source_index(
    evidence: ReplayClaimEvidence,
) -> dict[tuple[FigureSourceKind, str], FigureSourceRecord]:
    output: dict[tuple[FigureSourceKind, str], FigureSourceRecord] = {}

    def insert(kind: FigureSourceKind, identifier: str, row: FigureSourceRecord) -> None:
        key = (kind, identifier)
        if key in output:
            raise ValueError("figure evidence contains a duplicate source identifier")
        output[key] = row

    for row in evidence.persistent_aggregates:
        insert("persistent-aggregate", row.result_id, row)
    for row in evidence.persistent_crossovers:
        insert("persistent-crossover", row.crossover_id, row)
    for row in evidence.health_aggregates:
        insert("health-aggregate", row.result_id, row)
    for row in evidence.cluster_sensitivity:
        insert("cluster-sensitivity", row.sensitivity_id, row)
    for row in evidence.descriptor_aggregates:
        insert("descriptor-aggregate", descriptor_source_identifier(row), row)
    return output


def _figure_source_kind(projection: ReplayPublicClaimProjectionV1) -> FigureSourceKind:
    if projection.source_kind not in {
        "persistent-aggregate",
        "persistent-crossover",
        "health-aggregate",
        "cluster-sensitivity",
        "descriptor-aggregate",
    }:
        raise ValueError("non-figure source kind is assigned to an M5 figure")
    return cast(FigureSourceKind, projection.source_kind)


def _expected_mark_count(
    figure_id: FigureId,
    registry: ReplayPublicClaimProjectionsV1,
) -> int:
    return {
        "m5-persistent-panel-summary": 100,
        "m5-crossovers": 10,
        "m5-health-transfer": 43,
        "m5-descriptor-comparison": 67,
        "m5-cluster-sensitivity": registry.sensitivity_projection_count,
    }[figure_id]


def build_figure_specs(
    registry: ReplayPublicClaimProjectionsV1,
    evidence: ReplayClaimEvidence,
) -> tuple[ReplayFigureSpecV1, ...]:
    """Build all five canonical figure specs in frozen order."""

    validate_public_claim_projections(registry, evidence)
    sources = _source_index(evidence)
    specs: list[ReplayFigureSpecV1] = []
    for figure_id in M5_FIGURE_ORDER:
        configuration = _FIGURE_CONFIG[figure_id]
        projections = figure_projections(registry, figure_id)
        if len(projections) != _expected_mark_count(figure_id, registry):
            raise ValueError(f"{figure_id} has the wrong fixed projection count")
        marks: list[ReplayFigureMarkV1] = []
        for ordinal, projection in enumerate(projections):
            source_kind = _figure_source_kind(projection)
            try:
                source = sources[(source_kind, projection.source_identifier)]
            except KeyError as error:
                raise ValueError(
                    "figure projection references an unknown aggregate source"
                ) from error
            digest = sha256_digest(source)
            if projection.source_record_sha256 != digest:
                raise ValueError(
                    "figure projection source digest disagrees with aggregate evidence"
                )
            identity_sha256 = (
                None
                if isinstance(source, ReplayDescriptorAggregateV1)
                else source.replay_identity_sha256
            )
            marks.append(
                ReplayFigureMarkV1(
                    mark_ordinal=ordinal,
                    projection_id=projection.projection_id,
                    source_member=projection.source_member,
                    source_kind=source_kind,
                    source_identifier=projection.source_identifier,
                    source_record_sha256=digest,
                    projected_fields=tuple(field.field for field in projection.projected_fields),
                    replay_identity_sha256=identity_sha256,
                )
            )
        specs.append(
            ReplayFigureSpecV1(
                schema="ffb.m5-figure-spec/v1",
                release_id="m5-nuscenes-replay-v0.1.0",
                run_id=registry.run_id,
                replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
                replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
                figure_id=figure_id,
                figure_kind=configuration.kind,
                figure_file=f"figures/{figure_id}.svg",
                width_px=configuration.width,
                height_px=configuration.height,
                font_families=("Arial", "Helvetica", "sans-serif"),
                colors=_COLORS,
                units=configuration.units,
                axis_facets=configuration.facets,
                caption_boundary="registry-projected-values-and-literal-statuses-only",
                renderer_id="ffb.m5-deterministic-svg/v1",
                marks=tuple(marks),
                tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
                non_endorsement_footer=_FOOTER,
            )
        )
    return tuple(specs)


def canonical_figure_spec_bytes(spec: ReplayFigureSpecV1) -> bytes:
    """Serialize one figure spec with the repository canonical JSON contract."""

    return canonical_json_bytes(spec)


def canonical_figure_spec_files(
    specs: Sequence[ReplayFigureSpecV1],
) -> dict[str, bytes]:
    """Return the exact five canonical spec members keyed by release path."""

    ordered = tuple(specs)
    if tuple(spec.figure_id for spec in ordered) != M5_FIGURE_ORDER:
        raise ValueError("figure specs are incomplete or reordered")
    return {
        spec.figure_file.removesuffix(".svg") + ".spec.json": canonical_figure_spec_bytes(spec)
        for spec in ordered
    }


def _projected_value_text(
    projection: ReplayPublicClaimProjectionV1,
    field_name: str,
) -> str:
    field = next((row for row in projection.projected_fields if row.field == field_name), None)
    if field is None:
        raise ValueError("figure mark names an absent projected field")
    value = field.value
    if value is None:
        if projection.status_behavior == "literal-not-applicable":
            return "not-applicable"
        if projection.status_behavior == "literal-censored" and field_name in {
            "interval_lower",
            "interval_upper",
        }:
            return "censored"
        return "undefined"
    if field.rendering == ".6g":
        numeric = float(value)
        return format(0.0 if numeric == 0.0 else numeric, ".6g")
    if field.rendering == ".2f":
        numeric = float(value)
        return format(0.0 if numeric == 0.0 else numeric, ".2f")
    if field.rendering == "exact-integer":
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(int(cast(int | float, value)))
    if field.rendering == "machine-token":
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _selector_values(projection: ReplayPublicClaimProjectionV1) -> dict[str, str]:
    return {field.field: field.value for field in projection.selector_fields}


def _projected_values(projection: ReplayPublicClaimProjectionV1) -> dict[str, object]:
    return {field.field: field.value for field in projection.projected_fields}


def _numeric(projection: ReplayPublicClaimProjectionV1, field_name: str) -> float | None:
    value = _projected_values(projection).get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _domain(
    projections: Sequence[ReplayPublicClaimProjectionV1],
    fields: Sequence[str],
    *,
    include_zero: bool = True,
) -> tuple[float, float]:
    values = [
        value
        for projection in projections
        for field in fields
        if (value := _numeric(projection, field)) is not None
    ]
    if include_zero:
        values.append(0.0)
    if not values:
        return 0.0, 1.0
    lower, upper = min(values), max(values)
    if lower == upper:
        return lower - 0.5, upper + 0.5
    return lower, upper


def _scaled(value: float, domain: tuple[float, float], left: float, right: float) -> float:
    lower, upper = domain
    return left + (value - lower) * (right - left) / (upper - lower)


def _short(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[: maximum - 1] + "…"


def _svg_header(spec: ReplayFigureSpecV1, subtitle: str) -> list[str]:
    configuration = _FIGURE_CONFIG[spec.figure_id]
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {spec.width_px} '
            f'{spec.height_px}" width="{spec.width_px}" height="{spec.height_px}" role="img" '
            f'aria-label="{escape(configuration.title, quote=True)}">'
        ),
        (f'<rect x="0" y="0" width="{spec.width_px}" height="{spec.height_px}" fill="#F8FAFC"/>'),
        (
            '<text x="24" y="34" font-family="Arial, Helvetica, sans-serif" '
            'font-size="24" font-weight="700" fill="#0F172A">'
            f"{escape(configuration.title)}</text>"
        ),
        (
            '<text x="24" y="61" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="12" fill="#475569">{escape(subtitle)}</text>'
        ),
        (
            f'<line x1="24" y1="82" x2="{spec.width_px - 24}" y2="82" '
            'stroke="#CBD5E1" stroke-width="1"/>'
        ),
    ]


def _svg_footer(spec: ReplayFigureSpecV1) -> tuple[str, ...]:
    return (
        (
            f'<line x1="24" y1="{spec.height_px - 36}" x2="{spec.width_px - 24}" '
            f'y2="{spec.height_px - 36}" stroke="#CBD5E1" stroke-width="1"/>'
        ),
        (
            f'<text x="24" y="{spec.height_px - 16}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#64748B">'
            f"{escape(spec.non_endorsement_footer)}</text>"
        ),
        "</svg>",
    )


def _validated_figure_projections(
    spec: ReplayFigureSpecV1,
    registry: ReplayPublicClaimProjectionsV1,
) -> tuple[ReplayPublicClaimProjectionV1, ...]:
    if spec.figure_id not in M5_FIGURE_ORDER or spec.renderer_id != "ffb.m5-deterministic-svg/v1":
        raise ValueError("figure spec is not a canonical M5 deterministic SVG spec")
    projection_index = public_projection_by_id(registry)
    expected = figure_projections(registry, spec.figure_id)
    if tuple(mark.projection_id for mark in spec.marks) != tuple(
        projection.projection_id for projection in expected
    ):
        raise ValueError("figure spec mark order disagrees with the public registry")
    output: list[ReplayPublicClaimProjectionV1] = []
    for mark in spec.marks:
        try:
            projection = projection_index[mark.projection_id]
        except KeyError as error:
            raise ValueError("figure mark references an unknown public projection") from error
        if (
            mark.source_member != projection.source_member
            or mark.source_identifier != projection.source_identifier
            or mark.source_record_sha256 != projection.source_record_sha256
            or mark.source_kind != _figure_source_kind(projection)
            or mark.projected_fields != tuple(field.field for field in projection.projected_fields)
        ):
            raise ValueError("figure mark source binding disagrees with public projection")
        output.append(projection)
    return tuple(output)


def _render_persistent(
    spec: ReplayFigureSpecV1,
    projections: Sequence[ReplayPublicClaimProjectionV1],
) -> list[str]:
    lines = _svg_header(
        spec,
        "Point •; pointwise interval —; status, persistence, and signs + / 0 / - / ?.",
    )
    domains = {
        unit: _domain(
            tuple(projection for projection in projections if projection.unit == unit),
            ("estimate", "interval_lower", "interval_upper"),
        )
        for unit in ("m^2", "fraction")
    }
    column_width = (spec.width_px - 48) / 2
    for ordinal, projection in enumerate(projections):
        column, row = divmod(ordinal, 50)
        x = 24 + column * column_width
        y = 112 + row * 32
        selectors = _selector_values(projection)
        values = _projected_values(projection)
        selector = selectors["condition_selector"]
        facet = (
            "dropout"
            if selector.startswith("replay-camera-dropout:")
            else "common-mode"
            if selector.startswith("replay-common-mode-x:")
            else "ordinary-fault"
        )
        label = f"[{facet}] {selector} · {selectors['method_id']} · {selectors['metric_id']}"
        plot_left, plot_right = x + 295, x + 505
        estimate = _numeric(projection, "estimate")
        lower = _numeric(projection, "interval_lower")
        upper = _numeric(projection, "interval_upper")
        status = str(values["status"])
        persistence = str(values["persistence_label"])
        hypothesis = projection.hypothesis_id or "context"
        sign_values = tuple(
            values[field]
            for field in (
                "positive_scene_count",
                "zero_scene_count",
                "negative_scene_count",
                "undefined_scene_count",
            )
        )
        signs = (
            "n/a"
            if any(value is None for value in sign_values)
            else "/".join(str(value) for value in sign_values)
        )
        lines.extend(
            (
                f'<g class="persistent-row" data-facet="{facet}">',
                (
                    f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="8" fill="#0F172A">{escape(_short(label, 52))}</text>'
                ),
                (
                    f'<line class="zero-axis" x1="{_scaled(0.0, domains[projection.unit], plot_left, plot_right):.2f}" '
                    f'y1="{y - 9:.1f}" x2="{_scaled(0.0, domains[projection.unit], plot_left, plot_right):.2f}" '
                    f'y2="{y + 3:.1f}" stroke="#CBD5E1" stroke-width="1"/>'
                ),
            )
        )
        if estimate is None or lower is None or upper is None:
            lines.append(
                f'<text class="undefined" x="{plot_left:.1f}" y="{y:.1f}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="9" '
                'fill="#DC2626">undefined</text>'
            )
        else:
            lower_x = _scaled(lower, domains[projection.unit], plot_left, plot_right)
            upper_x = _scaled(upper, domains[projection.unit], plot_left, plot_right)
            estimate_x = _scaled(estimate, domains[projection.unit], plot_left, plot_right)
            lines.extend(
                (
                    f'<line class="interval" x1="{lower_x:.2f}" y1="{y - 3:.1f}" '
                    f'x2="{upper_x:.2f}" y2="{y - 3:.1f}" stroke="#2563EB" stroke-width="2"/>',
                    f'<circle class="point" cx="{estimate_x:.2f}" cy="{y - 3:.1f}" r="3" fill="#0F172A"/>',
                    (
                        f'<text x="{plot_left:.1f}" y="{y + 10:.1f}" '
                        'font-family="Arial, Helvetica, sans-serif" font-size="7" fill="#475569">'
                        f"{escape(_projected_value_text(projection, 'estimate'))} "
                        f"[{escape(_projected_value_text(projection, 'interval_lower'))}, "
                        f"{escape(_projected_value_text(projection, 'interval_upper'))}] "
                        f"{escape(projection.unit)}</text>"
                    ),
                )
            )
        lines.extend(
            (
                (
                    f'<text class="persistence" x="{x + 515:.1f}" y="{y - 2:.1f}" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="8" fill="#0F172A">'
                    f"{escape(hypothesis)} · {escape(status)} · {escape(persistence)}</text>"
                ),
                (
                    f'<text class="sign-partition" x="{x + 515:.1f}" y="{y + 9:.1f}" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="7" fill="#475569">'
                    f"signs +/0/-/? = {escape(signs)}</text>"
                ),
                "</g>",
            )
        )
    return lines


def _render_crossovers(
    spec: ReplayFigureSpecV1,
    projections: Sequence[ReplayPublicClaimProjectionV1],
) -> list[str]:
    lines = _svg_header(
        spec,
        "Separate physical-unit facets; ● point, — finite interval, → right-censored above maximum.",
    )
    panel_origins = {
        "m": (24, 110),
        "rad": (710, 110),
        "s": (24, 505),
        "std-scale": (710, 505),
    }
    for unit in ("m", "rad", "s", "std-scale"):
        facet = tuple(projection for projection in projections if projection.unit == unit)
        origin_x, origin_y = panel_origins[unit]
        lines.extend(
            (
                f'<g class="crossover-facet" data-unit="{escape(unit)}">',
                (
                    f'<rect x="{origin_x}" y="{origin_y}" width="650" height="340" '
                    'rx="8" fill="#FFFFFF" stroke="#CBD5E1"/>'
                ),
                (
                    f'<text x="{origin_x + 16}" y="{origin_y + 28}" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="16" '
                    f'font-weight="700" fill="#0F172A">facet: {escape(unit)}</text>'
                ),
            )
        )
        for row, projection in enumerate(facet):
            selectors = _selector_values(projection)
            y = origin_y + 76 + row * 66
            tested_maximum = float(selectors["tested_maximum"])
            plot_left, plot_right = origin_x + 285, origin_x + 590
            point = _numeric(projection, "point_estimate")
            lower = _numeric(projection, "interval_lower")
            upper = _numeric(projection, "interval_upper")
            status = _projected_value_text(projection, "status")
            label = f"{selectors['experiment_id']} · {selectors['direction']}"
            lines.extend(
                (
                    f'<text x="{origin_x + 16}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="10" fill="#0F172A">{escape(_short(label, 42))}</text>',
                    f'<line x1="{plot_left}" y1="{y - 4}" x2="{plot_right}" y2="{y - 4}" '
                    'stroke="#CBD5E1" stroke-width="2"/>',
                )
            )
            if lower is not None and upper is not None:
                lines.append(
                    f'<line class="crossover-interval" x1="{_scaled(lower, (0.0, tested_maximum), plot_left, plot_right):.2f}" '
                    f'y1="{y - 4}" x2="{_scaled(upper, (0.0, tested_maximum), plot_left, plot_right):.2f}" '
                    f'y2="{y - 4}" stroke="#2563EB" stroke-width="4"/>'
                )
            elif projection.status_behavior == "literal-positive-infinity" and lower is not None:
                start = _scaled(lower, (0.0, tested_maximum), plot_left, plot_right)
                lines.extend(
                    (
                        f'<line class="right-censored" x1="{start:.2f}" y1="{y - 4}" '
                        f'x2="{plot_right}" y2="{y - 4}" stroke="#D97706" stroke-width="4"/>',
                        f'<path class="censor-arrow" d="M {plot_right - 9} {y - 11} L {plot_right} {y - 4} '
                        f'L {plot_right - 9} {y + 3}" fill="none" stroke="#D97706" stroke-width="3"/>',
                    )
                )
            if point is not None:
                lines.append(
                    f'<circle class="crossover-point" cx="{_scaled(point, (0.0, tested_maximum), plot_left, plot_right):.2f}" '
                    f'cy="{y - 4}" r="5" fill="#0F172A"/>'
                )
            lines.append(
                f'<text class="crossover-status" x="{origin_x + 16}" y="{y + 17}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#475569">'
                f"{escape(status)} · point {escape(_projected_value_text(projection, 'point_estimate'))} · "
                f"interval [{escape(_projected_value_text(projection, 'interval_lower'))}, "
                f"{escape(_projected_value_text(projection, 'interval_upper'))}] · "
                f"{escape(_projected_value_text(projection, 'censoring'))}</text>"
            )
        lines.append("</g>")
    return lines


def _render_health(
    spec: ReplayFigureSpecV1,
    projections: Sequence[ReplayPublicClaimProjectionV1],
) -> list[str]:
    lines = _svg_header(
        spec,
        "Frozen experiment order; point •, pointwise interval —, hypothesis/control status retained.",
    )
    domain = _domain(projections, ("estimate", "interval_lower", "interval_upper"))
    previous_experiment = ""
    for ordinal, projection in enumerate(projections):
        y = 115 + ordinal * 38
        selectors = _selector_values(projection)
        values = _projected_values(projection)
        experiment = selectors["condition_selector"].split(":", 1)[0]
        estimate = _numeric(projection, "estimate")
        lower = _numeric(projection, "interval_lower")
        upper = _numeric(projection, "interval_upper")
        label = f"{selectors['condition_selector']}"
        plot_left, plot_right = 520, 980
        lines.extend(
            (
                '<g class="health-row">',
                f'<text x="24" y="{y}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="10" fill="#0F172A">{escape(label)}</text>',
                f'<line class="zero-axis" x1="{_scaled(0.0, domain, plot_left, plot_right):.2f}" '
                f'y1="{y - 12}" x2="{_scaled(0.0, domain, plot_left, plot_right):.2f}" '
                f'y2="{y + 4}" stroke="#CBD5E1"/>',
            )
        )
        if experiment != previous_experiment:
            lines.append(
                f'<text class="health-facet-label" x="380" y="{y}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="8" '
                f'font-weight="700" fill="#64748B">{escape(experiment)}</text>'
            )
            previous_experiment = experiment
        if estimate is None or lower is None or upper is None:
            lines.append(
                f'<text class="undefined" x="{plot_left}" y="{y}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="10" '
                'fill="#DC2626">undefined</text>'
            )
        else:
            lines.extend(
                (
                    f'<line class="health-interval" x1="{_scaled(lower, domain, plot_left, plot_right):.2f}" '
                    f'y1="{y - 4}" x2="{_scaled(upper, domain, plot_left, plot_right):.2f}" '
                    f'y2="{y - 4}" stroke="#2563EB" stroke-width="3"/>',
                    f'<circle class="health-point" cx="{_scaled(estimate, domain, plot_left, plot_right):.2f}" '
                    f'cy="{y - 4}" r="4" fill="#0F172A"/>',
                )
            )
        interpretation = str(values["persistence_label"])
        if values["nonpositive_control_supported"] is not None:
            interpretation = (
                "nonpositive-control-supported"
                if values["nonpositive_control_supported"] is True
                else "nonpositive-control-not-supported"
            )
        lines.extend(
            (
                f'<text class="health-status" x="1000" y="{y - 3}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#475569">'
                f"{escape(str(values['status']))} · {escape(interpretation)} · "
                f"{escape(_projected_value_text(projection, 'estimate'))} "
                f"[{escape(_projected_value_text(projection, 'interval_lower'))}, "
                f"{escape(_projected_value_text(projection, 'interval_upper'))}]</text>",
                (
                    f'<text class="hypothesis-badge" x="1370" y="{y - 3}" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="10" '
                    f'font-weight="700" fill="#0F172A">{escape((projection.hypothesis_id or "context").upper())}</text>'
                ),
            )
        )
        if selectors["condition_selector"].startswith("replay-common-mode-x:"):
            lines.append(
                f'<text class="common-mode-warning" x="1000" y="{y + 10}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="8" '
                'fill="#D97706">no uniquely faulty target</text>'
            )
        lines.append("</g>")
    return lines


def _render_descriptors(
    spec: ReplayFigureSpecV1,
    projections: Sequence[ReplayPublicClaimProjectionV1],
) -> list[str]:
    lines = _svg_header(
        spec,
        "Paired replay / M3 descriptive rows; replay-only quantities explicitly mark M3 unavailable.",
    )
    by_descriptor: dict[str, list[ReplayPublicClaimProjectionV1]] = {}
    for projection in projections:
        descriptor_id = _selector_values(projection)["descriptor_id"]
        by_descriptor.setdefault(descriptor_id, []).append(projection)
    descriptor_domains = {
        descriptor_id: _domain(rows, ("value",), include_zero=True)
        for descriptor_id, rows in by_descriptor.items()
    }
    for ordinal, projection in enumerate(projections):
        column, row = divmod(ordinal, 34)
        column_x = 24 + column * 738
        y = 112 + row * 42
        selectors = _selector_values(projection)
        descriptor_id = selectors["descriptor_id"]
        population = selectors["population"]
        statistic = selectors["statistic"]
        value = _numeric(projection, "value")
        paired = any(
            _selector_values(candidate)["population"] == "m3-main-test-comparator"
            for candidate in by_descriptor[descriptor_id]
        )
        label = f"{descriptor_id} · {statistic} · {population}"
        plot_left, plot_right = column_x + 350, column_x + 575
        lines.extend(
            (
                f'<g class="{"descriptor-pair" if paired else "descriptor-replay-only"}">',
                f'<text x="{column_x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="9" fill="#0F172A">{escape(_short(label, 55))}</text>',
                f'<line x1="{plot_left}" y1="{y - 4}" x2="{plot_right}" y2="{y - 4}" '
                'stroke="#CBD5E1" stroke-width="2"/>',
            )
        )
        if value is not None:
            x = _scaled(value, descriptor_domains[descriptor_id], plot_left, plot_right)
            lines.append(
                f'<circle class="descriptor-point" cx="{x:.2f}" cy="{y - 4}" r="4" '
                f'fill="{"#2563EB" if population == "nuscenes-mini-replay" else "#D97706"}"/>'
            )
        lines.append(
            f'<text class="descriptor-value" x="{plot_right + 10}" y="{y}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="9" fill="#475569">'
            f"{escape(_projected_value_text(projection, 'value'))} {escape(projection.unit)}</text>"
        )
        if not paired:
            lines.append(
                f'<text class="m3-unavailable" x="{column_x + 350}" y="{y + 11}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="8" '
                'fill="#64748B">M3 comparator unavailable (not modeled)</text>'
            )
        lines.append("</g>")
    return lines


def _render_sensitivity(
    spec: ReplayFigureSpecV1,
    projections: Sequence[ReplayPublicClaimProjectionV1],
) -> list[str]:
    lines = _svg_header(
        spec,
        "Source x opaque scene/log-group leave-out matrix; each cell is retained, X means undefined.",
    )
    source_order = tuple(
        dict.fromkeys(_selector_values(row)["source_result_id"] for row in projections)
    )
    cluster_order = tuple(f"scene-ordinal:{ordinal:02d}" for ordinal in range(10)) + tuple(
        f"log-group:{ordinal:02d}" for ordinal in range(spec.marks.__len__() // 26 - 10)
    )
    index = {
        (_selector_values(row)["source_result_id"], _selector_values(row)["cluster_id"]): row
        for row in projections
    }
    estimates = tuple(
        value for row in projections if (value := _numeric(row, "estimate")) is not None
    )
    maximum = max((abs(value) for value in estimates), default=1.0) or 1.0
    grid_left, grid_right = 455.0, float(spec.width_px - 30)
    cell_width = (grid_right - grid_left) / len(cluster_order)
    row_height = 55.0
    grid_top = 190.0
    for column, cluster_id in enumerate(cluster_order):
        x = grid_left + (column + 0.5) * cell_width
        lines.append(
            f'<text class="cluster-axis-label" x="{x:.2f}" y="170" '
            f'transform="rotate(-45 {x:.2f} 170)" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="9" fill="#475569">{escape(cluster_id)}</text>'
        )
    divider = grid_left + 10 * cell_width
    lines.append(
        f'<line class="cluster-kind-divider" x1="{divider:.2f}" y1="118" '
        f'x2="{divider:.2f}" y2="{grid_top + len(source_order) * row_height:.2f}" '
        'stroke="#0F172A" stroke-width="2"/>'
    )
    for row_ordinal, source_id in enumerate(source_order):
        y = grid_top + row_ordinal * row_height
        first = index[(source_id, cluster_order[0])]
        selectors = _selector_values(first)
        source_label = selectors["condition_selector"]
        hypothesis = selectors["hypothesis_id"]
        lines.append(
            f'<text class="source-axis-label" x="24" y="{y + 21:.2f}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#0F172A">'
            f"{escape(_short(source_label, 48))} · {escape(hypothesis.upper())}</text>"
        )
        for column, cluster_id in enumerate(cluster_order):
            projection = index[(source_id, cluster_id)]
            estimate = _numeric(projection, "estimate")
            x = grid_left + column * cell_width
            if estimate is None:
                fill, text = "#E2E8F0", "X"
            else:
                intensity = 0.25 + 0.75 * min(abs(estimate) / maximum, 1.0)
                base = (37, 99, 235) if estimate >= 0.0 else (217, 119, 6)
                fill = "#" + "".join(
                    f"{round(255 - (255 - channel) * intensity):02X}" for channel in base
                )
                text = _projected_value_text(projection, "estimate")
            lines.extend(
                (
                    f'<rect class="sensitivity-cell" data-status="{escape(_projected_value_text(projection, "status"))}" '
                    f'x="{x + 1:.2f}" y="{y:.2f}" width="{cell_width - 2:.2f}" '
                    f'height="{row_height - 5:.2f}" fill="{fill}" stroke="#FFFFFF"/>',
                    f'<text class="sensitivity-value" x="{x + cell_width / 2:.2f}" '
                    f'y="{y + 28:.2f}" text-anchor="middle" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="8" '
                    f'fill="#0F172A">{escape(text)}</text>',
                )
            )
    lines.extend(
        (
            '<text x="455" y="105" font-family="Arial, Helvetica, sans-serif" '
            'font-size="11" font-weight="700" fill="#0F172A">leave-one-scene-out</text>',
            f'<text x="{divider + 8:.2f}" y="105" font-family="Arial, Helvetica, sans-serif" '
            'font-size="11" font-weight="700" fill="#0F172A">leave-one-log-group-out</text>',
            '<text class="undefined-legend" x="24" y="1650" '
            'font-family="Arial, Helvetica, sans-serif" font-size="10" '
            'fill="#64748B">X = undefined leave-out retained</text>',
        )
    )
    return lines


def render_figure_svg(
    spec: ReplayFigureSpecV1,
    registry: ReplayPublicClaimProjectionsV1,
) -> bytes:
    """Render one figure-specific semantic SVG from only reviewed registry values."""

    projections = _validated_figure_projections(spec, registry)
    renderers = {
        "m5-persistent-panel-summary": _render_persistent,
        "m5-crossovers": _render_crossovers,
        "m5-health-transfer": _render_health,
        "m5-descriptor-comparison": _render_descriptors,
        "m5-cluster-sensitivity": _render_sensitivity,
    }
    lines = renderers[spec.figure_id](spec, projections)
    lines.extend(_svg_footer(spec))
    return ("\n".join(lines) + "\n").encode("utf-8")


render_replay_figure_svg = render_figure_svg


def render_figure_svgs(
    specs: Sequence[ReplayFigureSpecV1],
    registry: ReplayPublicClaimProjectionsV1,
) -> dict[str, bytes]:
    """Render all five SVGs in canonical path order."""

    ordered = tuple(specs)
    if tuple(spec.figure_id for spec in ordered) != M5_FIGURE_ORDER:
        raise ValueError("figure specs are incomplete or reordered")
    return {spec.figure_file: render_figure_svg(spec, registry) for spec in ordered}


def _binding_identity(
    source: FigureSourceRecord,
) -> tuple[ReplayExperimentIdentityV1 | None, str | None]:
    if isinstance(source, ReplayDescriptorAggregateV1):
        return None, None
    return source.identity, source.replay_identity_sha256


def build_figure_source_bindings(
    specs: Sequence[ReplayFigureSpecV1],
    svgs: Mapping[str, bytes],
    registry: ReplayPublicClaimProjectionsV1,
    evidence: ReplayClaimEvidence,
) -> tuple[ReplayFigureSourceBindingV1, ...]:
    """Bind every ordered figure mark to exact spec, SVG, and aggregate bytes."""

    expected_specs = build_figure_specs(registry, evidence)
    ordered = tuple(specs)
    if ordered != expected_specs:
        raise ValueError("figure bindings require the exact regenerated figure specs")
    expected_paths = tuple(spec.figure_file for spec in ordered)
    if tuple(svgs) != expected_paths:
        raise ValueError("rendered SVG mapping is incomplete or reordered")
    sources = _source_index(evidence)
    bindings: list[ReplayFigureSourceBindingV1] = []
    covered_identities: set[str] = set()
    for spec in ordered:
        spec_sha256 = hashlib.sha256(canonical_figure_spec_bytes(spec)).hexdigest()
        svg = svgs[spec.figure_file]
        expected_svg = render_figure_svg(spec, registry)
        if svg != expected_svg:
            raise ValueError("rendered SVG does not regenerate from its spec and registry")
        svg_sha256 = hashlib.sha256(svg).hexdigest()
        for mark in spec.marks:
            source = sources[(mark.source_kind, mark.source_identifier)]
            identity, identity_sha256 = _binding_identity(source)
            if identity_sha256 is not None:
                covered_identities.add(identity_sha256)
            bindings.append(
                ReplayFigureSourceBindingV1(
                    schema="ffb.replay-figure-source-binding/v1",
                    run_id=registry.run_id,
                    replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
                    replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
                    figure_id=spec.figure_id,
                    figure_kind=spec.figure_kind,
                    mark_ordinal=mark.mark_ordinal,
                    source_kind=mark.source_kind,
                    source_identifier=mark.source_identifier,
                    source_record_sha256=mark.source_record_sha256,
                    identity=identity,
                    replay_identity_sha256=identity_sha256,
                    figure_spec_sha256=spec_sha256,
                    rendered_svg_path=spec.figure_file,
                    rendered_svg_sha256=svg_sha256,
                    rendered_svg_byte_length=len(svg),
                    tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
                )
            )
    expected_identities = {
        replay_experiment_identity_sha256(identity) for identity in expected_replay_identities()
    }
    if covered_identities != expected_identities:
        raise ValueError("non-descriptor figure sources do not cover all 22 replay identities")
    if len(bindings) != sum(
        _expected_mark_count(figure_id, registry) for figure_id in M5_FIGURE_ORDER
    ):
        raise ValueError("figure binding count does not equal the exact mark union")
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class ReplayFigureBundle:
    """One regenerated set of five specs, five SVGs, and per-mark bindings."""

    specs: tuple[ReplayFigureSpecV1, ...]
    svgs: Mapping[str, bytes]
    bindings: tuple[ReplayFigureSourceBindingV1, ...]


def build_figure_bundle(
    registry: ReplayPublicClaimProjectionsV1,
    evidence: ReplayClaimEvidence,
) -> ReplayFigureBundle:
    """Build the complete deterministic M5 figure bundle."""

    specs = build_figure_specs(registry, evidence)
    svgs = render_figure_svgs(specs, registry)
    bindings = build_figure_source_bindings(specs, svgs, registry, evidence)
    return ReplayFigureBundle(specs=specs, svgs=MappingProxyType(svgs), bindings=bindings)


def validate_figure_bundle(
    specs: Sequence[ReplayFigureSpecV1],
    svgs: Mapping[str, bytes],
    bindings: Sequence[ReplayFigureSourceBindingV1],
    evidence: ReplayClaimEvidence,
) -> None:
    """Regenerate and byte-check every spec, SVG, and source binding."""

    registry = build_public_claim_projections(evidence)
    expected = build_figure_bundle(registry, evidence)
    if tuple(specs) != expected.specs:
        raise ValueError("figure specs do not regenerate from aggregate evidence")
    if dict(svgs) != dict(expected.svgs):
        raise ValueError("rendered SVGs do not regenerate byte-for-byte")
    if tuple(bindings) != expected.bindings:
        raise ValueError("figure source bindings do not regenerate from aggregate evidence")


__all__ = [
    "M5_FIGURE_ORDER",
    "ReplayFigureBundle",
    "build_figure_bundle",
    "build_figure_source_bindings",
    "build_figure_specs",
    "canonical_figure_spec_bytes",
    "canonical_figure_spec_files",
    "render_figure_svg",
    "render_figure_svgs",
    "render_replay_figure_svg",
    "validate_figure_bundle",
]
