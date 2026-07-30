"""Outcome-independent public projections and presentation text for M5.

This module deliberately consumes only the curated, aggregate-only replay
contracts.  Every selector and its order is fixed below from the M5 release
preregistration; no estimate, interval, status, or persistence label is used
to decide whether a record is public.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256,
    M5_HEALTH_HYPOTHESIS_COORDINATES,
    M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256,
    M5_PERSISTENT_HYPOTHESIS_COORDINATES,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_TRACKED_AGGREGATE_TERMS,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    ReplayProfileSummaryV1,
    ReplayRepeatVerificationV1,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_PRESENTATION_PLACEHOLDERS,
    ClaimProjectionGroup,
    ClaimSourceKind,
    FigureId,
    ReplayClaimSelectorFieldV1,
    ReplayProjectedFieldV1,
    ReplayPublicClaimProjectionsV1,
    ReplayPublicClaimProjectionV1,
    ReplaySoftwareVerificationV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_SHA256,
)
from fusion_fault_bench.replay_release_validation import (
    software_verification_test_subset_bytes,
)

_PERSISTENT_MEMBER = "machine/persistent-panel-aggregates.ndjson"
_CROSSOVER_MEMBER = "machine/persistent-panel-crossovers.ndjson"
_HEALTH_MEMBER = "machine/health-panel-aggregates.ndjson"
_SENSITIVITY_MEMBER = "machine/leave-one-cluster-sensitivity.ndjson"
_DESCRIPTOR_MEMBER = "machine/descriptor-aggregates.ndjson"
_PROFILE_MEMBER = "machine/replay-profile-summary.json"
_REPEAT_MEMBER = "machine/repeat-verification.json"
_SOFTWARE_MEMBER = "evidence/software-verification.json"
_FINAL_INDEX_MEMBER = "artifact/release-index.json"
_DROPOUT_NESTING_TEST_ID = "dropout-nesting-derivation"
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

M5_PRESENTATION_TEMPLATE_PATHS = (
    "presentation/README.md",
    "presentation/claim-evidence.md",
    "presentation/verification.md",
)

_PERSISTENT_FIGURE_ID: FigureId = "m5-persistent-panel-summary"
_CROSSOVER_FIGURE_ID: FigureId = "m5-crossovers"
_HEALTH_FIGURE_ID: FigureId = "m5-health-transfer"
_DESCRIPTOR_FIGURE_ID: FigureId = "m5-descriptor-comparison"
_SENSITIVITY_FIGURE_ID: FigureId = "m5-cluster-sensitivity"


def _format_selector_value(value: float, *, signed: bool) -> str:
    rendered = format(float(value), ".15g")
    return f"+{rendered}" if signed and value > 0.0 else rendered


def _symmetric_selectors(experiment_id: str, magnitudes: Sequence[float]) -> tuple[str, ...]:
    output = [f"{experiment_id}:0"]
    for magnitude in magnitudes:
        rendered = _format_selector_value(magnitude, signed=False)
        output.extend((f"{experiment_id}:-{rendered}", f"{experiment_id}:+{rendered}"))
    return tuple(output)


M5_ORDINARY_PERSISTENT_SELECTORS = (
    *_symmetric_selectors("replay-lidar-y-bias", (0.25, 0.5, 1.0, 2.0, 4.0)),
    *(
        f"replay-camera-noise-correctly-reported:{value}"
        for value in ("1", "1.25", "1.5", "2", "4")
    ),
    *(f"replay-camera-noise-underreported:{value}" for value in ("1", "1.25", "1.5", "2", "4")),
    *_symmetric_selectors("replay-camera-calibration-x", (0.25, 0.5, 1.0, 2.0, 4.0)),
    *_symmetric_selectors("replay-camera-calibration-yaw", (0.005, 0.01, 0.02, 0.04, 0.08)),
    *_symmetric_selectors("replay-camera-timestamp-offset", (0.05, 0.1, 0.2, 0.4, 0.8)),
)
M5_DROPOUT_PERSISTENT_SELECTORS = tuple(
    f"replay-camera-dropout:{value}" for value in ("0", "0.1", "0.25", "0.5", "0.75", "1")
)
M5_COMMON_MODE_PERSISTENT_SELECTORS = _symmetric_selectors(
    "replay-common-mode-x", (0.25, 0.5, 1.0, 2.0, 4.0)
)


def _health_selectors(
    experiment_id: str,
    values: Sequence[float],
    *,
    signed: bool,
) -> tuple[str, ...]:
    return tuple(
        f"{experiment_id}:{_format_selector_value(value, signed=signed)}" for value in values
    )


M5_HEALTH_FIGURE_SELECTORS = (
    *_health_selectors("replay-camera-output-y-bias", (-3.0, -0.75, 0.75, 3.0), signed=True),
    *_health_selectors("replay-lidar-output-y-bias", (-3.0, -0.75, 0.75, 3.0), signed=True),
    *_health_selectors("replay-camera-noise-underreported", (1.25, 3.0), signed=False),
    *_health_selectors("replay-lidar-noise-underreported", (1.25, 3.0), signed=False),
    *_health_selectors("replay-camera-noise-correctly-reported", (1.25, 3.0), signed=False),
    *_health_selectors("replay-lidar-noise-correctly-reported", (1.25, 3.0), signed=False),
    *_health_selectors("replay-camera-timestamp-offset", (-0.6, -0.15, 0.15, 0.6), signed=True),
    *_health_selectors("replay-lidar-timestamp-offset", (-0.6, -0.15, 0.15, 0.6), signed=True),
    *_health_selectors("replay-camera-dropout", (0.1, 0.5, 1.0), signed=False),
    *_health_selectors("replay-lidar-dropout", (0.1, 0.5, 1.0), signed=False),
    *_health_selectors("replay-camera-calibration-x", (-3.0, -0.75, 0.75, 3.0), signed=True),
    *_health_selectors("replay-camera-calibration-yaw", (-0.06, -0.015, 0.015, 0.06), signed=True),
    *_health_selectors("replay-common-mode-x", (-4.0, -1.0, 1.0, 4.0), signed=True),
    "replay-clean:0",
)

M5_SHARED_DESCRIPTOR_IDS = (
    "sample-count",
    "eligible-object-frame-count",
    "eligible-track-length-q50",
    "ego-range-q50",
    "ego-bearing-q50",
    "finite-difference-speed-q50",
    "reference-time-delta-q50",
)
M5_REPLAY_ONLY_DESCRIPTOR_IDS = (
    "support-all-annotations",
    "support-roi-pass",
    "support-camera-center-pass",
    "support-lidar-points-positive",
    "support-final-eligible",
    "unique-eligible-track-count",
    "camera-minus-lidar-acquisition-offset-q50",
    "zero-order-hold-velocity-fraction",
)
M5_DESCRIPTOR_STATISTICS = ("minimum", "median", "maximum")

_EXPECTED_CROSSOVER_COORDINATES = (
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

if (
    len(M5_ORDINARY_PERSISTENT_SELECTORS) != 54
    or len(M5_DROPOUT_PERSISTENT_SELECTORS) != 6
    or len(M5_COMMON_MODE_PERSISTENT_SELECTORS) != 11
    or len(M5_HEALTH_FIGURE_SELECTORS) != 43
):
    raise RuntimeError("M5 public selector constants are incomplete")


@dataclass(frozen=True, slots=True)
class ReplayClaimEvidence:
    """The aggregate-only inputs needed to regenerate all public M5 claims."""

    profile_summary: ReplayProfileSummaryV1
    descriptor_aggregates: tuple[ReplayDescriptorAggregateV1, ...]
    persistent_aggregates: tuple[ReplayPersistentAggregateV1, ...]
    persistent_crossovers: tuple[ReplayPersistentCrossoverV1, ...]
    health_aggregates: tuple[ReplayHealthAggregateV1, ...]
    cluster_sensitivity: tuple[ReplayClusterSensitivityV1, ...]
    repeat_verification: ReplayRepeatVerificationV1
    software_verification: ReplaySoftwareVerificationV1 | None = None


type PanelAggregate = ReplayPersistentAggregateV1 | ReplayHealthAggregateV1


def _require_unique_index[RowT](
    rows: Sequence[RowT],
    key: Callable[[RowT], object],
    *,
    label: str,
) -> dict[object, RowT]:
    index: dict[object, RowT] = {}
    for row in rows:
        coordinate = key(row)
        if coordinate in index:
            raise ValueError(f"{label} contains a duplicate public coordinate")
        index[coordinate] = row
    return index


def _aggregate_key(row: PanelAggregate) -> tuple[str, str, str, str, str, str]:
    return (
        row.condition_selector,
        row.method_id,
        row.metric_id,
        row.window,
        row.unit,
        row.aggregation,
    )


def _lookup_aggregate(
    index: Mapping[object, PanelAggregate],
    coordinate: tuple[str, str, str, str, str, str],
    *,
    label: str,
) -> PanelAggregate:
    try:
        return index[coordinate]
    except KeyError as error:
        raise ValueError(f"{label} is missing frozen coordinate {coordinate!r}") from error


def select_persistent_figure_rows(
    rows: Sequence[ReplayPersistentAggregateV1],
) -> tuple[ReplayPersistentAggregateV1, ...]:
    """Select the exact preregistered 100 M5-A rows in fixed selector order."""

    index = _require_unique_index(rows, _aggregate_key, label="persistent aggregates")
    selected: list[ReplayPersistentAggregateV1] = []
    for selector in M5_ORDINARY_PERSISTENT_SELECTORS:
        row = _lookup_aggregate(
            index,
            (selector, "fixed-fusion", "fused-minus-healthy", "full", "m^2", "equal-scene-mean"),
            label="persistent figure",
        )
        selected.append(cast(ReplayPersistentAggregateV1, row))
    dropout_coordinates = (
        ("fixed-fusion", "coverage", "fraction", "pooled-valid-eligible-count-ratio"),
        ("fixed-fusion", "conditional-matched-center-mse", "m^2", "pooled-valid-loss"),
        ("fault-target-drop-policy", "coverage", "fraction", "pooled-valid-eligible-count-ratio"),
        ("lidar-only", "coverage", "fraction", "pooled-valid-eligible-count-ratio"),
    )
    for selector in M5_DROPOUT_PERSISTENT_SELECTORS:
        for method_id, metric_id, unit, aggregation in dropout_coordinates:
            row = _lookup_aggregate(
                index,
                (selector, method_id, metric_id, "full", unit, aggregation),
                label="dropout figure",
            )
            selected.append(cast(ReplayPersistentAggregateV1, row))
    common_mode_coordinates = (
        ("fixed-fusion", "matched-center-mse"),
        ("camera-lidar-pair", "camera-lidar-disagreement-mse"),
    )
    for selector in M5_COMMON_MODE_PERSISTENT_SELECTORS:
        for method_id, metric_id in common_mode_coordinates:
            row = _lookup_aggregate(
                index,
                (selector, method_id, metric_id, "full", "m^2", "equal-scene-mean"),
                label="common-mode figure",
            )
            selected.append(cast(ReplayPersistentAggregateV1, row))
    if len(selected) != 100 or len({row.result_id for row in selected}) != 100:
        raise ValueError("persistent figure does not contain the exact 100 fixed rows")
    return tuple(selected)


def select_crossover_figure_rows(
    rows: Sequence[ReplayPersistentCrossoverV1],
) -> tuple[ReplayPersistentCrossoverV1, ...]:
    """Select all ten physical-axis crossover records in frozen order."""

    index = _require_unique_index(
        rows,
        lambda row: (
            row.identity.experiment_id,
            row.direction,
            row.severity_unit,
            float(row.tested_maximum),
        ),
        label="persistent crossovers",
    )
    if set(index) != set(_EXPECTED_CROSSOVER_COORDINATES):
        raise ValueError("crossover inputs differ from the exact ten preregistered coordinates")
    return tuple(index[coordinate] for coordinate in _EXPECTED_CROSSOVER_COORDINATES)


def select_health_figure_rows(
    rows: Sequence[ReplayHealthAggregateV1],
) -> tuple[ReplayHealthAggregateV1, ...]:
    """Select one fixed event-window gain row for every M5-B selector."""

    index = _require_unique_index(rows, _aggregate_key, label="health aggregates")
    selected = tuple(
        cast(
            ReplayHealthAggregateV1,
            _lookup_aggregate(
                index,
                (
                    selector,
                    "combined-health-gate",
                    "policy-gain-vs-fixed",
                    "event",
                    "m^2",
                    "equal-scene-mean",
                ),
                label="health figure",
            ),
        )
        for selector in M5_HEALTH_FIGURE_SELECTORS
    )
    if len({row.result_id for row in selected}) != 43:
        raise ValueError("health figure does not contain the exact 43 fixed rows")
    return selected


def descriptor_source_identifier(row: ReplayDescriptorAggregateV1) -> str:
    """Return the stable identifier for one descriptor coordinate."""

    category = "none" if row.category_label is None else row.category_label
    rendered = f"{row.population}:{row.descriptor_id}:{row.statistic}:{category}"
    if len(rendered) <= 256:
        return rendered
    return "replay-descriptor:" + sha256_digest(
        {
            "population": row.population,
            "descriptor_id": row.descriptor_id,
            "statistic": row.statistic,
            "category_label": row.category_label,
        }
    )


def select_descriptor_figure_rows(
    rows: Sequence[ReplayDescriptorAggregateV1],
) -> tuple[ReplayDescriptorAggregateV1, ...]:
    """Select the exact fixed 42 paired, 24 replay-only, and one count rows."""

    index = _require_unique_index(
        rows,
        lambda row: (row.population, row.descriptor_id, row.statistic, row.category_label),
        label="descriptor aggregates",
    )
    expected: list[tuple[str, str, str, None]] = []
    for descriptor_id in M5_SHARED_DESCRIPTOR_IDS:
        for population in ("nuscenes-mini-replay", "m3-main-test-comparator"):
            expected.extend(
                (population, descriptor_id, statistic, None)
                for statistic in M5_DESCRIPTOR_STATISTICS
            )
    for descriptor_id in M5_REPLAY_ONLY_DESCRIPTOR_IDS:
        expected.extend(
            ("nuscenes-mini-replay", descriptor_id, statistic, None)
            for statistic in M5_DESCRIPTOR_STATISTICS
        )
    expected.append(("nuscenes-mini-replay", "distinct-log-group-count", "count", None))
    try:
        selected = tuple(index[coordinate] for coordinate in expected)
    except KeyError as error:
        raise ValueError(
            f"descriptor figure is missing frozen coordinate {error.args[0]!r}"
        ) from error
    if len(selected) != 67 or len({descriptor_source_identifier(row) for row in selected}) != 67:
        raise ValueError("descriptor figure does not contain the exact 67 fixed rows")
    return selected


def select_sensitivity_sources(
    persistent_rows: Sequence[ReplayPersistentAggregateV1],
    health_rows: Sequence[ReplayHealthAggregateV1],
) -> tuple[PanelAggregate, ...]:
    """Select all 16 M5-A, eight M5-B directional, and two control sources."""

    persistent = tuple(
        row for row in persistent_rows if row.inference_role == "primary-directional"
    )
    health_directional = tuple(
        row for row in health_rows if row.inference_role == "primary-directional"
    )
    health_controls = tuple(
        row for row in health_rows if row.inference_role == "nonpositive-control"
    )
    if (len(persistent), len(health_directional), len(health_controls)) != (16, 8, 2):
        raise ValueError("sensitivity sources do not have the frozen 16/8/2 role partition")
    output: tuple[PanelAggregate, ...] = (*persistent, *health_directional, *health_controls)
    if len({row.result_id for row in output}) != 26:
        raise ValueError("sensitivity sources contain duplicate result identifiers")
    return output


def select_sensitivity_figure_rows(
    *,
    sources: Sequence[PanelAggregate],
    rows: Sequence[ReplayClusterSensitivityV1],
    distinct_log_group_count: int,
) -> tuple[ReplayClusterSensitivityV1, ...]:
    """Select every required LOSO/LOLO row, including undefined leave-outs."""

    if len(sources) != 26 or not 1 <= distinct_log_group_count <= 10:
        raise ValueError("sensitivity selection requires 26 sources and one to ten log groups")
    index = _require_unique_index(
        rows,
        lambda row: (row.source_result_id, row.cluster_kind, row.cluster_id),
        label="cluster sensitivity",
    )
    expected_keys: list[tuple[str, str, str]] = []
    source_by_id = {source.result_id: source for source in sources}
    for source in sources:
        expected_keys.extend(
            (source.result_id, "leave-one-scene-out", f"scene-ordinal:{ordinal:02d}")
            for ordinal in range(10)
        )
        expected_keys.extend(
            (source.result_id, "leave-one-log-group-out", f"log-group:{ordinal:02d}")
            for ordinal in range(distinct_log_group_count)
        )
    if set(index) != set(expected_keys):
        raise ValueError("cluster sensitivity does not cover exactly every required leave-out")
    selected = tuple(index[key] for key in expected_keys)
    for row in selected:
        source = source_by_id[row.source_result_id]
        if (
            row.source_record_sha256 != sha256_digest(source)
            or row.identity != source.identity
            or row.unit != source.unit
        ):
            raise ValueError("cluster sensitivity has an invalid aggregate source binding")
    return selected


def _selector_fields(**values: object) -> tuple[ReplayClaimSelectorFieldV1, ...]:
    return tuple(
        ReplayClaimSelectorFieldV1(field=field, value=str(value)) for field, value in values.items()
    )


def _projected(
    field: str,
    value: object,
    rendering: Literal["machine-token", ".6g", ".2f", "exact-integer", "literal-status"],
) -> ReplayProjectedFieldV1:
    return ReplayProjectedFieldV1(field=field, value=cast(Any, value), rendering=rendering)


def _status_behavior(
    status: str,
) -> Literal["defined-numeric", "literal-undefined", "literal-not-applicable"]:
    if status == "ok":
        return "defined-numeric"
    if status == "undefined":
        return "literal-undefined"
    if status == "not-applicable":
        return "literal-not-applicable"
    raise ValueError("aggregate projection has an unknown status")


def _projection_id(
    *,
    group: ClaimProjectionGroup,
    source_member: str,
    source_identifier: str,
    selectors: Sequence[ReplayClaimSelectorFieldV1],
) -> str:
    return "m5-projection-" + sha256_digest(
        {
            "schema": "ffb.m5-public-projection-coordinate/v1",
            "group": group,
            "source_member": source_member,
            "source_identifier": source_identifier,
            "selectors": [row.model_dump(mode="json") for row in selectors],
        }
    )


def _make_projection(
    *,
    group: ClaimProjectionGroup,
    source_member: str,
    source_kind: ClaimSourceKind,
    source_identifier: str,
    source_record_sha256: str | None,
    selectors: tuple[ReplayClaimSelectorFieldV1, ...],
    projected: tuple[ReplayProjectedFieldV1, ...],
    unit: str,
    status_behavior: Literal[
        "defined-numeric",
        "literal-undefined",
        "literal-not-applicable",
        "literal-censored",
        "literal-positive-infinity",
        "finalization-null-then-exact-integer",
    ],
    figure_ids: tuple[FigureId, ...] = (),
    hypothesis_id: str | None = None,
    public_claim_id: str | None = None,
) -> ReplayPublicClaimProjectionV1:
    return ReplayPublicClaimProjectionV1(
        schema="ffb.m5-public-claim-projection/v1",
        projection_id=_projection_id(
            group=group,
            source_member=source_member,
            source_identifier=source_identifier,
            selectors=selectors,
        ),
        public_claim_id=public_claim_id or f"m5-{group}",
        projection_group=group,
        source_member=source_member,
        source_kind=source_kind,
        source_identifier=source_identifier,
        source_record_sha256=source_record_sha256,
        selector_fields=selectors,
        projected_fields=projected,
        unit=unit,
        status_behavior=status_behavior,
        figure_ids=figure_ids,
        hypothesis_id=hypothesis_id,
    )


def _panel_projection(
    row: PanelAggregate,
    *,
    group: Literal["persistent-panel", "health-transfer"],
    source_member: str,
    source_kind: Literal["persistent-aggregate", "health-aggregate"],
    figure_id: FigureId,
) -> ReplayPublicClaimProjectionV1:
    selectors = _selector_fields(
        condition_selector=row.condition_selector,
        method_id=row.method_id,
        metric_id=row.metric_id,
        window=row.window,
        unit=row.unit,
        aggregation=row.aggregation,
    )
    projected = (
        _projected("status", row.status, "literal-status"),
        _projected("estimate", row.estimate, ".6g"),
        _projected("interval_lower", row.interval_lower, ".6g"),
        _projected("interval_upper", row.interval_upper, ".6g"),
        _projected("persistence_label", row.persistence_label, "literal-status"),
        _projected("positive_scene_count", row.positive_scene_count, "exact-integer"),
        _projected("zero_scene_count", row.zero_scene_count, "exact-integer"),
        _projected("negative_scene_count", row.negative_scene_count, "exact-integer"),
        _projected("undefined_scene_count", row.undefined_scene_count, "exact-integer"),
        _projected(
            "nonpositive_control_supported",
            row.nonpositive_control_supported,
            "literal-status",
        ),
    )
    return _make_projection(
        group=group,
        source_member=source_member,
        source_kind=source_kind,
        source_identifier=row.result_id,
        source_record_sha256=sha256_digest(row),
        selectors=selectors,
        projected=projected,
        unit=row.unit,
        status_behavior=_status_behavior(row.status),
        figure_ids=(figure_id,),
        hypothesis_id=row.hypothesis_id,
        public_claim_id=row.hypothesis_id or f"m5-{group}-context",
    )


def _validate_hypothesis_coordinates(
    persistent: Sequence[ReplayPersistentAggregateV1],
    health: Sequence[ReplayHealthAggregateV1],
) -> None:
    actual_persistent = {
        (
            row.hypothesis_id,
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.aggregation,
            row.inference_role,
            row.expected_direction,
        )
        for row in persistent
        if row.hypothesis_id is not None
    }
    actual_health = {
        (
            row.hypothesis_id,
            row.condition_selector,
            row.method_id,
            row.metric_id,
            row.window,
            row.unit,
            row.inference_role,
            row.expected_direction,
        )
        for row in health
        if row.hypothesis_id is not None
    }
    if actual_persistent != set(M5_PERSISTENT_HYPOTHESIS_COORDINATES):
        raise ValueError("persistent public rows do not mark all exact 33 M5-A hypotheses")
    if actual_health != set(M5_HEALTH_HYPOTHESIS_COORDINATES):
        raise ValueError("health public rows do not mark all exact 11 M5-B hypotheses")


def _validate_evidence_globals(evidence: ReplayClaimEvidence) -> None:
    records: tuple[object, ...] = (
        evidence.profile_summary,
        *evidence.descriptor_aggregates,
        *evidence.persistent_aggregates,
        *evidence.persistent_crossovers,
        *evidence.health_aggregates,
        *evidence.cluster_sensitivity,
        evidence.repeat_verification,
    )
    if any(
        getattr(row, "run_id", None) != evidence.profile_summary.run_id
        or getattr(row, "replay_intent_sha256", None) != M5_REPLAY_INTENT_SHA256
        or getattr(row, "replay_identity_set_sha256", None) != M5_REPLAY_IDENTITY_SET_SHA256
        for row in records
    ):
        raise ValueError("public claim evidence has inconsistent global bindings")
    if evidence.software_verification is None:
        raise ValueError("public claim evidence lacks software verification authority")


def _descriptor_projection(
    row: ReplayDescriptorAggregateV1,
    *,
    plotted: bool,
) -> ReplayPublicClaimProjectionV1:
    return _make_projection(
        group="descriptor-comparison" if plotted else "release-facts",
        source_member=_DESCRIPTOR_MEMBER,
        source_kind="descriptor-aggregate",
        source_identifier=descriptor_source_identifier(row),
        source_record_sha256=sha256_digest(row),
        selectors=_selector_fields(
            population=row.population,
            descriptor_id=row.descriptor_id,
            statistic=row.statistic,
            category_label=row.category_label or "none",
        ),
        projected=(
            _projected("status", row.status, "literal-status"),
            _projected("value", row.value, ".6g"),
        ),
        unit=row.unit,
        status_behavior=_status_behavior(row.status),
        figure_ids=(_DESCRIPTOR_FIGURE_ID,) if plotted else (),
        public_claim_id=("m5-descriptor-comparison" if plotted else "m5-descriptor-ledger"),
    )


def _defined_coverage(row: ReplayPersistentAggregateV1) -> tuple[float, float, float]:
    if (
        row.status != "ok"
        or row.estimate is None
        or row.interval_lower is None
        or row.interval_upper is None
    ):
        raise ValueError("dropout nesting requires defined fixed coverage sources")
    return row.estimate, row.interval_lower, row.interval_upper


def _dropout_nesting_projection(
    evidence: ReplayClaimEvidence,
    persistent: Sequence[ReplayPersistentAggregateV1],
) -> ReplayPublicClaimProjectionV1:
    verification = evidence.software_verification
    if verification is None:
        raise ValueError("dropout nesting lacks software verification authority")
    subset = software_verification_test_subset_bytes(
        verification,
        (_DROPOUT_NESTING_TEST_ID,),
    )
    index = {
        (row.condition_selector, row.method_id, row.metric_id): row
        for row in persistent
        if row.condition_selector in M5_DROPOUT_PERSISTENT_SELECTORS
    }
    fixed = tuple(
        index[(selector, "fixed-fusion", "coverage")]
        for selector in M5_DROPOUT_PERSISTENT_SELECTORS
    )
    target_drop = tuple(
        index[(selector, "fault-target-drop-policy", "coverage")]
        for selector in M5_DROPOUT_PERSISTENT_SELECTORS
    )
    lidar = tuple(
        index[(selector, "lidar-only", "coverage")] for selector in M5_DROPOUT_PERSISTENT_SELECTORS
    )
    fixed_values = tuple(_defined_coverage(row) for row in fixed)
    target_values = tuple(_defined_coverage(row) for row in target_drop)
    lidar_values = tuple(_defined_coverage(row) for row in lidar)
    fixed_nested = all(
        all(
            next_value <= previous_value
            for next_value, previous_value in zip(after, before, strict=True)
        )
        for before, after in pairwise(fixed_values)
    )
    target_matches_lidar = target_values == lidar_values
    if not fixed_nested or not target_matches_lidar:
        raise ValueError("dropout nesting relation failed for fixed coverage sources")
    selector_values: dict[str, object] = {
        "required_test_id": _DROPOUT_NESTING_TEST_ID,
        "named_authority_sha256": hashlib.sha256(subset).hexdigest(),
    }
    for ordinal, selector in enumerate(M5_DROPOUT_PERSISTENT_SELECTORS):
        selector_values[f"selector_{ordinal:02d}"] = selector
        for method_label, rows in (
            ("fixed", fixed),
            ("target_drop", target_drop),
            ("lidar", lidar),
        ):
            selector_values[f"{method_label}_{ordinal:02d}_sha256"] = sha256_digest(rows[ordinal])
    return _make_projection(
        group="dropout-nesting",
        source_member=_SOFTWARE_MEMBER,
        source_kind="software-verification",
        source_identifier=_DROPOUT_NESTING_TEST_ID,
        source_record_sha256=sha256_digest(verification),
        selectors=_selector_fields(**selector_values),
        projected=(
            _projected("all_coverage_sources_defined", True, "literal-status"),
            _projected("fixed_fusion_coverage_nested", fixed_nested, "literal-status"),
            _projected("target_drop_matches_lidar_only", target_matches_lidar, "literal-status"),
            _projected("dropout_nesting_passed", True, "literal-status"),
            _projected("coverage_source_count", 18, "exact-integer"),
        ),
        unit="relation",
        status_behavior="defined-numeric",
        public_claim_id="m5-dropout-nesting",
    )


def build_public_claim_projections(
    evidence: ReplayClaimEvidence,
) -> ReplayPublicClaimProjectionsV1:
    """Build the complete, fixed M5 public claim registry from curated records."""

    _validate_evidence_globals(evidence)
    persistent = select_persistent_figure_rows(evidence.persistent_aggregates)
    crossovers = select_crossover_figure_rows(evidence.persistent_crossovers)
    health = select_health_figure_rows(evidence.health_aggregates)
    descriptors = select_descriptor_figure_rows(evidence.descriptor_aggregates)
    sensitivity_sources = select_sensitivity_sources(
        evidence.persistent_aggregates,
        evidence.health_aggregates,
    )
    sensitivity = select_sensitivity_figure_rows(
        sources=sensitivity_sources,
        rows=evidence.cluster_sensitivity,
        distinct_log_group_count=evidence.profile_summary.distinct_log_group_count,
    )
    _validate_hypothesis_coordinates(persistent, health)

    projections: list[ReplayPublicClaimProjectionV1] = []
    projections.extend(
        _panel_projection(
            row,
            group="persistent-panel",
            source_member=_PERSISTENT_MEMBER,
            source_kind="persistent-aggregate",
            figure_id=_PERSISTENT_FIGURE_ID,
        )
        for row in persistent
    )
    for row in crossovers:
        selectors = _selector_fields(
            experiment_id=row.identity.experiment_id,
            direction=row.direction,
            severity_unit=row.severity_unit,
            tested_maximum=row.tested_maximum,
        )
        behavior: Literal["defined-numeric", "literal-censored", "literal-positive-infinity"]
        if row.status == "observed":
            behavior = "defined-numeric"
        elif row.status == "not-observed":
            behavior = "literal-positive-infinity"
        else:
            behavior = "literal-censored"
        projections.append(
            _make_projection(
                group="crossovers",
                source_member=_CROSSOVER_MEMBER,
                source_kind="persistent-crossover",
                source_identifier=row.crossover_id,
                source_record_sha256=sha256_digest(row),
                selectors=selectors,
                projected=(
                    _projected("status", row.status, "literal-status"),
                    _projected("point_estimate", row.point_estimate, ".6g"),
                    _projected("interval_lower", row.interval_lower, ".6g"),
                    _projected(
                        "interval_upper",
                        row.interval_upper,
                        ".6g" if isinstance(row.interval_upper, float) else "literal-status",
                    ),
                    _projected("censoring", row.censoring, "literal-status"),
                    _projected(
                        "bootstrap_crossing_fraction", row.bootstrap_crossing_fraction, ".6g"
                    ),
                ),
                unit=row.severity_unit,
                status_behavior=behavior,
                figure_ids=(_CROSSOVER_FIGURE_ID,),
            )
        )
    projections.extend(
        _panel_projection(
            row,
            group="health-transfer",
            source_member=_HEALTH_MEMBER,
            source_kind="health-aggregate",
            figure_id=_HEALTH_FIGURE_ID,
        )
        for row in health
    )
    for row in sensitivity:
        source = next(
            source for source in sensitivity_sources if source.result_id == row.source_result_id
        )
        projections.append(
            _make_projection(
                group="cluster-sensitivity",
                source_member=_SENSITIVITY_MEMBER,
                source_kind="cluster-sensitivity",
                source_identifier=row.sensitivity_id,
                source_record_sha256=sha256_digest(row),
                selectors=_selector_fields(
                    source_result_id=row.source_result_id,
                    condition_selector=source.condition_selector,
                    hypothesis_id=source.hypothesis_id or "none",
                    inference_role=source.inference_role,
                    expected_direction=source.expected_direction,
                    cluster_kind=row.cluster_kind,
                    cluster_id=row.cluster_id,
                ),
                projected=(
                    _projected("status", row.status, "literal-status"),
                    _projected("estimate", row.estimate, ".6g"),
                ),
                unit=row.unit,
                status_behavior=_status_behavior(row.status),
                figure_ids=(_SENSITIVITY_FIGURE_ID,),
            )
        )
    plotted_descriptor_ids = {descriptor_source_identifier(row) for row in descriptors}
    projections.extend(_descriptor_projection(row, plotted=True) for row in descriptors)
    projections.extend(
        _descriptor_projection(row, plotted=False)
        for row in evidence.descriptor_aggregates
        if descriptor_source_identifier(row) not in plotted_descriptor_ids
    )
    for row in evidence.profile_summary.resource_evidence:
        projections.append(
            _make_projection(
                group="resources",
                source_member=_PROFILE_MEMBER,
                source_kind="execution-resource",
                source_identifier=f"replay-resource:{row.run_label}",
                source_record_sha256=sha256_digest(row),
                selectors=_selector_fields(run_label=row.run_label),
                projected=(
                    _projected("elapsed_seconds", row.elapsed_seconds, ".2f"),
                    _projected("peak_rss_bytes", row.peak_rss_bytes, "exact-integer"),
                    _projected("wall_time_within_cap", row.wall_time_within_cap, "literal-status"),
                    _projected("peak_rss_within_cap", row.peak_rss_within_cap, "literal-status"),
                ),
                unit="seconds-and-bytes",
                status_behavior="defined-numeric",
            )
        )

    profile_facts: tuple[tuple[str, object, str, str], ...] = (
        ("scene_count", evidence.profile_summary.scene_count, "count", "exact-integer"),
        (
            "persistent_experiment_count",
            evidence.profile_summary.persistent_experiment_count,
            "count",
            "exact-integer",
        ),
        (
            "health_experiment_count",
            evidence.profile_summary.health_experiment_count,
            "count",
            "exact-integer",
        ),
        (
            "replay_experiment_count",
            evidence.profile_summary.replay_experiment_count,
            "count",
            "exact-integer",
        ),
        (
            "distinct_log_group_count",
            evidence.profile_summary.distinct_log_group_count,
            "count",
            "exact-integer",
        ),
        ("elapsed_seconds", evidence.profile_summary.elapsed_seconds, "s", ".2f"),
        ("peak_rss_bytes", evidence.profile_summary.peak_rss_bytes, "bytes", "exact-integer"),
    )
    for field, value, unit, rendering in profile_facts:
        projections.append(
            _make_projection(
                group="release-facts",
                source_member=_PROFILE_MEMBER,
                source_kind="profile-summary",
                source_identifier=f"replay-profile-summary:{field}",
                source_record_sha256=sha256_digest(evidence.profile_summary),
                selectors=_selector_fields(field=field),
                projected=(_projected(field, value, cast(Any, rendering)),),
                unit=unit,
                status_behavior="defined-numeric",
            )
        )
    repeat_facts = (
        ("mismatch_count", evidence.repeat_verification.mismatch_count),
        ("scientific_member_count", evidence.repeat_verification.scientific_member_count),
    )
    for field, value in repeat_facts:
        projections.append(
            _make_projection(
                group="release-facts",
                source_member=_REPEAT_MEMBER,
                source_kind="repeat-verification",
                source_identifier=f"replay-repeat-verification:{field}",
                source_record_sha256=sha256_digest(evidence.repeat_verification),
                selectors=_selector_fields(field=field),
                projected=(_projected(field, value, "exact-integer"),),
                unit="count",
                status_behavior="defined-numeric",
            )
        )

    projections.append(_dropout_nesting_projection(evidence, persistent))

    projections.append(
        _make_projection(
            group="finalization-metadata",
            source_member=_FINAL_INDEX_MEMBER,
            source_kind="release-index",
            source_identifier="machine-artifact-byte-length",
            source_record_sha256=None,
            selectors=_selector_fields(field="artifact_byte_length"),
            projected=(_projected("artifact_byte_length", None, "exact-integer"),),
            unit="bytes",
            status_behavior="finalization-null-then-exact-integer",
            public_claim_id="m5-finalization-metadata",
        )
    )
    return ReplayPublicClaimProjectionsV1(
        schema="ffb.m5-public-claim-projections/v1",
        release_id="m5-nuscenes-replay-v0.1.0",
        run_id=evidence.profile_summary.run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        persistent_hypothesis_count=33,
        persistent_hypothesis_coordinate_set_sha256=(
            M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256
        ),
        health_hypothesis_count=11,
        health_hypothesis_coordinate_set_sha256=M5_HEALTH_HYPOTHESIS_COORDINATE_SET_SHA256,
        persistent_figure_projection_count=100,
        crossover_projection_count=10,
        health_figure_projection_count=43,
        sensitivity_source_count=26,
        distinct_log_group_count=evidence.profile_summary.distinct_log_group_count,
        sensitivity_projection_count=len(sensitivity),
        descriptor_figure_projection_count=67,
        resource_record_count=2,
        projections=tuple(projections),
    )


def validate_public_claim_projections(
    registry: ReplayPublicClaimProjectionsV1,
    evidence: ReplayClaimEvidence,
) -> None:
    """Regenerate the registry and require exact semantic equality."""

    expected = build_public_claim_projections(evidence)
    if registry != expected:
        raise ValueError("public claim projection registry does not regenerate from evidence")


def _projection_fields(projection: ReplayPublicClaimProjectionV1) -> dict[str, object]:
    return {field.field: field.value for field in projection.projected_fields}


def _projection_selectors(projection: ReplayPublicClaimProjectionV1) -> dict[str, str]:
    return {field.field: field.value for field in projection.selector_fields}


def _estimate_triplet(
    projection: ReplayPublicClaimProjectionV1,
) -> tuple[float | None, float | None, float | None]:
    fields = _projection_fields(projection)
    values: list[float | None] = []
    for name in ("estimate", "interval_lower", "interval_upper"):
        value = fields.get(name)
        values.append(
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )
    return values[0], values[1], values[2]


def _require_one_projection(
    rows: Sequence[ReplayPublicClaimProjectionV1],
    *,
    condition_selector: str,
    method_id: str,
    metric_id: str,
) -> ReplayPublicClaimProjectionV1:
    matches = tuple(
        row
        for row in rows
        if (
            _projection_selectors(row).get("condition_selector") == condition_selector
            and _projection_selectors(row).get("method_id") == method_id
            and _projection_selectors(row).get("metric_id") == metric_id
        )
    )
    if len(matches) != 1:
        raise ValueError("claim registry lacks one frozen hypothesis comparison row")
    return matches[0]


def _h5_a5_result(rows: Sequence[ReplayPublicClaimProjectionV1]) -> str:
    selector = "replay-camera-dropout:1"
    fixed_coverage = _require_one_projection(
        rows,
        condition_selector=selector,
        method_id="fixed-fusion",
        metric_id="coverage",
    )
    fixed_loss = _require_one_projection(
        rows,
        condition_selector=selector,
        method_id="fixed-fusion",
        metric_id="conditional-matched-center-mse",
    )
    target_coverage = _require_one_projection(
        rows,
        condition_selector=selector,
        method_id="fault-target-drop-policy",
        metric_id="coverage",
    )
    lidar_coverage = _require_one_projection(
        rows,
        condition_selector=selector,
        method_id="lidar-only",
        metric_id="coverage",
    )
    coverage_rows = (fixed_coverage, target_coverage, lidar_coverage)
    if any(_projection_fields(row).get("status") != "ok" for row in coverage_rows):
        return "undefined-or-not-applicable"
    supported = (
        _estimate_triplet(fixed_coverage) == (0.0, 0.0, 0.0)
        and _projection_fields(fixed_loss).get("status") == "undefined"
        and _estimate_triplet(fixed_loss) == (None, None, None)
        and _estimate_triplet(target_coverage) == _estimate_triplet(lidar_coverage)
    )
    return "supported" if supported else "not-supported"


def _h5_a6_result(
    rows: Sequence[ReplayPublicClaimProjectionV1],
    all_rows: Sequence[ReplayPublicClaimProjectionV1],
) -> str:
    absolute_baseline = _require_one_projection(
        all_rows,
        condition_selector="replay-common-mode-x:0",
        method_id="fixed-fusion",
        metric_id="matched-center-mse",
    )
    absolute_endpoints = tuple(
        _require_one_projection(
            rows,
            condition_selector=selector,
            method_id="fixed-fusion",
            metric_id="matched-center-mse",
        )
        for selector in ("replay-common-mode-x:-4", "replay-common-mode-x:+4")
    )
    disagreement_rows = tuple(
        row
        for row in rows
        if _projection_selectors(row).get("metric_id") == "camera-lidar-disagreement-mse"
    )
    if len(disagreement_rows) != 11:
        raise ValueError("claim registry lacks all eleven H5-A6 disagreement rows")
    compared = (absolute_baseline, *absolute_endpoints, *disagreement_rows)
    if any(
        _projection_fields(row).get("status") != "ok"
        or any(value is None for value in _estimate_triplet(row))
        for row in compared
    ):
        return "undefined-or-not-applicable-diagnostic"
    baseline_loss = _estimate_triplet(absolute_baseline)[0]
    assert baseline_loss is not None
    endpoint_increase = all(
        cast(float, _estimate_triplet(row)[0]) > baseline_loss for row in absolute_endpoints
    )
    disagreement_baseline = _require_one_projection(
        disagreement_rows,
        condition_selector="replay-common-mode-x:0",
        method_id="camera-lidar-pair",
        metric_id="camera-lidar-disagreement-mse",
    )
    disagreement_unchanged = all(
        _estimate_triplet(row) == _estimate_triplet(disagreement_baseline)
        for row in disagreement_rows
    )
    supported = endpoint_increase and disagreement_unchanged
    return "supported-diagnostic" if supported else "not-supported-diagnostic"


def _hypothesis_result(
    hypothesis_id: str,
    rows: Sequence[ReplayPublicClaimProjectionV1],
    all_rows: Sequence[ReplayPublicClaimProjectionV1],
) -> dict[str, object]:
    statuses = tuple(str(_projection_fields(row)["status"]) for row in rows)
    persistence = tuple(str(_projection_fields(row)["persistence_label"]) for row in rows)
    control_results = tuple(
        _projection_fields(row)["nonpositive_control_supported"] for row in rows
    )
    if hypothesis_id == "h5-a5":
        result = _h5_a5_result(rows)
        role = "diagnostic"
    elif hypothesis_id == "h5-a6":
        result = _h5_a6_result(rows, all_rows)
        role = "common-mode-diagnostic-no-uniquely-faulty-target"
    elif hypothesis_id == "h5-b3":
        if any(status != "ok" for status in statuses):
            result = "undefined-nonpositive-control"
        elif all(value is True for value in control_results):
            result = "supported-nonpositive-control"
        else:
            result = "not-supported-nonpositive-control"
        role = "nonpositive-control"
    elif hypothesis_id in {"h5-a1", "h5-a2", "h5-a3", "h5-a4", "h5-b1", "h5-b2"}:
        if all(status != "ok" for status in statuses):
            result = "undefined"
        elif any(status != "ok" for status in statuses):
            result = "mixed-defined-and-undefined"
        elif all(label == "robustly-persistent" for label in persistence):
            result = "robustly-persistent"
        elif any(label == "non-persistent" for label in persistence):
            result = "non-persistent"
        else:
            result = "directionally-consistent"
        role = "directional"
    else:
        if all(status == "not-applicable" for status in statuses):
            result = "not-applicable"
        elif all(status == "undefined" for status in statuses):
            result = "undefined"
        elif any(status != "ok" for status in statuses):
            result = "mixed-defined-undefined-or-not-applicable"
        else:
            result = "defined-diagnostic"
        role = (
            "common-mode-diagnostic-no-uniquely-faulty-target"
            if hypothesis_id in {"h5-a6", "h5-b4"}
            else "diagnostic"
        )
    return {
        "result": result,
        "role": role,
        "statuses": statuses,
        "persistence_labels": persistence,
        "nonpositive_control_results": control_results,
        "projection_ids": tuple(row.projection_id for row in rows),
    }


def _hypothesis_results(registry: ReplayPublicClaimProjectionsV1) -> dict[str, object]:
    output: dict[str, object] = {}
    for hypothesis_id in _HYPOTHESIS_ORDER:
        rows = tuple(row for row in registry.projections if row.hypothesis_id == hypothesis_id)
        if not rows:
            raise ValueError(f"claim registry is missing {hypothesis_id}")
        output[hypothesis_id] = _hypothesis_result(hypothesis_id, rows, registry.projections)
    return output


def _m3_mechanism_scope(registry: ReplayPublicClaimProjectionsV1) -> str:
    rows = tuple(
        row
        for row in registry.projections
        if row.hypothesis_id in {"h5-a1", "h5-a2", "h5-a3", "h5-a4"}
    )
    robust = tuple(
        _projection_fields(row)["status"] == "ok"
        and _projection_fields(row)["persistence_label"] == "robustly-persistent"
        for row in rows
    )
    if robust and all(robust):
        return "global"
    if any(robust):
        return "partial"
    return "non-persistence"


def _registry_fact(
    registry: ReplayPublicClaimProjectionsV1,
    source_identifier: str,
    field_name: str,
) -> object:
    matches = tuple(
        field.value
        for projection in registry.projections
        if projection.source_identifier == source_identifier
        for field in projection.projected_fields
        if field.field == field_name
    )
    if len(matches) != 1:
        raise ValueError(f"claim registry lacks unique fact {source_identifier}:{field_name}")
    return matches[0]


def build_release_summary(
    registry: ReplayPublicClaimProjectionsV1,
    evidence: ReplayClaimEvidence,
) -> dict[str, Any]:
    """Build the compact, deterministic candidate release summary."""

    validate_public_claim_projections(registry, evidence)
    status_counts = {
        status: sum(row.status_behavior == status for row in registry.projections)
        for status in (
            "defined-numeric",
            "literal-undefined",
            "literal-not-applicable",
            "literal-censored",
            "literal-positive-infinity",
        )
    }
    nesting = tuple(
        row for row in registry.projections if row.projection_group == "dropout-nesting"
    )
    if len(nesting) != 1:
        raise ValueError("claim registry lacks one explicit dropout nesting result")
    return {
        "schema": "ffb.m5-release-summary/v1",
        "release_id": registry.release_id,
        "run_id": registry.run_id,
        "replay_intent_sha256": registry.replay_intent_sha256,
        "replay_identity_set_sha256": registry.replay_identity_set_sha256,
        "public_claim_projection_sha256": sha256_digest(registry),
        "projection_count": len(registry.projections),
        "persistent_hypothesis_count": registry.persistent_hypothesis_count,
        "health_hypothesis_count": registry.health_hypothesis_count,
        "persistent_figure_projection_count": registry.persistent_figure_projection_count,
        "crossover_projection_count": registry.crossover_projection_count,
        "health_figure_projection_count": registry.health_figure_projection_count,
        "sensitivity_source_count": registry.sensitivity_source_count,
        "sensitivity_projection_count": registry.sensitivity_projection_count,
        "descriptor_figure_projection_count": registry.descriptor_figure_projection_count,
        "descriptor_aggregate_count": sum(
            row.source_kind == "descriptor-aggregate" for row in registry.projections
        ),
        "resource_record_count": registry.resource_record_count,
        "distinct_log_group_count": registry.distinct_log_group_count,
        "status_behavior_counts": status_counts,
        "maximum_elapsed_seconds": _registry_fact(
            registry, "replay-profile-summary:elapsed_seconds", "elapsed_seconds"
        ),
        "maximum_peak_rss_bytes": _registry_fact(
            registry, "replay-profile-summary:peak_rss_bytes", "peak_rss_bytes"
        ),
        "repeat_mismatch_count": _registry_fact(
            registry, "replay-repeat-verification:mismatch_count", "mismatch_count"
        ),
        "scientific_member_count": _registry_fact(
            registry,
            "replay-repeat-verification:scientific_member_count",
            "scientific_member_count",
        ),
        "dropout_nesting_result": _projection_fields(nesting[0]),
        "hypothesis_results": _hypothesis_results(registry),
        "m3_mechanism_scope": _m3_mechanism_scope(registry),
        "machine_artifact_byte_length": None,
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
        "attribution_required": True,
        "non_endorsement": True,
        "claim_boundary": (
            "Matched-center estimator-output replay evidence on the complete fixed "
            "nuScenes-mini scene population; not detector, fleet, production, or safety evidence."
        ),
    }


def _machine_token(value: object) -> str:
    normalized = 0.0 if isinstance(value, float) and value == 0.0 else value
    return json.dumps(normalized, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def _identity_block() -> str:
    artifact, run, attestation, byte_length = M5_PRESENTATION_PLACEHOLDERS
    return "\n".join(
        (
            f"- Machine artifact SHA-256: `{artifact}`",
            f"- Machine run SHA-256: `{run}`",
            f"- Results-review attestation SHA-256: `{attestation}`",
            f"- Machine artifact bytes: `{byte_length}`",
        )
    )


def _assert_template_placeholders(files: Mapping[str, bytes]) -> None:
    allowed = set(M5_PRESENTATION_PLACEHOLDERS)
    for path, payload in files.items():
        text = payload.decode("utf-8")
        if any(text.count(placeholder) != 1 for placeholder in allowed):
            raise RuntimeError(f"{path} does not contain each M5 identity placeholder exactly once")
        discovered = {
            token for token in text.split("`") if token.startswith("@") and token.endswith("@")
        }
        if discovered != allowed:
            raise RuntimeError(f"{path} contains an undeclared presentation placeholder")


def _hypothesis_markdown(release_summary: Mapping[str, Any]) -> str:
    raw_results = release_summary.get("hypothesis_results")
    if not isinstance(raw_results, Mapping):
        raise ValueError("release summary lacks hypothesis results")
    lines: list[str] = []
    for hypothesis_id in _HYPOTHESIS_ORDER:
        raw = raw_results.get(hypothesis_id)
        if not isinstance(raw, Mapping):
            raise ValueError(f"release summary lacks {hypothesis_id}")
        statuses = ",".join(
            dict.fromkeys(str(value) for value in cast(Sequence[object], raw.get("statuses", ())))
        )
        persistence = ",".join(
            dict.fromkeys(
                str(value) for value in cast(Sequence[object], raw.get("persistence_labels", ()))
            )
        )
        controls = ",".join(
            dict.fromkeys(
                "true" if value is True else "false"
                for value in raw.get("nonpositive_control_results", ())
                if value is not None
            )
        )
        control_text = f"; control `{controls}`" if controls else ""
        lines.append(
            f"- {hypothesis_id.upper()}: `{raw.get('result')}`; role `{raw.get('role')}`; "
            f"status `{statuses}`; persistence `{persistence}`{control_text}."
        )
    lines.extend(
        (
            f"- M3 mechanism scope: `{release_summary.get('m3_mechanism_scope')}`. ",
            "  `global` is used only when every H5-A1 through H5-A4 row is robustly persistent.",
            "- H5-B3 remains a nonpositive control; it is not a positive transport claim.",
            "- H5-B4 is a common-mode diagnostic with no uniquely faulty target.",
        )
    )
    return "\n".join(lines)


def build_presentation_files(
    registry: ReplayPublicClaimProjectionsV1,
    release_summary: Mapping[str, Any],
) -> dict[str, bytes]:
    """Render the three reviewed Markdown templates with fixed placeholders."""

    if release_summary.get("public_claim_projection_sha256") != sha256_digest(registry):
        raise ValueError("release summary does not bind the public claim registry")
    group_counts = {
        group: sum(row.projection_group == group for row in registry.projections)
        for group in (
            "persistent-panel",
            "crossovers",
            "health-transfer",
            "cluster-sensitivity",
            "descriptor-comparison",
            "resources",
        )
    }
    hypothesis_markdown = _hypothesis_markdown(release_summary)
    readme = f"""# M5 nuScenes-mini replay evidence

This package reports the complete fixed aggregate projection without ranking,
top-k selection, or omission based on sign, interval, magnitude, status, or
persistence. It covers `{group_counts["persistent-panel"]}` persistent-panel
rows, `{group_counts["crossovers"]}` crossover rows,
`{group_counts["health-transfer"]}` health-transfer rows,
`{group_counts["cluster-sensitivity"]}` cluster leave-outs,
`{group_counts["descriptor-comparison"]}` descriptor rows, and
`{group_counts["resources"]}` complete-run resource records.
The figure uses only the fixed descriptor projection; the claim ledger retains
all `{release_summary["descriptor_aggregate_count"]}` complete descriptor records.

## Package identity

{_identity_block()}

## Figures

- [Persistent panel](figures/m5-persistent-panel-summary.svg)
- [Crossovers](figures/m5-crossovers.svg)
- [Health transfer](figures/m5-health-transfer.svg)
- [Descriptor comparison](figures/m5-descriptor-comparison.svg)
- [Cluster sensitivity](figures/m5-cluster-sensitivity.svg)

## Preregistered hypotheses

{hypothesis_markdown}

## Scope and limitations

The evidence concerns matched-center estimator-output loss under the declared
proxy on the complete fixed nuScenes-mini scene set. It is not raw-sensor or
detector performance, fleet generalization, a physical tolerance, planning or
collision benefit, production readiness, or safety evidence. Undefined,
not-applicable, censored, control, and non-persistent outcomes are retained.

Aggregates are provided under CC BY-NC-SA 4.0 plus Motional Dataset Terms;
attribution is required and no endorsement is implied.
"""
    ledger_lines = [
        "# M5 claim-evidence ledger",
        "",
        "Every numeric token below comes from the reviewed public projection registry.",
        "",
        "## Package identity",
        "",
        _identity_block(),
        "",
        "## Preregistered hypotheses",
        "",
        hypothesis_markdown,
        "",
        "## Ordered projections",
        "",
    ]
    for projection in registry.projections:
        selectors = ", ".join(f"{row.field}={row.value}" for row in projection.selector_fields)
        values = ", ".join(
            f"{row.field}={_machine_token(row.value)}" for row in projection.projected_fields
        )
        ledger_lines.append(
            f"- `{projection.projection_id}` `{projection.public_claim_id}`; "
            f"source `{projection.source_member}` / `{projection.source_identifier}`; "
            f"selector {selectors}; values {values}; unit `{projection.unit}`; "
            f"behavior `{projection.status_behavior}`."
        )
    ledger_lines.extend(
        (
            "",
            "Aggregates are provided under CC BY-NC-SA 4.0 plus Motional Dataset Terms; ",
            "attribution is required and no endorsement is implied.",
        )
    )
    verification = f"""# M5 replay verification

## Package identity

{_identity_block()}

## Frozen completeness facts

- Public projection semantic SHA-256: `{sha256_digest(registry)}`
- Projection rows: `{release_summary["projection_count"]}`
- Persistent hypotheses: `{registry.persistent_hypothesis_count}`
- Health hypotheses: `{registry.health_hypothesis_count}`
- Sensitivity sources: `{registry.sensitivity_source_count}`
- Distinct log groups: `{registry.distinct_log_group_count}`
- Repeat mismatches: `{release_summary["repeat_mismatch_count"]}`
- Scientific members compared: `{release_summary["scientific_member_count"]}`

## Preregistered hypotheses

{hypothesis_markdown}

These facts establish deterministic content consistency and recorded review
bindings. They are not cryptographic proof of independent execution, dataset
authentication, or syscall-level non-access. Package validation is designed
to run offline without a dataset root.

Aggregates are provided under CC BY-NC-SA 4.0 plus Motional Dataset Terms;
attribution is required and no endorsement is implied.
"""
    files = {
        "presentation/README.md": readme.encode("utf-8"),
        "presentation/claim-evidence.md": ("\n".join(ledger_lines) + "\n").encode("utf-8"),
        "presentation/verification.md": verification.encode("utf-8"),
    }
    if tuple(files) != M5_PRESENTATION_TEMPLATE_PATHS or any(
        not payload.endswith(b"\n") or b"\r" in payload for payload in files.values()
    ):
        raise RuntimeError("M5 presentation templates have an invalid path or newline shape")
    _assert_template_placeholders(files)
    return files


def public_projection_by_id(
    registry: ReplayPublicClaimProjectionsV1,
) -> dict[str, ReplayPublicClaimProjectionV1]:
    """Return the unique projection index used by deterministic figures."""

    output = {row.projection_id: row for row in registry.projections}
    if len(output) != len(registry.projections):
        raise ValueError("public claim registry contains duplicate projection identifiers")
    return output


def figure_projections(
    registry: ReplayPublicClaimProjectionsV1,
    figure_id: FigureId,
) -> tuple[ReplayPublicClaimProjectionV1, ...]:
    """Return fixed-order registry entries assigned to one canonical figure."""

    return tuple(row for row in registry.projections if figure_id in row.figure_ids)


__all__ = [
    "M5_COMMON_MODE_PERSISTENT_SELECTORS",
    "M5_DESCRIPTOR_STATISTICS",
    "M5_DROPOUT_PERSISTENT_SELECTORS",
    "M5_HEALTH_FIGURE_SELECTORS",
    "M5_ORDINARY_PERSISTENT_SELECTORS",
    "M5_PRESENTATION_TEMPLATE_PATHS",
    "M5_REPLAY_ONLY_DESCRIPTOR_IDS",
    "M5_SHARED_DESCRIPTOR_IDS",
    "ReplayClaimEvidence",
    "build_presentation_files",
    "build_public_claim_projections",
    "build_release_summary",
    "descriptor_source_identifier",
    "figure_projections",
    "public_projection_by_id",
    "select_crossover_figure_rows",
    "select_descriptor_figure_rows",
    "select_health_figure_rows",
    "select_persistent_figure_rows",
    "select_sensitivity_figure_rows",
    "select_sensitivity_sources",
    "validate_public_claim_projections",
]
