from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_replay_curation import _PLAN, _curate

from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_REPLAY_FIGURE_DEFINITIONS,
    M5_TRACKED_AGGREGATE_TERMS,
    ReplayDescriptorAggregateV1,
)
from fusion_fault_bench.contracts.replay_v1 import M5_HEALTH_EXPERIMENT_IDS
from fusion_fault_bench.replay_figures import (
    ReplayFigureError,
    build_replay_figure_bundle,
    select_replay_figure_sources,
    validate_replay_figure_bundle,
)

_SHARED_DESCRIPTORS = {
    "sample-count": "count",
    "eligible-object-frame-count": "count",
    "eligible-track-length-q50": "frames",
    "ego-range-q50": "m",
    "ego-bearing-q50": "rad",
    "finite-difference-speed-q50": "m/s",
    "reference-time-delta-q50": "s",
}
_REPLAY_ONLY_DESCRIPTORS = {
    "support-all-annotations": "count",
    "support-roi-pass": "count",
    "support-camera-center-pass": "count",
    "support-lidar-points-positive": "count",
    "support-final-eligible": "count",
    "unique-eligible-track-count": "count",
    "zero-order-hold-velocity-fraction": "fraction",
    "camera-minus-lidar-acquisition-offset-q50": "s",
}


def _descriptor_rows() -> tuple[ReplayDescriptorAggregateV1, ...]:
    evidence = _curate()

    def row(
        descriptor_id: str,
        population: str,
        statistic: str,
        unit: str,
        value: float,
    ) -> ReplayDescriptorAggregateV1:
        return ReplayDescriptorAggregateV1(
            schema="ffb.replay-descriptor-aggregate/v1",
            run_id=evidence.run.run_id,
            replay_intent_sha256=evidence.profile_summary.replay_intent_sha256,
            replay_identity_set_sha256=evidence.profile_summary.replay_identity_set_sha256,
            descriptor_id=descriptor_id,
            population=population,  # type: ignore[arg-type]
            population_count=10 if population == "nuscenes-mini-replay" else 200,
            statistic=statistic,  # type: ignore[arg-type]
            status="ok",
            value=value,
            unit=unit,  # type: ignore[arg-type]
            tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        )

    rows: list[ReplayDescriptorAggregateV1] = []
    for descriptor_ordinal, (descriptor_id, unit) in enumerate(_SHARED_DESCRIPTORS.items()):
        for statistic_ordinal, statistic in enumerate(("minimum", "median", "maximum")):
            for population_ordinal, population in enumerate(
                ("nuscenes-mini-replay", "m3-main-test-comparator")
            ):
                rows.append(
                    row(
                        descriptor_id,
                        population,
                        statistic,
                        unit,
                        float(1 + descriptor_ordinal + statistic_ordinal + population_ordinal),
                    )
                )
    for descriptor_ordinal, (descriptor_id, unit) in enumerate(_REPLAY_ONLY_DESCRIPTORS.items()):
        for statistic_ordinal, statistic in enumerate(("minimum", "median", "maximum")):
            value = (
                0.25 + statistic_ordinal * 0.25
                if unit == "fraction"
                else float(1 + descriptor_ordinal + statistic_ordinal)
            )
            rows.append(
                row(
                    descriptor_id,
                    "nuscenes-mini-replay",
                    statistic,
                    unit,
                    value,
                )
            )
    rows.append(
        row(
            "distinct-log-group-count",
            "nuscenes-mini-replay",
            "count",
            "count",
            2.0,
        )
    )
    assert len(rows) == 67
    return tuple(rows)


def _evidence():
    return replace(_curate(), descriptor_aggregates=_descriptor_rows())


def test_exact_five_figure_sources_and_bindings_are_complete() -> None:
    evidence = _evidence()
    bundle = build_replay_figure_bundle(evidence, plan=_PLAN)

    assert (
        tuple(
            (figure.figure_id, figure.figure_kind, figure.relative_svg_path)
            for figure in bundle.figures
        )
        == M5_REPLAY_FIGURE_DEFINITIONS
    )
    assert tuple(len(figure.sources) for figure in bundle.figures) == (
        100,
        10,
        43,
        67,
        338,
    )
    assert len(bundle.bindings) == 558
    assert len(bundle.files()) == 10
    assert (
        tuple(
            dict.fromkeys(
                binding.identity.experiment_id
                for binding in bundle.bindings
                if binding.figure_id == "m5-health-transfer" and binding.identity is not None
            )
        )
        == M5_HEALTH_EXPERIMENT_IDS
    )
    descriptor_bindings = tuple(
        binding for binding in bundle.bindings if binding.figure_id == "m5-descriptor-comparison"
    )
    assert all(
        binding.identity is None and binding.replay_identity_sha256 is None
        for binding in descriptor_bindings
    )
    validate_replay_figure_bundle(bundle, evidence=evidence, plan=_PLAN)


def test_selection_and_rendering_are_deterministic_under_source_reordering() -> None:
    evidence = _evidence()
    reversed_evidence = replace(
        evidence,
        descriptor_aggregates=tuple(reversed(evidence.descriptor_aggregates)),
        persistent_aggregates=tuple(reversed(evidence.persistent_aggregates)),
        persistent_crossovers=tuple(reversed(evidence.persistent_crossovers)),
        health_aggregates=tuple(reversed(evidence.health_aggregates)),
        cluster_sensitivity=tuple(reversed(evidence.cluster_sensitivity)),
    )

    assert build_replay_figure_bundle(evidence, plan=_PLAN) == build_replay_figure_bundle(
        reversed_evidence,
        plan=_PLAN,
    )


def test_specs_are_canonical_and_svgs_are_static_public_aggregate_views() -> None:
    bundle = build_replay_figure_bundle(_evidence(), plan=_PLAN)

    for figure in bundle.figures:
        spec = json.loads(figure.spec_bytes)
        assert spec["figure_id"] == figure.figure_id
        assert len(spec["marks"]) == len(figure.sources)
        assert tuple(mark["mark_ordinal"] for mark in spec["marks"]) == tuple(
            range(len(figure.sources))
        )
        assert b"<script" not in figure.svg_bytes
        assert b"href=" not in figure.svg_bytes
        assert b"/Users/" not in figure.spec_bytes + figure.svg_bytes
        assert b"NUSCENES_ROOT" not in figure.spec_bytes + figure.svg_bytes
        assert b"matched-center-estimator-output-proxy-only" in figure.spec_bytes
        assert b"no endorsement" in figure.svg_bytes


def test_missing_predeclared_source_and_tampering_fail_closed() -> None:
    evidence = _evidence()
    missing = replace(
        evidence,
        descriptor_aggregates=evidence.descriptor_aggregates[:-1],
    )
    with pytest.raises(ReplayFigureError, match="missing a predeclared source"):
        select_replay_figure_sources(missing, plan=_PLAN)

    bundle = build_replay_figure_bundle(evidence, plan=_PLAN)
    first = bundle.figures[0]
    tampered_figure = replace(first, svg_bytes=first.svg_bytes + b" ")
    tampered_bundle = replace(bundle, figures=(tampered_figure, *bundle.figures[1:]))
    with pytest.raises(ReplayFigureError, match="deterministic regeneration"):
        validate_replay_figure_bundle(tampered_bundle, evidence=evidence, plan=_PLAN)

    tampered_binding = bundle.bindings[0].model_copy(update={"source_record_sha256": "f" * 64})
    tampered_binding_bundle = replace(
        bundle,
        bindings=(tampered_binding, *bundle.bindings[1:]),
    )
    with pytest.raises(ReplayFigureError, match="deterministic regeneration"):
        validate_replay_figure_bundle(
            tampered_binding_bundle,
            evidence=evidence,
            plan=_PLAN,
        )
