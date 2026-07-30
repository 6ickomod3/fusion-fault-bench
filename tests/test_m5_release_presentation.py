from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from functools import cache

import pytest
from test_replay_artifacts import _request

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import ReplayDescriptorAggregateV1
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_PRESENTATION_PLACEHOLDERS,
    M5_SOFTWARE_VERIFICATION_CATEGORIES,
    ReplaySoftwareVerificationCheckV1,
    ReplaySoftwareVerificationV1,
)
from fusion_fault_bench.replay_claims import (
    M5_COMMON_MODE_PERSISTENT_SELECTORS,
    M5_DESCRIPTOR_STATISTICS,
    M5_DROPOUT_PERSISTENT_SELECTORS,
    M5_HEALTH_FIGURE_SELECTORS,
    M5_ORDINARY_PERSISTENT_SELECTORS,
    M5_REPLAY_ONLY_DESCRIPTOR_IDS,
    M5_SHARED_DESCRIPTOR_IDS,
    ReplayClaimEvidence,
    build_presentation_files,
    build_public_claim_projections,
    build_release_summary,
    select_sensitivity_sources,
    validate_public_claim_projections,
)
from fusion_fault_bench.replay_figures import (
    M5_FIGURE_ORDER,
    ReplayFigureBundle,
    build_figure_bundle,
    canonical_figure_spec_files,
    render_figure_svg,
    validate_figure_bundle,
)
from fusion_fault_bench.replay_release_validation import (
    M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK,
    M5_SOFTWARE_VERIFICATION_CHECK_IDS,
)

_DESCRIPTOR_UNITS = {
    "sample-count": "count",
    "eligible-object-frame-count": "count",
    "eligible-track-length-q50": "frames",
    "ego-range-q50": "m",
    "ego-bearing-q50": "rad",
    "finite-difference-speed-q50": "m/s",
    "reference-time-delta-q50": "s",
    "support-all-annotations": "count",
    "support-roi-pass": "count",
    "support-camera-center-pass": "count",
    "support-lidar-points-positive": "count",
    "support-final-eligible": "count",
    "unique-eligible-track-count": "count",
    "camera-minus-lidar-acquisition-offset-q50": "s",
    "zero-order-hold-velocity-fraction": "fraction",
    "distinct-log-group-count": "count",
}


def _descriptor(
    base: ReplayDescriptorAggregateV1,
    *,
    descriptor_id: str,
    population: str,
    statistic: str,
    value: float,
) -> ReplayDescriptorAggregateV1:
    return base.model_copy(
        update={
            "descriptor_id": descriptor_id,
            "population": population,
            "population_count": 10 if population == "nuscenes-mini-replay" else 200,
            "statistic": statistic,
            "category_label": None,
            "status": "ok",
            "value": value,
            "unit": _DESCRIPTOR_UNITS[descriptor_id],
        }
    )


def _software_verification() -> ReplaySoftwareVerificationV1:
    revision = "a" * 40
    checks = tuple(
        ReplaySoftwareVerificationCheckV1(
            check_id=check_id,
            category=category,
            command=("release-check", check_id),
            required_test_ids=M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[check_id],
            exit_status=0,
            output_sha256=hashlib.sha256(check_id.encode()).hexdigest(),
            output_normalization=("stable-command-output-with-runtime-paths-and-durations-removed"),
        )
        for check_id, category in zip(
            M5_SOFTWARE_VERIFICATION_CHECK_IDS,
            M5_SOFTWARE_VERIFICATION_CATEGORIES,
            strict=True,
        )
    )
    return ReplaySoftwareVerificationV1(
        schema="ffb.m5-software-verification/v1",
        release_id="m5-nuscenes-replay-v0.1.0",
        scientific_git_revision=revision,
        lockfile_sha256="b" * 64,
        package_version="0.1.0",
        implementation_snapshot_sha256="c" * 64,
        tooling_revision=revision,
        checks=checks,
    )


def _nested_dropout_rows(rows: tuple[object, ...]) -> tuple[object, ...]:
    fixed_values = (1.0, 0.9, 0.75, 0.5, 0.25, 0.0)
    output = []
    for row in rows:
        selector = row.condition_selector
        method_id = row.method_id
        metric_id = row.metric_id
        if selector in M5_DROPOUT_PERSISTENT_SELECTORS and metric_id == "coverage":
            ordinal = M5_DROPOUT_PERSISTENT_SELECTORS.index(selector)
            value = fixed_values[ordinal] if method_id == "fixed-fusion" else 1.0
            if method_id in {"fixed-fusion", "fault-target-drop-policy", "lidar-only"}:
                row = row.model_copy(
                    update={
                        "status": "ok",
                        "estimate": value,
                        "interval_lower": value,
                        "interval_upper": value,
                        "defined_bootstrap_replicates": 2000,
                    }
                )
        output.append(row)
    return tuple(output)


@cache
def _evidence() -> ReplayClaimEvidence:
    request = _request()
    base = request.descriptor_aggregates[0]
    rows: list[ReplayDescriptorAggregateV1] = []
    ordinal = 0
    for descriptor_id in M5_SHARED_DESCRIPTOR_IDS:
        for population in ("nuscenes-mini-replay", "m3-main-test-comparator"):
            for statistic in M5_DESCRIPTOR_STATISTICS:
                ordinal += 1
                rows.append(
                    _descriptor(
                        base,
                        descriptor_id=descriptor_id,
                        population=population,
                        statistic=statistic,
                        value=float(ordinal),
                    )
                )
    for descriptor_id in M5_REPLAY_ONLY_DESCRIPTOR_IDS:
        for statistic in M5_DESCRIPTOR_STATISTICS:
            ordinal += 1
            rows.append(
                _descriptor(
                    base,
                    descriptor_id=descriptor_id,
                    population="nuscenes-mini-replay",
                    statistic=statistic,
                    value=float(ordinal),
                )
            )
    rows.append(
        _descriptor(
            base,
            descriptor_id="distinct-log-group-count",
            population="nuscenes-mini-replay",
            statistic="count",
            value=float(request.profile_summary.distinct_log_group_count),
        )
    )
    rows.append(
        base.model_copy(
            update={
                "descriptor_id": "category-frequency",
                "population": "nuscenes-mini-replay",
                "population_count": 10,
                "statistic": "count",
                "category_label": "public-category",
                "status": "ok",
                "value": 3.0,
                "unit": "count",
            }
        )
    )
    return ReplayClaimEvidence(
        profile_summary=request.profile_summary,
        descriptor_aggregates=tuple(rows),
        persistent_aggregates=_nested_dropout_rows(request.persistent_aggregates),
        persistent_crossovers=request.persistent_crossovers,
        health_aggregates=request.health_aggregates,
        cluster_sensitivity=request.cluster_sensitivity,
        repeat_verification=request.repeat_verification,
        software_verification=_software_verification(),
    )


@cache
def _registry_and_bundle() -> tuple[object, ReplayFigureBundle]:
    registry = build_public_claim_projections(_evidence())
    return registry, build_figure_bundle(registry, _evidence())


def test_fixed_selector_registry_has_exact_outcome_independent_counts() -> None:
    evidence = _evidence()
    registry = build_public_claim_projections(evidence)
    counts = Counter(row.projection_group for row in registry.projections)

    assert len(M5_ORDINARY_PERSISTENT_SELECTORS) == 54
    assert len(M5_DROPOUT_PERSISTENT_SELECTORS) == 6
    assert len(M5_COMMON_MODE_PERSISTENT_SELECTORS) == 11
    assert len(M5_HEALTH_FIGURE_SELECTORS) == 43
    assert counts["persistent-panel"] == 100
    assert counts["crossovers"] == 10
    assert counts["health-transfer"] == 43
    assert counts["descriptor-comparison"] == 67
    assert counts["dropout-nesting"] == 1
    assert counts["cluster-sensitivity"] == 26 * (
        10 + evidence.profile_summary.distinct_log_group_count
    )
    assert counts["resources"] == 2
    descriptor_rows = tuple(
        row for row in registry.projections if row.source_kind == "descriptor-aggregate"
    )
    assert len(descriptor_rows) == len(evidence.descriptor_aggregates)
    assert sum(bool(row.figure_ids) for row in descriptor_rows) == 67
    assert any(
        row.public_claim_id == "m5-descriptor-ledger" and not row.figure_ids
        for row in descriptor_rows
    )
    nesting = tuple(
        row for row in registry.projections if row.projection_group == "dropout-nesting"
    )
    assert len(nesting) == 1
    assert nesting[0].source_kind == "software-verification"
    assert dict((field.field, field.value) for field in nesting[0].projected_fields) == {
        "all_coverage_sources_defined": True,
        "fixed_fusion_coverage_nested": True,
        "target_drop_matches_lidar_only": True,
        "dropout_nesting_passed": True,
        "coverage_source_count": 18,
    }
    assert sum(row.hypothesis_id is not None for row in registry.projections) == 44
    assert (
        len(select_sensitivity_sources(evidence.persistent_aggregates, evidence.health_aggregates))
        == 26
    )
    validate_public_claim_projections(registry, evidence)

    tampered_projection = registry.projections[0].model_copy(
        update={"public_claim_id": "tampered-claim"}
    )
    tampered = registry.model_copy(
        update={"projections": (tampered_projection, *registry.projections[1:])}
    )
    with pytest.raises(ValueError, match="does not regenerate"):
        validate_public_claim_projections(tampered, evidence)


def test_release_summary_and_three_templates_are_deterministic() -> None:
    evidence = _evidence()
    registry = build_public_claim_projections(evidence)
    summary = build_release_summary(registry, evidence)
    first = build_presentation_files(registry, summary)
    second = build_presentation_files(registry, summary)

    assert first == second
    assert tuple(first) == (
        "presentation/README.md",
        "presentation/claim-evidence.md",
        "presentation/verification.md",
    )
    assert summary["machine_artifact_byte_length"] is None
    assert summary["repeat_mismatch_count"] == 0
    assert summary["descriptor_aggregate_count"] == len(evidence.descriptor_aggregates)
    assert summary["m3_mechanism_scope"] == "global"
    hypothesis_results = summary["hypothesis_results"]
    assert tuple(hypothesis_results) == (
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
    assert hypothesis_results["h5-b3"]["role"] == "nonpositive-control"
    assert hypothesis_results["h5-b4"]["role"] == "common-mode-diagnostic-no-uniquely-faulty-target"
    for payload in first.values():
        text = payload.decode("utf-8")
        assert payload.endswith(b"\n")
        assert "\r" not in text
        for placeholder in M5_PRESENTATION_PLACEHOLDERS:
            assert text.count(placeholder) == 1
        assert "CC BY-NC-SA 4.0 plus Motional Dataset Terms" in text
        assert "no endorsement" in text
        for hypothesis_id in hypothesis_results:
            assert hypothesis_id.upper() in text
        assert "H5-B3 remains a nonpositive control" in text
        assert "H5-B4 is a common-mode diagnostic with no uniquely faulty target" in text
        assert "not-applicable" in text
    assert b"category-frequency" in first["presentation/claim-evidence.md"]


def test_undefined_hypothesis_state_is_retained_in_summary_and_every_template() -> None:
    evidence = _evidence()
    persistent = list(evidence.persistent_aggregates)
    index = next(
        ordinal
        for ordinal, row in enumerate(persistent)
        if row.hypothesis_id == "h5-a5" and row.metric_id == "conditional-matched-center-mse"
    )
    persistent[index] = persistent[index].model_copy(
        update={
            "status": "undefined",
            "estimate": None,
            "interval_lower": None,
            "interval_upper": None,
            "defined_bootstrap_replicates": 0,
            "persistence_label": "undefined",
        }
    )
    changed = replace(evidence, persistent_aggregates=tuple(persistent))
    registry = build_public_claim_projections(changed)
    summary = build_release_summary(registry, changed)
    assert summary["hypothesis_results"]["h5-a5"]["result"] == "supported"
    for payload in build_presentation_files(registry, summary).values():
        assert b"H5-A5" in payload
        assert b"undefined" in payload


def test_h5_a5_reports_a_retained_contradiction_instead_of_only_statuses() -> None:
    evidence = _evidence()
    persistent = list(evidence.persistent_aggregates)
    index = next(
        ordinal
        for ordinal, row in enumerate(persistent)
        if row.hypothesis_id == "h5-a5"
        and row.method_id == "fixed-fusion"
        and row.metric_id == "coverage"
    )
    persistent[index] = persistent[index].model_copy(
        update={"estimate": 0.1, "interval_lower": 0.1, "interval_upper": 0.1}
    )
    changed = replace(evidence, persistent_aggregates=tuple(persistent))
    registry = build_public_claim_projections(changed)
    summary = build_release_summary(registry, changed)

    assert summary["hypothesis_results"]["h5-a5"]["result"] == "not-supported"
    assert all(
        b"H5-A5: `not-supported`" in payload
        for payload in build_presentation_files(registry, summary).values()
    )


def test_h5_a6_reports_support_and_contradiction_from_frozen_comparisons() -> None:
    evidence = _evidence()
    persistent = list(evidence.persistent_aggregates)
    for ordinal, row in enumerate(persistent):
        if not row.condition_selector.startswith("replay-common-mode-x:"):
            continue
        if row.method_id == "fixed-fusion" and row.metric_id == "matched-center-mse":
            estimate = 1.0 if row.condition_selector.endswith(":0") else 2.0
        elif (
            row.method_id == "camera-lidar-pair"
            and row.metric_id == "camera-lidar-disagreement-mse"
        ):
            estimate = 0.5
        else:
            continue
        persistent[ordinal] = row.model_copy(
            update={
                "estimate": estimate,
                "interval_lower": estimate - 0.1,
                "interval_upper": estimate + 0.1,
            }
        )
    supported_evidence = replace(evidence, persistent_aggregates=tuple(persistent))
    registry = build_public_claim_projections(supported_evidence)
    supported = build_release_summary(registry, supported_evidence)
    assert supported["hypothesis_results"]["h5-a6"]["result"] == "supported-diagnostic"

    endpoint = next(
        ordinal
        for ordinal, row in enumerate(persistent)
        if row.hypothesis_id == "h5-a6" and row.method_id == "fixed-fusion"
    )
    persistent[endpoint] = persistent[endpoint].model_copy(
        update={"estimate": 0.5, "interval_lower": 0.4, "interval_upper": 0.6}
    )
    contradicted_evidence = replace(evidence, persistent_aggregates=tuple(persistent))
    registry = build_public_claim_projections(contradicted_evidence)
    contradicted = build_release_summary(registry, contradicted_evidence)
    assert contradicted["hypothesis_results"]["h5-a6"]["result"] == ("not-supported-diagnostic")


def test_dropout_nesting_rejects_relation_failure_and_missing_named_authority() -> None:
    evidence = _evidence()
    persistent = list(evidence.persistent_aggregates)
    index = next(
        ordinal
        for ordinal, row in enumerate(persistent)
        if row.condition_selector == "replay-camera-dropout:0.25"
        and row.method_id == "fixed-fusion"
        and row.metric_id == "coverage"
    )
    persistent[index] = persistent[index].model_copy(
        update={"estimate": 0.95, "interval_lower": 0.95, "interval_upper": 0.95}
    )
    with pytest.raises(ValueError, match="nesting relation failed"):
        build_public_claim_projections(replace(evidence, persistent_aggregates=tuple(persistent)))

    verification = evidence.software_verification
    assert verification is not None
    checks = list(verification.checks)
    check_index = next(
        ordinal
        for ordinal, check in enumerate(checks)
        if "dropout-nesting-derivation" in check.required_test_ids
    )
    checks[check_index] = checks[check_index].model_copy(
        update={
            "required_test_ids": tuple(
                test_id
                for test_id in checks[check_index].required_test_ids
                if test_id != "dropout-nesting-derivation"
            )
        }
    )
    missing_authority = verification.model_copy(update={"checks": tuple(checks)})
    with pytest.raises(ValueError, match="missing a required named test"):
        build_public_claim_projections(replace(evidence, software_verification=missing_authority))


def test_crossover_rendering_is_numeric_only_for_finite_interval_bounds() -> None:
    evidence = _evidence()
    first = evidence.persistent_crossovers[0].model_copy(
        update={
            "status": "observed",
            "point_curve_crossed": True,
            "point_estimate": 2.0,
            "interval_lower": 1.0,
            "interval_upper": 3.0,
            "censoring": "none",
            "bootstrap_crossing_count": 2000,
            "bootstrap_crossing_fraction": 1.0,
        }
    )
    registry = build_public_claim_projections(
        replace(
            evidence,
            persistent_crossovers=(first, *evidence.persistent_crossovers[1:]),
        )
    )
    crossovers = tuple(row for row in registry.projections if row.projection_group == "crossovers")
    finite_upper = next(
        field for field in crossovers[0].projected_fields if field.field == "interval_upper"
    )
    censored_upper = next(
        field for field in crossovers[1].projected_fields if field.field == "interval_upper"
    )
    assert finite_upper.rendering == ".6g"
    assert censored_upper.rendering == "literal-status"


def test_not_observed_crossover_does_not_invent_an_infinite_point() -> None:
    _registry, bundle = _registry_and_bundle()
    svg = bundle.svgs["figures/m5-crossovers.svg"]

    assert b"point undefined" in svg
    assert b"point positive-infinity" not in svg


def test_global_mechanism_wording_drops_after_any_a1_a4_failure() -> None:
    evidence = _evidence()
    persistent = list(evidence.persistent_aggregates)
    changed = next(
        row for row in persistent if row.hypothesis_id in {"h5-a1", "h5-a2", "h5-a3", "h5-a4"}
    )
    changed_index = persistent.index(changed)
    persistent[changed_index] = changed.model_copy(update={"persistence_label": "non-persistent"})
    changed_digest = sha256_digest(persistent[changed_index])
    sensitivity = tuple(
        row.model_copy(update={"source_record_sha256": changed_digest})
        if row.source_result_id == changed.result_id
        else row
        for row in evidence.cluster_sensitivity
    )
    changed_evidence = replace(
        evidence,
        persistent_aggregates=tuple(persistent),
        cluster_sensitivity=sensitivity,
    )
    registry = build_public_claim_projections(changed_evidence)
    summary = build_release_summary(registry, changed_evidence)
    assert summary["m3_mechanism_scope"] != "global"


def test_five_specs_svgs_and_every_mark_binding_regenerate() -> None:
    registry, bundle = _registry_and_bundle()
    registry = build_public_claim_projections(_evidence())
    specs = bundle.specs
    spec_files = canonical_figure_spec_files(specs)

    assert tuple(spec.figure_id for spec in specs) == M5_FIGURE_ORDER
    assert [len(spec.marks) for spec in specs] == [100, 10, 43, 67, 364]
    assert len(spec_files) == 5
    assert len(bundle.svgs) == 5
    assert len(bundle.bindings) == 584
    for spec in specs:
        spec_path = spec.figure_file.removesuffix(".svg") + ".spec.json"
        assert spec_files[spec_path] == canonical_json_bytes(spec)
        svg = bundle.svgs[spec.figure_file]
        assert svg == render_figure_svg(spec, registry)
        assert svg.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        assert b"<script" not in svg
        assert b"/Users/" not in svg
        assert b"2026-" not in svg
        assert hashlib.sha256(svg).hexdigest()
    persistent_svg = bundle.svgs["figures/m5-persistent-panel-summary.svg"]
    assert persistent_svg.count(b'class="persistent-row"') == 100
    assert b'class="interval"' in persistent_svg
    assert b'class="point"' in persistent_svg
    assert b'class="persistence"' in persistent_svg
    assert b'class="sign-partition"' in persistent_svg
    assert b"signs +/0/-/?" in persistent_svg

    crossover_svg = bundle.svgs["figures/m5-crossovers.svg"]
    for unit in (b"facet: m", b"facet: rad", b"facet: s", b"facet: std-scale"):
        assert unit in crossover_svg
    assert crossover_svg.count(b'class="crossover-facet"') == 4
    assert b'class="right-censored"' in crossover_svg
    assert b"positive-infinity" in crossover_svg

    health_svg = bundle.svgs["figures/m5-health-transfer.svg"]
    assert health_svg.count(b'class="health-row"') == 43
    for hypothesis_id in (b"H5-B1", b"H5-B2", b"H5-B3", b"H5-B4"):
        assert hypothesis_id in health_svg
    assert b"nonpositive-control" in health_svg
    assert b"no uniquely faulty target" in health_svg

    descriptor_svg = bundle.svgs["figures/m5-descriptor-comparison.svg"]
    assert descriptor_svg.count(b'class="descriptor-point"') == 67
    assert b"Paired replay / M3" in descriptor_svg
    assert b"M3 comparator unavailable (not modeled)" in descriptor_svg

    sensitivity_svg = bundle.svgs["figures/m5-cluster-sensitivity.svg"]
    assert sensitivity_svg.count(b'class="sensitivity-cell"') == 364
    assert b"scene-ordinal:00" in sensitivity_svg
    assert b"log-group:00" in sensitivity_svg
    assert b"X = undefined leave-out retained" in sensitivity_svg
    validate_figure_bundle(specs, bundle.svgs, bundle.bindings, _evidence())


def test_figure_validator_rejects_svg_and_binding_tampering() -> None:
    _, bundle = _registry_and_bundle()
    first_path = bundle.specs[0].figure_file
    bad_svgs = dict(bundle.svgs)
    bad_svgs[first_path] = bad_svgs[first_path].replace(b"#F8FAFC", b"#FFFFFF", 1)
    with pytest.raises(ValueError, match="rendered SVGs"):
        validate_figure_bundle(bundle.specs, bad_svgs, bundle.bindings, _evidence())

    bad_binding = bundle.bindings[0].model_copy(update={"source_record_sha256": "f" * 64})
    with pytest.raises(ValueError, match="source bindings"):
        validate_figure_bundle(
            bundle.specs,
            bundle.svgs,
            (bad_binding, *bundle.bindings[1:]),
            _evidence(),
        )
