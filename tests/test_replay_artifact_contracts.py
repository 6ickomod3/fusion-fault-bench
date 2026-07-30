from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256,
    M5_HEALTH_FIT_RUN_SHA256,
    M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256,
    M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256,
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_FIGURE_DEFINITIONS,
    M5_REPLAY_IDENTITY_SET,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    M5_TRACKED_AGGREGATE_TERMS,
    REPLAY_CURATED_ARTIFACT_CONTRACT,
    REPLAY_INDEXED_PATHS,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayExecutionResourceEvidenceV1,
    ReplayFigureRecordV1,
    ReplayFigureSourceBindingV1,
    ReplayHealthAggregateV1,
    ReplayIdentitySetV1,
    ReplayPayloadFileEntryV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    ReplayProfileSummaryV1,
    ReplayReleaseIndexV1,
    ReplayRepeatVerificationV1,
    ReplaySourceMemberCommitmentV1,
    ReplaySuccessV1,
    ReplayValidationCheckV1,
    ReplayValidationV1,
    replay_descriptor_source_id,
    replay_identity_set_digest,
    replay_release_index_json_schema,
    replay_resource_evidence_sha256,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
    expected_replay_identities,
    replay_experiment_identity_sha256,
)

_RUN_ID = f"run:{'1' * 64}"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _global() -> dict[str, object]:
    return {
        "run_id": _RUN_ID,
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
    }


def _identity(index: int) -> dict[str, object]:
    identity = expected_replay_identities()[index]
    return {
        **_global(),
        "identity": identity,
        "replay_identity_sha256": replay_experiment_identity_sha256(identity),
    }


def _resource(
    run_label: str,
    *,
    elapsed_seconds: float,
    peak_rss_bytes: int,
) -> ReplayExecutionResourceEvidenceV1:
    return ReplayExecutionResourceEvidenceV1(
        schema="ffb.replay-execution-resource-evidence/v1",
        **_global(),
        run_label=run_label,  # type: ignore[arg-type]
        local_artifact_sha256=_digest("local-artifact"),
        local_run_sha256=_digest(f"{run_label}-run"),
        environment_sha256=_digest("environment"),
        logical_command_sha256=_digest(f"{run_label}-command"),
        persisted_internal_elapsed_seconds=elapsed_seconds / 2.0,
        persisted_internal_peak_rss_bytes=peak_rss_bytes - 1,
        persisted_internal_measurement_scope=(
            "metadata-through-canonical-scientific-members-before-publication"
        ),
        tool_path="/usr/bin/time",
        tool_options=("-l",),
        parser_contract="ffb.darwin-time-l-strict/v1",
        raw_log_sha256=_digest(f"{run_label}-raw-log"),
        raw_log_byte_length=777,
        elapsed_seconds=elapsed_seconds,
        peak_rss_bytes=peak_rss_bytes,
        exit_status=0,
        scientific_replay_worker_count=1,
        cpu_process_scope="one-scientific-replay-worker-no-benchmark-multiprocessing",
        helper_process_policy=("sequential-provenance-and-resource-measurement-helpers-only"),
        accelerator_requested=False,
        wall_time_cap_seconds=1800.0,
        peak_rss_cap_bytes=1_073_741_824,
        wall_time_within_cap=True,
        peak_rss_within_cap=True,
        measurement_scope=(
            "operator-recorded-darwin-time-l-for-complete-replay-cli-lifetime;"
            "self-reported-not-independent-attestation"
        ),
    )


def _resources() -> tuple[
    ReplayExecutionResourceEvidenceV1,
    ReplayExecutionResourceEvidenceV1,
]:
    return (
        _resource("primary", elapsed_seconds=0.75, peak_rss_bytes=99),
        _resource("repeat", elapsed_seconds=1.0, peak_rss_bytes=100),
    )


def _aggregate(**updates: object) -> ReplayPersistentAggregateV1:
    value: dict[str, object] = {
        "schema": "ffb.replay-persistent-aggregate/v1",
        **_identity(0),
        "result_id": "result-00",
        "condition_id": "replay-lidar-y-bias",
        "condition_selector": "replay-lidar-y-bias:+4",
        "method_id": "fixed-fusion",
        "metric_id": "matched-center-mse",
        "window": "full",
        "inference_role": "primary-directional",
        "unit": "m^2",
        "status": "ok",
        "estimate": 1.0,
        "interval_lower": 0.5,
        "interval_upper": 1.5,
        "bootstrap_replicates": 2000,
        "defined_bootstrap_replicates": 2000,
        "confidence_level": 0.95,
        "interval_method": "paired-scene-percentile-pointwise",
        "aggregation": "equal-scene-mean",
        "scene_count": 10,
        "positive_scene_count": 8,
        "zero_scene_count": 0,
        "negative_scene_count": 2,
        "undefined_scene_count": 0,
        "expected_direction": "positive",
        "persistence_label": "robustly-persistent",
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
    }
    value.update(updates)
    return ReplayPersistentAggregateV1.model_validate(value)


def _descriptor(**updates: object) -> ReplayDescriptorAggregateV1:
    value: dict[str, object] = {
        "schema": "ffb.replay-descriptor-aggregate/v1",
        **_global(),
        "descriptor_id": "sample-count",
        "population": "nuscenes-mini-replay",
        "population_count": 10,
        "statistic": "minimum",
        "category_label": None,
        "status": "ok",
        "value": 16.0,
        "unit": "count",
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
    }
    value.update(updates)
    return ReplayDescriptorAggregateV1.model_validate(value)


def _health_aggregate(
    *,
    identity_index: int = 8,
    **updates: object,
) -> ReplayHealthAggregateV1:
    identity = expected_replay_identities()[identity_index]
    value: dict[str, object] = {
        "schema": "ffb.replay-health-aggregate/v1",
        **_identity(identity_index),
        "result_id": "health-result-00",
        "condition_id": identity.experiment_id,
        "condition_selector": f"{identity.experiment_id}:+3",
        "method_id": "combined-health-gate",
        "metric_id": "policy-gain-vs-fixed",
        "window": "event",
        "inference_role": "diagnostic",
        "unit": "m^2",
        "status": "ok",
        "estimate": 1.0,
        "interval_lower": 0.5,
        "interval_upper": 1.5,
        "bootstrap_replicates": 2000,
        "defined_bootstrap_replicates": 2000,
        "confidence_level": 0.95,
        "interval_method": "paired-scene-percentile-pointwise",
        "aggregation": "equal-scene-mean",
        "scene_count": 10,
        "positive_scene_count": None,
        "zero_scene_count": None,
        "negative_scene_count": None,
        "undefined_scene_count": None,
        "expected_direction": "none",
        "persistence_label": "not-applicable",
        "nonpositive_control_supported": None,
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
        "applicability_basis": "applicable",
        "recovery_support_compatible_scene_count": None,
    }
    value.update(updates)
    return ReplayHealthAggregateV1.model_validate(value)


def test_identity_set_is_the_exact_frozen_22_record_order() -> None:
    assert M5_REPLAY_IDENTITY_SET.identities == expected_replay_identities()
    assert len(M5_REPLAY_IDENTITY_SET.identities) == 22
    assert sha256_digest(M5_REPLAY_IDENTITY_SET) == M5_REPLAY_IDENTITY_SET_SHA256
    assert replay_identity_set_digest() == M5_REPLAY_IDENTITY_SET_SHA256
    assert replay_release_index_json_schema()["type"] == "object"

    for update in (
        {"replay_intent_sha256": _digest("wrong-intent")},
        {"identities": tuple(reversed(expected_replay_identities()))},
    ):
        with pytest.raises(ValidationError):
            ReplayIdentitySetV1.model_validate(
                {
                    "schema": "ffb.replay-identity-set/v1",
                    "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
                    "identities": expected_replay_identities(),
                    **update,
                }
            )


def test_global_identity_and_public_label_bindings_reject_substitution() -> None:
    for update in (
        {"replay_intent_sha256": _digest("wrong-intent")},
        {"replay_identity_set_sha256": _digest("wrong-set")},
        {"replay_identity_sha256": _digest("wrong-identity")},
        {"result_id": "sample_token"},
        {"condition_id": "annotation_token"},
        {"method_id": "log_token"},
        {"metric_id": "filepath"},
    ):
        with pytest.raises(ValidationError):
            _aggregate(**update)

    for private_label in (
        "nuscenes:scene-0061",
        "sample_token",
        "annotation_token",
        "instance_token",
        "calibrated_sensor_token",
        "ego_pose_token",
        "log_token",
        "filename",
        "filepath",
        "private/path",
        r"private\path",
    ):
        with pytest.raises(ValidationError):
            _descriptor(category_label=private_label)


def test_descriptor_status_population_and_privacy_contract() -> None:
    assert _descriptor().value == 16.0
    assert (
        _descriptor(
            population="m3-main-test-comparator",
            population_count=200,
            statistic="not-modeled",
            status="not-applicable",
            value=None,
            unit="unitless",
        ).status
        == "not-applicable"
    )
    for update in (
        {"population_count": 200},
        {"status": "ok", "value": None},
        {"status": "not-applicable", "value": 1.0, "statistic": "not-modeled"},
        {"status": "not-applicable", "value": None, "statistic": "minimum"},
        {"descriptor_id": "sample_token"},
    ):
        with pytest.raises(ValidationError):
            _descriptor(**update)


def test_descriptor_figure_source_id_discriminates_the_complete_coordinate() -> None:
    base = _descriptor(descriptor_id="shared-descriptor")
    variants = (
        base,
        _descriptor(
            descriptor_id="shared-descriptor",
            population="m3-main-test-comparator",
            population_count=200,
        ),
        _descriptor(descriptor_id="shared-descriptor", statistic="median"),
        _descriptor(
            descriptor_id="shared-descriptor",
            statistic="count",
            category_label="category-a",
        ),
    )
    source_ids = tuple(replay_descriptor_source_id(record) for record in variants)
    assert len(set(source_ids)) == len(variants)
    assert replay_descriptor_source_id(base) == replay_descriptor_source_id(base)
    assert all(source_id.startswith("replay-descriptor-") for source_id in source_ids)


def test_profile_summary_requires_frozen_sources_and_cpu_only_caps() -> None:
    profile = ReplayProfileSummaryV1(
        schema="ffb.replay-profile-summary/v1",
        **_global(),
        release_id=M5_REPLAY_RELEASE_ID,
        replay_intent_byte_sha256=M5_REPLAY_INTENT_BYTE_SHA256,
        persistent_source_matrix_sha256=M5_PERSISTENT_MATRIX_SHA256,
        health_fit_artifact_sha256=M5_HEALTH_FIT_SHA256,
        health_fit_run_sha256=M5_HEALTH_FIT_RUN_SHA256,
        dataset_profile="official-nuscenes-v1.0-mini",
        adapter_profile="nuscenes-mini-matched-centers-v1",
        scene_count=10,
        persistent_experiment_count=8,
        health_experiment_count=14,
        replay_experiment_count=22,
        distinct_log_group_count=2,
        all_scenes_have_base_support=True,
        all_health_schedules_valid=True,
        raw_sensor_payload_reads=0,
        scientific_replay_worker_count=1,
        gpu_used=False,
        torch_imported=False,
        cuda_used=False,
        resource_evidence=_resources(),
        peak_rss_bytes=100,
        elapsed_seconds=1.0,
        dataset_root_serialized=False,
        dataset_bytes_authenticated=False,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        attribution_and_non_endorsement_required=True,
    )

    for field_name in (
        "replay_intent_byte_sha256",
        "persistent_source_matrix_sha256",
        "health_fit_artifact_sha256",
        "health_fit_run_sha256",
    ):
        with pytest.raises(ValidationError):
            ReplayProfileSummaryV1.model_validate(
                profile.model_copy(update={field_name: _digest(field_name)}).model_dump(
                    mode="python", by_alias=True
                )
            )
    with pytest.raises(ValidationError):
        ReplayProfileSummaryV1.model_validate(
            profile.model_copy(update={"gpu_used": True}).model_dump(mode="python", by_alias=True)
        )
    with pytest.raises(ValidationError):
        ReplayProfileSummaryV1.model_validate(
            profile.model_copy(update={"peak_rss_bytes": 1024**3}).model_dump(
                mode="python", by_alias=True
            )
        )
    assert replay_resource_evidence_sha256(profile.resource_evidence) == (
        replay_resource_evidence_sha256(_resources())
    )
    for update in (
        {"resource_evidence": tuple(reversed(profile.resource_evidence))},
        {"elapsed_seconds": 0.75},
        {"peak_rss_bytes": 99},
    ):
        with pytest.raises(ValidationError):
            ReplayProfileSummaryV1.model_validate(
                {
                    **profile.model_dump(mode="python", by_alias=True),
                    **update,
                }
            )


def test_aggregate_enforces_interval_support_scene_partition_and_persistence() -> None:
    assert _aggregate().persistence_label == "robustly-persistent"
    for update in (
        {"defined_bootstrap_replicates": 1950},
        {"positive_scene_count": 7, "negative_scene_count": 2},
        {
            "positive_scene_count": 7,
            "negative_scene_count": 2,
            "undefined_scene_count": 1,
        },
    ):
        with pytest.raises(ValidationError):
            _aggregate(**update)


def test_aggregate_rejects_wrong_bindings_support_shapes_and_claims() -> None:
    null_interval = {
        "estimate": None,
        "interval_lower": None,
        "interval_upper": None,
    }
    null_signs = {
        "positive_scene_count": None,
        "zero_scene_count": None,
        "negative_scene_count": None,
        "undefined_scene_count": None,
    }
    assert (
        _aggregate(
            status="undefined",
            **null_interval,
            **null_signs,
            defined_bootstrap_replicates=1000,
            persistence_label="undefined",
        ).status
        == "undefined"
    )
    assert (
        _aggregate(
            status="not-applicable",
            **null_interval,
            **null_signs,
            defined_bootstrap_replicates=0,
            inference_role="descriptive",
            expected_direction="none",
            persistence_label="undefined",
        ).status
        == "not-applicable"
    )
    assert (
        _aggregate(
            **null_signs,
            inference_role="diagnostic",
            expected_direction="none",
            persistence_label="not-applicable",
        ).inference_role
        == "diagnostic"
    )

    invalid_updates = (
        {"condition_id": "replay-camera-noise-underreported"},
        {"condition_selector": "replay-camera-noise-underreported:+4"},
        {"estimate": None},
        {"interval_lower": None},
        {"interval_upper": None},
        {"interval_lower": 2.0, "interval_upper": 1.0},
        {"defined_bootstrap_replicates": 1950},
        {
            "unit": "fraction",
            "estimate": -0.1,
            "interval_lower": 0.0,
            "interval_upper": 0.5,
        },
        {
            "unit": "fraction",
            "estimate": 0.5,
            "interval_lower": 0.0,
            "interval_upper": 1.1,
        },
        {"status": "undefined"},
        {
            "status": "not-applicable",
            **null_interval,
            **null_signs,
            "defined_bootstrap_replicates": 1,
            "inference_role": "descriptive",
            "expected_direction": "none",
            "persistence_label": "undefined",
        },
        {"positive_scene_count": None},
        {"positive_scene_count": 7, "negative_scene_count": 2},
        {
            **null_signs,
            "inference_role": "primary-directional",
        },
        {
            "positive_scene_count": 7,
            "negative_scene_count": 2,
            "undefined_scene_count": 1,
        },
        {"expected_direction": "none"},
        {"nonpositive_control_supported": False},
        {
            "inference_role": "nonpositive-control",
            "expected_direction": "positive",
            "persistence_label": "not-applicable",
            "nonpositive_control_supported": False,
        },
        {
            "inference_role": "nonpositive-control",
            "expected_direction": "nonpositive",
            "persistence_label": "robustly-persistent",
            "nonpositive_control_supported": False,
        },
        {
            "inference_role": "nonpositive-control",
            "expected_direction": "nonpositive",
            "persistence_label": "not-applicable",
            "nonpositive_control_supported": True,
        },
        {
            "inference_role": "diagnostic",
            "expected_direction": "positive",
            "persistence_label": "not-applicable",
            **null_signs,
        },
        {
            "inference_role": "diagnostic",
            "expected_direction": "none",
            "persistence_label": "robustly-persistent",
            **null_signs,
        },
        {
            "inference_role": "diagnostic",
            "expected_direction": "none",
            "persistence_label": "not-applicable",
            "nonpositive_control_supported": False,
            **null_signs,
        },
    )
    for update in invalid_updates:
        with pytest.raises(ValidationError):
            _aggregate(**update)

    with pytest.raises(ValidationError):
        _aggregate(
            status="undefined",
            **null_interval,
            **null_signs,
            defined_bootstrap_replicates=1000,
            inference_role="nonpositive-control",
            expected_direction="nonpositive",
            persistence_label="undefined",
            nonpositive_control_supported=False,
        )


def test_fraction_aggregate_is_bounded_and_undefined_rows_carry_no_estimates() -> None:
    with pytest.raises(ValidationError):
        _aggregate(unit="fraction", estimate=1.1, interval_lower=0.9, interval_upper=1.2)
    undefined = _aggregate(
        status="undefined",
        estimate=None,
        interval_lower=None,
        interval_upper=None,
        defined_bootstrap_replicates=1000,
        positive_scene_count=None,
        zero_scene_count=None,
        negative_scene_count=None,
        undefined_scene_count=None,
        persistence_label="undefined",
    )
    assert undefined.status == "undefined"
    with pytest.raises(ValidationError):
        _aggregate(status="undefined", persistence_label="undefined")


def test_percentile_interval_need_not_contain_the_point_estimate() -> None:
    aggregate = _aggregate(
        estimate=0.25,
        interval_lower=0.5,
        interval_upper=1.5,
        persistence_label="robustly-persistent",
    )
    assert aggregate.estimate < aggregate.interval_lower


def test_nonpositive_control_cannot_claim_persistence() -> None:
    control = _aggregate(
        inference_role="nonpositive-control",
        estimate=-0.5,
        interval_lower=-1.0,
        interval_upper=0.0,
        expected_direction="nonpositive",
        persistence_label="not-applicable",
        nonpositive_control_supported=True,
    )
    assert control.nonpositive_control_supported is True
    with pytest.raises(ValidationError):
        _aggregate(
            inference_role="nonpositive-control",
            estimate=-0.5,
            interval_lower=-1.0,
            interval_upper=0.0,
            expected_direction="nonpositive",
            persistence_label="robustly-persistent",
            nonpositive_control_supported=True,
        )


def test_panel_and_health_applicability_contracts_cover_all_three_bases() -> None:
    health_identity = expected_replay_identities()[8]
    with pytest.raises(ValidationError):
        _aggregate(
            **_identity(8),
            condition_id=health_identity.experiment_id,
            condition_selector=f"{health_identity.experiment_id}:+3",
        )
    with pytest.raises(ValidationError):
        _health_aggregate(identity_index=0)

    null_shape = {
        "status": "not-applicable",
        "estimate": None,
        "interval_lower": None,
        "interval_upper": None,
        "defined_bootstrap_replicates": 0,
        "positive_scene_count": None,
        "zero_scene_count": None,
        "negative_scene_count": None,
        "undefined_scene_count": None,
        "inference_role": "descriptive",
        "expected_direction": "none",
        "persistence_label": "undefined",
    }
    structural = _health_aggregate(
        identity_index=20,
        condition_selector="replay-common-mode-x:+4",
        metric_id="gap-vs-frame-oracle",
        applicability_basis="structural-unavailable",
        recovery_support_compatible_scene_count=None,
        **null_shape,
    )
    assert structural.applicability_basis == "structural-unavailable"
    for update in (
        {"applicability_basis": "applicable"},
        {"recovery_support_compatible_scene_count": 0},
        {
            "status": "undefined",
            "applicability_basis": "structural-unavailable",
        },
    ):
        value = structural.model_dump(mode="python", by_alias=True)
        value.update(update)
        with pytest.raises(ValidationError):
            ReplayHealthAggregateV1.model_validate(value)

    recovery = _health_aggregate(
        metric_id="frame-oracle-recoverable-loss-fraction",
        unit="unitless",
        aggregation="unclipped-recovery-ratio",
        applicability_basis="applicable",
        recovery_support_compatible_scene_count=10,
    )
    assert recovery.recovery_support_compatible_scene_count == 10
    support_incompatible = _health_aggregate(
        metric_id="frame-oracle-recoverable-loss-fraction",
        unit="unitless",
        aggregation="unclipped-recovery-ratio",
        applicability_basis="support-incompatible",
        recovery_support_compatible_scene_count=9,
        **null_shape,
    )
    assert support_incompatible.status == "not-applicable"
    for update in (
        {"recovery_support_compatible_scene_count": None},
        {"recovery_support_compatible_scene_count": 9},
        {
            "recovery_support_compatible_scene_count": 10,
            "applicability_basis": "support-incompatible",
        },
    ):
        value = recovery.model_dump(mode="python", by_alias=True)
        value.update(update)
        with pytest.raises(ValidationError):
            ReplayHealthAggregateV1.model_validate(value)

    ordinary = _health_aggregate()
    for update in (
        {"applicability_basis": "support-incompatible"},
        {"recovery_support_compatible_scene_count": 10},
    ):
        value = ordinary.model_dump(mode="python", by_alias=True)
        value.update(update)
        with pytest.raises(ValidationError):
            ReplayHealthAggregateV1.model_validate(value)
    ordinary_na = ordinary.model_dump(mode="python", by_alias=True)
    ordinary_na.update(null_shape)
    with pytest.raises(ValidationError):
        ReplayHealthAggregateV1.model_validate(ordinary_na)


def test_crossover_excludes_dropout_and_common_mode_and_requires_finite_root_shape() -> None:
    value: dict[str, object] = {
        "schema": "ffb.replay-persistent-crossover/v1",
        **_identity(0),
        "crossover_id": "crossover-00",
        "direction": "increase",
        "severity_unit": "m",
        "tested_maximum": 4.0,
        "status": "observed",
        "point_curve_crossed": True,
        "point_estimate": 1.0,
        "interval_lower": 0.5,
        "interval_upper": 1.5,
        "censoring": "none",
        "bootstrap_replicates": 2000,
        "bootstrap_crossing_count": 2000,
        "bootstrap_crossing_fraction": 1.0,
        "confidence_level": 0.95,
        "interval_method": "right-censored-paired-scene-percentile",
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
    }
    assert ReplayPersistentCrossoverV1.model_validate(value).point_estimate == 1.0
    for update in (
        {"crossover_id": "sample_token"},
        {"bootstrap_crossing_fraction": 0.5},
        {"status": "undetermined"},
        {"interval_lower": None},
        {"interval_upper": "positive-infinity"},
        {"censoring": "mixed-bootstrap"},
        {"point_estimate": -0.1},
        {"point_estimate": 4.1},
        {"interval_lower": -0.1},
        {"interval_upper": 4.1},
        {"interval_lower": 2.0, "interval_upper": 1.0},
    ):
        with pytest.raises(ValidationError):
            ReplayPersistentCrossoverV1.model_validate({**value, **update})
    percentile_outside_point = ReplayPersistentCrossoverV1.model_validate(
        {
            **value,
            "point_estimate": 0.25,
            "interval_lower": 0.5,
            "interval_upper": 1.5,
        }
    )
    assert percentile_outside_point.point_estimate < percentile_outside_point.interval_lower
    with pytest.raises(ValidationError):
        ReplayPersistentCrossoverV1.model_validate({**value, **_identity(6)})
    with pytest.raises(ValidationError):
        ReplayPersistentCrossoverV1.model_validate({**value, "point_estimate": None})
    with pytest.raises(ValidationError):
        ReplayPersistentCrossoverV1.model_validate(
            {**value, "interval_lower": -0.1, "interval_upper": 4.1}
        )
    undetermined = ReplayPersistentCrossoverV1.model_validate(
        {
            **value,
            "status": "undetermined",
            "point_estimate": 1.0,
            "interval_lower": None,
            "interval_upper": None,
            "censoring": "mixed-bootstrap",
            "bootstrap_crossing_count": 1000,
            "bootstrap_crossing_fraction": 0.5,
        }
    )
    assert undetermined.point_curve_crossed
    with pytest.raises(ValidationError):
        ReplayPersistentCrossoverV1.model_validate(
            {
                **undetermined.model_dump(mode="python", by_alias=True),
                "point_estimate": 4.1,
            }
        )
    not_observed = ReplayPersistentCrossoverV1.model_validate(
        {
            **value,
            "status": "not-observed",
            "point_curve_crossed": False,
            "point_estimate": None,
            "interval_lower": 4.0,
            "interval_upper": "positive-infinity",
            "censoring": "right-above-tested-maximum",
            "bootstrap_crossing_count": 0,
            "bootstrap_crossing_fraction": 0.0,
        }
    )
    assert not_observed.interval_upper == "positive-infinity"
    for update in (
        {"point_estimate": 1.0},
        {"interval_lower": 3.0},
        {"interval_upper": None},
        {"censoring": "mixed-bootstrap"},
    ):
        with pytest.raises(ValidationError):
            ReplayPersistentCrossoverV1.model_validate(
                {
                    **not_observed.model_dump(mode="python", by_alias=True),
                    **update,
                }
            )
    for update in (
        {"point_curve_crossed": False},
        {"interval_lower": 0.5},
        {"interval_upper": 1.5},
        {"censoring": "none"},
    ):
        with pytest.raises(ValidationError):
            ReplayPersistentCrossoverV1.model_validate(
                {
                    **undetermined.model_dump(mode="python", by_alias=True),
                    **update,
                }
            )


def test_cluster_sensitivity_uses_opaque_ordinals_and_rejects_scene_names() -> None:
    base = {
        "schema": "ffb.replay-cluster-sensitivity/v1",
        **_identity(0),
        "sensitivity_id": "sensitivity-00",
        "source_result_id": "result-00",
        "source_record_sha256": _digest("result"),
        "cluster_kind": "leave-one-log-group-out",
        "cluster_id": "log-group:00",
        "status": "ok",
        "estimate": 1.0,
        "unit": "m^2",
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
    }
    assert ReplayClusterSensitivityV1.model_validate(base).cluster_id == "log-group:00"
    assert (
        ReplayClusterSensitivityV1.model_validate(
            {
                **base,
                "cluster_kind": "leave-one-scene-out",
                "cluster_id": "scene-ordinal:00",
            }
        ).cluster_id
        == "scene-ordinal:00"
    )
    assert (
        ReplayClusterSensitivityV1.model_validate(
            {
                **base,
                "status": "undefined",
                "estimate": None,
            }
        ).status
        == "undefined"
    )
    for update in (
        {"cluster_kind": "leave-one-scene-out", "cluster_id": "scene-0061"},
        {"cluster_id": "log-group:0"},
        {"cluster_id": "log-group:aa"},
        {"cluster_id": f"log-group:{chr(0xFF11)}{chr(0xFF12)}"},
        {"status": "ok", "estimate": None},
        {"status": "undefined", "estimate": 1.0},
        {"sensitivity_id": "instance_token"},
        {"source_result_id": "ego_pose_token"},
    ):
        with pytest.raises(ValidationError):
            ReplayClusterSensitivityV1.model_validate({**base, **update})


def test_figure_records_bind_public_aggregate_labels() -> None:
    value = {
        "schema": "ffb.replay-figure-record/v1",
        **_identity(0),
        "figure_id": "persistent-panel-summary",
        "figure_kind": "panel-summary",
        "source_result_id": "result-00",
        "source_record_sha256": _digest("source"),
        "figure_spec_sha256": _digest("spec"),
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
    }
    assert ReplayFigureRecordV1.model_validate(value).figure_kind == "panel-summary"
    for field_name, private_value in (
        ("figure_id", "filename"),
        ("source_result_id", "calibrated_sensor_token"),
    ):
        with pytest.raises(ValidationError):
            ReplayFigureRecordV1.model_validate({**value, field_name: private_value})


def test_figure_source_binding_authenticates_exact_public_figure_and_source_shape() -> None:
    identity = expected_replay_identities()[0]
    value = {
        "schema": "ffb.replay-figure-source-binding/v1",
        **_global(),
        "figure_id": "m5-persistent-panel-summary",
        "figure_kind": "panel-summary",
        "mark_ordinal": 0,
        "source_kind": "persistent-aggregate",
        "source_id": "result-00",
        "source_record_sha256": _digest("source"),
        "identity": identity,
        "replay_identity_sha256": replay_experiment_identity_sha256(identity),
        "figure_spec_sha256": _digest("spec"),
        "rendered_svg_path": "figures/m5-persistent-panel-summary.svg",
        "rendered_svg_sha256": _digest("svg"),
        "rendered_svg_byte_length": 123,
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
    }
    binding = ReplayFigureSourceBindingV1.model_validate(value)
    assert binding.mark_ordinal == 0
    assert M5_REPLAY_FIGURE_DEFINITIONS == (
        (
            "m5-persistent-panel-summary",
            "panel-summary",
            "figures/m5-persistent-panel-summary.svg",
        ),
        ("m5-crossovers", "crossover", "figures/m5-crossovers.svg"),
        ("m5-health-transfer", "health-transfer", "figures/m5-health-transfer.svg"),
        (
            "m5-descriptor-comparison",
            "descriptor-comparison",
            "figures/m5-descriptor-comparison.svg",
        ),
        (
            "m5-cluster-sensitivity",
            "cluster-sensitivity",
            "figures/m5-cluster-sensitivity.svg",
        ),
    )

    for update in (
        {"figure_kind": "health-transfer"},
        {"rendered_svg_path": "figures/m5-health-transfer.svg"},
        {"source_kind": "descriptor-aggregate"},
        {"identity": None},
        {"replay_identity_sha256": None},
        {"replay_identity_sha256": _digest("wrong-identity")},
        {"mark_ordinal": -1},
        {"source_id": "calibrated_sensor_token"},
        {"rendered_svg_path": "/Users/private/figure.svg"},
        {"rendered_svg_byte_length": 0},
    ):
        with pytest.raises(ValidationError):
            ReplayFigureSourceBindingV1.model_validate({**value, **update})


def test_descriptor_figure_binding_has_no_synthetic_replay_identity() -> None:
    value = {
        "schema": "ffb.replay-figure-source-binding/v1",
        **_global(),
        "figure_id": "m5-descriptor-comparison",
        "figure_kind": "descriptor-comparison",
        "mark_ordinal": 0,
        "source_kind": "descriptor-aggregate",
        "source_id": "descriptor-00",
        "source_record_sha256": _digest("source"),
        "identity": None,
        "replay_identity_sha256": None,
        "figure_spec_sha256": _digest("spec"),
        "rendered_svg_path": "figures/m5-descriptor-comparison.svg",
        "rendered_svg_sha256": _digest("svg"),
        "rendered_svg_byte_length": 123,
        "tracked_aggregate_terms": M5_TRACKED_AGGREGATE_TERMS,
    }
    assert ReplayFigureSourceBindingV1.model_validate(value).identity is None
    with pytest.raises(ValidationError, match="synthetic identity"):
        ReplayFigureSourceBindingV1.model_validate(
            {
                **value,
                "identity": expected_replay_identities()[0],
                "replay_identity_sha256": replay_experiment_identity_sha256(
                    expected_replay_identities()[0]
                ),
            }
        )


def test_validation_is_exact_ordered_conjunction() -> None:
    checks = tuple(
        ReplayValidationCheckV1(
            check_id=check_id,
            passed=True,
            evidence_sha256=_digest(check_id),
        )
        for check_id in M5_RELEASE_VALIDATION_CHECK_IDS
    )
    value = {
        "schema": "ffb.replay-validation/v1",
        **_global(),
        "checks": checks,
        "scene_count": 10,
        "replay_experiment_count": 22,
        "raw_sensor_payload_reads": 0,
        "all_checks_passed": True,
    }
    assert ReplayValidationV1.model_validate(value).all_checks_passed
    with pytest.raises(ValidationError):
        ReplayValidationV1.model_validate({**value, "checks": tuple(reversed(checks))})
    failed = list(checks)
    failed[0] = failed[0].model_copy(update={"passed": False})
    with pytest.raises(ValidationError):
        ReplayValidationV1.model_validate({**value, "checks": tuple(failed)})
    assert not ReplayValidationV1.model_validate(
        {
            **value,
            "checks": tuple(failed),
            "all_checks_passed": False,
        }
    ).all_checks_passed
    with pytest.raises(ValidationError):
        ReplayValidationV1.model_validate({**value, "all_checks_passed": False})


def test_commitment_and_repeat_equality_cannot_be_asserted_contradictorily() -> None:
    commitment = ReplaySourceMemberCommitmentV1(
        schema="ffb.replay-source-member-commitment/v1",
        **_global(),
        relative_role="aggregate-members",
        primary_byte_length=10,
        repeat_byte_length=10,
        primary_record_count=1,
        repeat_record_count=1,
        primary_sha256=_digest("member"),
        repeat_sha256=_digest("member"),
        equal=True,
    )
    with pytest.raises(ValidationError):
        ReplaySourceMemberCommitmentV1.model_validate(
            commitment.model_copy(update={"repeat_byte_length": 11}).model_dump(
                mode="python", by_alias=True
            )
        )
    unequal = ReplaySourceMemberCommitmentV1.model_validate(
        {
            **commitment.model_dump(mode="python", by_alias=True),
            "repeat_byte_length": 11,
            "equal": False,
        }
    )
    assert unequal.equal is False
    with pytest.raises(ValidationError):
        ReplaySourceMemberCommitmentV1.model_validate(
            {
                **commitment.model_dump(mode="python", by_alias=True),
                "relative_role": "filepath",
            }
        )

    repeat = {
        "schema": "ffb.replay-repeat-verification/v1",
        **_global(),
        "primary_local_artifact_sha256": _digest("local-artifact"),
        "repeat_local_artifact_sha256": _digest("local-artifact"),
        "primary_run_sha256": _digest("primary"),
        "repeat_run_sha256": _digest("repeat"),
        "source_member_commitments_sha256": _digest("commitments"),
        "scientific_member_count": 1,
        "mismatch_count": 0,
        "scientific_members_all_equal": True,
        "run_records_distinct": True,
        "source_paths_and_inodes_independent": True,
        "same_named_cpu_environment": True,
        "evidence_scope": (
            "distinct-path-inode-run-and-member-consistency-not-cryptographic-proof"
        ),
        "all_checks_passed": True,
    }
    assert ReplayRepeatVerificationV1.model_validate(repeat).all_checks_passed
    for update in (
        {"scientific_member_count": 1, "mismatch_count": 2},
        {"scientific_members_all_equal": False},
        {
            "repeat_run_sha256": repeat["primary_run_sha256"],
            "run_records_distinct": True,
        },
        {"run_records_distinct": False},
        {"source_paths_and_inodes_independent": False},
        {"same_named_cpu_environment": False},
    ):
        with pytest.raises(ValidationError):
            ReplayRepeatVerificationV1.model_validate({**repeat, **update})

    valid_failures = (
        {
            "mismatch_count": 1,
            "scientific_members_all_equal": False,
            "all_checks_passed": False,
        },
        {
            "repeat_run_sha256": repeat["primary_run_sha256"],
            "run_records_distinct": False,
            "all_checks_passed": False,
        },
        {
            "source_paths_and_inodes_independent": False,
            "all_checks_passed": False,
        },
        {
            "same_named_cpu_environment": False,
            "all_checks_passed": False,
        },
    )
    for update in valid_failures:
        assert not ReplayRepeatVerificationV1.model_validate({**repeat, **update}).all_checks_passed
    independently_committed_artifacts = ReplayRepeatVerificationV1.model_validate(
        {
            **repeat,
            "repeat_local_artifact_sha256": _digest("different-local-artifact"),
        }
    )
    assert independently_committed_artifacts.all_checks_passed
    assert (
        independently_committed_artifacts.primary_local_artifact_sha256
        != independently_committed_artifacts.repeat_local_artifact_sha256
    )


def test_release_index_and_success_bind_exact_identity_set() -> None:
    entries = tuple(
        ReplayPayloadFileEntryV1(
            path=path,  # type: ignore[arg-type]
            byte_length=100,
            sha256=_digest(path),
            record_count=1 if path.endswith(".ndjson") else None,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        )
        for path in REPLAY_INDEXED_PATHS
    )
    index = ReplayReleaseIndexV1(
        schema="ffb.replay-release-index/v1",
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
        release_id=M5_REPLAY_RELEASE_ID,
        run_id=_RUN_ID,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        identities=expected_replay_identities(),
        persistent_source_matrix_sha256=M5_PERSISTENT_MATRIX_SHA256,
        persistent_condition_selector_count=71,
        persistent_condition_selector_set_sha256=(
            "9bc53e38bb910e50e4111c7a72103edf980568d52739f64c58e13a5fcd735cac"
        ),
        persistent_aggregate_coordinate_count=464,
        persistent_aggregate_coordinate_set_sha256=(M5_PERSISTENT_AGGREGATE_COORDINATE_SET_SHA256),
        persistent_hypothesis_coordinate_count=33,
        persistent_hypothesis_coordinate_set_sha256=(
            M5_PERSISTENT_HYPOTHESIS_COORDINATE_SET_SHA256
        ),
        health_fit_artifact_sha256=M5_HEALTH_FIT_SHA256,
        health_fit_run_sha256=M5_HEALTH_FIT_RUN_SHA256,
        health_condition_selector_count=43,
        health_condition_selector_set_sha256=(
            "4b09fbbcff924ab94a17c94984e67bd06d6a0e56df43a95094dec64d60d3376c"
        ),
        health_aggregate_coordinate_count=14_988,
        health_aggregate_coordinate_set_sha256=M5_HEALTH_AGGREGATE_COORDINATE_SET_SHA256,
        health_hypothesis_coordinate_count=11,
        health_hypothesis_coordinate_set_sha256=(
            "6d73453c8cef65e90bc6d4cb1fe972bce976a43a924046dcaafdbdc57b7f7cb8"
        ),
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        files=entries,
    )
    assert tuple(entry.path for entry in index.files) == REPLAY_INDEXED_PATHS
    for field_name in (
        "replay_intent_sha256",
        "replay_identity_set_sha256",
        "persistent_source_matrix_sha256",
        "persistent_condition_selector_set_sha256",
        "persistent_aggregate_coordinate_set_sha256",
        "persistent_hypothesis_coordinate_set_sha256",
        "health_fit_artifact_sha256",
        "health_fit_run_sha256",
        "health_condition_selector_set_sha256",
        "health_aggregate_coordinate_set_sha256",
        "health_hypothesis_coordinate_set_sha256",
    ):
        with pytest.raises(ValidationError):
            ReplayReleaseIndexV1.model_validate(
                index.model_copy(update={field_name: _digest(field_name)}).model_dump(
                    mode="python", by_alias=True
                )
            )
    with pytest.raises(ValidationError):
        ReplayReleaseIndexV1.model_validate(
            index.model_copy(
                update={"identities": tuple(reversed(expected_replay_identities()))}
            ).model_dump(mode="python", by_alias=True)
        )
    with pytest.raises(ValidationError):
        ReplayReleaseIndexV1.model_validate(
            index.model_copy(update={"files": tuple(reversed(entries))}).model_dump(
                mode="python", by_alias=True
            )
        )
    ndjson_entry = next(entry for entry in entries if entry.path.endswith(".ndjson"))
    json_entry = next(entry for entry in entries if not entry.path.endswith(".ndjson"))
    for entry, update in (
        (ndjson_entry, {"record_count": None}),
        (json_entry, {"record_count": 1}),
        (ndjson_entry, {"replay_identity_set_sha256": _digest("wrong-entry-set")}),
    ):
        with pytest.raises(ValidationError):
            ReplayPayloadFileEntryV1.model_validate(
                {
                    **entry.model_dump(mode="python"),
                    **update,
                }
            )

    success = ReplaySuccessV1(
        schema="ffb.replay-success/v1",
        release_artifact_sha256=_digest("artifact"),
        run_sha256=_digest("run"),
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
    )
    with pytest.raises(ValidationError):
        ReplaySuccessV1.model_validate(
            success.model_copy(update={"replay_identity_set_sha256": _digest("other")}).model_dump(
                mode="python", by_alias=True
            )
        )
    with pytest.raises(ValidationError):
        ReplaySuccessV1.model_validate(
            success.model_copy(update={"replay_intent_sha256": _digest("other-intent")}).model_dump(
                mode="python", by_alias=True
            )
        )
