from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import fusion_fault_bench.replay_resources as replay_resources
from fusion_fault_bench.artifacts import canonical_json_bytes, compute_run_record_digest
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_FIT_RUN_SHA256,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    M5_TRACKED_AGGREGATE_TERMS,
    ReplayExecutionResourceEvidenceV1,
    ReplayProfileSummaryV1,
    replay_execution_resource_evidence_json_schema,
    replay_resource_evidence_sha256,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.replay_resources import (
    ReplayResourceEvidenceError,
    ReplayResourceRunBinding,
    import_replay_execution_resource_evidence,
    parse_darwin_time_l,
    replay_environment_sha256,
    replay_logical_command_sha256,
)

_RESOURCE_LABELS = (
    "maximum resident set size",
    "average shared memory size",
    "average unshared data size",
    "average unshared stack size",
    "page reclaims",
    "page faults",
    "swaps",
    "block input operations",
    "block output operations",
    "messages sent",
    "messages received",
    "signals received",
    "voluntary context switches",
    "involuntary context switches",
    "instructions retired",
    "cycles elapsed",
    "peak memory footprint",
)
_RUN_ID = f"run:{'a' * 64}"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _time_log(
    *,
    elapsed: str = "20.00",
    user: str = "15.00",
    system: str = "1.00",
    peak_rss_bytes: int = 300_000_000,
) -> bytes:
    values = {
        label: (peak_rss_bytes if label == "maximum resident set size" else index)
        for index, label in enumerate(_RESOURCE_LABELS)
    }
    lines = [
        f"{elapsed:>12} real {user:>12} user {system:>12} sys",
        *(f"{values[label]:20d}  {label}" for label in _RESOURCE_LABELS),
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.12.13",
        os_name="Darwin",
        os_release="25.5.0",
        machine="arm64",
        cpu_model="Test CPU",
        logical_cpu_count=4,
        memory_bytes=8 * 1024**3,
    )


def _run(label: str) -> RunRecordV1Alpha1:
    started_at = datetime(2026, 1, 2, tzinfo=UTC) + (
        timedelta(minutes=1) if label == "repeat" else timedelta()
    )
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=_RUN_ID,
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        package_version="0.1.0",
        git_revision="1" * 40,
        source_dirty=False,
        lockfile_sha256="2" * 64,
        command=(
            "ffb",
            "replay",
            "run",
            "--output-dir",
            f"reports/generated/{label}",
        ),
        environment=_environment(),
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=10),
        status="succeeded",
        artifact_sha256="0" * 64,
    )


def _binding(
    label: str,
    *,
    internal_elapsed: float = 10.0,
    internal_rss: int = 200_000_000,
) -> ReplayResourceRunBinding:
    run = _run(label)
    return ReplayResourceRunBinding(
        run=run,
        local_artifact_sha256=_digest("shared-local-artifact"),
        local_run_sha256=compute_run_record_digest(canonical_json_bytes(run)),
        persisted_internal_elapsed_seconds=internal_elapsed,
        persisted_internal_peak_rss_bytes=internal_rss,
        persisted_internal_measurement_scope=(
            "metadata-through-canonical-scientific-members-before-publication"
        ),
    )


def _import_pair(
    tmp_path: Path,
    *,
    primary_log: bytes | None = None,
    repeat_log: bytes | None = None,
    primary_binding: ReplayResourceRunBinding | None = None,
    repeat_binding: ReplayResourceRunBinding | None = None,
) -> tuple[ReplayExecutionResourceEvidenceV1, ReplayExecutionResourceEvidenceV1]:
    primary_path = tmp_path / "primary.time-l.txt"
    repeat_path = tmp_path / "repeat.time-l.txt"
    primary_path.write_bytes(_time_log() if primary_log is None else primary_log)
    repeat_path.write_bytes(
        _time_log(elapsed="25.00", peak_rss_bytes=350_000_000) if repeat_log is None else repeat_log
    )
    return import_replay_execution_resource_evidence(
        primary_log_path=primary_path,
        repeat_log_path=repeat_path,
        primary=_binding("primary") if primary_binding is None else primary_binding,
        repeat=_binding("repeat") if repeat_binding is None else repeat_binding,
    )


def test_strict_darwin_time_l_parser_accepts_only_the_complete_canonical_log() -> None:
    raw = _time_log()
    parsed = parse_darwin_time_l(raw)
    assert parsed.elapsed_seconds == 20.0
    assert parsed.peak_rss_bytes == 300_000_000

    lines = raw.splitlines()
    malformed = (
        raw[:-1],
        raw.replace(b"\n", b"\r\n"),
        raw.replace(b" ", b"\t", 1),
        raw + b"extra line\n",
        b"\n".join(lines[:-1]) + b"\n",
        b"\n".join((lines[0], lines[2], lines[1], *lines[3:])) + b"\n",
        raw.replace(b"20.00 real", b"20,00 real"),
        raw.replace(b"20.00 real", b"Elapsed (wall clock) time: 20.00"),
        raw.replace(b"      20.00 real", b"       20.00 real"),
        raw.replace(b"300000000  maximum", b"0300000000  maximum"),
        raw.replace(b"20.00 real", b"nan real"),
        raw.replace(b"20.00 real", b"0.00 real"),
        raw.replace(b"           300000000", b"                   0", 1),
        raw.replace(b"cycles elapsed", "cycles élapsed".encode()),
    )
    for candidate in malformed:
        with pytest.raises(ReplayResourceEvidenceError):
            parse_darwin_time_l(candidate)


def test_resource_import_binds_exact_runs_environment_commands_and_raw_logs(
    tmp_path: Path,
) -> None:
    resources = _import_pair(tmp_path)
    primary, repeat = resources

    assert tuple(record.run_label for record in resources) == ("primary", "repeat")
    assert primary.elapsed_seconds == 20.0
    assert repeat.elapsed_seconds == 25.0
    assert primary.peak_rss_bytes == 300_000_000
    assert repeat.peak_rss_bytes == 350_000_000
    assert primary.local_artifact_sha256 == repeat.local_artifact_sha256
    assert primary.local_run_sha256 == _binding("primary").local_run_sha256
    assert repeat.local_run_sha256 == _binding("repeat").local_run_sha256
    assert primary.environment_sha256 == replay_environment_sha256(_environment())
    assert primary.logical_command_sha256 == replay_logical_command_sha256(
        tuple(_run("primary").command)
    )
    assert primary.logical_command_sha256 == repeat.logical_command_sha256
    assert primary.raw_log_sha256 == hashlib.sha256(_time_log()).hexdigest()
    assert primary.raw_log_byte_length == len(_time_log())
    assert replay_execution_resource_evidence_json_schema()["type"] == "object"

    serialized = canonical_json_bytes(primary)
    assert os.fspath(tmp_path).encode() not in serialized
    assert b"time-l.txt" not in serialized
    assert sha256_digest(
        {
            "schema": "ffb.replay-execution-resource-evidence-set/v1",
            "records": [record.model_dump(mode="json", by_alias=True) for record in resources],
        }
    ) == replay_resource_evidence_sha256(resources)


def test_logical_command_rejects_non_m5_and_noncanonical_output_arguments() -> None:
    invalid_commands = (
        ("fusion-fault-bench", "replay", "curate"),
        ("ffb", "replay", "run"),
        ("ffb", "replay", "run", "--output-dir", "/private/output"),
        ("ffb", "replay", "run", "--output-dir", "reports/generated"),
        ("ffb", "replay", "run", "--output-dir", "reports/generated/../private"),
        ("ffb", "replay", "run", "--output-dir", "reports/generated//primary"),
        ("ffb", "replay", "run", "--output-dir", "reports/generated/private output"),
        (
            "ffb",
            "replay",
            "run",
            "--output-dir",
            "reports/generated/primary",
            "--extra",
        ),
    )
    for command in invalid_commands:
        with pytest.raises(ValueError):
            replay_logical_command_sha256(command)


def test_profile_requires_exact_resource_order_global_bindings_and_maxima(
    tmp_path: Path,
) -> None:
    resources = _import_pair(tmp_path)
    profile = ReplayProfileSummaryV1(
        schema="ffb.replay-profile-summary/v1",
        run_id=_RUN_ID,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
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
        resource_evidence=resources,
        peak_rss_bytes=350_000_000,
        elapsed_seconds=25.0,
        dataset_root_serialized=False,
        dataset_bytes_authenticated=False,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        attribution_and_non_endorsement_required=True,
    )
    assert profile.elapsed_seconds == 25.0
    assert profile.peak_rss_bytes == 350_000_000

    for update in (
        {"resource_evidence": tuple(reversed(resources))},
        {"resource_evidence": (resources[0], resources[0])},
        {"elapsed_seconds": 24.0},
        {"elapsed_seconds": 26.0},
        {"peak_rss_bytes": 300_000_000},
        {"peak_rss_bytes": 351_000_000},
    ):
        with pytest.raises(ValidationError):
            ReplayProfileSummaryV1.model_validate(
                {
                    **profile.model_dump(mode="python", by_alias=True),
                    **update,
                }
            )


def test_import_rejects_caps_at_equality_and_external_underreporting(
    tmp_path: Path,
) -> None:
    cases = (
        {
            "primary_log": _time_log(elapsed="1800.00"),
        },
        {
            "primary_log": _time_log(peak_rss_bytes=1_073_741_824),
        },
        {
            "primary_log": _time_log(elapsed="20.00"),
            "primary_binding": _binding("primary", internal_elapsed=20.01),
        },
        {
            "primary_log": _time_log(peak_rss_bytes=300_000_000),
            "primary_binding": _binding("primary", internal_rss=300_000_001),
        },
    )
    for index, kwargs in enumerate(cases):
        case_root = tmp_path / f"case-{index}"
        case_root.mkdir()
        with pytest.raises(ReplayResourceEvidenceError):
            _import_pair(case_root, **kwargs)  # type: ignore[arg-type]


def test_resource_contract_rejects_bool_as_int_nonfinite_and_mutated_bindings(
    tmp_path: Path,
) -> None:
    primary = _import_pair(tmp_path)[0]
    raw = primary.model_dump(mode="python", by_alias=True)
    for update in (
        {"elapsed_seconds": float("nan")},
        {"persisted_internal_elapsed_seconds": float("inf")},
        {"peak_rss_bytes": True},
        {"persisted_internal_peak_rss_bytes": True},
        {"raw_log_byte_length": True},
        {"wall_time_cap_seconds": 1799.0},
        {"wall_time_within_cap": False},
        {"peak_rss_within_cap": False},
        {"cpu_process_scope": "all-helper-processes"},
        {"helper_process_policy": "parallel-benchmark-workers"},
        {"environment_sha256": "x" * 64},
    ):
        with pytest.raises(ValidationError):
            ReplayExecutionResourceEvidenceV1.model_validate({**raw, **update})

    valid_binding = _binding("primary")
    with pytest.raises(ValueError):
        ReplayResourceRunBinding(
            run=valid_binding.run,
            local_artifact_sha256=valid_binding.local_artifact_sha256,
            local_run_sha256=_digest("wrong-run"),
            persisted_internal_elapsed_seconds=(valid_binding.persisted_internal_elapsed_seconds),
            persisted_internal_peak_rss_bytes=(valid_binding.persisted_internal_peak_rss_bytes),
            persisted_internal_measurement_scope=(
                valid_binding.persisted_internal_measurement_scope
            ),
        )


def test_resource_log_reader_rejects_symlinks_hardlinks_reuse_and_path_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.time-l.txt"
    target.write_bytes(_time_log())
    repeat = tmp_path / "repeat.time-l.txt"
    repeat.write_bytes(_time_log(elapsed="25.00", peak_rss_bytes=350_000_000))
    symlink = tmp_path / "symlink.time-l.txt"
    symlink.symlink_to(target)
    with pytest.raises(ReplayResourceEvidenceError, match="symlink"):
        import_replay_execution_resource_evidence(
            primary_log_path=symlink,
            repeat_log_path=repeat,
            primary=_binding("primary"),
            repeat=_binding("repeat"),
        )

    hardlink = tmp_path / "hardlink.time-l.txt"
    os.link(target, hardlink)
    with pytest.raises(ReplayResourceEvidenceError, match="private regular"):
        import_replay_execution_resource_evidence(
            primary_log_path=target,
            repeat_log_path=repeat,
            primary=_binding("primary"),
            repeat=_binding("repeat"),
        )
    hardlink.unlink()

    with pytest.raises(ReplayResourceEvidenceError, match="independent files"):
        import_replay_execution_resource_evidence(
            primary_log_path=target,
            repeat_log_path=target,
            primary=_binding("primary"),
            repeat=_binding("repeat"),
        )

    original_snapshot = vars(replay_resources)["_component_snapshot"]
    calls = 0

    def changed_snapshot(path: Path):
        nonlocal calls
        calls += 1
        snapshot = original_snapshot(path)
        if calls == 2:
            other = tmp_path / "other"
            other.write_bytes(b"different")
            return (*snapshot[:-1], (snapshot[-1][0], other.stat()))
        return snapshot

    monkeypatch.setattr(replay_resources, "_component_snapshot", changed_snapshot)
    with pytest.raises(ReplayResourceEvidenceError, match="path changed"):
        import_replay_execution_resource_evidence(
            primary_log_path=target,
            repeat_log_path=repeat,
            primary=_binding("primary"),
            repeat=_binding("repeat"),
        )


def test_resource_pair_rejects_different_run_identity_or_environment(tmp_path: Path) -> None:
    primary_path = tmp_path / "primary.time-l.txt"
    repeat_path = tmp_path / "repeat.time-l.txt"
    primary_path.write_bytes(_time_log())
    repeat_path.write_bytes(_time_log())
    repeat_binding = _binding("repeat")
    different_run = repeat_binding.run.model_copy(update={"run_id": f"run:{'b' * 64}"})
    object.__setattr__(repeat_binding, "run", different_run)

    with pytest.raises(ReplayResourceEvidenceError, match="run identity or environment"):
        import_replay_execution_resource_evidence(
            primary_log_path=primary_path,
            repeat_log_path=repeat_path,
            primary=_binding("primary"),
            repeat=repeat_binding,
        )
