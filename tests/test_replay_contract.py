from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_HEALTH_IDENTITY_BINDINGS,
    M5_HEALTH_PANEL_ID,
    M5_PERSISTENT_IDENTITY_BINDINGS,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_PERSISTENT_PANEL_ID,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_PATH,
    M5_REPLAY_INTENT_SHA256,
    M5_SCENE_NAMES,
    REPLAY_EXPERIMENT_IDENTITY_ADAPTER,
    ReplayExperimentIdentityV1,
    expected_replay_identities,
    load_replay_intent,
    replay_experiment_identity_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def _copy_intent(tmp_path: Path) -> Path:
    destination = tmp_path / M5_REPLAY_INTENT_PATH
    destination.parent.mkdir(parents=True)
    shutil.copy2(ROOT / M5_REPLAY_INTENT_PATH, destination)
    return destination


def test_loads_exact_byte_and_canonical_frozen_replay_intent() -> None:
    loaded = load_replay_intent(source_root=ROOT)

    assert loaded.path == ROOT / M5_REPLAY_INTENT_PATH
    assert loaded.intent_sha256 == M5_REPLAY_INTENT_SHA256
    assert loaded.byte_sha256 == M5_REPLAY_INTENT_BYTE_SHA256
    assert loaded.scene_names == M5_SCENE_NAMES
    assert loaded.persistent_identity_bindings == M5_PERSISTENT_IDENTITY_BINDINGS
    assert loaded.health_identity_bindings == M5_HEALTH_IDENTITY_BINDINGS


def test_exact_identity_order_uses_each_m5_a_manifest_and_the_m5_b_fit() -> None:
    identities = expected_replay_identities()
    persistent = identities[: len(M5_PERSISTENT_IDENTITY_BINDINGS)]
    health = identities[len(M5_PERSISTENT_IDENTITY_BINDINGS) :]

    assert tuple((item.experiment_id, item.source_sha256) for item in persistent) == tuple(
        (item.experiment_id, item.source_sha256) for item in M5_PERSISTENT_IDENTITY_BINDINGS
    )
    assert all(item.panel_id == M5_PERSISTENT_PANEL_ID for item in persistent)
    assert all(item.source_sha256 != M5_PERSISTENT_MATRIX_SHA256 for item in persistent)
    assert tuple((item.experiment_id, item.source_sha256) for item in health) == tuple(
        (item.experiment_id, M5_HEALTH_FIT_SHA256) for item in M5_HEALTH_IDENTITY_BINDINGS
    )
    assert all(item.panel_id == M5_HEALTH_PANEL_ID for item in health)
    assert len({replay_experiment_identity_sha256(item) for item in identities}) == len(identities)


def test_identity_rejects_matrix_digest_or_another_experiment_manifest() -> None:
    first = M5_PERSISTENT_IDENTITY_BINDINGS[0]
    second = M5_PERSISTENT_IDENTITY_BINDINGS[1]

    for wrong_source in (M5_PERSISTENT_MATRIX_SHA256, second.source_sha256):
        with pytest.raises(ValidationError, match="exact M5-A manifest"):
            ReplayExperimentIdentityV1(
                schema="ffb.replay-experiment-identity/v1",
                replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
                panel_id=M5_PERSISTENT_PANEL_ID,
                source_sha256=wrong_source,
                experiment_id=first.experiment_id,
            )


def test_identity_schema_is_strict_and_forbids_cross_panel_binding() -> None:
    value = {
        "schema": "ffb.replay-experiment-identity/v1",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "panel_id": M5_HEALTH_PANEL_ID,
        "source_sha256": M5_HEALTH_FIT_SHA256,
        "experiment_id": M5_HEALTH_IDENTITY_BINDINGS[0].experiment_id,
    }
    parsed = REPLAY_EXPERIMENT_IDENTITY_ADAPTER.validate_python(value)
    assert parsed.model_dump(mode="json", by_alias=True) == value

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        REPLAY_EXPERIMENT_IDENTITY_ADAPTER.validate_python({**value, "unexpected": True})
    with pytest.raises(ValidationError, match="frozen M5-B fit"):
        REPLAY_EXPERIMENT_IDENTITY_ADAPTER.validate_python(
            {
                **value,
                "experiment_id": M5_PERSISTENT_IDENTITY_BINDINGS[0].experiment_id,
            }
        )


def test_loader_rejects_duplicate_keys_before_content_authentication(
    tmp_path: Path,
) -> None:
    path = _copy_intent(tmp_path)
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace(
            '  "benchmark_id": "m5-nuscenes-mini-replay-v1",',
            (
                '  "benchmark_id": "m5-nuscenes-mini-replay-v1",\n'
                '  "benchmark_id": "m5-nuscenes-mini-replay-v1",'
            ),
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_replay_intent(source_root=tmp_path)


def test_loader_rejects_semantically_equal_but_not_byte_frozen_json(
    tmp_path: Path,
) -> None:
    path = _copy_intent(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="bytes are not preregistered"):
        load_replay_intent(source_root=tmp_path)


@pytest.mark.parametrize(
    "path",
    (Path("examples/replay/other.json"), Path("../m5-replay.json")),
)
def test_loader_rejects_noncanonical_paths(tmp_path: Path, path: Path) -> None:
    _copy_intent(tmp_path)

    with pytest.raises(ValueError, match="replay intent"):
        load_replay_intent(path, source_root=tmp_path)
