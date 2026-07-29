from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.contracts.procedural_profile_v1 import (
    SplitId,
    load_procedural_profile,
)
from fusion_fault_bench.reference.procedural import (
    independent_fault_uniforms,
    reference_eligibility_mask,
    reference_latent_state,
    reference_truth,
)
from fusion_fault_bench.rng import draw_latent_uniforms
from fusion_fault_bench.scenarios.procedural import (
    generate_procedural_sequence,
    generate_procedural_sequences,
)

PROFILE_ROOT = Path("examples/profiles")


@pytest.mark.parametrize(
    ("filename", "split"),
    [
        ("constant-velocity-front-roi-v1.json", "train"),
        ("constant-velocity-front-roi-v1.json", "validation"),
        ("constant-velocity-front-roi-v1.json", "test"),
        ("constant-velocity-fov-edge-v1.json", "test"),
        ("constant-velocity-ci-smoke-v1.json", "test"),
    ],
)
def test_production_sequence_matches_independent_latent_and_eligibility(
    filename: str,
    split: SplitId,
) -> None:
    profile = load_procedural_profile(PROFILE_ROOT / filename)
    sequence = generate_procedural_sequence(
        profile,
        split=split,
        sequence_index=1,
        data_master_seed=1729,
    )
    uniforms = draw_latent_uniforms(
        data_master_seed=1729,
        sequence_id=sequence.sequence_id,
        object_count=profile.source.object_count,
    )
    reference_state = reference_latent_state(profile.profile_id, split, uniforms)
    reference_centers = reference_truth(
        reference_state,
        frame_count=profile.source.frame_count,
        frame_period_s=profile.source.frame_period_s,
    )
    reference_mask = reference_eligibility_mask(
        reference_centers,
        x_min_m=5.0,
        x_max_m=60.0,
        abs_y_max_m=40.0,
        camera_half_fov_rad=0.7,
    )
    assert sequence.initial_xy_m == pytest.approx(reference_state.initial_xy_m)
    assert sequence.velocity_xy_mps == pytest.approx(reference_state.velocity_xy_mps)
    assert sequence.truth_xy_m == pytest.approx(reference_centers)
    assert np.array_equal(sequence.eligibility_mask, reference_mask)
    assert sequence.eligible_object_frame_count == int(np.count_nonzero(reference_mask))
    assert sequence.eligible_object_frame_ids[0].startswith("000000:object:")
    assert len(sequence.eligibility_sha256) == 64


def test_sequence_draws_are_paired_deterministic_and_immutable() -> None:
    profile = load_procedural_profile(PROFILE_ROOT / "constant-velocity-front-roi-v1.json")
    first = generate_procedural_sequence(
        profile,
        split="test",
        sequence_index=0,
        data_master_seed=1729,
    )
    second = generate_procedural_sequence(
        profile,
        split="test",
        sequence_index=0,
        data_master_seed=1729,
    )
    assert first.sequence_id == "procedural:constant-velocity-front-roi-v1:test:000000"
    for first_array, second_array in (
        (first.truth_xy_m, second.truth_xy_m),
        (first.camera_standard_normal_xy, second.camera_standard_normal_xy),
        (first.lidar_standard_normal_xy, second.lidar_standard_normal_xy),
        (first.fault_uniform_by_frame, second.fault_uniform_by_frame),
    ):
        assert first_array.tobytes() == second_array.tobytes()
        assert not first_array.flags.writeable
        with pytest.raises(ValueError):
            first_array.setflags(write=True)
    assert not np.array_equal(
        first.camera_standard_normal_xy,
        first.lidar_standard_normal_xy,
    )
    assert (
        first.fault_uniform_by_frame.tobytes()
        == independent_fault_uniforms(
            data_master_seed=1729,
            sequence_id=first.sequence_id,
            frame_count=48,
        ).tobytes()
    )


def test_sequence_prefix_has_no_retries_or_exclusions() -> None:
    profile = load_procedural_profile(PROFILE_ROOT / "constant-velocity-ci-smoke-v1.json")
    sequences = generate_procedural_sequences(
        profile,
        split="test",
        sequence_count=4,
        data_master_seed=1729,
    )
    assert tuple(sequence.sequence_index for sequence in sequences) == (0, 1, 2, 3)
    assert len({sequence.sequence_id for sequence in sequences}) == 4
    assert all(sequence.eligible_object_frame_count == 24 for sequence in sequences)


def test_sequence_generator_rejects_out_of_contract_requests() -> None:
    profile = load_procedural_profile(PROFILE_ROOT / "constant-velocity-ci-smoke-v1.json")
    with pytest.raises(ValueError, match="outside the declared split"):
        generate_procedural_sequence(
            profile,
            split="test",
            sequence_index=4,
            data_master_seed=1729,
        )
    with pytest.raises(ValueError, match="declared split count"):
        generate_procedural_sequences(
            profile,
            split="test",
            sequence_count=5,
            data_master_seed=1729,
        )
