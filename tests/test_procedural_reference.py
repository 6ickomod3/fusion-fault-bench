from __future__ import annotations

import math

import numpy as np
import pytest

from fusion_fault_bench.reference.procedural import (
    AffineLossMoments,
    ReferenceLatentState,
    affine_signed_contrast_moments,
    affine_squared_loss_moments,
    covariance_six_se_bound,
    equal_sequence_population_moments,
    independent_dropout_mask,
    independent_fault_stream_seed,
    independent_fault_uniforms,
    mean_six_se_bound,
    reference_eligibility_mask,
    reference_latent_state,
    reference_truth,
    timestamp_displacement_xy,
    variance_six_se_bound,
    yaw_displacement_xy,
)
from fusion_fault_bench.rng import derive_stream_seed


def test_reference_main_profile_exact_mappings_and_support() -> None:
    uniforms = np.linspace(0.01, 0.99, 24, dtype=np.float64).reshape(6, 4)
    train = reference_latent_state("constant-velocity-front-roi-v1", "train", uniforms)
    validation = reference_latent_state(
        "constant-velocity-front-roi-v1",
        "validation",
        uniforms,
    )
    test = reference_latent_state("constant-velocity-front-roi-v1", "test", uniforms)

    assert train.initial_xy_m[0, 0] == pytest.approx(10.0 + 18.0 * uniforms[0, 0])
    assert train.initial_xy_m[0, 1] == pytest.approx(-3.5 + 0.25 * (2 * uniforms[0, 1] - 1))
    assert validation.velocity_xy_mps[1, 1] == pytest.approx(-(1.5 + 1.5 * uniforms[1, 3]))
    assert test.initial_xy_m[:, 1] == pytest.approx(
        np.asarray([-7.0, -4.0, -1.0, 1.0, 4.0, 7.0]) + 0.25 * (2.0 * uniforms[:, 1] - 1.0)
    )
    truth = reference_truth(test, frame_count=48, frame_period_s=0.1)
    assert truth.shape == (48, 6, 2)
    assert truth[47] == pytest.approx(
        test.initial_xy_m + 4.7 * test.velocity_xy_mps,
    )
    eligible = reference_eligibility_mask(
        truth,
        x_min_m=5.0,
        x_max_m=60.0,
        abs_y_max_m=40.0,
        camera_half_fov_rad=0.7,
    )
    assert eligible.shape == (48, 6)
    assert np.all(eligible)
    with pytest.raises(ValueError, match="unknown split"):
        reference_latent_state("constant-velocity-front-roi-v1", "unknown", uniforms)  # type: ignore[arg-type]


def test_reference_edge_and_smoke_profiles() -> None:
    edge_uniforms = np.full((4, 4), 0.5, dtype=np.float64)
    edge = reference_latent_state("constant-velocity-fov-edge-v1", "test", edge_uniforms)
    edge_bearings = np.arctan2(edge.initial_xy_m[:, 1], edge.initial_xy_m[:, 0])
    assert np.abs(edge_bearings) == pytest.approx(np.full(4, 0.6875))
    edge_truth = reference_truth(edge, frame_count=48, frame_period_s=0.1)
    assert np.all(
        reference_eligibility_mask(
            edge_truth,
            x_min_m=5.0,
            x_max_m=60.0,
            abs_y_max_m=40.0,
            camera_half_fov_rad=0.7,
        )
    )

    smoke_uniforms = np.full((3, 4), 0.5, dtype=np.float64)
    smoke = reference_latent_state("constant-velocity-ci-smoke-v1", "test", smoke_uniforms)
    assert smoke.initial_xy_m == pytest.approx(np.asarray([[15.0, -2.0], [15.0, 0.0], [15.0, 2.0]]))
    assert smoke.velocity_xy_mps == pytest.approx(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="only the test split"):
        reference_latent_state("constant-velocity-fov-edge-v1", "train", edge_uniforms)
    with pytest.raises(ValueError, match="only the test split"):
        reference_latent_state("constant-velocity-ci-smoke-v1", "validation", smoke_uniforms)


def test_reference_fault_uniforms_match_only_at_contract_boundary() -> None:
    sequence_id = "procedural:constant-velocity-front-roi-v1:test:000003"
    expected_seed = derive_stream_seed(
        data_master_seed=1729,
        stream_name="fault",
        sequence_id=sequence_id,
    )
    assert (
        independent_fault_stream_seed(
            data_master_seed=1729,
            sequence_id=sequence_id,
        )
        == expected_seed
    )
    uniforms = independent_fault_uniforms(
        data_master_seed=1729,
        sequence_id=sequence_id,
        frame_count=48,
    )
    assert uniforms.shape == (48,)
    masks = [independent_dropout_mask(uniforms, probability) for probability in (0, 0.25, 1)]
    assert not np.any(masks[0])
    assert np.all(masks[2])
    assert np.all(masks[0] <= masks[1])
    assert np.all(masks[1] <= masks[2])


def test_affine_loss_and_contrast_moments_match_monte_carlo() -> None:
    first_matrix = np.asarray([[0.8, 0.0, 0.2, 0.0], [0.0, 0.8, 0.0, 0.2]])
    first_bias = np.asarray([0.3, -0.1])
    second_matrix = np.asarray([[0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.3]])
    second_bias = np.zeros(2)
    first = affine_squared_loss_moments(first_matrix, first_bias)
    contrast = affine_signed_contrast_moments(
        first_matrix,
        first_bias,
        second_matrix,
        second_bias,
    )
    generator = np.random.default_rng(91)
    draws = generator.standard_normal((400_000, 4))
    first_errors = draws @ first_matrix.T + first_bias
    second_errors = draws @ second_matrix.T + second_bias
    first_losses = np.sum(first_errors**2, axis=1)
    contrast_losses = first_losses - np.sum(second_errors**2, axis=1)
    assert np.mean(first_losses) == pytest.approx(first.expected_m2, abs=0.01)
    assert np.var(first_losses) == pytest.approx(first.variance_m4, abs=0.03)
    assert np.mean(contrast_losses) == pytest.approx(contrast.expected_m2, abs=0.01)
    assert np.var(contrast_losses) == pytest.approx(contrast.variance_m4, abs=0.03)


def test_equal_sequence_population_uses_sequence_means() -> None:
    result = equal_sequence_population_moments(
        (
            (AffineLossMoments(1.0, 4.0),),
            (AffineLossMoments(2.0, 1.0), AffineLossMoments(4.0, 9.0)),
        )
    )
    assert result.expected_m2 == pytest.approx(2.0)
    expected_variance = (4.0 + (1.0 + 9.0) / 4.0) / 4.0
    assert result.standard_error_m2 == pytest.approx(math.sqrt(expected_variance))
    assert result.sequence_count == 2
    assert result.object_frame_count == 3


def test_preregistered_six_se_bounds_and_geometry_oracles() -> None:
    assert mean_six_se_bound(standard_deviation=2.0, sample_count=100) == pytest.approx(1.2)
    assert variance_six_se_bound(variance=4.0, sample_count=101) == pytest.approx(
        24.0 * math.sqrt(0.02)
    )
    assert covariance_six_se_bound(
        first_standard_deviation=1.0,
        second_standard_deviation=0.3,
        sample_count=101,
    ) == pytest.approx(0.18)
    point = (30.0, -4.0)
    yaw = 0.04
    displacement = yaw_displacement_xy(point, yaw)
    displaced_norm = math.hypot(*displacement)
    assert displaced_norm == pytest.approx(2.0 * math.hypot(*point) * math.sin(abs(yaw) / 2.0))
    assert timestamp_displacement_xy((4.0, -2.0), 0.25) == (-1.0, 0.5)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: reference_truth(
                reference_latent_state(
                    "constant-velocity-ci-smoke-v1",
                    "test",
                    np.full((3, 4), 0.5),
                ),
                frame_count=0,
                frame_period_s=0.1,
            ),
            "frame_count",
        ),
        (
            lambda: independent_fault_stream_seed(data_master_seed=-1, sequence_id="sequence"),
            "unsigned 128-bit",
        ),
        (
            lambda: independent_dropout_mask(np.asarray([0.5]), 1.5),
            "probability",
        ),
        (
            lambda: affine_squared_loss_moments(np.zeros((3, 4)), np.zeros(2)),
            "matrix",
        ),
        (
            lambda: equal_sequence_population_moments(()),
            "at least one",
        ),
        (
            lambda: variance_six_se_bound(variance=1.0, sample_count=1),
            "at least two",
        ),
        (
            lambda: reference_latent_state(
                "constant-velocity-ci-smoke-v1",
                "test",
                np.zeros((2, 4)),
            ),
            "shape",
        ),
        (
            lambda: reference_latent_state(
                "constant-velocity-ci-smoke-v1",
                "test",
                np.full((3, 4), np.nan),
            ),
            r"\[0, 1\)",
        ),
        (
            lambda: reference_latent_state(  # type: ignore[arg-type]
                "unknown-profile",
                "test",
                np.zeros((3, 4)),
            ),
            "unknown profile",
        ),
        (
            lambda: reference_truth(
                ReferenceLatentState(np.zeros((1, 2)), np.zeros((1, 2))),
                frame_count=1,
                frame_period_s=0.0,
            ),
            "frame_period",
        ),
        (
            lambda: reference_truth(
                ReferenceLatentState(np.zeros((1, 3)), np.zeros((1, 3))),
                frame_count=1,
                frame_period_s=0.1,
            ),
            "aligned",
        ),
        (
            lambda: reference_truth(
                ReferenceLatentState(
                    np.asarray([[np.nan, 0.0]]),
                    np.zeros((1, 2)),
                ),
                frame_count=1,
                frame_period_s=0.1,
            ),
            "finite",
        ),
        (
            lambda: reference_eligibility_mask(
                np.zeros((2, 2)),
                x_min_m=0.0,
                x_max_m=1.0,
                abs_y_max_m=1.0,
                camera_half_fov_rad=0.5,
            ),
            "finite",
        ),
        (
            lambda: reference_eligibility_mask(
                np.zeros((1, 1, 2)),
                x_min_m=math.nan,
                x_max_m=1.0,
                abs_y_max_m=1.0,
                camera_half_fov_rad=0.5,
            ),
            "finite",
        ),
        (
            lambda: reference_eligibility_mask(
                np.zeros((1, 1, 2)),
                x_min_m=2.0,
                x_max_m=1.0,
                abs_y_max_m=1.0,
                camera_half_fov_rad=0.5,
            ),
            "spatial",
        ),
        (
            lambda: reference_eligibility_mask(
                np.zeros((1, 1, 2)),
                x_min_m=0.0,
                x_max_m=1.0,
                abs_y_max_m=1.0,
                camera_half_fov_rad=math.pi,
            ),
            "camera_half_fov",
        ),
        (
            lambda: independent_fault_stream_seed(
                data_master_seed=1,
                sequence_id="",
            ),
            "nonempty",
        ),
        (
            lambda: independent_fault_uniforms(
                data_master_seed=1,
                sequence_id="sequence",
                frame_count=0,
            ),
            "frame_count",
        ),
        (
            lambda: independent_dropout_mask(np.zeros((1, 1)), 0.5),
            "vector",
        ),
        (
            lambda: independent_dropout_mask(np.asarray([-0.1]), 0.5),
            r"\[0, 1\)",
        ),
        (
            lambda: affine_squared_loss_moments(np.zeros((2, 4)), np.zeros(3)),
            "bias",
        ),
        (
            lambda: affine_squared_loss_moments(np.zeros((2, 0)), np.zeros(2)),
            "positive finite",
        ),
        (
            lambda: affine_squared_loss_moments(
                np.zeros((2, 2)),
                np.asarray([np.nan, 0.0]),
            ),
            "bias must be finite",
        ),
        (
            lambda: affine_signed_contrast_moments(
                np.zeros((2, 2)),
                np.zeros(2),
                np.zeros((2, 3)),
                np.zeros(2),
            ),
            "same shape",
        ),
        (
            lambda: equal_sequence_population_moments(((),)),
            "eligible row",
        ),
        (
            lambda: mean_six_se_bound(
                standard_deviation=0.0,
                sample_count=1,
            ),
            "standard_deviation",
        ),
        (
            lambda: mean_six_se_bound(
                standard_deviation=1.0,
                sample_count=0,
            ),
            "sample_count",
        ),
        (
            lambda: variance_six_se_bound(
                variance=0.0,
                sample_count=2,
            ),
            "variance",
        ),
        (
            lambda: covariance_six_se_bound(
                first_standard_deviation=0.0,
                second_standard_deviation=1.0,
                sample_count=2,
            ),
            "standard deviations",
        ),
        (
            lambda: covariance_six_se_bound(
                first_standard_deviation=1.0,
                second_standard_deviation=1.0,
                sample_count=1,
            ),
            "at least two",
        ),
        (
            lambda: yaw_displacement_xy((1.0,), 0.1),
            "two coordinates",
        ),
        (
            lambda: yaw_displacement_xy((1.0, math.nan), 0.1),
            "finite",
        ),
        (
            lambda: timestamp_displacement_xy((1.0,), 0.1),
            "two coordinates",
        ),
        (
            lambda: timestamp_displacement_xy((1.0, math.nan), 0.1),
            "finite",
        ),
    ],
)
def test_reference_rejects_invalid_inputs(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        assert callable(call)
        call()
