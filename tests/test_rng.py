from __future__ import annotations

import numpy as np
import pytest

from fusion_fault_bench.rng import derive_stream_seed, draw_standard_normal_xy

SEQUENCE_ID = "analytic:one-object-static-local-error-gaussian-v1:000000"


@pytest.mark.parametrize(
    ("stream_name", "expected_seed"),
    [
        ("latent", 0x87A8ECDBD577CAE88FFC65D72B9DE155),
        ("camera", 0xD323E2933299EE0D031CC2817B5F2B06),
        ("lidar", 0xBF39E5D2AFBA277F452129FF8843B660),
        ("fault", 0x59D324BBB5D1E2FC76B07F2D372A81CE),
    ],
)
def test_renamed_sequence_stream_seed_goldens(stream_name, expected_seed: int) -> None:
    assert (
        derive_stream_seed(
            data_master_seed=1729,
            stream_name=stream_name,
            sequence_id=SEQUENCE_ID,
        )
        == expected_seed
    )


@pytest.mark.parametrize(
    ("stream_name", "expected_bits"),
    [
        ("camera", [0x3FFBB473C1FC6729, 0x3FD24F20987CD2C7]),
        ("lidar", [0x3FDEB3B08EC45829, 0xBFFE0570BC40F083]),
    ],
)
def test_renamed_sequence_first_draw_float64_bit_goldens(
    stream_name,
    expected_bits: list[int],
) -> None:
    values = draw_standard_normal_xy(
        data_master_seed=1729,
        stream_name=stream_name,
        sequence_id=SEQUENCE_ID,
        object_frame_count=1,
    )

    assert values.dtype == np.float64
    assert values.shape == (1, 2)
    assert values.view(np.uint64).ravel().tolist() == expected_bits


def test_vectorized_draw_preserves_the_first_row_and_is_reproducible() -> None:
    first = draw_standard_normal_xy(
        data_master_seed=1729,
        stream_name="camera",
        sequence_id=SEQUENCE_ID,
        object_frame_count=3,
    )
    second = draw_standard_normal_xy(
        data_master_seed=1729,
        stream_name="camera",
        sequence_id=SEQUENCE_ID,
        object_frame_count=3,
    )

    assert np.array_equal(first, second)
    assert first[0].view(np.uint64).tolist() == [
        0x3FFBB473C1FC6729,
        0x3FD24F20987CD2C7,
    ]


@pytest.mark.parametrize("data_master_seed", [-1, 2**128, True, 1.5])
def test_seed_derivation_rejects_non_uint128_master_seed(data_master_seed) -> None:
    with pytest.raises(ValueError, match="unsigned 128-bit"):
        derive_stream_seed(
            data_master_seed=data_master_seed,
            stream_name="camera",
            sequence_id=SEQUENCE_ID,
        )


@pytest.mark.parametrize("stream_name", ["", "thermal"])
def test_seed_derivation_rejects_unknown_stream(stream_name) -> None:
    with pytest.raises(ValueError):
        derive_stream_seed(
            data_master_seed=1729,
            stream_name=stream_name,
            sequence_id=SEQUENCE_ID,
        )


@pytest.mark.parametrize("sequence_id", ["", "\ud800"])
def test_seed_derivation_rejects_invalid_sequence_id(sequence_id: str) -> None:
    with pytest.raises(ValueError):
        derive_stream_seed(
            data_master_seed=1729,
            stream_name="camera",
            sequence_id=sequence_id,
        )


@pytest.mark.parametrize("object_frame_count", [0, -1, True, 1.5])
def test_draw_rejects_invalid_object_frame_count(object_frame_count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        draw_standard_normal_xy(
            data_master_seed=1729,
            stream_name="camera",
            sequence_id=SEQUENCE_ID,
            object_frame_count=object_frame_count,
        )
