"""Deterministic named data streams for the frozen v0.1 RNG contract."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import numpy.typing as npt

type DataStreamName = Literal["latent", "camera", "lidar", "fault"]
type FloatArray = npt.NDArray[np.float64]

_DOMAIN = b"fusion-fault-bench/rng/v1"
_UINT32_LIMIT = 2**32
_UINT128_LIMIT = 2**128
_DATA_STREAM_NAMES = frozenset({"latent", "camera", "lidar", "fault"})


def _utf8_field(value: str, *, field_name: str) -> bytes:
    if not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    if len(encoded) >= _UINT32_LIMIT:
        raise ValueError(f"{field_name} is too long for uint32 framing")
    return encoded


def derive_stream_seed(
    *,
    data_master_seed: int,
    stream_name: DataStreamName,
    sequence_id: str,
) -> int:
    """Derive one unsigned 128-bit PCG64DXSM seed from a named sequence stream."""

    if type(data_master_seed) is not int or not 0 <= data_master_seed < _UINT128_LIMIT:
        raise ValueError("data_master_seed must be an unsigned 128-bit integer")
    if stream_name not in _DATA_STREAM_NAMES:
        raise ValueError(f"unknown v0.1 data stream: {stream_name!r}")

    stream_bytes = _utf8_field(stream_name, field_name="stream_name")
    sequence_bytes = _utf8_field(sequence_id, field_name="sequence_id")
    payload = b"".join(
        (
            _DOMAIN,
            b"\x00",
            data_master_seed.to_bytes(16, byteorder="big", signed=False),
            len(stream_bytes).to_bytes(4, byteorder="big", signed=False),
            stream_bytes,
            len(sequence_bytes).to_bytes(4, byteorder="big", signed=False),
            sequence_bytes,
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], byteorder="big")


def draw_standard_normal_xy(
    *,
    data_master_seed: int,
    stream_name: DataStreamName,
    sequence_id: str,
    object_frame_count: int,
) -> FloatArray:
    """Draw the exact row-major float64 ``(object_frame_count, 2)`` normal array."""

    if type(object_frame_count) is not int or object_frame_count <= 0:
        raise ValueError("object_frame_count must be a positive integer")
    seed = derive_stream_seed(
        data_master_seed=data_master_seed,
        stream_name=stream_name,
        sequence_id=sequence_id,
    )
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    return generator.standard_normal(
        size=(object_frame_count, 2),
        dtype=np.float64,
    )
