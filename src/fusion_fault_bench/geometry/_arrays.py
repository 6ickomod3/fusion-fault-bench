"""Internal construction of float64 arrays that cannot be made writeable."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


def immutable_float64_copy(value: npt.ArrayLike) -> FloatArray:
    """Copy into an immutable-bytes buffer, preserving the observed shape."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)
    assert not result.flags.writeable
    return result
