"""Array type aliases shared by every module of the library."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]

__all__ = ["FloatArray", "IntArray"]
