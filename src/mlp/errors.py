"""Exception hierarchy of the library.

The library raises, it never prints: only the ``main()`` of the scripts
catches. Every message states what was expected *and* what was received,
and wrapping an underlying exception is done with ``raise ... from e`` so
that the cause is preserved.
"""


class MLPError(Exception):
    """Base class of every error raised by the library."""


class ConfigurationError(MLPError, ValueError):
    """Unknown name, invalid combination or invalid hyperparameter."""


class ShapeError(MLPError, ValueError):
    """Inconsistent dimensions between X, y and the layers."""


class NotBuiltError(MLPError, RuntimeError):
    """Operation requiring a built or compiled model (forward, summary)."""


class TrainingDivergedError(MLPError, ArithmeticError):
    """The loss became NaN or infinite during training."""


class DatasetError(MLPError):
    """Malformed CSV file or unknown label."""


class ModelFileError(MLPError):
    """Invalid model file: bad JSON, unknown version or wrong shapes."""


__all__ = [
    "MLPError",
    "ConfigurationError",
    "ShapeError",
    "NotBuiltError",
    "TrainingDivergedError",
    "DatasetError",
    "ModelFileError",
]
