"""Record of the loss and metrics of a training, epoch after epoch."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from mlp.errors import ModelFileError


def _optional_epoch(data: Mapping[str, Any], key: str) -> int | None:
    """Return ``data[key]``, a positive integer or None when absent."""
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"expected a positive integer or null for {key!r}, "
            f"received {value!r}"
        )
    return value


@dataclass
class History:
    """Per-epoch values of the loss and metrics, on train and validation.

    Both dictionaries map a name (``"loss"``, ``"accuracy"``...) to one
    value per epoch. `valid` stays empty when no validation data was
    given to the training.

    `best_epoch` (1-based) is the best epoch seen by the early stopping,
    None without early stopping. `stopped_epoch` is the epoch at which
    the early stopping ended the training, None when it did not.
    """

    train: dict[str, list[float]] = field(default_factory=dict)
    valid: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int | None = None
    stopped_epoch: int | None = None

    @property
    def epochs(self) -> int:
        """Return the number of epochs recorded."""
        return len(self.train.get("loss", []))

    def append(
        self,
        train: Mapping[str, float],
        valid: Mapping[str, float] | None = None,
    ) -> None:
        """Record the values of one epoch."""
        for name, value in train.items():
            self.train.setdefault(name, []).append(float(value))
        for name, value in (valid or {}).items():
            self.valid.setdefault(name, []).append(float(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serializable copy of the history."""
        return {
            "train": {name: list(v) for name, v in self.train.items()},
            "valid": {name: list(v) for name, v in self.valid.items()},
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "History":
        """Rebuild a history from the output of to_dict().

        `best_epoch` and `stopped_epoch` default to None, as in the
        files written before early stopping existed. Raise
        ModelFileError when `data` does not have that layout.
        """
        try:
            return cls(
                train={str(k): [float(x) for x in v]
                       for k, v in data["train"].items()},
                valid={str(k): [float(x) for x in v]
                       for k, v in data["valid"].items()},
                best_epoch=_optional_epoch(data, "best_epoch"),
                stopped_epoch=_optional_epoch(data, "stopped_epoch"),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ModelFileError(
                "expected a history of the form "
                "{'train': {name: [float]}, 'valid': {name: [float]}, "
                "'best_epoch': int | null, 'stopped_epoch': int | null}, "
                f"received: {e!r}"
            ) from e


__all__ = ["History"]
