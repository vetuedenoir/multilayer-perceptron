"""Record of the loss and metrics of a training, epoch after epoch."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from mlp.errors import ModelFileError


@dataclass
class History:
    """Per-epoch values of the loss and metrics, on train and validation.

    Both dictionaries map a name (``"loss"``, ``"accuracy"``...) to one
    value per epoch. `valid` stays empty when no validation data was
    given to the training.
    """

    train: dict[str, list[float]] = field(default_factory=dict)
    valid: dict[str, list[float]] = field(default_factory=dict)

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

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """Return a JSON serializable copy of the history."""
        return {
            "train": {name: list(v) for name, v in self.train.items()},
            "valid": {name: list(v) for name, v in self.valid.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "History":
        """Rebuild a history from the output of to_dict().

        Raise ModelFileError when `data` does not have that layout.
        """
        try:
            return cls(
                train={str(k): [float(x) for x in v]
                       for k, v in data["train"].items()},
                valid={str(k): [float(x) for x in v]
                       for k, v in data["valid"].items()},
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ModelFileError(
                "expected a history of the form "
                "{'train': {name: [float]}, 'valid': {name: [float]}}, "
                f"received: {e!r}"
            ) from e


__all__ = ["History"]
