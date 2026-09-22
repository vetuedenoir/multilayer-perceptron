#!/usr/bin/env python3
"""Load a trained model, predict a dataset and evaluate the prediction.

Third of the three programs of the subject. Nothing about the network is
declared here: the architecture, the weights and the scaler all come
from the model file written by ``train.py``. The prediction is evaluated
with the binary cross-entropy, as the subject requires, plus the metrics
the model was compiled with.
"""

import argparse
import sys
from typing import Final, Sequence

import numpy as np

from mlp.data import LABELS, read_dataset, to_arrays
from mlp.errors import MLPError, ModelFileError, ShapeError
from mlp.losses import binary_crossentropy
from mlp.metrics import METRICS
from mlp.preprocessing import transform
from mlp.registry import get_from_registry
from mlp.serialization import load_model
from mlp.types import FloatArray

PROG: Final[str] = "predict.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line, `argv` defaulting to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Evaluate a trained model on a dataset.",
    )
    parser.add_argument("--model", default="model.json",
                        help="the model file written by train.py "
                             "(default: %(default)s)")
    parser.add_argument("--dataset", default="data_valid.csv",
                        help="the dataset to predict "
                             "(default: %(default)s)")
    return parser.parse_args(argv)


def positive_probability(proba: FloatArray) -> FloatArray:
    """Return the probability of the positive class, as ``(m, 1)``.

    A single output column already is that probability (sigmoid). With
    two columns (softmax), the positive class is the second one, the
    first being its complement.

    Raise ShapeError on more than two columns: the binary cross-entropy
    has no meaning there.
    """
    if proba.ndim != 2 or proba.shape[1] not in (1, 2):
        raise ShapeError(
            "expected the output of a binary classifier, of shape (m, 1) "
            f"or (m, 2), received {proba.shape}"
        )
    return proba[:, -1:]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program and return its exit status."""
    args = parse_args(argv)
    try:
        model, scaler, labels, _, _ = load_model(args.model)
        if tuple(labels) != LABELS:
            raise ModelFileError(
                f"expected a model trained on the classes {LABELS}, "
                f"received one trained on {tuple(labels)}"
            )
        x, y = to_arrays(read_dataset(args.dataset))
        x = transform(scaler, x)

        proba = model.predict_proba(x)
        y_pred = model.predict_classes(x)
        loss = binary_crossentropy(
            y.reshape(-1, 1).astype(np.float64), positive_probability(proba))
        scores = {name: get_from_registry(METRICS, name, "metric")(y, y_pred)
                  for name in model.metrics}
    except (MLPError, OSError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1

    print(f"samples: {len(y)}")
    print(f"binary cross-entropy: {loss:.4f}")
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
