"""The path the subject requires: softmax output + categorical CE.

Trained on the real dataset, split and scaled the way ``train.py`` does,
with one-hot targets of shape ``(m, 2)``. The sigmoid + binary CE path
is trained on the same split as a reference: both must learn equally.
"""

from pathlib import Path

import numpy as np
import pytest

from mlp.data import LABELS, read_dataset, split_dataset, to_arrays
from mlp.history import History
from mlp.layers import DenseLayer
from mlp.losses import binary_crossentropy
from mlp.network import Model
from mlp.preprocessing import fit_scaler, one_hot, transform
from mlp.types import FloatArray, IntArray

DATASET = Path(__file__).resolve().parent.parent / "data.csv"
EPOCHS = 20
SEED = 14
MIN_ACCURACY = 0.93

Split = tuple[FloatArray, IntArray, FloatArray, IntArray]


@pytest.fixture(scope="module")
def data() -> Split:
    """Return ``(x_train, labels_train, x_valid, labels_valid)``.

    80/20 split of data.csv, scaled with statistics of the training set
    only.
    """
    x, labels = to_arrays(read_dataset(DATASET))
    train, valid = split_dataset(len(x), 0.8, np.random.default_rng(42))
    scaler = fit_scaler("standard", x[train])
    return (transform(scaler, x[train]), labels[train],
            transform(scaler, x[valid]), labels[valid])


def train(output: str, loss: str, data: Split) -> tuple[Model, History]:
    """Train a 24-24 ReLU network ending with `output` on `data`."""
    x_train, labels_train, x_valid, labels_valid = data
    units = 1 if loss == "binaryCrossentropy" else len(LABELS)

    def targets(labels: IntArray) -> FloatArray:
        if units == 1:
            return labels.reshape(-1, 1).astype(np.float64)
        return one_hot(labels, units)

    model = Model([
        DenseLayer(24, "relu", "heUniform"),
        DenseLayer(24, "relu", "heUniform"),
        DenseLayer(units, output, "heUniform"),
    ])
    model.compile(loss, metrics=["accuracy", "f1"], learning_rate=0.0314)
    history = model.fit(
        x_train, targets(labels_train),
        validation_data=(x_valid, targets(labels_valid)),
        epochs=EPOCHS, batch_size=8, seed=SEED, verbose=False,
    )
    return model, history


@pytest.fixture(scope="module")
def softmax(data: Split) -> tuple[Model, History]:
    """Return the softmax + CCE model trained on `data`."""
    return train("softmax", "categoricalCrossentropy", data)


@pytest.fixture(scope="module")
def sigmoid(data: Split) -> tuple[Model, History]:
    """Return the sigmoid + BCE model trained on `data`."""
    return train("sigmoid", "binaryCrossentropy", data)


def test_softmax_outputs_a_distribution(
    softmax: tuple[Model, History],
    data: Split,
) -> None:
    """Two columns of probabilities summing to 1."""
    proba = softmax[0].predict_proba(data[2])
    assert proba.shape == (len(data[2]), len(LABELS))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_softmax_loss_decreases(softmax: tuple[Model, History]) -> None:
    """Both losses end lower than after the first epoch."""
    history = softmax[1]
    assert history.train["loss"][-1] < history.train["loss"][0]
    assert history.valid["loss"][-1] < history.valid["loss"][0]


def test_softmax_accuracy(softmax: tuple[Model, History]) -> None:
    """The validation accuracy reaches the expected level."""
    assert softmax[1].valid["accuracy"][-1] > MIN_ACCURACY


def test_softmax_matches_sigmoid(
    softmax: tuple[Model, History],
    sigmoid: tuple[Model, History],
) -> None:
    """Both output layers learn the same problem equally well."""
    for name in ("accuracy", "f1"):
        gap = abs(softmax[1].valid[name][-1] - sigmoid[1].valid[name][-1])
        assert gap < 0.05, name


def test_softmax_binary_crossentropy_is_its_categorical_one(
    softmax: tuple[Model, History],
    data: Split,
) -> None:
    """On 2 classes, the BCE of the M column equals the CCE.

    This is what predict.py relies on to report the binary cross-entropy
    of a softmax model.
    """
    model, history = softmax
    x_valid, labels_valid = data[2], data[3]
    proba = model.predict_proba(x_valid)
    bce = binary_crossentropy(
        labels_valid.reshape(-1, 1).astype(np.float64), proba[:, -1:])
    assert bce == pytest.approx(history.valid["loss"][-1], rel=1e-6)
