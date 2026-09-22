"""Tests of the model: life cycle, training loop and helpers."""

import numpy as np
import pytest
from conftest import numerical_gradient, relative_error

from mlp.errors import (
    ConfigurationError,
    NotBuiltError,
    ShapeError,
    TrainingDivergedError,
)
from mlp.history import History
from mlp.layers import DenseLayer
from mlp.losses import BINARY_CROSSENTROPY, CATEGORICAL_CROSSENTROPY
from mlp.network import (
    Model,
    check_targets,
    format_epoch_line,
    iter_batches,
    to_class_labels,
)
from mlp.types import FloatArray

TOLERANCE = 1e-7
FEATURES = 4


def binary_model() -> Model:
    """Return a small sigmoid model, not built."""
    return Model([
        DenseLayer(5, "relu", "heUniform"),
        DenseLayer(3, "relu", "heUniform"),
        DenseLayer(1, "sigmoid", "glorotUniform"),
    ])


def softmax_model(classes: int = 2) -> Model:
    """Return a small softmax model, not built."""
    return Model([
        DenseLayer(5, "leaky_relu", "heUniform"),
        DenseLayer(classes, "softmax", "glorotUniform"),
    ])


def separable_data(
    samples: int = 60,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Return (x, y) linearly separable, y of shape (m, 1)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples, FEATURES))
    y = (x[:, :1] + x[:, 1:2] > 0).astype(np.float64)
    return x, y


# Life cycle ----------------------------------------------------------------


def test_fit_before_compile_raises() -> None:
    """Without a loss there is nothing to minimise."""
    x, y = separable_data()
    with pytest.raises(NotBuiltError, match="compile"):
        binary_model().fit(x, y, epochs=1, batch_size=8, verbose=False)


def test_summary_before_build_raises() -> None:
    """Before build, the shapes of the weights are unknown."""
    with pytest.raises(NotBuiltError, match="built"):
        binary_model().summary()


def test_forward_before_build_raises() -> None:
    """The model refuses to run on layers without weights."""
    with pytest.raises(NotBuiltError):
        binary_model().forward(np.zeros((1, FEATURES)))


def test_summary_after_build() -> None:
    """Summary counts the weights and biases of every layer."""
    model = binary_model()
    model.build(FEATURES, seed=0)
    summary = model.summary()
    total = (5 * 4 + 5) + (3 * 5 + 3) + (1 * 3 + 1)
    assert f"total parameters: {total}" in summary
    assert "params=[(5, 4), (1, 5)]" in summary


def test_build_is_idempotent() -> None:
    """A second build keeps the existing weights."""
    model = binary_model()
    model.build(FEATURES, seed=0)
    before = [p.copy() for layer in model.layers for p in layer.params()]
    model.build(FEATURES, seed=1)
    after = [p for layer in model.layers for p in layer.params()]
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_compile_twice() -> None:
    """Compile does not mutate the layers, so it can be called again."""
    model = binary_model()
    model.compile("binaryCrossentropy", metrics=["accuracy"])
    model.compile("BinaryCrossentropy", metrics=["accuracy", "f1"])
    assert model.metrics == ["accuracy", "f1"]
    x, y = separable_data()
    model.fit(x, y, epochs=1, batch_size=8, seed=0, verbose=False)


def test_second_fit_resumes_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second fit never rebuilds, hence never reinitialises, a layer."""
    model = binary_model()
    model.compile("binaryCrossentropy", learning_rate=0.05)
    x, y = separable_data()
    first = model.fit(x, y, epochs=5, batch_size=8, seed=0, verbose=False)

    def forbidden(*args: object) -> None:
        raise AssertionError("build() called on an already built layer")

    for layer in model.layers:
        monkeypatch.setattr(layer, "build", forbidden)
    second = model.fit(x, y, epochs=5, batch_size=8, seed=0, verbose=False)
    assert second.train["loss"][0] < first.train["loss"][0]


def test_softmax_on_one_unit_raises() -> None:
    """Softmax over a single unit is always 1."""
    model = Model([DenseLayer(3, "relu"), DenseLayer(1, "softmax")])
    with pytest.raises(ConfigurationError, match="at least 2 units"):
        model.compile("categoricalCrossentropy")


def test_softmax_with_binary_crossentropy_raises() -> None:
    """The pairing softmax + BCE is refused at compile time."""
    with pytest.raises(ConfigurationError, match="softmax"):
        softmax_model().compile("binaryCrossentropy")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"loss": "mse"}, "loss"),
        ({"loss": "binaryCrossentropy", "optimizer": "adam"}, "optimizer"),
        ({"loss": "binaryCrossentropy", "metrics": ["auc"]}, "metric"),
    ],
)
def test_compile_unknown_names_raise(
    kwargs: dict[str, str],
    match: str,
) -> None:
    """Every unknown name goes through the registry error."""
    with pytest.raises(ConfigurationError, match=match):
        binary_model().compile(**kwargs)  # type: ignore[arg-type]


def test_compile_empty_model_raises() -> None:
    """An empty model has no output layer to check."""
    with pytest.raises(ConfigurationError):
        Model().compile("binaryCrossentropy")


def test_add_rejects_non_layers() -> None:
    """Only objects implementing the Layer protocol are accepted."""
    with pytest.raises(TypeError):
        Model().add("dense")  # type: ignore[arg-type]


def test_len() -> None:
    """len() is the number of layers."""
    assert len(binary_model()) == 3


# Training ------------------------------------------------------------------


def test_fit_learns_a_separable_problem() -> None:
    """The loss decreases and the accuracy ends high."""
    model = binary_model()
    model.compile("binaryCrossentropy", metrics=["accuracy", "f1"],
                  learning_rate=0.1)
    x, y = separable_data(200)
    x_valid, y_valid = separable_data(50, seed=1)
    history = model.fit(x, y, epochs=30, batch_size=8, seed=0,
                        validation_data=(x_valid, y_valid), verbose=False)
    assert history.epochs == 30
    assert set(history.valid) == {"loss", "accuracy", "f1"}
    assert history.train["loss"][-1] < history.train["loss"][0]
    assert history.valid["accuracy"][-1] > 0.9


def test_softmax_model_learns() -> None:
    """The fused softmax + CCE path trains on one-hot targets."""
    model = softmax_model()
    model.compile("categoricalCrossentropy", learning_rate=0.1)
    x, y = separable_data(200)
    y_one_hot = np.hstack([1.0 - y, y])
    history = model.fit(x, y_one_hot, epochs=30, batch_size=8, seed=0,
                        verbose=False)
    assert history.train["loss"][-1] < history.train["loss"][0]
    assert history.train["accuracy"][-1] > 0.9


def test_fit_is_reproducible_with_a_seed() -> None:
    """Same seed, same initialisation, same batches, same history."""
    x, y = separable_data()
    histories = []
    for _ in range(2):
        model = binary_model()
        model.compile("binaryCrossentropy")
        histories.append(model.fit(x, y, epochs=3, batch_size=8, seed=42,
                                   verbose=False))
    assert histories[0] == histories[1]


def test_divergence_raises() -> None:
    """A NaN or infinite loss stops the training with a clear error."""
    model = binary_model()
    model.compile("binaryCrossentropy", learning_rate=1e300)
    x, y = separable_data()
    with pytest.raises(TrainingDivergedError, match="epoch 1"):
        with np.errstate(all="ignore"):
            model.fit(1e300 * x, y, epochs=3, batch_size=8, seed=0,
                      verbose=False)


def test_fit_rejects_wrong_targets() -> None:
    """The BCE needs a (m, 1) target."""
    model = binary_model()
    model.compile("binaryCrossentropy")
    x, y = separable_data()
    with pytest.raises(ShapeError):
        model.fit(x, np.hstack([y, y]), epochs=1, batch_size=8,
                  verbose=False)
    with pytest.raises(ShapeError, match="samples"):
        model.fit(x, y[:-1], epochs=1, batch_size=8, verbose=False)


@pytest.mark.parametrize("name", ["epochs", "batch_size"])
def test_fit_rejects_non_positive_hyperparameters(name: str) -> None:
    """Epochs and batch size must be positive integers."""
    model = binary_model()
    model.compile("binaryCrossentropy")
    x, y = separable_data()
    sizes = {"epochs": 1, "batch_size": 8, name: 0}
    with pytest.raises(ConfigurationError, match=name):
        model.fit(x, y, epochs=sizes["epochs"],
                  batch_size=sizes["batch_size"], verbose=False)


@pytest.mark.parametrize(
    ("output", "loss"),
    [("sigmoid", "binaryCrossentropy"), ("softmax", "categoricalCrossentropy"),
     ("sigmoid", "categoricalCrossentropy")],
)
def test_backward_matches_finite_differences(output: str, loss: str) -> None:
    """The gradients of the whole network match finite differences.

    Covers both fused paths and a non fused one (sigmoid on 2 units with
    the categorical cross-entropy).
    """
    units = 1 if loss == "binaryCrossentropy" else 2
    model = Model([
        DenseLayer(3, "sigmoid", "glorotNormal"),
        DenseLayer(units, output, "glorotNormal"),
    ])
    model.compile(loss)
    model.build(FEATURES, seed=0)
    x, y = separable_data(6)
    if units == 2:
        y = np.hstack([1.0 - y, y])

    compiled_loss = (BINARY_CROSSENTROPY if units == 1
                     else CATEGORICAL_CROSSENTROPY)
    grad_fn = model._output_grad
    assert grad_fn is not None
    model.backward(grad_fn(y, model.forward(x)), from_loss=model._fused)

    for layer in model.layers:
        for param, grad in zip(layer.params(), layer.grads()):
            numeric = numerical_gradient(
                lambda _: compiled_loss.forward(y, model.forward(x)), param)
            assert relative_error(grad, numeric) < TOLERANCE


def test_predict_and_evaluate() -> None:
    """predict_classes gives labels, evaluate the loss and metrics."""
    model = binary_model()
    model.compile("binaryCrossentropy", metrics=["accuracy", "precision"])
    model.build(FEATURES, seed=0)
    x, y = separable_data()
    labels = model.predict_classes(x)
    assert labels.shape == (x.shape[0],)
    assert model.predict_proba(x).shape == (x.shape[0], 1)
    loss, metrics = model.evaluate(x, y)
    assert loss > 0.0
    assert set(metrics) == {"accuracy", "precision"}


# Pure helpers --------------------------------------------------------------


def test_iter_batches_covers_every_sample_once() -> None:
    """The batches are a partition of the dataset."""
    x = np.arange(10, dtype=np.float64).reshape(10, 1)
    batches = list(iter_batches(x, x, 3, np.random.default_rng(0)))
    assert [len(b[0]) for b in batches] == [3, 3, 3, 1]
    seen = np.concatenate([b[0] for b in batches]).ravel()
    assert sorted(seen) == list(range(10))
    np.testing.assert_array_equal(
        np.concatenate([b[0] for b in batches]),
        np.concatenate([b[1] for b in batches]),
    )


def test_iter_batches_without_rng_keeps_the_order() -> None:
    """No generator, no shuffle."""
    x = np.arange(5, dtype=np.float64).reshape(5, 1)
    first, _ = next(iter_batches(x, x, 2))
    np.testing.assert_array_equal(first.ravel(), [0.0, 1.0])


def test_to_class_labels() -> None:
    """One column is thresholded, several are argmaxed."""
    np.testing.assert_array_equal(
        to_class_labels(np.array([[0.2], [0.5], [0.9]])), [0, 1, 1])
    np.testing.assert_array_equal(
        to_class_labels(np.array([[0.7, 0.3], [0.1, 0.9]])), [0, 1])
    with pytest.raises(ShapeError):
        to_class_labels(np.array([0.2, 0.8]))


def test_check_targets() -> None:
    """Each loss gets the target shape and output units it needs."""
    check_targets(BINARY_CROSSENTROPY, np.zeros((4, 1)), 1)
    check_targets(CATEGORICAL_CROSSENTROPY, np.zeros((4, 3)), 3)
    with pytest.raises(ShapeError, match="1 unit"):
        check_targets(BINARY_CROSSENTROPY, np.zeros((4, 1)), 2)
    with pytest.raises(ShapeError, match="one-hot"):
        check_targets(CATEGORICAL_CROSSENTROPY, np.zeros((4, 1)), 1)
    with pytest.raises(ShapeError, match="3 units"):
        check_targets(CATEGORICAL_CROSSENTROPY, np.zeros((4, 3)), 2)


def test_format_epoch_line() -> None:
    """Train values, then validation ones after a bar."""
    line = format_epoch_line(3, 60, {"loss": 0.21041, "accuracy": 0.5},
                             {"loss": 0.1987})
    assert line == ("epoch  3/60 - train_loss: 0.2104, "
                    "train_accuracy: 0.5000 | valid_loss: 0.1987")
    assert "|" not in format_epoch_line(1, 1, {"loss": 1.0})


def test_history_round_trip() -> None:
    """to_dict and from_dict are inverse of each other."""
    history = History()
    history.append({"loss": 0.5, "accuracy": 0.8}, {"loss": 0.6})
    history.append({"loss": 0.4, "accuracy": 0.9}, {"loss": 0.5})
    assert history.epochs == 2
    assert History.from_dict(history.to_dict()) == history
