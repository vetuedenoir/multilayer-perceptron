"""Tests of the optimizers: update rules, state, registry, integration."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from mlp.data import read_dataset, split_dataset, to_arrays
from mlp.errors import ConfigurationError, ModelFileError, ShapeError
from mlp.layers import DenseLayer
from mlp.network import Model
from mlp.optimizers import (
    OPTIMIZERS,
    SGD,
    Adam,
    Momentum,
    Optimizer,
    RMSprop,
    adam_step,
    make_optimizer,
    momentum_step,
    rmsprop_step,
    sgd_step,
)
from mlp.preprocessing import fit_scaler, one_hot, transform
from mlp.serialization import (
    TrainingConfig,
    load_model,
    model_from_dict,
    model_to_dict,
    save_model,
)
from mlp.types import FloatArray

DATASET = Path(__file__).resolve().parent.parent / "data.csv"
NAMES = ("sgd", "momentum", "nesterov", "rmsprop", "adam")

P = np.array([[1.0, 2.0], [3.0, 4.0]])
G = np.array([[1.0, -2.0], [2.0, 1.0]])
LR = 0.1


def arrays(*values: FloatArray) -> list[FloatArray]:
    """Return copies of `values`, to hand to a step function."""
    return [v.copy() for v in values]


# Pure update rules, against values computed by hand ------------------------


def test_sgd_step() -> None:
    """Plain SGD: p - lr * g."""
    (p,) = sgd_step([P], [G], LR)
    np.testing.assert_allclose(p, [[0.9, 2.2], [2.8, 3.9]])


@pytest.mark.parametrize(
    ("nesterov", "expected"),
    [
        (False, [[1.04, 2.1], [2.8, 3.82]]),
        (True, [[0.986, 2.19], [2.62, 3.838]]),
    ],
)
def test_momentum_step(nesterov: bool, expected: list[list[float]]) -> None:
    """Velocity 0.9 v - 0.1 g, then p + v or p + 0.9 v - 0.1 g."""
    g = np.array([[0.5, -1.0], [2.0, 0.0]])
    v0 = np.array([[0.1, 0.0], [0.0, -0.2]])
    (p,), (v,) = momentum_step([P], [g], [v0], LR, 0.9, nesterov)
    np.testing.assert_allclose(v, [[0.04, 0.1], [-0.2, -0.18]])
    np.testing.assert_allclose(p, expected)


def test_rmsprop_step() -> None:
    """s0 = g**2 keeps s = g**2: the step is lr * sign(g)."""
    (p,), (s,) = rmsprop_step([P], [G], [G * G], LR, 0.9, 0.0)
    np.testing.assert_allclose(s, G * G)
    np.testing.assert_allclose(p, P - LR * np.sign(G))


def test_adam_step() -> None:
    """Second step from zero moments, beta1 = 0.5 and beta2 = 0.75.

    m_hat = 0.5 g / 0.75 = 2/3 g and v_hat = 0.25 g**2 / 0.4375
    = 4/7 g**2, so the step is lr * sqrt(7) / 3 * sign(g).
    """
    zeros = np.zeros_like(P)
    (p,), (m,), (v,) = adam_step([P], [G], [zeros], [zeros], 2, LR,
                                 0.5, 0.75, 0.0)
    np.testing.assert_allclose(m, 0.5 * G)
    np.testing.assert_allclose(v, 0.25 * G * G)
    np.testing.assert_allclose(p, P - LR * np.sqrt(7.0) / 3.0 * np.sign(G))


def test_adam_first_step_is_lr_times_sign() -> None:
    """The bias correction makes the first step lr * sign(g)."""
    g = np.array([[1e-3, -50.0], [7.0, -0.2]])
    zeros = np.zeros_like(P)
    (p,), _, _ = adam_step([P], [g], [zeros], [zeros], 1, LR,
                           0.9, 0.999, 1e-8)
    np.testing.assert_allclose(P - p, LR * np.sign(g), rtol=1e-4)


def test_adam_step_number_starts_at_one() -> None:
    """A step t = 0 would divide by 1 - beta**0 = 0."""
    with pytest.raises(ConfigurationError, match="t >= 1"):
        adam_step([P], [G], [P], [P], 0, LR, 0.9, 0.999, 1e-8)


@pytest.mark.parametrize(
    "step",
    [
        lambda p, g, s: sgd_step(p, g, LR),
        lambda p, g, s: momentum_step(p, g, s, LR, 0.9, False),
        lambda p, g, s: momentum_step(p, g, s, LR, 0.9, True),
        lambda p, g, s: rmsprop_step(p, g, s, LR, 0.9, 1e-8),
        lambda p, g, s: adam_step(p, g, s, s, 3, LR, 0.9, 0.999, 1e-8),
    ],
    ids=NAMES,
)
def test_steps_are_pure(step: Callable[..., Any]) -> None:
    """No input array is modified."""
    params, grads, state = arrays(P), arrays(G), arrays(G * G)
    step(params, grads, state)
    assert np.array_equal(params[0], P)
    assert np.array_equal(grads[0], G)
    assert np.array_equal(state[0], G * G)


def test_steps_need_one_gradient_per_parameter() -> None:
    """A missing gradient is a shape error, not a silent zip."""
    with pytest.raises(ShapeError, match="one gradient per parameter"):
        sgd_step([P, P], [G], LR)


def test_nesterov_differs_from_momentum() -> None:
    """The look-ahead form already differs at the first step."""
    classic, nesterov = Momentum(LR), Momentum(LR, nesterov=True)
    p_classic = classic.step([P], [G])
    p_nesterov = nesterov.step([P], [G])
    assert not np.allclose(p_classic[0], p_nesterov[0])
    assert not np.allclose(classic.step(p_classic, [G])[0],
                           nesterov.step(p_nesterov, [G])[0])


@pytest.mark.parametrize("nesterov", [False, True])
def test_zero_momentum_is_sgd(nesterov: bool) -> None:
    """With momentum = 0, every step is a plain SGD step."""
    momentum, sgd = Momentum(LR, momentum=0.0, nesterov=nesterov), SGD(LR)
    p_momentum, p_sgd = [P], [P]
    for _ in range(3):
        p_momentum = momentum.step(p_momentum, [G])
        p_sgd = sgd.step(p_sgd, [G])
        np.testing.assert_allclose(p_momentum[0], p_sgd[0])


# Classes: state, convergence, validation ------------------------------------


@pytest.mark.parametrize(
    ("name", "learning_rate"),
    [("sgd", 0.1), ("momentum", 0.05), ("nesterov", 0.05),
     ("rmsprop", 0.01), ("adam", 0.1)],
)
def test_convergence_on_a_quadratic(name: str, learning_rate: float) -> None:
    """Minimise 1/2 ||p - p*||**2, whose gradient is p - p*."""
    target = np.array([[1.0, -2.0], [3.0, 0.5]])
    optimizer = make_optimizer(name, learning_rate)
    params = [np.zeros_like(target)]
    for _ in range(500):
        params = optimizer.step(params, [params[0] - target])
    np.testing.assert_allclose(params[0], target, atol=1e-2)


@pytest.mark.parametrize("name", NAMES)
def test_state_shapes_are_checked(name: str) -> None:
    """New shapes without reset() are refused; reset() allows them."""
    optimizer = make_optimizer(name, LR)
    optimizer.step([P, P], [G, G])
    if name == "sgd":
        optimizer.step([P], [G])  # stateless: anything goes
        return
    with pytest.raises(ShapeError, match="2 parameters, received 1"):
        optimizer.step([P], [G])
    with pytest.raises(ShapeError, match="shape"):
        optimizer.step([P, np.zeros(3)], [G, np.zeros(3)])
    optimizer.reset()
    optimizer.step([np.zeros(3)], [np.ones(3)])


@pytest.mark.parametrize("name", NAMES)
def test_reset_restarts_from_scratch(name: str) -> None:
    """After reset(), the same steps give the same parameters."""
    optimizer = make_optimizer(name, LR)
    first = optimizer.step([P], [G])
    optimizer.step(first, [G])
    optimizer.reset()
    assert np.array_equal(optimizer.step([P], [G])[0], first[0])


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: SGD(0.0), "learning rate"),
        (lambda: Adam(float("nan")), "learning rate"),
        (lambda: Momentum(LR, momentum=1.0), "momentum"),
        (lambda: Momentum(LR, momentum=-0.1), "momentum"),
        (lambda: Momentum(LR, nesterov=1), "nesterov"),  # type: ignore
        (lambda: RMSprop(LR, rho=0.0), "rho"),
        (lambda: RMSprop(LR, rho=1.0), "rho"),
        (lambda: RMSprop(LR, epsilon=0.0), "epsilon"),
        (lambda: Adam(LR, beta1=1.0), "beta1"),
        (lambda: Adam(LR, beta2=-0.5), "beta2"),
        (lambda: Adam(LR, epsilon="1e-8"), "epsilon"),  # type: ignore
    ],
)
def test_invalid_hyperparameters(
    factory: Callable[[], Optimizer],
    match: str,
) -> None:
    """Every hyperparameter out of its bounds is refused."""
    with pytest.raises(ConfigurationError, match=match):
        factory()


def test_zero_betas_and_momentum_are_allowed() -> None:
    """The lower bounds of momentum, beta1 and beta2 are inclusive."""
    Momentum(LR, momentum=0.0)
    Adam(LR, beta1=0.0, beta2=0.0)


# Registry ----------------------------------------------------------------


def test_registry_names() -> None:
    """Every documented name, plus the aliases, builds its optimizer."""
    assert {"sgd", "momentum", "nesterov", "rmsprop", "adam", "SGD",
            "RMSProp", "Adam"} <= set(OPTIMIZERS)
    assert make_optimizer("nesterov", LR) == Momentum(LR, nesterov=True)
    assert make_optimizer("RMSProp", LR) == RMSprop(LR)


@pytest.mark.parametrize("name", NAMES)
def test_config_rebuilds_the_optimizer(name: str) -> None:
    """get_config() then make_optimizer() gives the same optimizer."""
    config = make_optimizer(name, LR).get_config()
    assert config["name"] == name
    rebuilt = make_optimizer(config["name"], config["learning_rate"],
                             **config["hyperparameters"])
    assert rebuilt.get_config() == config


def test_unknown_hyperparameter_lists_the_accepted_ones() -> None:
    """A typo is caught with the list of what the optimizer takes."""
    with pytest.raises(ConfigurationError,
                       match="'beta_1'.*Accepted: 'beta1', 'beta2'"):
        make_optimizer("adam", LR, beta_1=0.8)
    with pytest.raises(ConfigurationError, match="Accepted: none"):
        make_optimizer("sgd", LR, momentum=0.9)


def test_unknown_optimizer() -> None:
    """An unknown name goes through the registry error."""
    with pytest.raises(ConfigurationError, match="Available values"):
        make_optimizer("adagrad", LR)


# Integration with the model ------------------------------------------------


def separable_data(samples: int = 60) -> tuple[FloatArray, FloatArray]:
    """Return (x, y), y being 1 where x0 + x1 > 0."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(samples, 4))
    return x, (x[:, :1] + x[:, 1:2] > 0).astype(np.float64)


def small_model(optimizer: str | Optimizer = "sgd") -> Model:
    """Return a compiled 5-3-1 sigmoid model."""
    model = Model([DenseLayer(5, "relu"), DenseLayer(3, "relu"),
                   DenseLayer(1, "sigmoid")])
    model.compile("binaryCrossentropy", optimizer=optimizer,
                  metrics=["accuracy"], learning_rate=0.05)
    return model


class Recorder:
    """An SGD recording the number of parameters of every step."""

    def __init__(self) -> None:
        """Wrap a plain SGD."""
        self.sgd = SGD(0.05)
        self.calls: list[list[tuple[int, ...]]] = []

    @property
    def name(self) -> str:
        """Return the name of the wrapped optimizer."""
        return self.sgd.name

    def step(self, params: Any, grads: Any) -> list[FloatArray]:
        """Record the shapes, then delegate."""
        self.calls.append([p.shape for p in params])
        return self.sgd.step(params, grads)

    def reset(self) -> None:
        """Delegate."""

    def get_config(self) -> Any:
        """Delegate."""
        return self.sgd.get_config()


def test_update_passes_every_parameter_at_once() -> None:
    """One step per batch, with the flat list in layer order."""
    recorder = Recorder()
    model = small_model(recorder)
    x, y = separable_data()
    model.fit(x, y, epochs=2, batch_size=20, seed=0, verbose=False)
    assert len(recorder.calls) == 2 * 3
    assert recorder.calls[0] == [(5, 4), (1, 5), (3, 5), (1, 3),
                                 (1, 3), (1, 1)]


def test_flat_update_is_the_per_layer_sgd() -> None:
    """With SGD, the flat update is bit for bit the per-layer one."""
    model = small_model()
    x, y = separable_data()
    model.build(4, seed=0)
    model.backward(model.forward(x) - y, from_loss=True)
    expected = [sgd_step(layer.params(), layer.grads(), 0.05)
                for layer in model.layers]
    model.update()
    for layer, params in zip(model.layers, expected):
        for actual, wanted in zip(layer.params(), params):
            assert np.array_equal(actual, wanted)


def test_compile_with_a_name_gives_a_fresh_state() -> None:
    """A name builds a new optimizer; an instance is kept as it is."""
    adam = Adam(0.01)
    model = small_model(adam)
    x, y = separable_data()
    model.fit(x, y, epochs=1, batch_size=20, seed=0, verbose=False)
    assert model.get_compile_config()["optimizer"]["name"] == "adam"
    assert adam._t == 3  # the model trained with this very instance
    model.compile("binaryCrossentropy", optimizer=adam)
    model.fit(x, y, epochs=1, batch_size=20, seed=0, verbose=False)
    assert adam._t == 6  # kept, state included
    model.compile("binaryCrossentropy", optimizer="adam")
    model.fit(x, y, epochs=1, batch_size=20, seed=0, verbose=False)
    assert adam._t == 6  # replaced by a new one


@pytest.mark.parametrize("name", NAMES)
def test_every_optimizer_trains_a_model(name: str) -> None:
    """The loss decreases with every optimizer."""
    model = small_model(make_optimizer(name, 0.01))
    x, y = separable_data()
    history = model.fit(x, y, epochs=30, batch_size=10, seed=0,
                        verbose=False)
    assert history.train["loss"][-1] < history.train["loss"][0]


# Saving ---------------------------------------------------------------------


def test_round_trip_of_an_adam_model(tmp_path: Path) -> None:
    """Hyperparameters come back, and the model evaluates the same."""
    model = small_model(Adam(0.01, beta1=0.8, epsilon=1e-7))
    x, y = separable_data()
    model.fit(x, y, epochs=5, batch_size=10, seed=0, verbose=False)
    scaler = fit_scaler("standard", x)
    training: TrainingConfig = {"epochs": 5, "batch_size": 10, "seed": 0,
                                "early_stopping": None}
    save_model(tmp_path / "model.json", model, scaler, ["B", "M"], training)
    loaded = load_model(tmp_path / "model.json").model
    assert loaded.get_compile_config()["optimizer"] == {
        "name": "adam", "learning_rate": 0.01,
        "hyperparameters": {"beta1": 0.8, "beta2": 0.999, "epsilon": 1e-7}}
    assert loaded.evaluate(x, y) == model.evaluate(x, y)


def saved_dict(optimizer: Optimizer) -> dict[str, Any]:
    """Return the model dict of a trained small model."""
    model = small_model(optimizer)
    x, y = separable_data()
    model.fit(x, y, epochs=1, batch_size=10, seed=0, verbose=False)
    return model_to_dict(model, fit_scaler("standard", x), ["B", "M"],
                         {"epochs": 1, "batch_size": 10, "seed": 0,
                          "early_stopping": None})


def test_missing_hyperparameters_take_the_defaults() -> None:
    """A file written before step 9 has none: defaults are used."""
    data = saved_dict(Momentum(0.01, momentum=0.5))
    del data["compile"]["optimizer"]["hyperparameters"]
    config = model_from_dict(data).model.get_compile_config()
    assert config["optimizer"]["hyperparameters"] == {"momentum": 0.9,
                                                      "nesterov": False}


@pytest.mark.parametrize(
    ("hyperparameters", "match"),
    [
        ({"momentum": 1.5}, "momentum"),
        ({"rho": 0.9}, "Accepted"),
        ([0.9], "hyperparameters"),
    ],
)
def test_invalid_hyperparameters_in_file(
    hyperparameters: object,
    match: str,
) -> None:
    """Invalid hyperparameters make an invalid model file."""
    data = saved_dict(Momentum(0.01))
    data["compile"]["optimizer"]["hyperparameters"] = hyperparameters
    with pytest.raises(ModelFileError, match=match):
        model_from_dict(data)


# On the real dataset --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "learning_rate"),
    [("adam", 0.001), ("nesterov", 0.01)],
)
def test_accuracy_on_the_dataset(name: str, learning_rate: float) -> None:
    """Adam and Nesterov reach the accuracy of the reference SGD."""
    x, labels = to_arrays(read_dataset(DATASET))
    train, valid = split_dataset(len(x), 0.8, np.random.default_rng(42))
    scaler = fit_scaler("standard", x[train])
    model = Model([DenseLayer(24, "relu", "heUniform"),
                   DenseLayer(24, "relu", "heUniform"),
                   DenseLayer(2, "softmax", "heUniform")])
    model.compile("categoricalCrossentropy",
                  optimizer=make_optimizer(name, learning_rate))
    history = model.fit(
        transform(scaler, x[train]), one_hot(labels[train], 2),
        validation_data=(transform(scaler, x[valid]),
                         one_hot(labels[valid], 2)),
        epochs=30, batch_size=8, seed=14, verbose=False,
    )
    assert history.valid["accuracy"][-1] >= 0.96
