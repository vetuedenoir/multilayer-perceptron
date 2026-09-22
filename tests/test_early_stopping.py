"""Tests of the early stopping: pure decision, fit integration, saving."""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from test_model import binary_model, separable_data
from test_serialization import LABELS, trained_softmax

from mlp.early_stopping import (
    EarlyStopping,
    EarlyStoppingState,
    check_monitor,
    initial_state,
    is_improvement,
    split_monitor,
    update_state,
)
from mlp.errors import ConfigurationError, ModelFileError
from mlp.history import History
from mlp.layers import DenseLayer
from mlp.network import Model, format_early_stop_line
from mlp.preprocessing import fit_standard
from mlp.serialization import (
    TrainingConfig,
    load_model,
    model_from_dict,
    model_to_dict,
    save_model,
)
from mlp.types import FloatArray

STOPPING = EarlyStopping(monitor="valid_loss", patience=3, min_delta=0.1,
                         mode="auto", restore_best_weights=True)


def run(
    values: Sequence[float],
    config: EarlyStopping,
) -> tuple[EarlyStoppingState, int | None]:
    """Feed `values` epoch after epoch; return the state and stop epoch."""
    state = initial_state(config)
    for epoch, value in enumerate(values, start=1):
        state, stop = update_state(state, value, epoch, config)
        if stop:
            return state, epoch
    return state, None


# Configuration --------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"patience": 0}, "patience"),
        ({"patience": 1.5}, "patience"),
        ({"patience": True}, "patience"),
        ({"min_delta": -0.1}, "min_delta"),
        ({"min_delta": math.nan}, "min_delta"),
        ({"mode": "lowest"}, "mode"),
        ({"monitor": "loss"}, "monitored"),
        ({"monitor": "test_loss"}, "monitored"),
        ({"monitor": "valid_"}, "monitored"),
        ({"restore_best_weights": 1}, "restore_best_weights"),
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any], match: str) -> None:
    """Every invalid field is refused when the configuration is made."""
    with pytest.raises(ConfigurationError, match=match):
        EarlyStopping(**kwargs)


@pytest.mark.parametrize(
    ("monitor", "mode", "resolved"),
    [
        ("valid_loss", "auto", "min"),
        ("train_loss", "auto", "min"),
        ("valid_accuracy", "auto", "max"),
        ("train_f1", "auto", "max"),
        ("valid_accuracy", "min", "min"),
        ("valid_loss", "max", "max"),
    ],
)
def test_resolved_mode(monitor: str, mode: str, resolved: str) -> None:
    """Auto minimises a loss and maximises a metric."""
    assert EarlyStopping(monitor=monitor, mode=mode).resolved_mode() == \
        resolved


def test_split_monitor() -> None:
    """Only the first underscore separates the set from the quantity."""
    assert split_monitor("valid_loss") == ("valid", "loss")
    assert split_monitor("train_my_metric") == ("train", "my_metric")


def test_config_round_trip() -> None:
    """to_config and from_config are inverse of each other."""
    assert EarlyStopping.from_config(STOPPING.to_config()) == STOPPING
    with pytest.raises(ConfigurationError, match="fields"):
        EarlyStopping.from_config({**STOPPING.to_config(), "extra": 1})


def test_check_monitor() -> None:
    """Valid values need validation data, metrics must be compiled."""
    check_monitor(EarlyStopping("train_accuracy"), ["accuracy"], False)
    with pytest.raises(ConfigurationError, match="without validation"):
        check_monitor(EarlyStopping("valid_loss"), [], False)
    with pytest.raises(ConfigurationError,
                       match="valid_loss, valid_accuracy"):
        check_monitor(EarlyStopping("valid_auc"), ["accuracy"], True)


# Pure decision --------------------------------------------------------------


def test_is_improvement() -> None:
    """Strictly better by more than min_delta, in the right direction."""
    assert is_improvement(0.5, math.inf, 0.0, "min")
    assert is_improvement(0.5, 0.7, 0.1, "min")
    assert not is_improvement(0.65, 0.7, 0.1, "min")
    assert not is_improvement(0.7, 0.7, 0.0, "min")
    assert is_improvement(0.9, 0.7, 0.1, "max")
    assert not is_improvement(0.75, 0.7, 0.1, "max")
    with pytest.raises(ConfigurationError):
        is_improvement(0.5, 0.7, 0.0, "auto")


def test_initial_state() -> None:
    """The worst possible value, no epoch, no wait."""
    assert initial_state(EarlyStopping()) == \
        EarlyStoppingState(math.inf, 0, 0)
    assert initial_state(EarlyStopping("valid_accuracy")).best == -math.inf


def test_stops_at_best_epoch_plus_patience() -> None:
    """Decrease then plateau: stop exactly `patience` epochs later."""
    config = EarlyStopping(patience=3)
    state, stopped = run([1.0, 0.8, 0.6, 0.7, 0.65, 0.61, 0.5], config)
    assert (state.best, state.best_epoch, stopped) == (0.6, 3, 6)


def test_improvement_below_min_delta_is_ignored() -> None:
    """0.95 after 1.0 is not better by more than 0.1: it does not count."""
    state, stopped = run([1.0, 0.95, 0.92, 0.91], STOPPING)
    assert (state.best, state.best_epoch, stopped) == (1.0, 1, 4)


def test_improvement_resets_the_wait() -> None:
    """The patience counts consecutive epochs without improvement."""
    config = EarlyStopping(patience=2)
    state, stopped = run([1.0, 1.1, 0.9, 1.0, 0.8, 0.85], config)
    assert (state.best_epoch, state.wait, stopped) == (5, 1, None)


def test_max_mode_on_accuracy() -> None:
    """An accuracy is maximised."""
    config = EarlyStopping("valid_accuracy", patience=2)
    state, stopped = run([0.8, 0.9, 0.85, 0.9], config)
    assert (state.best, state.best_epoch, stopped) == (0.9, 2, 4)


def test_update_state_is_pure() -> None:
    """The given state is never modified."""
    state = initial_state(STOPPING)
    update_state(state, 0.5, 1, STOPPING)
    assert state == EarlyStoppingState(math.inf, 0, 0)


def test_format_early_stop_line() -> None:
    """The final line tells where it stopped and what was kept."""
    assert format_early_stop_line(EarlyStopping(), 27, 0.08123, 37) == (
        "early stopping at epoch 37, best epoch 27 (valid_loss 0.0812), "
        "weights restored")
    config = EarlyStopping("valid_f1", restore_best_weights=False)
    assert format_early_stop_line(config, 5, 0.9, None) == (
        "early stopping not triggered, best epoch 5 (valid_f1 0.9000), "
        "last weights kept")


# Integration with fit -------------------------------------------------------


def noisy_data(
    samples: int,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Return (x, y) with noisy labels: easy to overfit."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples, 4))
    noise = rng.normal(size=(samples, 1))
    y = (x[:, :1] + x[:, 1:2] + noise > 0).astype(np.float64)
    return x, y


def overfitting_model() -> Model:
    """Return a model far too large for 40 noisy samples."""
    model = Model([DenseLayer(32, "relu"), DenseLayer(32, "relu"),
                   DenseLayer(1, "sigmoid")])
    model.compile("binaryCrossentropy", metrics=["accuracy"],
                  learning_rate=0.1)
    return model


def fit_overfitting(
    early_stopping: EarlyStopping | None,
    epochs: int = 200,
) -> tuple[Model, History, tuple[FloatArray, FloatArray]]:
    """Train the overfitting model; return it, its history, valid data."""
    model = overfitting_model()
    valid = noisy_data(40, seed=1)
    history = model.fit(*noisy_data(40, seed=0), epochs=epochs,
                        batch_size=4, seed=0, validation_data=valid,
                        verbose=False, early_stopping=early_stopping)
    return model, history, valid


def test_restores_the_best_weights() -> None:
    """The model evaluates exactly to the loss of its best epoch."""
    model, history, valid = fit_overfitting(EarlyStopping(patience=5))
    assert history.best_epoch is not None
    assert history.stopped_epoch == history.best_epoch + 5
    assert history.epochs == history.stopped_epoch < 200
    assert model.evaluate(*valid)[0] == \
        history.valid["loss"][history.best_epoch - 1]


def test_without_restore_keeps_the_last_weights() -> None:
    """restore_best_weights=False: the loss of the last epoch."""
    config = EarlyStopping(patience=5, restore_best_weights=False)
    model, history, valid = fit_overfitting(config)
    assert history.stopped_epoch is not None
    assert history.best_epoch is not None
    assert history.best_epoch < history.stopped_epoch
    assert model.evaluate(*valid)[0] == history.valid["loss"][-1]


def test_restores_at_the_end_of_the_epochs_too() -> None:
    """Not triggered: the best weights are restored all the same."""
    config = EarlyStopping(patience=1000)
    model, history, valid = fit_overfitting(config, epochs=30)
    assert history.stopped_epoch is None
    assert history.best_epoch is not None
    assert history.best_epoch < 30
    assert model.evaluate(*valid)[0] == \
        history.valid["loss"][history.best_epoch - 1]


def test_monitor_a_training_metric() -> None:
    """A train_* value works without validation data."""
    model = overfitting_model()
    history = model.fit(*noisy_data(40, seed=0), epochs=50, batch_size=4,
                        seed=0, verbose=False,
                        early_stopping=EarlyStopping("train_accuracy",
                                                     patience=3))
    assert history.best_epoch is not None
    assert history.valid == {}


@pytest.mark.parametrize(
    ("monitor", "validation", "match"),
    [
        ("valid_loss", False, "without validation"),
        ("valid_f1", True, "cannot monitor 'valid_f1'"),
    ],
)
def test_fit_refuses_unrecorded_monitor(
    monitor: str,
    validation: bool,
    match: str,
) -> None:
    """Checked before any training: nothing is built."""
    model = overfitting_model()
    x, y = noisy_data(10, seed=0)
    with pytest.raises(ConfigurationError, match=match):
        model.fit(x, y, epochs=1, batch_size=4, verbose=False,
                  validation_data=(x, y) if validation else None,
                  early_stopping=EarlyStopping(monitor))
    assert not model.built


def test_verbose_prints_the_final_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One line after the epochs, only with early stopping."""
    model = overfitting_model()
    x, y = noisy_data(20, seed=0)
    model.fit(x, y, epochs=2, batch_size=4, seed=0, validation_data=(x, y),
              early_stopping=EarlyStopping(patience=5))
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("early stopping not triggered, best epoch")


def test_no_early_stopping_is_unchanged() -> None:
    """Without early stopping: the very numbers of before this feature.

    The constants were recorded with the code of step 7.
    """
    x, y = separable_data()
    model = binary_model()
    model.compile("binaryCrossentropy", metrics=["accuracy"],
                  learning_rate=0.05)
    history = model.fit(x, y, epochs=5, batch_size=8, seed=42,
                        validation_data=separable_data(30, seed=1),
                        verbose=False)
    assert history.train["loss"][-1] == 0.5823599432804191
    assert history.valid["loss"][-1] == 0.6108945021350702
    assert history.valid["accuracy"][-1] == 0.6333333333333333
    assert (history.best_epoch, history.stopped_epoch) == (None, None)


def test_untriggered_early_stopping_gives_the_same_history() -> None:
    """Watching without stopping nor restoring changes no number."""
    histories = []
    for config in (None, EarlyStopping(patience=1000,
                                       restore_best_weights=False)):
        _, history, _ = fit_overfitting(config, epochs=20)
        histories.append((history.train, history.valid))
    assert histories[0] == histories[1]


# Saving ---------------------------------------------------------------------


def test_round_trip_of_an_early_stopped_model(tmp_path: Path) -> None:
    """Best and stopped epochs, and the configuration, come back."""
    config = EarlyStopping(patience=5)
    model, history, valid = fit_overfitting(config)
    scaler = fit_standard(noisy_data(40, seed=0)[0])
    training: TrainingConfig = {"epochs": 200, "batch_size": 4, "seed": 0,
                                "early_stopping": config.to_config()}
    path = tmp_path / "model.json"
    save_model(path, model, scaler, LABELS, training)
    loaded = load_model(path)
    assert loaded.history == history
    assert loaded.history.best_epoch == history.best_epoch
    assert loaded.history.stopped_epoch == history.stopped_epoch
    assert loaded.training == training
    assert loaded.model.evaluate(*valid) == model.evaluate(*valid)
    content = json.loads(path.read_text())
    assert content["format_version"] == 2
    assert content["training"]["early_stopping"]["patience"] == 5


def v2_dict() -> dict[str, Any]:
    """Return the dict of a trained model, through JSON like on disk."""
    model, scaler = trained_softmax()
    training: TrainingConfig = {"epochs": 5, "batch_size": 8, "seed": 0,
                                "early_stopping": STOPPING.to_config()}
    content: dict[str, Any] = json.loads(
        json.dumps(model_to_dict(model, scaler, LABELS, training)))
    return content


def test_version_1_file_still_loads() -> None:
    """A file written before early stopping has none."""
    data = v2_dict()
    data["format_version"] = 1
    del data["training"]["early_stopping"]
    del data["training"]["history"]["best_epoch"]
    del data["training"]["history"]["stopped_epoch"]
    loaded = model_from_dict(data)
    assert loaded.training["early_stopping"] is None
    assert (loaded.history.best_epoch, loaded.history.stopped_epoch) == \
        (None, None)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("patience", 0, "patience"),
        ("mode", "lowest", "mode"),
        ("unknown", 1, "fields"),
    ],
)
def test_invalid_early_stopping_in_file(
    key: str,
    value: object,
    match: str,
) -> None:
    """The stored configuration is validated like a new one."""
    data = v2_dict()
    data["training"]["early_stopping"][key] = value
    with pytest.raises(ModelFileError, match=match):
        model_from_dict(data)


def test_version_2_requires_the_early_stopping_key() -> None:
    """Only a version 1 file may omit it."""
    data = v2_dict()
    del data["training"]["early_stopping"]
    with pytest.raises(ModelFileError, match="early_stopping"):
        model_from_dict(data)


@pytest.mark.parametrize("value", [0, -1, 1.5, "3", True])
def test_invalid_best_epoch_in_file(value: object) -> None:
    """best_epoch is a positive integer or null."""
    data = v2_dict()
    data["training"]["history"]["best_epoch"] = value
    with pytest.raises(ModelFileError, match="best_epoch"):
        model_from_dict(data)
