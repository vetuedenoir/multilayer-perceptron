"""Learning curves. The only module of the library importing matplotlib."""

from typing import Final, Mapping

import matplotlib.pyplot as plt

from mlp.history import History

METRIC_COLORS: Final[Mapping[str, str]] = {
    "loss": "tab:blue",
    "accuracy": "tab:green",
    "precision": "tab:red",
    "recall": "tab:purple",
    "f1": "tab:orange",
}
DEFAULT_COLOR: Final[str] = "tab:gray"
TRAIN_STYLE: Final[str] = "-"
VALID_STYLE: Final[str] = "--"
BEST_EPOCH_COLOR: Final[str] = "black"
BEST_EPOCH_STYLE: Final[str] = ":"


def plot_history(history: History, title: str = "Training history") -> None:
    """Show the loss on the left and the other metrics on the right.

    Solid lines are the training values, dashed lines the validation
    ones (drawn only when the history has some). A dotted vertical line
    marks the best epoch of an early-stopped training.
    """
    epochs = range(1, history.epochs + 1)
    fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title)

    for name in history.train:
        ax = ax_loss if name == "loss" else ax_metrics
        color = METRIC_COLORS.get(name, DEFAULT_COLOR)
        label = name.capitalize()
        ax.plot(epochs, history.train[name], label=f"Train {label}",
                color=color, linestyle=TRAIN_STYLE)
        if name in history.valid:
            ax.plot(epochs, history.valid[name],
                    label=f"Validation {label}",
                    color=color, linestyle=VALID_STYLE)

    if history.best_epoch is not None:
        for ax in (ax_loss, ax_metrics):
            ax.axvline(history.best_epoch, label="Best epoch",
                       color=BEST_EPOCH_COLOR, linestyle=BEST_EPOCH_STYLE)

    ax_loss.set_title("Loss")
    ax_loss.set_ylabel("Loss")
    ax_metrics.set_title("Metrics")
    ax_metrics.set_ylabel("Score")
    for ax in (ax_loss, ax_metrics):
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.show()


__all__ = ["plot_history"]
