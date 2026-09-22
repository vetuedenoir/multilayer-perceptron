"""Generic lookup helpers for the registries of the library.

Two separate functions instead of a single one returning ``list | str``:
a union return type cannot be exploited by the caller (nor by mypy),
whereas ``get_from_registry`` returns a ``T`` and
``get_many_from_registry`` a ``list[T]``.
"""

from typing import Mapping, Sequence, TypeVar

from mlp.errors import ConfigurationError

T = TypeVar("T")


def _available(registry: Mapping[str, T]) -> str:
    """Return the valid keys of `registry` as a printable list."""
    return ", ".join(repr(key) for key in registry if key)


def get_from_registry(
    registry: Mapping[str, T],
    name: str,
    category: str,
) -> T:
    """Return the `registry` entry named `name`.

    Raise ConfigurationError if `name` is not a key of `registry`,
    `category` being the human readable name of the registry.
    """
    if not isinstance(name, str):
        raise ConfigurationError(
            f"expected a string for {category}, "
            f"received {type(name).__name__}"
        )
    if name not in registry:
        raise ConfigurationError(
            f"unknown {category}: {name!r}. "
            f"Available values: {_available(registry)}"
        )
    return registry[name]


def get_many_from_registry(
    registry: Mapping[str, T],
    names: Sequence[str],
    category: str,
) -> list[T]:
    """Return the `registry` entries named by `names`, in order.

    Raise ConfigurationError on the first unknown name.
    """
    if isinstance(names, str):
        raise ConfigurationError(
            f"expected a sequence of names for {category}, "
            f"received a single string {names!r}"
        )
    return [get_from_registry(registry, name, category) for name in names]


__all__ = ["get_from_registry", "get_many_from_registry"]
