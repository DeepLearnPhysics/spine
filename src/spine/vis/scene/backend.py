"""Backend registration for renderer-neutral scenes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["get_backend", "register_backend", "render_scene"]


_BACKENDS: dict[str, Callable[[], Any]] = {}


def register_backend(
    name: str, factory: Callable[[], Any], *, replace: bool = False
) -> None:
    """Register a scene backend factory under a stable name.

    Parameters
    ----------
    name : str
        Name used to resolve the backend.
    factory : callable
        Zero-argument callable which constructs the backend.
    replace : bool, default False
        Whether to replace an existing registration with the same name.

    Raises
    ------
    ValueError
        If ``name`` is empty or already registered without replacement.
    """
    # Validate the public registry key and guard against accidental collisions
    if not name:
        raise ValueError("Backend names cannot be empty.")
    if name in _BACKENDS and not replace:
        raise ValueError(f"Scene backend `{name}` is already registered.")

    # Store factories rather than instances so renderers do not share state
    _BACKENDS[name] = factory


def get_backend(name: str) -> Any:
    """Construct a registered scene backend.

    Parameters
    ----------
    name : str
        Name of the backend to construct.

    Returns
    -------
    Any
        Newly constructed backend instance.

    Raises
    ------
    ValueError
        If no backend is registered under ``name``.
    """
    # Import the built-in backend lazily to keep the registry extensible
    if name == "plotly" and name not in _BACKENDS:
        from .plotly import PlotlyBackend

        register_backend("plotly", PlotlyBackend)

    # Report discoverable names when backend resolution fails
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS)) or "none"
        raise ValueError(f"Unknown scene backend `{name}`. Available: {available}.")

    # Construct a fresh backend for this render operation
    return _BACKENDS[name]()


def render_scene(scene: Any, backend: str | Any = "plotly", **kwargs: Any) -> Any:
    """Render a scene without calling its convenience method.

    Parameters
    ----------
    scene : Scene
        Renderer-neutral scene to convert.
    backend : str or object, default 'plotly'
        Registered backend name or object exposing a ``render`` method.
    **kwargs : dict, optional
        Backend-specific rendering options.

    Returns
    -------
    Any
        Backend-specific scene representation.
    """
    # Resolve named backends while accepting directly configured instances
    renderer = get_backend(backend) if isinstance(backend, str) else backend
    return renderer.render(scene, **kwargs)
