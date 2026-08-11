"""Backend registration for renderer-neutral scenes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["get_backend", "register_backend", "render_scene"]


_BACKENDS: dict[str, Callable[[], Any]] = {}


def register_backend(
    name: str, factory: Callable[[], Any], *, replace: bool = False
) -> None:
    """Register a scene backend factory under a stable name."""
    if not name:
        raise ValueError("Backend names cannot be empty.")
    if name in _BACKENDS and not replace:
        raise ValueError(f"Scene backend `{name}` is already registered.")
    _BACKENDS[name] = factory


def get_backend(name: str) -> Any:
    """Construct a registered scene backend."""
    if name == "plotly" and name not in _BACKENDS:
        from .plotly import PlotlyBackend

        register_backend("plotly", PlotlyBackend)
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS)) or "none"
        raise ValueError(f"Unknown scene backend `{name}`. Available: {available}.")
    return _BACKENDS[name]()


def render_scene(scene: Any, backend: str | Any = "plotly", **kwargs: Any) -> Any:
    """Render a scene without calling its convenience method."""
    renderer = get_backend(backend) if isinstance(backend, str) else backend
    return renderer.render(scene, **kwargs)
