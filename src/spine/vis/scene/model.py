"""Renderer-neutral scene primitives for three-dimensional visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = ["PointLayer", "PointStyle", "Scene", "SceneView"]


@dataclass(frozen=True)
class PointStyle:
    """Display hints for a point layer.

    The values intentionally describe common rendering concepts rather than
    backend-specific objects. Backends may ignore unsupported hints.
    """

    size: float = 2.0
    opacity: float | None = None
    colorscale: str | list | None = None
    cmin: float | None = None
    cmax: float | None = None


@dataclass
class PointLayer:
    """A contiguous point cloud and its renderer-independent attributes.

    ``object_offsets`` stores the boundaries of domain objects within the
    point buffer. A fast renderer can keep the layer in one GPU buffer, while
    a backend such as Plotly can optionally materialize one trace per object.
    """

    positions: np.ndarray
    name: str | None = None
    values: Any = None
    hovertext: Any = None
    object_ids: np.ndarray | None = None
    object_offsets: np.ndarray | None = None
    attributes: Mapping[str, np.ndarray] = field(default_factory=dict)
    style: PointStyle = field(default_factory=PointStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Point positions must have shape (N, 3).")
        self.positions = np.ascontiguousarray(positions)
        count = len(self.positions)

        if self.object_ids is not None:
            object_ids = np.asarray(self.object_ids, dtype=np.int32)
            if object_ids.shape != (count,):
                raise ValueError("Point object IDs must have shape (N,).")
            self.object_ids = np.ascontiguousarray(object_ids)

        if self.object_offsets is not None:
            offsets = np.asarray(self.object_offsets, dtype=np.int64)
            if offsets.ndim != 1 or len(offsets) == 0:
                raise ValueError("Object offsets must be a non-empty 1D array.")
            if offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) < 0):
                raise ValueError(
                    "Object offsets must be monotonic and span the point buffer."
                )
            self.object_offsets = np.ascontiguousarray(offsets)

        normalized = {}
        for name, value in self.attributes.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != count:
                raise ValueError(
                    f"Point attribute `{name}` must have N values in its first axis."
                )
            normalized[name] = np.ascontiguousarray(array)
        self.attributes = normalized

        if self.values is not None and not np.isscalar(self.values):
            values = np.asarray(self.values)
            if values.ndim == 0 or values.shape[0] != count:
                raise ValueError("Point values must be scalar or have shape (N, ...).")
            self.values = np.ascontiguousarray(values)

    @property
    def point_count(self) -> int:
        """Number of points in this layer."""
        return len(self.positions)

    @property
    def object_count(self) -> int:
        """Number of domain objects represented by this layer."""
        if self.object_offsets is None:
            return 0
        return len(self.object_offsets) - 1


@dataclass
class SceneView:
    """One independently viewable collection of scene layers."""

    name: str
    layers: list[PointLayer] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    """Renderer-neutral scene produced from SPINE domain objects."""

    views: list[SceneView] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, backend: str | Any = "plotly", **kwargs: Any) -> Any:
        """Render the scene using a registered backend or backend instance."""
        from .backend import get_backend

        renderer = get_backend(backend) if isinstance(backend, str) else backend
        if not hasattr(renderer, "render"):
            raise TypeError(
                "A scene backend must provide a `render(scene, ...)` method."
            )
        return renderer.render(self, **kwargs)
