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

    Attributes
    ----------
    size : float, default 2.0
        Point size in display pixels.
    opacity : float, optional
        Point opacity in the range ``[0, 1]``.
    colorscale : str or list, optional
        Named or explicit color scale used to map point values.
    cmin : float, optional
        Lower bound of the color scale.
    cmax : float, optional
        Upper bound of the color scale.
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

    Attributes
    ----------
    positions : np.ndarray
        Contiguous ``(N, 3)`` point coordinates stored as ``float32``.
    name : str, optional
        Layer label displayed by rendering backends.
    values : scalar or np.ndarray, optional
        Shared or per-point values used for coloring.
    hovertext : scalar or sequence, optional
        Shared or per-point hover labels.
    object_ids : np.ndarray, optional
        Per-point domain-object identifiers stored as ``int32``.
    object_offsets : np.ndarray, optional
        Monotonic ``(M + 1,)`` boundaries for ``M`` domain objects.
    attributes : mapping, optional
        Named per-point arrays available for filtering or recoloring.
    style : PointStyle, optional
        Renderer-neutral point display hints.
    metadata : dict, optional
        Layer-level semantic metadata.
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
        """Validate and normalize point-layer buffers."""
        # Normalize coordinates to the compact layout expected by GPU backends
        positions = np.asarray(self.positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Point positions must have shape (N, 3).")
        self.positions = np.ascontiguousarray(positions)
        count = len(self.positions)

        # Normalize optional domain-object membership for picking and grouping
        if self.object_ids is not None:
            object_ids = np.asarray(self.object_ids, dtype=np.int32)
            if object_ids.shape != (count,):
                raise ValueError("Point object IDs must have shape (N,).")
            self.object_ids = np.ascontiguousarray(object_ids)

        # Validate the compressed object-boundary representation
        if self.object_offsets is not None:
            offsets = np.asarray(self.object_offsets, dtype=np.int64)
            if offsets.ndim != 1 or len(offsets) == 0:
                raise ValueError("Object offsets must be a non-empty 1D array.")
            if offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) < 0):
                raise ValueError(
                    "Object offsets must be monotonic and span the point buffer."
                )
            self.object_offsets = np.ascontiguousarray(offsets)

        # Require every named attribute to align with the point buffer
        normalized = {}
        for name, value in self.attributes.items():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != count:
                raise ValueError(
                    f"Point attribute `{name}` must have N values in its first axis."
                )
            normalized[name] = np.ascontiguousarray(array)
        self.attributes = normalized

        # Normalize non-scalar color values while preserving shared scalars
        if self.values is not None and not np.isscalar(self.values):
            values = np.asarray(self.values)
            if values.ndim == 0 or values.shape[0] != count:
                raise ValueError("Point values must be scalar or have shape (N, ...).")
            self.values = np.ascontiguousarray(values)

    @property
    def point_count(self) -> int:
        """Return the number of points in this layer."""
        return len(self.positions)

    @property
    def object_count(self) -> int:
        """Return the number of domain objects represented by this layer."""
        if self.object_offsets is None:
            return 0
        return len(self.object_offsets) - 1


@dataclass
class SceneView:
    """One independently viewable collection of scene layers.

    Attributes
    ----------
    name : str
        Human-readable view label.
    layers : list[PointLayer], optional
        Ordered point layers included in the view.
    metadata : dict, optional
        View-level semantic metadata.
    """

    name: str
    layers: list[PointLayer] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    """Renderer-neutral scene produced from SPINE domain objects.

    Attributes
    ----------
    views : list[SceneView], optional
        Independently viewable layer collections.
    metadata : dict, optional
        Scene-level semantic metadata.
    """

    views: list[SceneView] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, backend: str | Any = "plotly", **kwargs: Any) -> Any:
        """Render the scene using a registered backend or backend instance.

        Parameters
        ----------
        backend : str or object, default 'plotly'
            Registered backend name or object exposing a ``render`` method.
        **kwargs : dict, optional
            Backend-specific rendering options.

        Returns
        -------
        Any
            Backend-specific scene representation.

        Raises
        ------
        TypeError
            If a provided backend object does not expose ``render``.
        """
        from .backend import get_backend

        # Resolve names lazily while allowing callers to inject backend instances
        renderer = get_backend(backend) if isinstance(backend, str) else backend
        if not hasattr(renderer, "render"):
            raise TypeError(
                "A scene backend must provide a `render(scene, ...)` method."
            )

        # Delegate output construction without coupling the scene to one library
        return renderer.render(self, **kwargs)
