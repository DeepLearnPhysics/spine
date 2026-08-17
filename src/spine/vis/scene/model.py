"""Renderer-neutral scene primitives for three-dimensional visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = [
    "BoxLayer",
    "LineLayer",
    "LineStyle",
    "MarkerLayer",
    "MeshLayer",
    "MeshStyle",
    "PointLayer",
    "PointStyle",
    "Scene",
    "SceneView",
    "VectorLayer",
]


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


@dataclass(frozen=True)
class LineStyle:
    """Display hints shared by line and vector layers.

    Attributes
    ----------
    width : float, default 2.0
        Line width in display pixels.
    color : Any, optional
        Shared line color or per-segment color values.
    opacity : float, optional
        Layer opacity in the range ``[0, 1]``.
    colorscale : str or list, optional
        Named or explicit color scale used to map numeric values.
    cmin : float, optional
        Lower bound of the color scale.
    cmax : float, optional
        Upper bound of the color scale.
    """

    width: float = 2.0
    color: Any = None
    opacity: float | None = None
    colorscale: str | list | None = None
    cmin: float | None = None
    cmax: float | None = None


@dataclass(frozen=True)
class MeshStyle:
    """Display hints for indexed triangle meshes.

    Attributes
    ----------
    color : Any, optional
        Shared mesh color.
    opacity : float, optional
        Mesh opacity in the range ``[0, 1]``.
    wireframe : bool, default False
        Whether a backend should emphasize mesh edges instead of faces.
    colorscale : str or list, optional
        Named or explicit color scale used to map numeric values.
    cmin : float, optional
        Lower bound of the color scale.
    cmax : float, optional
        Upper bound of the color scale.
    """

    color: Any = None
    opacity: float | None = None
    wireframe: bool = False
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
class MarkerLayer(PointLayer):
    """Discrete point glyphs with a renderer-neutral symbol hint.

    Attributes
    ----------
    symbol : str, default ``"circle"``
        Generic marker symbol such as ``"circle"``, ``"diamond"`` or
        ``"circle-open"``.
    """

    symbol: str = "circle"


@dataclass
class LineLayer:
    """Independent line segments stored as ``(N, 2, 3)`` coordinates.

    Attributes
    ----------
    segments : np.ndarray
        Line segment endpoints with shape ``(N, 2, 3)``.
    name : str, optional
        Layer label.
    values : scalar or np.ndarray, optional
        Shared or per-vector values used for coloring.
    values : scalar or np.ndarray, optional
        Shared or per-segment color values.
    hovertext : scalar or sequence, optional
        Shared or per-segment hover labels.
    object_ids : np.ndarray, optional
        Domain-object identifier per segment.
    style : LineStyle, optional
        Renderer-neutral line display hints.
    metadata : dict, optional
        Layer-level semantic metadata.
    """

    segments: np.ndarray
    name: str | None = None
    values: Any = None
    hovertext: Any = None
    object_ids: np.ndarray | None = None
    style: LineStyle = field(default_factory=LineStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize line-segment buffers."""
        segments = np.asarray(self.segments, dtype=np.float32)
        if segments.ndim != 3 or segments.shape[1:] != (2, 3):
            raise ValueError("Line segments must have shape (N, 2, 3).")
        self.segments = np.ascontiguousarray(segments)
        if self.object_ids is not None:
            object_ids = np.asarray(self.object_ids, dtype=np.int32)
            if object_ids.shape != (len(segments),):
                raise ValueError("Line object IDs must have shape (N,).")
            self.object_ids = np.ascontiguousarray(object_ids)

        # Normalize the portable one-value-per-segment color representation
        if self.values is not None and not np.isscalar(self.values):
            values = np.asarray(self.values)
            if values.ndim != 1 or len(values) != len(segments):
                raise ValueError(
                    "Line values must be scalar or provide one value per segment."
                )
            self.values = np.ascontiguousarray(values)


@dataclass
class VectorLayer:
    """Vector glyphs represented by origins and direction vectors.

    Attributes
    ----------
    origins : np.ndarray
        Vector origins with shape ``(N, 3)``.
    vectors : np.ndarray
        Direction vectors with shape ``(N, 3)``.
    name : str, optional
        Layer label.
    hovertext : scalar or sequence, optional
        Shared or per-vector hover labels.
    object_ids : np.ndarray, optional
        Domain-object identifier per vector.
    scale : float, default 1.0
        Common vector-length scale.
    head_size : float, default 0.25
        Arrow-head length relative to the displayed vector.
    style : LineStyle, optional
        Renderer-neutral vector display hints.
    metadata : dict, optional
        Layer-level semantic metadata.
    """

    origins: np.ndarray
    vectors: np.ndarray
    name: str | None = None
    values: Any = None
    hovertext: Any = None
    object_ids: np.ndarray | None = None
    scale: float = 1.0
    head_size: float = 0.25
    style: LineStyle = field(default_factory=LineStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize vector buffers."""
        origins = np.asarray(self.origins, dtype=np.float32)
        vectors = np.asarray(self.vectors, dtype=np.float32)
        if origins.ndim != 2 or origins.shape[1:] != (3,):
            raise ValueError("Vector origins must have shape (N, 3).")
        if vectors.shape != origins.shape:
            raise ValueError("Vector directions must match origin shape.")
        self.origins = np.ascontiguousarray(origins)
        self.vectors = np.ascontiguousarray(vectors)
        if self.object_ids is not None:
            object_ids = np.asarray(self.object_ids, dtype=np.int32)
            if object_ids.shape != (len(origins),):
                raise ValueError("Vector object IDs must have shape (N,).")
            self.object_ids = np.ascontiguousarray(object_ids)

        # Normalize the portable one-value-per-vector color representation
        if self.values is not None and not np.isscalar(self.values):
            values = np.asarray(self.values)
            if values.ndim != 1 or len(values) != len(origins):
                raise ValueError(
                    "Vector values must be scalar or provide one value per vector."
                )
            self.values = np.ascontiguousarray(values)


@dataclass
class MeshLayer:
    """Indexed triangle mesh with optional per-vertex values.

    Attributes
    ----------
    vertices : np.ndarray
        Mesh vertices with shape ``(N, 3)``.
    faces : np.ndarray
        Triangle vertex indices with shape ``(M, 3)``.
    name : str, optional
        Layer label.
    values : scalar or np.ndarray, optional
        Shared or per-vertex color values.
    hovertext : scalar or sequence, optional
        Shared or per-vertex hover labels.
    style : MeshStyle, optional
        Renderer-neutral mesh display hints.
    metadata : dict, optional
        Layer-level semantic metadata.
    """

    vertices: np.ndarray
    faces: np.ndarray
    name: str | None = None
    values: Any = None
    hovertext: Any = None
    style: MeshStyle = field(default_factory=MeshStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize indexed mesh buffers."""
        vertices = np.asarray(self.vertices, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("Mesh vertices must have shape (N, 3).")
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("Mesh faces must have shape (M, 3).")
        if len(faces) and (np.min(faces) < 0 or np.max(faces) >= len(vertices)):
            raise ValueError("Mesh face indices must reference existing vertices.")
        self.vertices = np.ascontiguousarray(vertices)
        self.faces = np.ascontiguousarray(faces)

        # Per-vertex values are the portable mesh-color representation
        if self.values is not None and not np.isscalar(self.values):
            values = np.asarray(self.values)
            if values.ndim == 0 or values.shape[0] != len(vertices):
                raise ValueError(
                    "Mesh values must be scalar or have one value per vertex."
                )
            self.values = np.ascontiguousarray(values)


@dataclass
class BoxLayer:
    """Axis-aligned boxes represented by lower and upper corners.

    Attributes
    ----------
    bounds : np.ndarray
        Box bounds with shape ``(N, 2, 3)``.
    name : str, optional
        Layer label.
    values : scalar or np.ndarray, optional
        Shared or per-box color values.
    hovertext : scalar or sequence, optional
        Shared or per-box hover labels.
    object_ids : np.ndarray, optional
        Domain-object identifier per box.
    draw_faces : bool, default False
        Whether to draw filled faces rather than wireframes.
    line_style : LineStyle, optional
        Wireframe display hints.
    mesh_style : MeshStyle, optional
        Filled-face display hints.
    metadata : dict, optional
        Layer-level semantic metadata.
    """

    bounds: np.ndarray
    name: str | None = None
    values: Any = None
    hovertext: Any = None
    object_ids: np.ndarray | None = None
    draw_faces: bool = False
    line_style: LineStyle = field(default_factory=LineStyle)
    mesh_style: MeshStyle = field(default_factory=MeshStyle)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize box buffers."""
        bounds = np.asarray(self.bounds, dtype=np.float32)
        if bounds.ndim != 3 or bounds.shape[1:] != (2, 3):
            raise ValueError("Box bounds must have shape (N, 2, 3).")
        if len(bounds) and np.any(bounds[:, 1] < bounds[:, 0]):
            raise ValueError("Box upper bounds must not be below lower bounds.")
        self.bounds = np.ascontiguousarray(bounds)
        if self.object_ids is not None:
            object_ids = np.asarray(self.object_ids, dtype=np.int32)
            if object_ids.shape != (len(bounds),):
                raise ValueError("Box object IDs must have shape (N,).")
            self.object_ids = np.ascontiguousarray(object_ids)

        # Per-box values retain one color value for each compact box primitive
        if self.values is not None and not np.isscalar(self.values):
            values = np.asarray(self.values)
            if values.ndim == 0 or values.shape[0] != len(bounds):
                raise ValueError("Box values must be scalar or have one value per box.")
            self.values = np.ascontiguousarray(values)


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
    layers: list[Any] = field(default_factory=list)
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
