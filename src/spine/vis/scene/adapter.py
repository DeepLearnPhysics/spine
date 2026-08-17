"""Adapters from established Plotly traces to renderer-neutral scene layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .model import (
    LineLayer,
    LineStyle,
    MarkerLayer,
    MeshLayer,
    MeshStyle,
    PointStyle,
    VectorLayer,
)

__all__ = ["plotly_trace_to_layer"]


def _value(obj: Any, name: str, default: Any = None) -> Any:
    """Read a graph-object property without requiring a concrete class.

    Parameters
    ----------
    obj : Any
        Plotly graph object or property container.
    name : str
        Property name to read.
    default : Any, optional
        Value returned when the property is absent or ``None``.

    Returns
    -------
    Any
        Resolved property value.
    """
    value = getattr(obj, name, default)
    return default if value is None else value


def _positions(trace: Any) -> np.ndarray:
    """Stack Plotly coordinate arrays into contiguous positions.

    Parameters
    ----------
    trace : plotly.graph_objs.BaseTraceType
        Trace exposing ``x``, ``y`` and ``z`` coordinates.

    Returns
    -------
    np.ndarray
        Contiguous ``(N, 3)`` single-precision positions.
    """
    return np.column_stack((trace.x, trace.y, trace.z)).astype(np.float32)


def _line_segments(points: np.ndarray) -> np.ndarray:
    """Expand NaN-separated polylines into independent line segments.

    Parameters
    ----------
    points : np.ndarray
        Plotly polyline vertices with non-finite separator rows.

    Returns
    -------
    np.ndarray
        Line segment endpoints with shape ``(N, 2, 3)``.
    """
    # A non-finite row terminates the current polyline and prevents a segment
    # from being drawn across Plotly's separator.
    segments = []
    previous = None
    for point in points:
        if not np.all(np.isfinite(point)):
            previous = None
            continue
        if previous is not None:
            segments.append((previous, point))
        previous = point
    return np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)


def _line_values(points: np.ndarray, values: Any) -> Any:
    """Reduce Plotly vertex colors to one value per line segment.

    Parameters
    ----------
    points : np.ndarray
        Plotly polyline vertices with non-finite separator rows.
    values : Any
        Shared color or one numeric color value per vertex.

    Returns
    -------
    Any
        Shared input color or one averaged value per finite segment.

    Raises
    ------
    ValueError
        If vertex colors do not align with ``points``.
    """
    if values is None or np.isscalar(values):
        return values
    values = np.asarray(values)
    if len(values) != len(points):
        raise ValueError("Scatter3d line colors must align with line vertices.")
    segment_values = []
    previous = None
    for index, point in enumerate(points):
        if not np.all(np.isfinite(point)):
            previous = None
            continue
        if previous is not None:
            segment_values.append((values[previous] + values[index]) / 2)
        previous = index
    return np.asarray(segment_values)


def _mesh_faces(trace: Any, vertices: np.ndarray) -> np.ndarray:
    """Return explicit triangle faces, resolving convex hulls eagerly.

    Parameters
    ----------
    trace : plotly.graph_objs.Mesh3d
        Plotly mesh containing explicit faces or an implicit convex hull.
    vertices : np.ndarray
        Mesh vertex coordinates.

    Returns
    -------
    np.ndarray
        Triangle indexes with shape ``(N, 3)``.

    Raises
    ------
    ValueError
        If explicit indexes are inconsistent or a supported hull cannot be
        constructed.
    """
    i, j, k = _value(trace, "i", []), _value(trace, "j", []), _value(trace, "k", [])
    if len(i) > 0 or len(j) > 0 or len(k) > 0:
        if not (len(i) == len(j) == len(k)):
            raise ValueError("Mesh3d i, j and k arrays must have matching lengths.")
        return np.column_stack((i, j, k)).astype(np.int32)

    # Plotly computes an implicit convex hull in the browser when alphahull=0.
    # Resolve it here so renderer-neutral consumers receive an actual mesh.
    alphahull = _value(trace, "alphahull", 0)
    if alphahull not in (0, -1):
        raise ValueError(
            "Renderer-neutral conversion only supports explicit faces or "
            "convex Mesh3d hulls."
        )
    if len(vertices) < 4:
        return np.empty((0, 3), dtype=np.int32)
    try:
        return np.ascontiguousarray(ConvexHull(vertices).simplices, dtype=np.int32)
    except QhullError as error:
        raise ValueError(
            "Mesh3d vertices do not define a three-dimensional hull."
        ) from error


def plotly_trace_to_layer(trace: Any, **metadata: Any) -> Any:
    """Convert a supported Plotly 3D trace to a neutral scene layer.

    Parameters
    ----------
    trace : plotly.graph_objs.BaseTraceType
        Scatter3d, Mesh3d or Cone trace.
    **metadata : dict, optional
        Semantic metadata attached to the resulting layer.

    Returns
    -------
    object
        Matching neutral layer.

    Raises
    ------
    TypeError
        If the trace type or scatter mode is unsupported.
    """
    trace_type = _value(trace, "type", "")
    name = _value(trace, "name")
    hovertext = _value(trace, "hovertext", _value(trace, "text"))

    if trace_type == "scatter3d":
        mode = _value(trace, "mode", "markers")
        points = _positions(trace)
        if "lines" in mode:
            line = _value(trace, "line", {})
            line_color = _value(line, "color")
            fixed_color = line_color if isinstance(line_color, str) else None
            values = (
                None if fixed_color is not None else _line_values(points, line_color)
            )
            return LineLayer(
                _line_segments(points),
                name=name,
                values=values,
                hovertext=hovertext,
                style=LineStyle(
                    width=float(_value(line, "width", 2.0)),
                    color=fixed_color,
                    opacity=_value(trace, "opacity"),
                    colorscale=_value(line, "colorscale"),
                    cmin=_value(line, "cmin"),
                    cmax=_value(line, "cmax"),
                ),
                metadata=metadata,
            )
        if "markers" in mode:
            marker = _value(trace, "marker", {})
            return MarkerLayer(
                points,
                name=name,
                values=_value(marker, "color"),
                hovertext=hovertext,
                style=PointStyle(
                    size=float(_value(marker, "size", 2.0)),
                    opacity=_value(marker, "opacity", _value(trace, "opacity")),
                    colorscale=_value(marker, "colorscale"),
                    cmin=_value(marker, "cmin"),
                    cmax=_value(marker, "cmax"),
                ),
                symbol=_value(marker, "symbol", "circle"),
                metadata=metadata,
            )
        raise TypeError(f"Unsupported Scatter3d mode: {mode}.")

    if trace_type == "mesh3d":
        vertices = _positions(trace)
        faces = _mesh_faces(trace, vertices)
        return MeshLayer(
            vertices,
            faces,
            name=name,
            values=_value(trace, "intensity"),
            hovertext=hovertext,
            style=MeshStyle(
                color=_value(trace, "color"),
                opacity=_value(trace, "opacity"),
                colorscale=_value(trace, "colorscale"),
                cmin=_value(trace, "cmin"),
                cmax=_value(trace, "cmax"),
            ),
            metadata=metadata,
        )

    if trace_type == "cone":
        vectors = np.column_stack((trace.u, trace.v, trace.w))
        colorscale = _value(trace, "colorscale")
        color = colorscale[0][1] if colorscale else None
        return VectorLayer(
            _positions(trace),
            vectors,
            name=name,
            hovertext=hovertext,
            style=LineStyle(color=color, opacity=_value(trace, "opacity")),
            metadata=metadata,
        )

    raise TypeError(f"Unsupported Plotly trace type: {trace_type}.")
