"""Tools to draw 2D or 3D point clouds."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import plotly.graph_objs as go

from spine.constants import COORD_COLS

from .utils import ColorInput, HoverTextInput, NumericOrSequence, is_scalar_sequence

__all__ = ["scatter_points", "scatter_points_2d", "scatter_points_3d"]


def _prepare_point_trace_inputs(
    points: np.ndarray,
    color: ColorInput = None,
    markersize: NumericOrSequence = 2,
    linewidth: float = 2,
    colorscale: str | list | None = None,
    cmin: int | float | None = None,
    cmax: int | float | None = None,
    opacity: float | None = None,
    hovertext: HoverTextInput = None,
    hovertemplate: str | None = None,
    mode: str = "markers",
    marker: dict[str, Any] | None = None,
    line: dict[str, Any] | None = None,
    dim: Literal[2, 3] = 3,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any] | None,
    dict[str, Any] | None,
    HoverTextInput,
    str,
]:
    """Prepare the shared Plotly inputs for 2D and 3D point traces.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2+)`` array of point coordinates.
    color : Union[str, int, float, Sequence], optional
        Color of markers or lines. Can be one shared scalar value or one value
        per point.
    markersize : Union[int, float, Sequence], default 2
        Marker size, provided either as one shared numeric value or one value
        per point.
    linewidth : float, default 2
        Line width.
    colorscale : Union[str, list], optional
        Plotly colorscale specifier.
    cmin : Union[int, float], optional
        Minimum of the color range.
    cmax : Union[int, float], optional
        Maximum of the color range.
    opacity : float, optional
        Marker opacity.
    hovertext : Union[int, float, str, Sequence], optional
        Marker hover labels. Can be one shared label or one label per point.
    hovertemplate : str, optional
        Plotly hover formatting string.
    mode : str, default ``"markers"``
        Plotly drawing mode.
    marker : dict, optional
        Plotly marker style dictionary.
    line : dict, optional
        Plotly line style dictionary.
    dim : Literal[2, 3], default 3
        Requested display dimension.

    Returns
    -------
    Tuple[dict, dict, dict, Any, str]
        Coordinate dictionary, marker dictionary, line dictionary, hovertext,
        and hovertemplate.
    """
    if dim not in [2, 3]:
        raise ValueError("This function only supports dimension 2 or 3.")
    if points.shape[1] == 2:
        dim = 2

    coord_cols = COORD_COLS[:dim]
    if points.shape[1] == dim:
        coord_cols = np.arange(dim)

    if hovertext is None and color is not None and is_scalar_sequence(color):
        hovertext = [f"Value: {color_value}" for color_value in color]

    if hovertemplate is None:
        hovertemplate = "x: %{x}<br>y: %{y}"
        if dim == 3:
            hovertemplate += "<br>z: %{z}"
        if hovertext is not None:
            if is_scalar_sequence(hovertext):
                hovertemplate += "<br>%{text}"
            else:
                hovertemplate += f"<br>{hovertext}"
                hovertext = None

    if "markers" in mode and marker is None:
        marker = {
            "size": markersize,
            "color": color,
            "opacity": opacity,
            "colorscale": colorscale,
            "cmin": cmin,
            "cmax": cmax,
        }

    if "lines" in mode and line is None:
        line = {"width": linewidth, "color": color}
        if dim == 3:
            line.update(
                {
                    "colorscale": colorscale,
                    "cmin": cmin,
                    "cmax": cmax,
                }
            )

    axes = ["x", "y", "z"][:dim]
    pos_dict = {axis: points[:, coord_cols[i]] for i, axis in enumerate(axes)}

    return pos_dict, marker, line, hovertext, hovertemplate


def scatter_points_2d(
    points: np.ndarray,
    color: ColorInput = None,
    markersize: NumericOrSequence = 2,
    linewidth: float = 2,
    colorscale: str | list | None = None,
    cmin: int | float | None = None,
    cmax: int | float | None = None,
    opacity: float | None = None,
    hovertext: HoverTextInput = None,
    hovertemplate: str | None = None,
    mode: str = "markers",
    marker: dict[str, Any] | None = None,
    line: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[go.Scatter]:
    """Scatter 2D points and their labels.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2+)`` array of point coordinates.
    color : Union[str, int, float, Sequence], optional
        Color of markers or lines. Can be one shared scalar value or one value
        per point.
    markersize : Union[int, float, Sequence], default 2
        Marker size, provided either as one shared numeric value or one value
        per point.
    linewidth : float, default 2
        Line width.
    colorscale : Union[str, list], optional
        Plotly colorscale specifier.
    cmin : Union[int, float], optional
        Minimum of the color range.
    cmax : Union[int, float], optional
        Maximum of the color range.
    opacity : float, optional
        Marker opacity.
    hovertext : Union[int, float, str, Sequence], optional
        Marker hover labels. Can be one shared label or one label per point.
    hovertemplate : str, optional
        Plotly hover formatting string.
    mode : str, default ``"markers"``
        Plotly drawing mode.
    marker : dict, optional
        Plotly marker style dictionary.
    line : dict, optional
        Plotly line style dictionary.
    **kwargs : Any
        Additional keyword arguments forwarded to :class:`plotly.graph_objs.Scatter`.

    Returns
    -------
    List[go.Scatter]
        Single 2D scatter trace wrapped in a list.
    """
    pos_dict, marker, line, hovertext, hovertemplate = _prepare_point_trace_inputs(
        points,
        color=color,
        markersize=markersize,
        linewidth=linewidth,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        opacity=opacity,
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        mode=mode,
        marker=marker,
        line=line,
        dim=2,
    )

    return [
        go.Scatter(
            mode=mode,
            marker=marker,
            line=line,
            text=hovertext,
            hovertemplate=hovertemplate,
            **pos_dict,
            **kwargs,
        )
    ]


def scatter_points_3d(
    points: np.ndarray,
    color: ColorInput = None,
    markersize: NumericOrSequence = 2,
    linewidth: float = 2,
    colorscale: str | list | None = None,
    cmin: int | float | None = None,
    cmax: int | float | None = None,
    opacity: float | None = None,
    hovertext: HoverTextInput = None,
    hovertemplate: str | None = None,
    mode: str = "markers",
    marker: dict[str, Any] | None = None,
    line: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[go.Scatter3d]:
    """Scatter 3D points and their labels.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3+)`` array of point coordinates.
    color : Union[str, int, float, Sequence], optional
        Color of markers or lines. Can be one shared scalar value or one value
        per point.
    markersize : Union[int, float, Sequence], default 2
        Marker size, provided either as one shared numeric value or one value
        per point.
    linewidth : float, default 2
        Line width.
    colorscale : Union[str, list], optional
        Plotly colorscale specifier.
    cmin : Union[int, float], optional
        Minimum of the color range.
    cmax : Union[int, float], optional
        Maximum of the color range.
    opacity : float, optional
        Marker opacity.
    hovertext : Union[int, float, str, Sequence], optional
        Marker hover labels. Can be one shared label or one label per point.
    hovertemplate : str, optional
        Plotly hover formatting string.
    mode : str, default ``"markers"``
        Plotly drawing mode.
    marker : dict, optional
        Plotly marker style dictionary.
    line : dict, optional
        Plotly line style dictionary.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :class:`plotly.graph_objs.Scatter3d`.

    Returns
    -------
    List[go.Scatter3d]
        Single 3D scatter trace wrapped in a list.
    """
    pos_dict, marker, line, hovertext, hovertemplate = _prepare_point_trace_inputs(
        points,
        color=color,
        markersize=markersize,
        linewidth=linewidth,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        opacity=opacity,
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        mode=mode,
        marker=marker,
        line=line,
        dim=3,
    )

    return [
        go.Scatter3d(
            mode=mode,
            marker=marker,
            line=line,
            text=hovertext,
            hovertemplate=hovertemplate,
            **pos_dict,
            **kwargs,
        )
    ]


def scatter_points(
    points: np.ndarray,
    color: ColorInput = None,
    markersize: NumericOrSequence = 2,
    linewidth: float = 2,
    colorscale: str | list | None = None,
    cmin: int | float | None = None,
    cmax: int | float | None = None,
    opacity: float | None = None,
    hovertext: HoverTextInput = None,
    hovertemplate: str | None = None,
    dim: Literal[2, 3] = 3,
    mode: str = "markers",
    marker: dict[str, Any] | None = None,
    line: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[go.Scatter] | list[go.Scatter3d]:
    """Dispatch to the explicit 2D or 3D point-scatter implementation.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2+)`` array of point coordinates.
    color : Union[str, int, float, Sequence], optional
        Color of markers or lines. Can be one shared scalar value or one value
        per point.
    markersize : Union[int, float, Sequence], default 2
        Marker size, provided either as one shared numeric value or one value
        per point.
    linewidth : float, default 2
        Line width.
    colorscale : Union[str, list], optional
        Plotly colorscale specifier.
    cmin : Union[int, float], optional
        Minimum of the color range.
    cmax : Union[int, float], optional
        Maximum of the color range.
    opacity : float, optional
        Marker opacity.
    hovertext : Union[int, float, str, Sequence], optional
        Marker hover labels. Can be one shared label or one label per point.
    hovertemplate : str, optional
        Plotly hover formatting string.
    dim : Literal[2, 3], default 3
        Requested display dimension. If ``points`` only has two coordinate
        columns, the 2D implementation is used regardless.
    mode : str, default ``"markers"``
        Plotly drawing mode.
    marker : dict, optional
        Plotly marker style dictionary.
    line : dict, optional
        Plotly line style dictionary.
    **kwargs : Any
        Additional keyword arguments forwarded to the concrete Plotly trace.

    Returns
    -------
    Union[list[go.Scatter], list[go.Scatter3d]]
        Single homogeneous trace sequence produced by the 2D or 3D helper.
    """
    if dim not in [2, 3]:
        raise ValueError("This function only supports dimension 2 or 3.")

    if dim == 2 or points.shape[1] == 2:
        return scatter_points_2d(
            points,
            color=color,
            markersize=markersize,
            linewidth=linewidth,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            hovertext=hovertext,
            hovertemplate=hovertemplate,
            mode=mode,
            marker=marker,
            line=line,
            **kwargs,
        )

    return scatter_points_3d(
        points,
        color=color,
        markersize=markersize,
        linewidth=linewidth,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        opacity=opacity,
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        mode=mode,
        marker=marker,
        line=line,
        **kwargs,
    )
