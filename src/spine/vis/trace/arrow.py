"""Module to draw 3D arrows."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from plotly.colors import sample_colorscale
from plotly import graph_objs as go

from .point import scatter_points_3d
from .utils import (
    ColorInput,
    HoverTextInput,
    is_scalar_sequence,
    select_scalar_or_sequence,
)

__all__ = ["scatter_arrows"]


def scatter_arrows(
    points: np.ndarray,
    directions: np.ndarray,
    length: float = 10.0,
    tip_ratio: float = 0.25,
    color: ColorInput = None,
    hovertext: HoverTextInput = None,
    line: dict[str, Any] | None = None,
    linewidth: float = 5,
    colorscale: str | list | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    name: str | None = None,
) -> list[go.Scatter3d | go.Cone]:
    """Converts a list of points and directions into a set of arrows.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) Array of point coordinates
    directions : np.ndarray
        (N, 3) Array of arrow direction vectors
    length : float, default 5.0
        Length of the arrows
    tip_ratio : float, defautl 0.05
        Relative arrow tip size w.r.t. its full length
    color : Union[str, int, float, Sequence], optional
        Color of the arrows, either as one shared scalar value or one value
        per arrow.
    hovertext : Union[int, float, str, Sequence], optional
        Text associated with the arrows, either as one shared label or one
        label per arrow.
    line : dict, optional
        Arrow trunk line property dictionary
    linewidth : float, default 2
        Width of the arrow trunk lines
    colorscale : str or list, optional
        Color scale used to map numeric per-arrow colors.
    cmin : float, optional
        Lower bound of the arrow color scale.
    cmax : float, optional
        Upper bound of the arrow color scale.
    name : name
        Name of the traces
    """
    # Process color and hovertext information for the arrows
    color_trunks, hovertext_trunks = color, hovertext
    if is_scalar_sequence(color):
        if len(color) != len(points):
            raise ValueError(
                "If providing a list of colors for the arrows, "
                "its length must match the number of points."
            )
        color_trunks = np.repeat(np.asarray(color), 3)

    hovertext_arrows = []
    for i, direction in enumerate(directions):
        vx, vy, vz = direction
        ht = f"vx: {vx:0.3f}<br>vy: {vy:0.3f}<br>vz: {vz:0.3f}"
        if hovertext is not None:
            if not is_scalar_sequence(hovertext):
                ht += f"<br>{hovertext}"
            else:
                ht += f"<br>{select_scalar_or_sequence(hovertext, i)}"

        hovertext_arrows.append(ht)

    hovertext_trunks = np.repeat(np.asarray(hovertext_arrows), 3)

    legendgroup = "group_" + str(time.time())

    # Initialize the arrow trunks
    vertices = np.empty((0, 3), dtype=points.dtype)
    if len(points) > 0:
        vertices = []
        for point, direction in zip(points, directions):
            vertices.extend([point, point + length * direction, [None, None, None]])

        vertices = np.vstack(vertices)

    traces = scatter_points_3d(
        vertices,
        color=color_trunks,
        hovertext=hovertext_trunks,
        line=line,
        linewidth=linewidth,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        mode="lines",
        hovertemplate="%{text}",
        name=name,
        legendgroup=legendgroup,
    )

    # Process color information for the arrow tips
    # Initialize the arrow tips. Plotly cones do not support one categorical
    # scalar per cone, so materialize colored tips individually when needed.
    ends = points + (1 - tip_ratio / 2) * length * directions
    directions = tip_ratio * length * directions
    tip_colors = [color] * len(points)
    if is_scalar_sequence(color):
        values = np.asarray(color)
        if values.dtype.kind in "biuf":
            low = np.min(values) if cmin is None and len(values) else cmin
            high = np.max(values) if cmax is None and len(values) else cmax
            low = 0.0 if low is None else low
            high = 1.0 if high is None else high
            fractions = (values - low) / ((high - low) or 1.0)
            color_map = "Viridis" if colorscale is None else colorscale
            tip_colors = sample_colorscale(color_map, fractions)
        else:
            tip_colors = values.tolist()

    for index, tip_color in enumerate(tip_colors):
        tip_color = tip_color if isinstance(tip_color, str) else "black"
        traces.append(
            go.Cone(
                x=ends[index : index + 1, 0],
                y=ends[index : index + 1, 1],
                z=ends[index : index + 1, 2],
                u=directions[index : index + 1, 0],
                v=directions[index : index + 1, 1],
                w=directions[index : index + 1, 2],
                showscale=False,
                showlegend=False,
                sizemode="raw",
                colorscale=[(0, tip_color), (1, tip_color)],
                hovertext=[hovertext_arrows[index]],
                hovertemplate="%{hovertext}",
                name=name,
                legendgroup=legendgroup,
            )
        )

    # Return
    return traces
