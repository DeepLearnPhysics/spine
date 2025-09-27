"""Module to draw 3D arrows."""

import time

import numpy as np
from plotly import graph_objs as go

from .point import scatter_points

__all__ = ["scatter_arrows"]


def scatter_arrows(
    points,
    directions,
    length=10.0,
    tip_ratio=0.25,
    color=None,
    hovertext=None,
    line=None,
    linewidth=5,
    name=None,
):
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
    color : Union[str, float], optional
        Color of the arrow
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with the arrow
    line : dict, optional
        Arrow trunk line property dictionary
    linewidth : float, default 2
        Width of the arrow trunk lines
    name : name
        Name of the traces
    """
    # Process color and hovertext information for the arrows
    color_trunks, hovertext_trunks = color, hovertext
    if color is not None and not np.isscalar(color):
        color_trunks = np.repeat(color, 3)

    hovertext_arrows = []
    for i in range(len(directions)):
        vx, vy, vz = directions[i]
        ht = f"vx: {vx:0.3f}<br>vy: {vy:0.3f}<br>vz: {vz:0.3f}"
        if hovertext is not None:
            if np.isscalar(hovertext):
                ht += f"<br>{hovertext}"
            else:
                ht += f"<br>{hovertext[i]}"

        hovertext_arrows.append(ht)

    hovertext_trunks = np.repeat(hovertext_arrows, 3)

    legendgroup = "group_" + str(time.time())

    # Initialize the arrow trunks
    vertices = np.empty((0, 3), dtype=points.dtype)
    if len(points) > 0:
        vertices = []
        for point, direction in zip(points, directions):
            vertices.extend([point, point + length * direction, [None, None, None]])

        vertices = np.vstack(vertices)

    traces = scatter_points(
        vertices,
        color=color_trunks,
        hovertext=hovertext_trunks,
        line=line,
        linewidth=linewidth,
        mode="lines",
        hovertemplate="%{text}",
        name=name,
        legendgroup=legendgroup,
    )

    # Process color information for the arrow tips
    colorscale = None
    if color is not None and isinstance(color, str):
        colorscale = [(0, color), (1, color)]
    else:
        colorscale = [(0, "black"), (1, "black")]

    # Intitialize the arrow tips
    ends = points + (1 - tip_ratio / 2) * length * directions
    directions = tip_ratio * length * directions
    traces += [
        go.Cone(
            x=ends[:, 0],
            y=ends[:, 1],
            z=ends[:, 2],
            u=directions[:, 0],
            v=directions[:, 1],
            w=directions[:, 2],
            showscale=False,
            showlegend=False,
            sizemode="raw",
            colorscale=colorscale,
            hovertext=hovertext_arrows,
            hovertemplate="%{hovertext}",
            name=name,
            legendgroup=legendgroup,
        )
    ]

    # Return
    return traces
