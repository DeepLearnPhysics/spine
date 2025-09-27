"""Tools to draw a (labeled) point cloud."""

import numpy as np
import plotly.graph_objs as go

from spine.utils.globals import COORD_COLS

__all__ = ["scatter_points"]


def scatter_points(
    points,
    color=None,
    markersize=2,
    linewidth=2,
    colorscale=None,
    cmin=None,
    cmax=None,
    opacity=None,
    hovertext=None,
    hovertemplate=None,
    dim=3,
    mode="markers",
    marker=None,
    line=None,
    **kwargs,
):
    """Scatters points and their labels.

    Produces :class:`plotly.graph_objs.Scatter3d` or
    :class:`plotly.graph_objs.Scatter` trace object to be drawn in plotly. The
    object is nested to be fed directly to a :class:`plotly.graph_objs.Figure`
    or :func:`plotly.offline.iplot`. All of the regular plotly parameters are
    available.

    It can scatter points individually (default `mode`) or it can draw lines
    between the provided points (`mode='lines'` option).

    Parameters
    ----------
    points : np.ndarray
        (N, 2+) array of N points of (..., x, y, [z],...) coordinate information
    color : Union[str, np.ndarray], optional
        Color of markers/lines or (N) list of color of markers/lines
    markersize : float, default 2
        Marker size
    linewidth : float, default 2
        Line width
    colorscale : Union[str, List[str], List[List[float, str]], optional
        Plotly colorscale specifier for the markers
    cmin : Union[int, float], optional
        Minimum of the color range
    cmax : Union[int, float], optional
        Maximum of the color range
    opacity : float
        Marker opacity
    hovertext : Union[List[str], List[int]], optional
        (N) List of labels associated with each marker
    hovertemplate : str, optional
        Hover information formatting
    dim : int, default 3
        Dimension (can either be 2 or 3)
    mode : str, default 'markers'
        Drawing mode
    marker : dict, optional
        Marker style configuration dictionary
    line : dict, optional
       Line style configuration dictionary
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D

    Returns
    -------
    List[go.Scatter3d]
        (1) List with one graph of the input points
    """
    # Check the dimension for compatibility
    if dim not in [2, 3]:
        raise ValueError("This function only supports dimension 2 or 3.")
    if points.shape[1] == 2:
        dim = 2

    # Get the coordinate column locations in the input tensor
    coord_cols = COORD_COLS
    if dim == 2:
        coord_cols = COORD_COLS[:2]
    if points.shape[1] == dim:
        coord_cols = np.arange(dim)

    # If there is no hovertext, print the color as part of the hovertext
    if hovertext is None and color is not None and not isinstance(color, str):
        hovertext = [f"Value: {c}" for c in color]

    # Update hovertemplate
    if hovertemplate is None:
        hovertemplate = "x: %{x}<br>y: %{y}"
        if dim == 3:
            hovertemplate += "<br>z: %{z}"
        if hovertext is not None:
            if not np.isscalar(hovertext):
                hovertemplate += "<br>%{text}"
            else:
                hovertemplate += f"<br>{hovertext}"
                hovertext = None

    # Initialize the marker/line object depending on mode
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
        if dim == 2:
            line = {"width": linewidth, "color": color}
        else:
            line = {
                "width": linewidth,
                "color": color,
                "colorscale": colorscale,
                "cmin": cmin,
                "cmax": cmax,
            }

    # Initialize and return
    axes = ["x", "y", "z"][:dim]
    pos_dict = {a: points[:, coord_cols[i]] for i, a in enumerate(axes)}

    if dim == 2:
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
    else:
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
