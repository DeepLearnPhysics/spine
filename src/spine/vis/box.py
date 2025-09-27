"""Defines functions used to draw finite-sized boxes.

These tools are typically used to represent the extent of a voxel or
a voxel neighborhood in an image. In the context of the Point-Proposal Network,
this helps represent the region proposed by the network at layers
deeper than the original resolution of the image.

The :func:`box_trace` function is also used to represent the extent of the
active volume of the modules that make up a detector.
"""

import time

import numpy as np
import plotly.graph_objs as go

__all__ = ["scatter_boxes"]


def box_trace(
    lower,
    upper,
    draw_faces=False,
    line=None,
    linewidth=None,
    color=None,
    cmin=None,
    cmax=None,
    colorscale=None,
    intensity=None,
    hovertext=None,
    showscale=False,
    **kwargs,
):
    """Function which produces a plotly trace of a box given its lower bounds
    and upper bounds in x, y and z.

    Parameters
    ----------
    lower : np.ndarray
        (3) Vector of lower boundaries in x, z and z
    upper : np.ndarray
        (3) Vector of upper boundaries in x, z and z
    draw_faces : bool, default False
        Weather or not to draw the box faces, or only the edges
    line : dict, optional
        Dictionary which specifies box line properties
    linewidth : int, optional
        Width of the box edge lines
    color : Union[str, np.ndarray], optional
        Color of box
    cmin : float, optional
        Minimum value of the color range
    cmax : float, optional
        Maximum value of the color range
    colorscale : Union[str, dict]
        Colorscale
    intensity : Union[int, float], optional
        Color intensity of the box along the colorscale axis
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with the box
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    **kwargs : dict, optional
        List of additional arguments to pass to
        :class:`plotly.graph_objs.Scatter3D` or
        :class:`plotly.graph_objs.Mesh3D`, depending on what the `draw_faces`
        parameter is set to.

    Returns
    -------
    Union[plotly.graph_objs.Scatter3D, plotly.graph_objs.Mesh3D]
        Box trace
    """
    # Check the parameters
    assert (
        len(lower) == len(upper) == 3
    ), "Must specify 3 values for both lower and upper boundaries."
    assert np.all(
        np.asarray(upper) > np.asarray(lower)
    ), "Each upper boundary should be greater than its lower counterpart."

    # List of box vertices in the edges that join them in the box mesh
    box_vertices = np.array(
        [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]]
    ).T
    box_edge_index = np.array(
        [[0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6], [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]]
    )
    box_tri_index = np.array(
        [
            [0, 6, 3, 2, 7, 6, 1, 1, 5, 5, 6, 7],
            [6, 0, 0, 0, 4, 4, 7, 7, 4, 0, 2, 3],
            [2, 4, 1, 3, 5, 7, 5, 3, 0, 1, 7, 2],
        ]
    )

    # List of scaled vertices
    vertices = lower + box_vertices * (upper - lower)

    # Update hovertemplate style
    hovertemplate = "x: %{x}<br>y: %{y}<br>z: %{z}"
    if hovertext is not None:
        if not np.isscalar(hovertext):
            hovertemplate += "<br>%{text}"
        else:
            hovertemplate += f"<br>{hovertext}"
            hovertext = None

    # Create the box trace
    if not draw_faces:
        # Build a list of box edges to draw (padded with None values to break
        # them from each other)
        edges = np.full((3 * box_edge_index.shape[1], 3), None)
        edges[np.arange(0, edges.shape[0], 3)] = vertices[box_edge_index[0]]
        edges[np.arange(1, edges.shape[0], 3)] = vertices[box_edge_index[1]]

        # Build a line property, if needed
        if (
            color is not None
            or linewidth is not None
            or cmin is not None
            or cmax is not None
            or colorscale is not None
        ):
            assert line is None, (
                "Must not specify `line` when providing `color`, "
                "`linewidth`, `cmin` or `cmax` independently."
            )
            if color is not None and not isinstance(color, str):
                color = np.full(len(edges), color)

            line = {
                "color": color,
                "width": linewidth,
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": colorscale,
            }

        # Return trace
        trace = go.Scatter3d(
            x=edges[:, 0],
            y=edges[:, 1],
            z=edges[:, 2],
            mode="lines",
            line=line,
            hovertext=hovertext,
            hovertemplate=hovertemplate,
            **kwargs,
        )

    else:
        # If the color is a number, must be specified as an intensity
        if color is not None and not isinstance(color, str):
            assert intensity is None, "Must not provide both `color` and `intensity`."
            intensity = np.full(len(vertices), color)
            color = None

        trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=box_tri_index[0],
            j=box_tri_index[1],
            k=box_tri_index[2],
            color=color,
            intensity=intensity,
            showscale=showscale,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            hovertext=hovertext,
            hovertemplate=hovertemplate,
            **kwargs,
        )

    # Return trace
    return trace


def box_traces(
    lowers,
    uppers,
    draw_faces=False,
    color=None,
    linewidth=None,
    hovertext=None,
    cmin=None,
    cmax=None,
    shared_legend=True,
    legendgroup=None,
    showlegend=True,
    group_name=None,
    name=None,
    **kwargs,
):
    """Function which produces a list of plotly traces of boxes given a list of
    lower bounds and upper bounds in x, y and z.

    Parameters
    ----------
    lowers : np.ndarray
        (N, 3) List of vector of lower boundaries in x, z and z
    uppers : np.ndarray
        (N, 3) List of vector of upper boundaries in x, z and z
    draw_faces : bool, default False
        Weather or not to draw the box faces, or only the edges
    color : Union[str, np.ndarray], optional
        Color of boxes or list of color of boxes
    linewidth : int, default 2
        Width of the box edge lines
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with every box or each box
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    shared_legend : bool, default True
        If True, the plotly legend of all boxes is shared as one
    legendgroup : str, optional
        Legend group to be shared between all boxes
    showlegend : bool, default `True`
        Whether to show legends on not
    name : str, optional
        Name of the trace(s)
    **kwargs : dict, optional
        List of additional arguments to pass to
        :class:`plotly.graph_objs.Scatter3D` or
        :class:`plotly.graph_objs.Mesh3D`, depending on what the `draw_faces`
        parameter is set to.

    Returns
    -------
    Union[List[plotly.graph_objs.Scatter3D], List[plotly.graph_objs.Mesh3D]]
        Box traces
    """
    # Check the parameters
    assert len(lowers) == len(
        uppers
    ), "Provide as many upper boundary vector as their lower counterpart."
    assert (
        color is None or np.isscalar(color) or len(color) == len(lowers)
    ), "Specify one color for all boxes, or one color per box."
    assert (
        hovertext is None or np.isscalar(hovertext) or len(hovertext) == len(lowers)
    ), "Specify one hovertext for all boxes, or one hovertext per box."

    # If one color is provided per box, give an associated hovertext
    if hovertext is None and color is not None and not np.isscalar(color):
        hovertext = [f"Value: {v:0.3f}" for v in color]

    # If cmin/cmax are not provided, must build them so that all boxes
    # share the same colorscale range (not guaranteed otherwise)
    if color is not None and not np.isscalar(color) and len(color) > 0:
        if cmin is None:
            cmin = np.min(color)
        if cmax is None:
            cmax = np.max(color)

    # If the legend is to be shared, make sure there is a common legend group
    if shared_legend and legendgroup is None:
        legendgroup = "group_" + str(time.time())

    # Loop over the list of box boundaries
    traces = []
    for i, (lower, upper) in enumerate(zip(lowers, uppers)):
        # Fetch the right color/hovertext combination
        col, hov = color, hovertext
        if color is not None and not np.isscalar(color):
            col = color[i]
        if hovertext is not None and not np.isscalar(hovertext):
            hov = hovertext[i]

        # If the legend is shared, only draw the legend of the first trace
        if shared_legend:
            showlegend = showlegend and i == 0
            name_i = name
        else:
            name_i = f"{name} {i}"

        # Append list of traces
        traces.append(
            box_trace(
                lower,
                upper,
                draw_faces,
                linewidth=linewidth,
                color=col,
                hovertext=hov,
                cmin=cmin,
                cmax=cmax,
                legendgroup=legendgroup,
                showlegend=showlegend,
                name=name_i,
                **kwargs,
            )
        )

    return traces


def scatter_boxes(
    coords,
    dimension,
    draw_faces=True,
    color="orange",
    hovertext=None,
    linewidth=2,
    shared_legend=True,
    **kwargs,
):
    """Function which produces a list of plotly traces of boxes given a list of
    coordinates and a box dimension.

    This function assumes that the coordinates represent the lower bounds of
    the voxels they point at. This follows the `MinkowskiEngine` convention,
    which is the package used for space convolutions. This can be used to
    represent the PPN regions of interest in a space compressed by a factor
    (b_x, b_y, b_z) from the original image resolution.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) Coordinates of in multiples of box lengths in each dimension
    dimension : Union[float, np.ndarray]
        Dimensions of the boxes. Specify it as either a single number (for
        cubes) or an array of values in each dimension, i.e. (b_x, b_y, b_z)
    draw_faces : bool, default True
        Weather or not to draw the box faces, or only the edges
    color : Union[str, np.ndarray], default 'orange'
        Color of boxes or list of color of boxes
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with every box or each box
    linewidth : int, default 2
        Width of the box edge lines
    shared_legend : bool, default True
        If True, the plotly legend of all boxes is shared as one
    **kwargs : dict, optional
        List of additional arguments to pass to
        :class:`plotly.graph_objs.Scatter3D` or
        :class:`plotly.graph_objs.Mesh3D`, depending on what the `draw_faces`
        parameter is set to.

    Returns
    -------
    Union[List[plotly.graph_objs.Scatter3D], List[plotly.graph_objs.Mesh3D]]
        Box traces
    """
    # Check the input
    if not np.isscalar(dimension):
        assert len(dimension) == 3, "Must specify three dimensions for the box size."
        dimension = np.asarray(dimension)

    # Compute the lower and upper boundaries, return box traces
    lowers = coords
    uppers = coords + dimension

    return box_traces(
        lowers, uppers, draw_faces, color, linewidth, hovertext, shared_legend, **kwargs
    )
