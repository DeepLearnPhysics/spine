"""Module to draw cylinders."""

import time

import numpy as np
import plotly.graph_objs as go

__all__ = ["cylinder_traces"]


def cylinder_trace(
    centroid,
    axis,
    height,
    diameter,
    num_samples=10,
    color=None,
    intensity=None,
    hovertext=None,
    showscale=False,
    **kwargs,
):
    """Draw a cylinder centered at a given position.

    Parameters
    ----------
    centroid : np.ndarray
        (3) Centroid of the cylinder
    axis : np.ndarray
        (3) Axis direction of the cylinder
    height : float
        Height of the cylinder
    diameter : float
        Diameter of the cylinder
    num_samples : int, default 10
        Number of points sampled along theta and h in the cylindrical coordinate
        system of the cylinder. A larger number increases the resolution.
    color : Union[str, float], optional
        Color of cylinder
    intensity : Union[int, float], optional
        Color intensity of the cylinder along the colorscale axis
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with the cylinder
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    **kwargs : dict, optional
        Additional parameters to pass to the underlying
        :class:`plotly.graph_objs.Mesh3d` object
    """
    # Compute the points on a unit cylinder
    phi = np.linspace(0, 2 * np.pi, num=num_samples)
    h = np.linspace(-0.5, 0.5, num=num_samples)
    phi, h = np.meshgrid(phi, h)
    x = 0.5 * np.cos(phi)
    y = 0.5 * np.sin(phi)
    z = h
    unit_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Compute the rotation matrix which aligns the z axis to the cylinder axis
    axis = axis / np.linalg.norm(axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    rotmat = np.eye(3)
    if (axis != z_axis).any():
        v = np.cross(z_axis, axis)
        c = np.dot(z_axis, axis)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotmat = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

    # Compute the scaling vectors for radius and height
    scale = np.diag([diameter, diameter, height])

    # Compute the cylinder points
    cyl_points = centroid + np.dot(unit_points.dot(scale), rotmat)

    # Convert the color provided to a set of intensities, if needed
    if color is not None and not isinstance(color, str):
        assert intensity is None, "Must not provide both `color` and `intensity`."
        intensity = np.full(len(cyl_points), color)
        color = None

    # Update hovertemplate style
    hovertemplate = "x: %{x}<br>y: %{y}<br>z: %{z}"
    if hovertext is not None:
        if not np.isscalar(hovertext):
            hovertemplate += "<br>%{text}"
        else:
            hovertemplate += f"<br>{hovertext}"
            hovertext = None

    # Append Mesh3d object
    return go.Mesh3d(
        x=cyl_points[:, 0],
        y=cyl_points[:, 1],
        z=cyl_points[:, 2],
        color=color,
        intensity=intensity,
        alphahull=0,
        showscale=showscale,
        hovertemplate=hovertemplate,
        **kwargs,
    )


def cylinder_traces(
    centroids,
    axis,
    height,
    diameter,
    color=None,
    hovertext=None,
    cmin=None,
    cmax=None,
    shared_legend=True,
    legendgroup=None,
    showlegend=True,
    name=None,
    **kwargs,
):
    """Function which produces a list of plotly traces of cylinders given a
    list of centroids and one covariance matrix in x, y and z.

    Parameters
    ----------
    centroids : np.ndarray
        (N, 3) Positions of each of the cylinder centroids
    axis : np.ndarray
        (3,) or (N, 3) Axis direction of the cylinders
    height : Union[float, np.ndarray]
        Height of the cylinders
    diameter : Union[float, np.ndarray]
        Diameter of the cylinders
    color : Union[str, np.ndarray], optional
        Color of cylinders or list of color of cylinders
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with every cylinder or each cylinder
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    shared_legend : bool, default True
        If True, the plotly legend of all ellipsoids is shared as one
    legendgroup : str, optional
        Legend group to be shared between all cylinders
    showlegend : bool, default `True`
        Whether to show legends on not
    name : str, optional
        Name of the trace(s)
    **kwargs : dict, optional
        List of additional arguments to pass to the underlying list of
        :class:`plotly.graph_objs.Mesh3D`

    Returns
    -------
    Union[List[plotly.graph_objs.Mesh3D]]
        Ellipsoid traces
    """
    # Check the parameters
    assert (
        color is None or np.isscalar(color) or len(color) == len(centroids)
    ), "Specify one color for all cylinders, or one color per cylinder."
    assert (
        hovertext is None or np.isscalar(hovertext) or len(hovertext) == len(centroids)
    ), "Specify one hovertext for all cylinders, or one hovertext per cylinder."
    assert axis.shape == (3,) or axis.shape == (
        len(centroids),
        3,
    ), "Specify one axis for all cylinders, or one axis per cylinder."
    assert np.isscalar(height) or len(height) == len(
        centroids
    ), "Specify one height for all cylinders, or one height per cylinder."
    assert np.isscalar(diameter) or len(diameter) == len(
        centroids
    ), "Specify one diameter for all cylinders, or one diameter per cylinder."

    # If one color is provided per cylinder, give an associated hovertext
    if hovertext is None and isinstance(color, (list, tuple, np.ndarray)):
        hovertext = [f"Value: {v:0.3f}" for v in color]

    # If cmin/cmax are not provided, must build them so that all cylinders
    # share the same colorscale range (not guaranteed otherwise)
    if color is not None and isinstance(color, (list, tuple, np.ndarray)):
        if len(color) > 0:
            if cmin is None:
                cmin = np.min(color)
            if cmax is None:
                cmax = np.max(color)

    # If the legend is to be shared, make sure there is a common legend group
    if shared_legend and legendgroup is None:
        legendgroup = "group_" + str(time.time())

    # Loop over the list of cylinder centroids
    traces = []
    col, hov = color, hovertext
    for i, centroid in enumerate(centroids):
        # Fetch the right color/hovertext combination
        if color is not None and isinstance(color, (list, tuple, np.ndarray)):
            col = color[i]
        if hovertext is not None and isinstance(hovertext, (list, tuple, np.ndarray)):
            hov = hovertext[i]

        # If the legend is shared, only draw the legend of the first trace
        if shared_legend:
            showlegend = showlegend and i == 0
            name_i = name
        else:
            name_i = f"{name} {i}"

        # If any of the axis, height or diameter are arrays, fetch the right one
        axis_i = axis
        if len(axis.shape) == 2:
            axis_i = axis[i]
        height_i = height
        if isinstance(height, (list, tuple, np.ndarray)):
            height_i = height[i]
        diameter_i = diameter
        if isinstance(diameter, (list, tuple, np.ndarray)):
            diameter_i = diameter[i]

        # Append list of traces
        traces.append(
            cylinder_trace(
                centroid=centroid,
                axis=axis_i,
                height=height_i,
                diameter=diameter_i,
                contour=None,
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
