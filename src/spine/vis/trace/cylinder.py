"""Module to draw cylinders."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import plotly.graph_objs as go

from .utils import (
    ColorInput,
    HoverTextInput,
    IntensityInput,
    NumericOrSequence,
    is_scalar_sequence,
    require_matching_length,
    rotation_matrix_from_z,
    select_numeric_or_sequence,
    select_scalar_or_sequence,
)

__all__ = ["cylinder_trace", "cylinder_traces"]


def cylinder_trace(
    centroid: np.ndarray,
    axis: np.ndarray,
    height: float,
    diameter: float,
    num_samples: int = 10,
    color: ColorInput = None,
    intensity: IntensityInput = None,
    hovertext: HoverTextInput = None,
    showscale: bool = False,
    **kwargs: Any,
) -> go.Mesh3d:
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
    color : Union[str, int, float, Sequence], optional
        Color of the cylinder. Can be a single Plotly color or numeric value.
    intensity : Union[int, float, Sequence], optional
        Color intensity of the cylinder along the colorscale axis. Can be a
        single numeric value or a per-vertex sequence.
    hovertext : Union[int, float, str, Sequence], optional
        Text associated with the cylinder. Can be a scalar label or a
        per-vertex sequence of labels.
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
    rotmat = rotation_matrix_from_z(axis)

    # Compute the scaling vectors for radius and height
    scale = np.diag([diameter, diameter, height])

    # Compute the cylinder points
    cyl_points = centroid + np.dot(unit_points.dot(scale), rotmat)

    # Convert the color provided to a set of intensities, if needed
    if color is not None and not isinstance(color, str):
        if intensity is not None:
            raise ValueError("Must not provide both `color` and `intensity`.")
        intensity = np.full(len(cyl_points), color)
        color = None

    # Update hovertemplate style
    hovertemplate = "x: %{x}<br>y: %{y}<br>z: %{z}"
    if hovertext is not None:
        if is_scalar_sequence(hovertext):
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
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        **kwargs,
    )


def cylinder_traces(
    centroids: np.ndarray,
    axis: np.ndarray,
    height: NumericOrSequence,
    diameter: NumericOrSequence,
    color: ColorInput = None,
    hovertext: HoverTextInput = None,
    cmin: float | None = None,
    cmax: float | None = None,
    shared_legend: bool = True,
    legendgroup: str | None = None,
    showlegend: bool = True,
    name: str | None = None,
    **kwargs: Any,
) -> list[go.Mesh3d]:
    """Function which produces a list of plotly traces of cylinders given a
    list of centroids and one shared or per-cylinder geometric description.

    Parameters
    ----------
    centroids : np.ndarray
        (N, 3) Positions of each of the cylinder centroids
    axis : np.ndarray
        (3,) or (N, 3) Axis direction of the cylinders
    height : Union[int, float, Sequence]
        Height of the cylinders, either as one shared value or one value per
        cylinder.
    diameter : Union[int, float, Sequence]
        Diameter of the cylinders, either as one shared value or one value per
        cylinder.
    color : Union[str, int, float, Sequence], optional
        Color of the cylinders, either as one shared value or one value per
        cylinder.
    hovertext : Union[int, float, str, Sequence], optional
        Text associated with the cylinders, either as one shared label or one
        label per cylinder.
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
        Cylinder traces
    """
    # Check the parameters
    if axis.shape != (3,) and axis.shape != (len(centroids), 3):
        raise ValueError(
            "Specify one axis for all cylinders, or one axis per cylinder."
        )
    require_matching_length(
        height,
        len(centroids),
        "Specify one height for all cylinders, or one height per cylinder.",
    )
    require_matching_length(
        diameter,
        len(centroids),
        "Specify one diameter for all cylinders, or one diameter per cylinder.",
    )
    require_matching_length(
        color,
        len(centroids),
        "Specify one color for all cylinders, or one color per cylinder.",
    )
    require_matching_length(
        hovertext,
        len(centroids),
        "Specify one hovertext for all cylinders, or one hovertext per cylinder.",
    )

    # If one color is provided per cylinder, give an associated hovertext
    if hovertext is None and is_scalar_sequence(color):
        hovertext = [f"Value: {v:0.3f}" for v in color]

    # If cmin/cmax are not provided, must build them so that all cylinders
    # share the same colorscale range (not guaranteed otherwise)
    if color is not None and is_scalar_sequence(color):
        if len(color) > 0:
            if cmin is None:
                cmin = np.min(np.asarray(color))
            if cmax is None:
                cmax = np.max(np.asarray(color))

    # If the legend is to be shared, make sure there is a common legend group
    if shared_legend and legendgroup is None:
        legendgroup = "group_" + str(time.time())

    # Loop over the list of cylinder centroids
    traces = []
    col, hov = color, hovertext
    for i, centroid in enumerate(centroids):
        # Fetch the right color/hovertext combination
        col = select_scalar_or_sequence(color, i)
        hov = select_scalar_or_sequence(hovertext, i)

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
        height_i = select_numeric_or_sequence(height, i)
        diameter_i = select_numeric_or_sequence(diameter, i)

        # Append list of traces
        traces.append(
            cylinder_trace(
                centroid=centroid,
                axis=axis_i,
                height=float(height_i),
                diameter=float(diameter_i),
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
