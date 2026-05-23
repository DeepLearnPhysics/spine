"""Module to convert a point cloud into an ellipsoidal envelope."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import plotly.graph_objs as go
from scipy.special import gammaincinv  # pylint: disable=E0611

from .utils import (
    ColorInput,
    HoverTextInput,
    IntensityInput,
    is_scalar_sequence,
    require_matching_length,
    select_scalar_or_sequence,
)

__all__ = ["ellipsoid_trace", "ellipsoid_traces"]


def ellipsoid_trace(
    points: np.ndarray | None = None,
    centroid: np.ndarray | None = None,
    covmat: np.ndarray | None = None,
    contour: float = 0.5,
    num_samples: int = 10,
    color: ColorInput = None,
    intensity: IntensityInput = None,
    hovertext: HoverTextInput = None,
    showscale: bool = False,
    **kwargs: Any,
) -> go.Mesh3d:
    """Converts a cloud of points or a covariance matrix into a 3D ellipsoid.

    This function uses the centroid and the covariance matrix of a cloud of
    points to define an ellipsoid which would encompass a user-defined fraction
    `contour` of all the points, were the points distributed following
    a 3D Gaussian.

    Parameters
    ----------
    points : np.ndarray, optional
        (N, 3) Array of point coordinates
    centroid : np.ndarray, optional
        (3) Centroid
    covmat : np.ndarray, optional
        (3, 3) Covariance matrix which defines the ellipsoid shape
    contour : float, default 0.5
        Fraction of the points contained in the ellipsoid, under the
        Gaussian distribution assumption
    num_samples : int, default 10
        Number of points sampled along theta and phi in the spherical coordinate
        system of the ellipsoid. A larger number increases the resolution.
    color : Union[str, int, float, Sequence], optional
        Color of the ellipsoid. Can be a single Plotly color or numeric value.
    intensity : Union[int, float, Sequence], optional
        Color intensity of the ellipsoid along the colorscale axis. Can be a
        single numeric value or a per-vertex sequence.
    hovertext : Union[int, float, str, Sequence], optional
        Text associated with the ellipsoid. Can be a scalar label or a
        per-vertex sequence of labels.
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    **kwargs : dict, optional
        Additional parameters to pass to the underlying
        :class:`plotly.graph_objs.Mesh3d` object
    """
    # Ensure that either a cloud of points or a covariance matrix is provided
    if (points is not None) == (centroid is not None and covmat is not None):
        raise ValueError(
            "Must provide either `points` or both `centroid` and `covmat`."
        )

    # Compute the points on a unit sphere
    phi = np.linspace(0, 2 * np.pi, num=num_samples)
    theta = np.linspace(-np.pi / 2, np.pi / 2, num=num_samples)
    phi, theta = np.meshgrid(phi, theta)
    x = np.cos(theta) * np.sin(phi)
    y = np.cos(theta) * np.cos(phi)
    z = np.sin(theta)
    unit_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Get the centroid and the covariance matrix, if needed
    covmat_provided = True
    if covmat is None:
        covmat_provided = False
        if points is None or len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError("The `points` parameter should be a (N, 3) array.")

        if len(points) > 1:
            centroid = np.mean(points, axis=0)
            covmat = np.cov((points - centroid).T)
        else:
            centroid = points[0]
            covmat = np.zeros((3, 3))

    # Diagonalize the covariance matrix, get rotation matrix
    w, v = np.linalg.eigh(covmat)
    diag = np.diag(np.sqrt(w))
    rotmat = np.dot(diag, v.T)

    # Compute the radius corresponding to the contour probability and rotate
    # the points into the basis of the covariance matrix
    radius = 1.0
    if not covmat_provided:
        if not 0.0 < contour < 1.0:
            raise ValueError("The `contour` parameter should be a probability.")
        radius = np.sqrt(2 * gammaincinv(1.5, contour))

    ell_points = centroid + radius * np.dot(unit_points, rotmat)

    # Convert the color provided to a set of intensities, if needed
    if color is not None and not isinstance(color, str):
        if intensity is not None:
            raise ValueError("Must not provide both `color` and `intensity`.")
        intensity = np.full(len(ell_points), color)
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
        x=ell_points[:, 0],
        y=ell_points[:, 1],
        z=ell_points[:, 2],
        color=color,
        intensity=intensity,
        alphahull=0,
        showscale=showscale,
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        **kwargs,
    )


def ellipsoid_traces(
    centroids: np.ndarray,
    covmat: np.ndarray,
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
    """Function which produces a list of plotly traces of ellipsoids given a
    list of centroids and one covariance matrix in x, y and z.

    Parameters
    ----------
    centroids : np.ndarray
        (N, 3) Positions of each of the ellipsoid centroids
    covmat : np.ndarray
        (3, 3) Covariance matrix which defines any of the base ellipsoid shape
    color : Union[str, int, float, Sequence], optional
        Color of the ellipsoids, either as one shared value or one value per
        ellipsoid.
    hovertext : Union[int, float, str, Sequence], optional
        Text associated with the ellipsoids, either as one shared label or one
        label per ellipsoid.
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    shared_legend : bool, default True
        If True, the plotly legend of all ellipsoids is shared as one
    legendgroup : str, optional
        Legend group to be shared between all ellipsoids
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
    require_matching_length(
        color,
        len(centroids),
        "Specify one color for all ellipsoids, or one color per ellipsoid.",
    )
    require_matching_length(
        hovertext,
        len(centroids),
        "Specify one hovertext for all ellipsoids, or one hovertext per ellipsoid.",
    )

    # If one color is provided per ellipsoid, give an associated hovertext
    if hovertext is None and is_scalar_sequence(color):
        hovertext = [f"Value: {v:0.3f}" for v in color]

    # If cmin/cmax are not provided, must build them so that all ellipsoids
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

    # Loop over the list of ellipsoid centroids
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

        # Append list of traces
        traces.append(
            ellipsoid_trace(
                centroid=centroid,
                covmat=covmat,
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
