"""Module to convert a point cloud into an ellipsoidal envelope."""

import time

import numpy as np
from scipy.special import gammaincinv # pylint: disable=E0611
import plotly.graph_objs as go


def ellipsoid_trace(points=None, centroid=None, covmat=None, contour=0.5,
                    num_samples=10, color=None, intensity=None, hovertext=None,
                    showscale=False, **kwargs):
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
    color : Union[str, float], optional
        Color of ellipse
    intensity : Union[int, float], optional
        Color intensity of the box along the colorscale axis
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with the box
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    **kwargs : dict, optional
        Additional parameters to pass to the underlying
        :class:`plotly.graph_objs.Mesh3d` object
    """
    # Ensure that either a cloud of points or a covariance matrix is provided
    assert (points is not None) ^ (centroid is not None and covmat is not None), (
            "Must provide either `points` or both `centroid` and `covmat`.")

    # Compute the points on a unit sphere
    phi = np.linspace(0, 2*np.pi, num=num_samples)
    theta = np.linspace(-np.pi/2, np.pi/2, num=num_samples)
    phi, theta = np.meshgrid(phi, theta)
    x = np.cos(theta) * np.sin(phi)
    y = np.cos(theta) * np.cos(phi)
    z = np.sin(theta)
    unit_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Get the centroid and the covariance matrix, if needed
    if points is not None:
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
    radius = 1.
    if contour is not None:
        assert contour > 0. and contour < 1., (
                "The `contour` parameter should be a probability.")
        radius = np.sqrt(2*gammaincinv(1.5, contour))

    ell_points = centroid + radius*np.dot(unit_points, rotmat)

    # Convert the color provided to a set of intensities, if needed
    if color is not None and not isinstance(color, str):
        assert intensity is None, (
                "Must not provide both `color` and `intensity`.")
        intensity = np.full(len(ell_points), color)
        color = None

    # Update hovertemplate style
    hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z}'
    if hovertext is not None:
        if not np.isscalar(hovertext):
            hovertemplate += '<br>%{text}'
        else:
            hovertemplate += f'<br>{hovertext}'
            hovertext = None

    # Append Mesh3d object
    return go.Mesh3d(
        x=ell_points[:, 0], y=ell_points[:, 1], z=ell_points[:, 2],
        color=color, intensity=intensity, alphahull=0, showscale=showscale,
        hovertemplate=hovertemplate, **kwargs)


def ellipsoid_traces(centroids, covmat, color=None, hovertext=None, cmin=None,
                     cmax=None, shared_legend=True, legendgroup=None,
                     showlegend=True, name=None, **kwargs):
    """Function which produces a list of plotly traces of ellipsoids given a
    list of centroids and one covariance matrix in x, y and z.

    Parameters
    ----------
    centroids : np.ndarray
        (N, 3) Positions of each of the ellipsoid centroids
    covmat : np.ndarray
        (3, 3) Covariance matrix which defines any of the base ellipsoid shape
    color : Union[str, np.ndarray], optional
        Color of ellipsoids or list of color of ellispoids
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with every ellipsoid or each ellipsoid
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    shared_legend : bool, default True
        If True, the plotly legend of all ellipsoids is shared as one
    legendgroup : str, optional
        Legend group to be shared between all boxes
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
    assert color is None or np.isscalar(color) or len(color) == len(centroids), (
            "Specify one color for all ellipsoids, or one color per ellipsoid.")
    assert (hovertext is None or np.isscalar(hovertext) or
            len(hovertext) == len(centroids)), (
            "Specify one hovertext for all ellipsoids, or one hovertext per "
            "ellipsoid.")

    # If one color is provided per ellipsoid, give an associated hovertext
    if hovertext is None and color is not None and not np.isscalar(color):
        hovertext = [f'Value: {v:0.3f}' for v in color]

    # If cmin/cmax are not provided, must build them so that all ellipsoids
    # share the same colorscale range (not guaranteed otherwise)
    if color is not None and not np.isscalar(color) and len(color) > 0:
        if cmin is None:
            cmin = np.min(color)
        if cmax is None:
            cmax = np.max(color)

    # If the legend is to be shared, make sure there is a common legend group
    if shared_legend and legendgroup is None:
        legendgroup = 'group_' + str(time.time())

    # Loop over the list of ellipsoid centroids
    traces = []
    col, hov = color, hovertext
    for i, centroid in enumerate(centroids):
        # Fetch the right color/hovertext combination
        if color is not None and not np.isscalar(color):
            col = color[i]
        if hovertext is not None and not np.isscalar(hovertext):
            hov = hovertext[i]

        # If the legend is shared, only draw the legend of the first trace
        if shared_legend:
            showlegend = showlegend and i == 0
            name_i = name
        else:
            name_i = f'{name} {i}'

        # Append list of traces
        traces.append(ellipsoid_trace(
            centroid=centroid, covmat=covmat, contour=None, color=col,
            hovertext=hov, cmin=cmin, cmax=cmax, legendgroup=legendgroup,
            showlegend=showlegend, name=name_i, **kwargs))

    return traces
