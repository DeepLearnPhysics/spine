"""Module to convert a point cloud into an ellipsoidal envelope."""

import numpy as np
from scipy.special import gammaincinv # pylint: disable=E0611
import plotly.graph_objs as go


def ellipsoid_trace(points, contour=0.5, num_samples=10, color=None,
                    showscale=False, **kwargs):
    """Converts a cloud of points into a 3D ellipsoid.

    This function uses the centroid and the covariance matrix of a cloud of
    points to define an ellipsoid which would encompass a user-defined fraction
    `contour` of all the points, were the points distributed following
    a 3D Gaussian.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) Array of point coordinates
    contour : float, default 0.5
        Fraction of the points contained in the ellipsoid, under the
        Gaussian distribution assumption
    num_samples : int, default 10
        Number of points sampled along theta and phi in the spherical coordinate
        system of the ellipsoid. A larger number increases the resolution.
    color : Union[str, float], optional
        Color of ellipse
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    **kwargs : dict, optional
        Additional parameters to pass to the 
    """
    # Compute the points on a unit sphere
    phi = np.linspace(0, 2*np.pi, num=num_samples)
    theta = np.linspace(-np.pi/2, np.pi/2, num=num_samples)
    phi, theta = np.meshgrid(phi, theta)
    x = np.cos(theta) * np.sin(phi)
    y = np.cos(theta) * np.cos(phi)
    z = np.sin(theta)
    unit_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Get the centroid and the covariance matrix
    centroid = np.mean(points, axis=0)
    covmat = np.cov((points - centroid).T)

    # Diagonalize the covariance matrix, get rotation matrix
    w, v = np.linalg.eigh(covmat)
    diag = np.diag(np.sqrt(w))
    rotmat = np.dot(diag, v.T)

    # Compute the radius corresponding to the contour probability and rotate
    # the points into the basis of the covariance matrix
    assert contour > 0. and contour < 1., (
            "The `contour` parameter should be a probability.")
    radius = np.sqrt(2*gammaincinv(1.5, contour))
    ell_points = centroid + radius*np.dot(unit_points, rotmat)

    # Convert the color provided to a set of intensities
    intensity = None
    if color is not None:
        assert np.isscalar('color'), (
                "Should provide a single color for the ellipsoid.")
        intensity = [color]*len(ell_points)

    # Append Mesh3d object
    return go.Mesh3d(
        x=ell_points[:, 0], y=ell_points[:, 1], z=ell_points[:, 2],
        intensity=intensity, alphahull=0, showscale=showscale, **kwargs)
