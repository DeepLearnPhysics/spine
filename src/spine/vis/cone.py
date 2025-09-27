"""Module to convert a point cloud into an cone envelope."""

import numpy as np
from plotly import graph_objs as go

from spine.math.decomposition import principal_components

__all__ = ["cone_trace"]


def cone_trace(
    points,
    fraction=0.5,
    num_samples=10,
    color=None,
    hovertext=None,
    showscale=False,
    **kwargs,
):
    """Converts a cloud of points into a 3D cone.

    This function uses the PCA and the average angle w.r.t. to the point
    of maximum curvature as a basis to construct a cone.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) Array of point coordinates
    fraction : float, default 0.5
        Fraction of the points contained in the cone (angle quantile)
    num_samples : int, default 10
        Number of points sampled along h and phi in the conical coordinate
        system of the cone. A larger number increases the resolution.
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    color : Union[str, float], optional
        Color of the cone
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with the cone
    **kwargs : dict, optional
        Additional parameters to pass to the
    """
    # Get the centroid and the principal components
    centroid = np.mean(points, axis=0)
    pcomp = principal_components(points)
    rotmat = np.flip(pcomp, axis=0)
    paxis = pcomp[0]

    # Collapse point cloud onto principal axis, find end points
    points_pa = np.dot(points, paxis)
    end_ids = np.argmin(points_pa), np.argmax(points_pa)

    # Find the directions w.r.t. each end point
    centroid = np.mean(points, axis=0)
    dirs = np.vstack([centroid - points[end_id] for end_id in end_ids])
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]

    # Find a quantile angle w.r.t. to each direction
    dots = np.zeros((2, len(points)), dtype=points.dtype)
    for i, end_id in enumerate(end_ids):
        for j, point in enumerate(points):
            if j != end_id:
                diff = point - points[end_id]
                dots[i, j] = np.dot(diff / np.linalg.norm(diff), dirs[i])

    assert (
        fraction > 0.0 and fraction < 1.0
    ), "The `fraction` parameter should be a probability."
    angles = np.arccos(dots)
    means = np.mean(angles, axis=1)
    quantiles = np.quantile(angles, fraction, axis=1)

    # The point with the lowest mean angle is the start, select it
    start_id = np.argmin(means)
    start_pos = points[end_ids[start_id]]

    # Define the cone main axis length and its opening angle
    length = abs(points_pa[end_ids[1]] - points_pa[end_ids[0]])
    theta = quantiles[start_id]

    # Compute the points on a cone with half-opening angle theta
    r = np.linspace(0, 1, num=num_samples)
    phi = np.linspace(0, 2 * np.pi, num=num_samples)
    r, phi = np.meshgrid(r, phi)
    x = r * np.tan(theta) * np.cos(phi)
    y = r * np.tan(theta) * np.sin(phi)
    z = r
    unit_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Rotate and offset the cone
    cone_points = start_pos + length * np.dot(unit_points, rotmat)

    # Convert the color provided to a set of intensities
    if color is not None:
        assert np.isscalar("color"), "Should provide a single color for the cone."
        intensity = [color] * len(cone_points)

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
        x=cone_points[:, 0],
        y=cone_points[:, 1],
        z=cone_points[:, 2],
        intensity=intensity,
        alphahull=0,
        showscale=showscale,
        hovertemplate=hovertemplate,
        **kwargs,
    )
