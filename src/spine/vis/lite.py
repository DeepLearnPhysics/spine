"""Produces representations of Lite data structures.

It represents:
- Lite tracks as simple lines
- Lite showers as simple cones
- Lite interactions as collections of lines and cones
"""

import time

import numpy as np
from plotly import graph_objs as go

from spine.utils.globals import TRACK_SHP
from spine.utils.shower import shower_angle_quantile, shower_long_quantile

__all__ = ["scatter_lite"]


def scatter_lite(objects, **kwargs):
    """Produces plotly traces for Lite objects.

    Parameters
    ----------
    objects : List[spine.data.out.OutBase]
        List of lite objects to visualize
    **kwargs : dict, optional
        Additional parameters to pass to the plotly trace objects

    Returns
    -------
    List[object]
        List of plotly trace objects representing the Lite objects
    """
    # Dispatch
    if len(objects) == 0:
        traces = []
    elif hasattr(objects[0], "particles"):
        traces = scatter_lite_interactions(objects, **kwargs)
    else:
        traces = scatter_lite_particles(objects, **kwargs)

    return traces


def scatter_lite_interactions(
    interactions,
    color=None,
    hovertext=None,
    name=None,
    cmin=None,
    cmax=None,
    shared_legend=True,
    **kwargs,
):
    """Produces plotly traces for Lite interactions.

    Parameters
    ----------
    interactions : List[spine.data.out.InteractionLite]
        List of lite interactions to visualize
    color : Union[str, float], optional
        Color of the interaction trace
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with the interaction trace
    name : Union[str, List[str]], optional
        Name of the interaction or of each interaction
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    shared_legend : bool, default True
        If `True` put all interaction traces under a single shared legend
    **kwargs : dict, optional
        Additional parameters to pass to the plot
    """
    # If cmin/cmax are not provided, must build them so that all clusters
    # share the same colorscale range (not guaranteed otherwise)
    if color is not None and isinstance(color, (list, tuple, np.ndarray)):
        assert len(color) == len(
            interactions
        ), "If providing a list of colors, must provide one per interaction."
        if len(color) > 0 and not isinstance(color[0], str):
            if cmin is None:
                cmin = np.min(color)
            if cmax is None:
                cmax = np.max(color)

    # Loop over interaction objects
    traces = []
    inter_color = color
    inter_hovertext = hovertext
    inter_name = name
    group_name = "group_" + str(time.time())
    for i, inter in enumerate(interactions):
        # If a separate color is given for each interaction, use it
        if isinstance(color, (list, tuple, np.ndarray)):
            inter_color = color[i]

        # If a separate hovertext is given for each interaction, use it
        if isinstance(hovertext, (list, tuple, np.ndarray)):
            inter_hovertext = hovertext[i]

        # If a separate name is given for each interaction, use it
        if isinstance(name, (list, tuple, np.ndarray)):
            inter_name = name[i]
        elif isinstance(name, str) and not shared_legend:
            inter_name = f"{name} {i}"

        # Set legend group if shared_legend is True
        legendgroup, showlegend = None, True
        if shared_legend:
            legendgroup = group_name
            showlegend = i == 0

        # Draw all particles in the interaction
        traces.extend(
            scatter_lite_particles(
                inter.particles,
                color=inter_color,
                hovertext=inter_hovertext,
                name=inter_name,
                cmin=cmin,
                cmax=cmax,
                shared_legend=True,  # Always share legend within interaction
                legendgroup=legendgroup,
                showlegend=showlegend,
                **kwargs,
            )
        )

    return traces


def scatter_lite_particles(
    particles,
    color=None,
    hovertext=None,
    showscale=False,
    linewidth=5.0,
    cone_num_samples=10,
    name=None,
    cmin=None,
    cmax=None,
    colorscale=None,
    legendgroup=None,
    showlegend=True,
    shared_legend=True,
    **kwargs,
):
    """Produces plotly traces for Lite particles.

    Parameters
    ----------
    particles : List[spine.data.out.ParticleBase]
        List of lite particles to visualize
    color : Union[str, float], optional
        Color of the particle trace
    hovertext : Union[int, str, np.ndarray], optional
        Text associated with the particle trace
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    cone_num_samples : int, default 10
        Number of samples to use for the cone mesh
    name : Union[str, List[str]], optional
        Name of the particle or of each particle
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    colorscale : List[Union[str, float]], optional
        Colorscale of the particle trace
    legendgroup : str, optional
        Legend group name for the trace
    showlegend : bool, optional
        Whether to show the legend for the trace
    shared_legend : bool, default True
        If `True` put all particle traces under a single shared legend
    **kwargs : dict, optional
        Additional parameters to pass to the plotly trace objects

    Returns
    -------
    List[object]
        List of plotly trace objects representing the Lite particle
    """
    # If cmin/cmax are not provided, must build them so that all clusters
    # share the same colorscale range (not guaranteed otherwise)
    if color is not None and isinstance(color, (list, tuple, np.ndarray)):
        assert len(color) == len(
            particles
        ), "If providing a list of colors, must provide one per particle."
        if len(color) > 0 and not isinstance(color[0], str):
            if cmin is None:
                cmin = np.min(color)
            if cmax is None:
                cmax = np.max(color)

    # Loop over particle objects
    traces = []
    part_color = color
    part_hovertext = hovertext
    part_name = name
    part_showlegend = showlegend
    part_legendgroup = legendgroup
    group_name = "group_" + str(time.time())
    for i, particle in enumerate(particles):
        # If a separate color is given for each particle, use it
        if isinstance(color, (list, tuple, np.ndarray)):
            part_color = color[i]

        # If a separate hovertext is given for each particle, use it
        if isinstance(hovertext, (list, tuple, np.ndarray)):
            part_hovertext = hovertext[i]

        # If a separate name is given for each particle, use it
        if isinstance(name, (list, tuple, np.ndarray)):
            part_name = name[i]
        elif isinstance(name, str) and not shared_legend:
            part_name = f"{name} {i}"

        # Set legend group if shared_legend is True
        if legendgroup is None:
            if shared_legend:
                part_legendgroup = group_name
            else:
                part_legendgroup = group_name + f"_{i}"
        if showlegend and shared_legend:
            part_showlegend = i == 0

        # Initialize the object
        if particle.shape == TRACK_SHP:
            traces.append(
                track_line_trace(
                    start_point=particle.start_point,
                    end_point=particle.end_point,
                    color=part_color,
                    hovertext=part_hovertext,
                    name=part_name,
                    legendgroup=part_legendgroup,
                    showlegend=False,  # Dummy trace for legend
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                    linewidth=linewidth,
                    **kwargs,
                )
            )

        else:
            traces.append(
                em_cone_trace(
                    start_point=particle.start_point,
                    direction=particle.start_dir,
                    energy=particle.ke,
                    num_samples=cone_num_samples,
                    color=part_color,
                    hovertext=part_hovertext,
                    name=part_name,
                    showscale=showscale,
                    legendgroup=part_legendgroup,
                    showlegend=False,  # Dummy trace for legend
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                    **kwargs,
                )
            )

        # Add a dummy trace to show the appropriate color in the legend
        if part_showlegend:
            traces.append(
                legend_trace(
                    color=part_color,
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                    legendgroup=part_legendgroup,
                    name=part_name,
                )
            )

    return traces


def track_line_trace(
    start_point,
    end_point,
    line=None,
    color=None,
    hovertext=None,
    colorscale=None,
    cmin=None,
    cmax=None,
    linewidth=5.0,
    **kwargs,
):
    """Generates a line trace representing a track between two points.

    Parameters
    ----------
    start_point : np.ndarray
        (3,) Array representing the starting point of the track.
    end_point : np.ndarray
        (3,) Array representing the ending point of the track.
    line : dict, optional
        Dictionary defining line properties (e.g., width, dash style)
    color : Union[str, float], optional
        Color of the line trace
    hovertext : Union[int, str], optional
        Text associated with the line trace
    colorscale : List[Union[str, float]], optional
        Colorscale of the line trace
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    linewidth : float, default 2.0
        Width of the line trace
    hovertext : Union[int, str], optional
        Text associated with the line trace.
    **kwargs : dict, optional
        Additional parameters to pass to the plot
    """
    # Define line properties
    if line is None:
        line = {}
    if color is not None:
        assert np.isscalar(color), "Should provide a single color for the line."
        line["color"] = [color, color]  # One per line endpoint
    if linewidth is not None:
        line["width"] = linewidth
    if colorscale is not None:
        line["colorscale"] = colorscale
    if cmin is not None:
        line["cmin"] = cmin
    if cmax is not None:
        line["cmax"] = cmax

    # Update hovertemplate style
    hovertemplate = "x: %{x}<br>y: %{y}<br>z: %{z}"
    if hovertext is not None:
        if not np.isscalar(hovertext):
            hovertemplate += "<br>%{text}"
        else:
            hovertemplate += f"<br>{hovertext}"
            hovertext = None

    # Append the line trace
    return go.Scatter3d(
        x=[start_point[0], end_point[0]],
        y=[start_point[1], end_point[1]],
        z=[start_point[2], end_point[2]],
        mode="lines",
        line=line,
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        hoverinfo="text",
        **kwargs,
    )


def em_cone_trace(
    start_point,
    direction,
    energy,
    num_samples=10,
    color=None,
    intensity=None,
    hovertext=None,
    showscale=False,
    **kwargs,
):
    """Generates a cone trace representing an electromagnetic shower.

    Parameters
    ----------
    start : np.ndarray
        (3,) Array representing the starting point of the shower.
    direction : np.ndarray
        (3,) Array representing the direction vector of the shower.
    energy : float
        Energy of the shower in MeV.
    num_samples : int, default 10
        Number of samples to use for the cone mesh.
    color : Union[str, float], optional
        Color of the cone trace.
    intensity : Union[str, float, np.ndarray], optional
        Intensity of the cone colors
    hovertext : Union[int, str], optional
        Text associated with the cone trace.
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    **kwargs : dict, optional
        Additional parameters to pass to the plotly trace object.

    Returns
    -------
    object
        Plotly Mesh3d trace representing the electromagnetic shower cone.
    """
    # Get the shower length from the 90th quantile of the longitudinal profile
    length = shower_long_quantile(energy=energy, quantile=0.68)

    # Get the shower half-opening angle from the 90th quantile of the angular profile
    theta = shower_angle_quantile(energy=energy, quantile=0.68)

    # Compute the points on a cone with half-opening angle theta
    r = np.linspace(0, 1, num=num_samples)
    phi = np.linspace(0, 2 * np.pi, num=num_samples)
    r, phi = np.meshgrid(r, phi)
    x = r * np.tan(theta) * np.cos(phi)
    y = r * np.tan(theta) * np.sin(phi)
    z = r
    unit_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Get the rotation matrix to point the cone in the direction of the shower
    # This uses the Rodrigues' rotation formula
    rotmat = np.eye(3)
    z_axis = np.array([0.0, 0.0, 1.0])
    if not np.allclose(direction, z_axis):
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        vx = np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
        )  # Cross-product matrix
        rotmat = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))

    # Rotate and offset the cone
    cone_points = start_point + length * np.dot(unit_points, rotmat.T)

    # Convert the color provided to a set of intensities
    if color is not None:
        assert intensity is None, "Provide either `color` or `intensity`, not both."
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

    # Return the Mesh3d object
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


def legend_trace(
    color,
    cmin=None,
    cmax=None,
    colorscale=None,
    legendgroup=None,
    name=None,
):
    """Generates a dummy trace to show in the legend.

    Parameters
    ----------
    color : Union[str, float]
        Color of the legend trace
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    colorscale : List[Union[str, float]], optional
        Colorscale of the legend trace
    **kwargs : dict, optional
        Additional parameters to pass to the plotly trace object

    Returns
    -------
    object
        Plotly Scatter3d trace representing the legend entry.
    """
    return go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode="markers",
        marker=dict(color=[color], cmin=cmin, cmax=cmax, colorscale=colorscale),
        legendgroup=legendgroup,
        showlegend=True,
        name=name,
    )
