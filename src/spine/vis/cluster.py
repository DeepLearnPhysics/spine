"""Tools to draw voxelized data organized in clusts."""

import time

import numpy as np

from .cone import cone_trace
from .ellipsoid import ellipsoid_trace
from .hull import hull_trace
from .point import scatter_points

__all__ = ["scatter_clusters"]


def scatter_clusters(
    points,
    clusts,
    color=None,
    hovertext=None,
    single_trace=False,
    name=None,
    mode="scatter",
    cmin=None,
    cmax=None,
    shared_legend=True,
    **kwargs,
):
    """Arranges points in clusters and scatters them and their cluster labels.

    Produces :class:`plotly.graph_objs.Scatter3d` trace object to be drawn
    in plotly. The object is nested to be fed directly to a
    :class:`plotly.graph_objs.Figure` or :func:`plotly.offline.iplot`.
    All of the regular plotly parameters are available.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of N points of (..., x, y, z,...) coordinate information
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    color : Union[str, np.ndarray], optional
        Color of markers or (N/C) list of color of markers or clusters
    hovertext : Union[List[str], List[int]], optional
        (N/C) List of labels associated with each marker or cluster
    single_trace : bool, default False
        If `True`, combine all clusters into a single plotly trace
    name : Union[str, List[str]], optional
        Name of the clusters or of each cluster
    mode : str, default 'scatter'
        Drawing mode; one of 'circle', 'scatter', 'ellipsoid', 'cone' or 'hull'
    cmin : float, optional
        Minimum value along the color scale
    cmax : float, optional
        Maximum value along the color scale
    shared_legend : bool, default True
        If `True` put all cluster traces under a single shared legend
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D

    Returns
    -------
    List[go.Scatter3d]
        (1/C) List with one combined trace or one trace per cluster
    """
    # Build the point coordinate sets
    coords = [points[c] for c in clusts]

    # Get a single cluster index value per points
    counts = [len(c) for c in clusts]
    clust_ids = np.arange(len(clusts))

    # Build the color vectors
    has_labels = False
    if color is not None:
        has_labels = True
        if np.isscalar(color):
            color = [color] * len(clusts)
        elif len(color) == len(points) and len(points) != len(clusts):
            color = [color[c] for c in clusts]
        elif len(color) == len(clusts):
            if mode == "scatter" and len(color) > 0 and np.isscalar(color[0]):
                color = [[color[i]] * len(c) for i, c in enumerate(clusts)]
        else:
            raise ValueError(
                "The `color` attribute should be provided as a scalar, "
                "one value per point or one value per cluster."
            )

    else:
        if mode != "scatter":
            color = clust_ids
        else:
            color = [[clust_ids[i]] * len(c) for i, c in enumerate(clusts)]

    # Build the hovertext vectors
    if hovertext is not None:
        if np.isscalar(hovertext):
            hovertext = [hovertext] * len(clusts)
        elif len(hovertext) == len(points) and len(points) != len(clusts):
            hovertext = [hovertext[c] for c in clusts]
        elif len(hovertext) == len(hovertext):
            if mode == "scatter" and len(hovertext) > 0 and np.isscalar(hovertext[0]):
                hovertext = [[hovertext[i]] * len(c) for i, c in enumerate(clusts)]
        elif len(hovertext) != len(clusts):
            raise ValueError(
                "The `hovertext` attribute should be provided as a scalar, "
                "one value per point or one value per cluster."
            )

    else:
        hovertext = [f"Cluster ID: {i:.0f}" for i in clust_ids]
        if has_labels and len(color):
            if np.isscalar(color[0]):
                for i, hc in enumerate(hovertext):
                    fmt = ".0f" if float(color[i]).is_integer() else ".2f"
                    hovertext[i] = hc + f"<br>Label: {color[i]:{fmt}}"
            else:
                for i, hc in enumerate(hovertext):
                    hovertext[i] = [hc + f"<br>Value: {v:0.3f}" for v in color[i]]
        elif mode == "scatter":
            hovertext = [[hovertext[i]] * len(c) for i, c in enumerate(clusts)]

    # If requested, combine all clusters into a single trace
    if single_trace:
        # Check that we are operating in the expected mode
        assert mode in [
            "circle",
            "scatter",
        ], "Can only combine in one trace in 'circle' or 'scatter' mode."
        assert (
            shared_legend
        ), "Cannot split legend when merging all clusters in one trace."

        # Aggregate the coordinates, color and hovertext
        if mode == "circle":
            # Define the nodes as circles centered in the centroid of each
            # cluster and of radius proportional to the sqrt of the cluster size
            centroids = np.empty((len(coords), 3), dtype=np.float32)
            for i, coord in enumerate(coords):
                centroids[i] = np.mean(coord, axis=0)
            sizes = np.sqrt(counts)

            return scatter_points(
                centroids,
                name=name,
                color=color,
                markersize=sizes,
                hovertext=hovertext,
                cmin=cmin,
                cmax=cmax,
                **kwargs,
            )

        else:
            if len(coords):
                coords = np.vstack(coords)
            else:
                coords = np.empty((0, 3), dtype=np.float32)

            if color is not None and len(color):
                color = np.concatenate(color)
            if hovertext is not None and len(hovertext):
                hovertext = np.concatenate(hovertext)

            return scatter_points(
                coords,
                color=color,
                hovertext=hovertext,
                name=name,
                cmin=cmin,
                cmax=cmax,
                **kwargs,
            )

    # If cmin/cmax are not provided, must build them so that all clusters
    # share the same colorscale range (not guaranteed otherwise)
    if color is not None and not np.isscalar(color) and len(color) > 0:
        if cmin is None:
            if np.isscalar(color[0]):
                cmin = np.min(color)
            else:
                cmin = np.min(np.concatenate(color))

        if cmax is None:
            if np.isscalar(color[0]):
                cmax = np.max(color)
            else:
                cmax = np.max(np.concatenate(color))

    # Loop over the list of clusters
    traces = []
    group_name = "group_" + str(time.time())
    for i, coord in enumerate(coords):
        # If the legend is shared, only draw the legend of the first trace
        legendgroup, showlegend, name_i = None, True, name
        if shared_legend:
            legendgroup = group_name
            showlegend = i == 0
        elif name is not None:
            if np.isscalar(name):
                name_i = f"{name} {i}"
            else:
                assert len(name) == len(clusts), (
                    "When providing the name as a list, there should be "
                    "one name per cluster."
                )
                name_i = name[i]

        # Dispatch
        if mode == "circle":
            centroid = np.mean(coord, axis=0)[None, :]
            size = np.sqrt(counts[i])
            traces += scatter_points(
                centroid,
                name=name_i,
                color=color[i],
                hovertext=hovertext[i],
                cmin=cmin,
                cmax=cmax,
                markersize=size,
                legendgroup=legendgroup,
                showlegend=showlegend,
                **kwargs,
            )

        elif mode == "scatter":
            traces += scatter_points(
                coord,
                name=name_i,
                color=color[i],
                hovertext=hovertext[i],
                cmin=cmin,
                cmax=cmax,
                legendgroup=legendgroup,
                showlegend=showlegend,
                **kwargs,
            )

        elif mode == "ellipsoid":
            traces.append(
                ellipsoid_trace(
                    coord,
                    name=name_i,
                    color=color[i],
                    hovertext=hovertext[i],
                    cmin=cmin,
                    cmax=cmax,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    **kwargs,
                )
            )

        elif mode == "cone":
            traces.append(
                cone_trace(
                    coord,
                    name=name_i,
                    color=color[i],
                    hovertext=hovertext[i],
                    cmin=cmin,
                    cmax=cmax,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    **kwargs,
                )
            )

        elif mode == "hull":
            traces.append(
                hull_trace(
                    coord,
                    name=name_i,
                    color=color[i],
                    hovertext=hovertext[i],
                    cmin=cmin,
                    cmax=cmax,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    **kwargs,
                )
            )

        else:
            raise ValueError(
                f"Cluster drawing mode not recognized: {mode}. Must be one "
                "of 'circle', 'sctatter', 'ellipsoid', 'cone' or 'hull'."
            )

    return traces
