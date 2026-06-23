"""Tools to draw voxelized data organized in clusts."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import plotly.graph_objs as go

from .cone import cone_trace
from .ellipsoid import ellipsoid_trace
from .hull import hull_trace
from .point import scatter_points_3d
from .utils import (
    ColorInput,
    HoverTextInput,
    is_scalar_sequence,
    select_scalar_or_sequence,
)

__all__ = ["scatter_clusters"]


def scatter_clusters(
    points: np.ndarray,
    clusts: list[np.ndarray],
    color: ColorInput = None,
    hovertext: HoverTextInput = None,
    single_trace: bool = False,
    name: str | list[str] | None = None,
    mode: str = "scatter",
    cmin: float | None = None,
    cmax: float | None = None,
    shared_legend: bool = True,
    **kwargs: Any,
) -> list[go.Scatter3d] | list[go.Mesh3d]:
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
    color : Union[str, int, float, Sequence], optional
        Color of the markers, provided either as one shared scalar value, one
        value per point, one value per cluster, or pre-grouped per-cluster
        point values in ``"scatter"`` mode.
    hovertext : Union[int, float, str, Sequence], optional
        Hover labels, provided either as one shared scalar label, one label per
        point, one label per cluster, or pre-grouped per-cluster labels in
        ``"scatter"`` mode.
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
    Union[List[go.Scatter3d], List[go.Mesh3d]]
        (1/C) List with one combined trace or one trace per cluster
    """
    # Build the point coordinate sets
    coords = [points[c] for c in clusts]

    # Get a single cluster index value per points
    counts = [len(c) for c in clusts]
    clust_ids = np.arange(len(clusts))

    # Normalize the color input to one value per cluster, with scatter-mode
    # inputs expanded to one value per point within each cluster.
    has_labels = False
    color_by_cluster: list[Any]
    if color is not None:
        has_labels = True
        if not is_scalar_sequence(color):
            color_by_cluster = [color] * len(clusts)
        elif len(color) == len(points) and len(points) != len(clusts):
            color_by_cluster = [np.asarray(color)[c] for c in clusts]
        elif len(color) == len(clusts):
            color_by_cluster = list(color)
            if (
                mode == "scatter"
                and len(color) > 0
                and not is_scalar_sequence(color[0])
            ):
                color_by_cluster = [[color[i]] * len(c) for i, c in enumerate(clusts)]
        else:
            raise ValueError(
                "The `color` attribute should be provided as a scalar, "
                "one value per point or one value per cluster."
            )
    else:
        if mode != "scatter":
            color_by_cluster = list(clust_ids)
        else:
            color_by_cluster = [[clust_ids[i]] * len(c) for i, c in enumerate(clusts)]

    # Normalize the hovertext input to one value per cluster, with scatter-mode
    # inputs expanded to one label per point within each cluster.
    hovertext_by_cluster: list[Any] | None = None
    if hovertext is not None:
        if not is_scalar_sequence(hovertext):
            hovertext_by_cluster = [hovertext] * len(clusts)
        elif len(hovertext) == len(points) and len(points) != len(clusts):
            hovertext_by_cluster = [np.asarray(hovertext)[c] for c in clusts]
        elif len(hovertext) == len(clusts):
            hovertext_by_cluster = list(hovertext)
            if (
                mode == "scatter"
                and len(hovertext) > 0
                and not is_scalar_sequence(hovertext[0])
            ):
                hovertext_by_cluster = [
                    [hovertext[i]] * len(c) for i, c in enumerate(clusts)
                ]
        elif len(hovertext) != len(clusts):
            raise ValueError(
                "The `hovertext` attribute should be provided as a scalar, "
                "one value per point or one value per cluster."
            )
    else:
        base_hovertext = [f"Cluster ID: {i:.0f}" for i in clust_ids]
        if (
            has_labels
            and len(color_by_cluster)
            and not isinstance(color_by_cluster[0], str)
        ):
            if not is_scalar_sequence(color_by_cluster[0]):
                hovertext_by_cluster = []
                for i, hover_label in enumerate(base_hovertext):
                    fmt = ".0f" if float(color_by_cluster[i]).is_integer() else ".2f"
                    hovertext_by_cluster.append(
                        hover_label + f"<br>Label: {color_by_cluster[i]:{fmt}}"
                    )
            else:
                hovertext_by_cluster = []
                for i, hover_label in enumerate(base_hovertext):
                    hovertext_by_cluster.append(
                        [
                            hover_label + f"<br>Value: {v:0.3f}"
                            for v in color_by_cluster[i]
                        ]
                    )
        elif mode == "scatter":
            hovertext_by_cluster = [
                [base_hovertext[i]] * len(c) for i, c in enumerate(clusts)
            ]
        else:
            hovertext_by_cluster = base_hovertext

    # If requested, combine all clusters into a single trace
    if single_trace:
        # Check that we are operating in the expected mode
        if mode not in ["circle", "scatter"]:
            raise ValueError(
                "Can only combine in one trace in 'circle' or 'scatter' mode."
            )
        if not shared_legend:
            raise ValueError(
                "Cannot split legend when merging all clusters in one trace."
            )

        # Aggregate the coordinates, color and hovertext
        if mode == "circle":
            # Define the nodes as circles centered in the centroid of each
            # cluster and of radius proportional to the sqrt of the cluster size
            centroids = np.empty((len(coords), 3), dtype=np.float32)
            for i, coord in enumerate(coords):
                centroids[i] = np.mean(coord, axis=0)
            sizes = np.sqrt(np.asarray(counts, dtype=np.float32))

            return scatter_points_3d(
                centroids,
                name=name,
                color=color_by_cluster,
                markersize=sizes,
                hovertext=hovertext_by_cluster,
                cmin=cmin,
                cmax=cmax,
                **kwargs,
            )

        else:
            if len(coords):
                coords = np.vstack(coords)
            else:
                coords = np.empty((0, 3), dtype=np.float32)

            merged_color = color_by_cluster
            if len(color_by_cluster):
                if is_scalar_sequence(color_by_cluster[0]):
                    merged_color = np.concatenate(color_by_cluster)
                else:
                    merged_color = np.concatenate(
                        [
                            np.asarray([color_by_cluster[i]] * len(clusts[i]))
                            for i in range(len(clusts))
                        ]
                    )
            merged_hovertext = hovertext_by_cluster
            if hovertext_by_cluster is not None and len(hovertext_by_cluster):
                if is_scalar_sequence(hovertext_by_cluster[0]):
                    merged_hovertext = np.concatenate(hovertext_by_cluster)
                else:
                    merged_hovertext = np.concatenate(
                        [
                            np.asarray([hovertext_by_cluster[i]] * len(clusts[i]))
                            for i in range(len(clusts))
                        ]
                    )

            return scatter_points_3d(
                coords,
                color=merged_color,
                hovertext=merged_hovertext,
                name=name,
                cmin=cmin,
                cmax=cmax,
                **kwargs,
            )

    # If cmin/cmax are not provided, must build them so that all clusters
    # share the same colorscale range (not guaranteed otherwise)
    if len(color_by_cluster) > 0 and not isinstance(color_by_cluster[0], str):
        if cmin is None:
            if not is_scalar_sequence(color_by_cluster[0]):
                cmin = np.min(color_by_cluster)
            else:
                cmin = np.min(np.concatenate(color_by_cluster))

        if cmax is None:
            if not is_scalar_sequence(color_by_cluster[0]):
                cmax = np.max(color_by_cluster)
            else:
                cmax = np.max(np.concatenate(color_by_cluster))

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
            if not is_scalar_sequence(name):
                name_i = f"{name} {i}"
            else:
                if len(name) != len(clusts):
                    raise ValueError(
                        "When providing the name as a list, there should be "
                        "one name per cluster."
                    )
                name_i = name[i]

        # Dispatch
        color_i = select_scalar_or_sequence(color_by_cluster, i)
        hovertext_i = select_scalar_or_sequence(hovertext_by_cluster, i)
        if mode == "circle":
            centroid = np.mean(coord, axis=0)[None, :]
            size = float(np.sqrt(counts[i]))
            traces += scatter_points_3d(
                centroid,
                name=name_i,
                color=color_i,
                hovertext=hovertext_i,
                cmin=cmin,
                cmax=cmax,
                markersize=size,
                legendgroup=legendgroup,
                showlegend=showlegend,
                **kwargs,
            )

        elif mode == "scatter":
            traces += scatter_points_3d(
                coord,
                name=name_i,
                color=color_i,
                hovertext=hovertext_i,
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
                    color=color_i,
                    hovertext=hovertext_i,
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
                    color=color_i,
                    hovertext=hovertext_i,
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
                    color=color_i,
                    hovertext=hovertext_i,
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
                "of 'circle', 'scatter', 'ellipsoid', 'cone' or 'hull'."
            )

    return traces
