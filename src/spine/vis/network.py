"""Tools to draw voxelized data organized in clusts."""

import numpy as np

from spine.math.distance import closest_pair
from spine.utils.globals import COORD_COLS

from .cluster import scatter_clusters
from .point import scatter_points

__all__ = ["network_topology", "network_schematic"]


def network_topology(
    points,
    clusts,
    edge_index,
    clust_labels=None,
    edge_labels=None,
    mode="scatter",
    color=None,
    line=None,
    linewidth=2,
    name=None,
    **kwargs,
):
    """Network 3D topological representation in Euclidean space.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of N points of (..., x, y, z,...) coordinate information
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    edge_index : np.ndarray
        (E, 2) List of connections between clusters
    clust_labels : np.ndarray, optional
        (C) List of cluster labels
    edge_labels : np.ndarray, optional
        (E) List of edge labels
    mode : str, default 'scatter'
        Drawing mode; one of 'circle', 'scatter', 'ellipsoid', 'cone' or 'hull'
    color : Union[str, np.ndarray], optional
        Color of clusters or (C) list of color of clusters
    line : dict, optional
        Line property dictionary
    linewidth : float, default 2
        Width of the edge lines
    name : str, optional
        Name of the network
    **kwargs : dict, optional
        List of additional arguments to pass to plotly

    Returns
    -------
    List[Union[plotly.graph_objs.Scatter3d, plotly.graph_objs.Mesh3d]]
        Node and edge traces in the same list
    """
    # Fetch the list of point coordinates
    if points.shape[1] != 3:
        points = points[:, COORD_COLS]

    # Check that color is not passed directly, ambiguous for a network
    assert (
        color is None
    ), "Use `clust_labels` instead of `color` to specify node colors."

    # Set the prefix to add to the trace names
    prefix = f"{name}" if name is not None else "Graph"
    node_name = name if edge_index is None else f"{prefix} nodes"
    edge_name = f"{prefix} edges"

    # Define the trace(s) associated with the graph nodes
    single_trace = mode in ["circle", "scatter"]
    traces = scatter_clusters(
        points,
        clusts,
        color=clust_labels,
        single_trace=single_trace,
        name=node_name,
        mode=mode,
        **kwargs,
    )

    # Define the trace associated with graph edges
    edge_vertices = np.empty((0, 3), dtype=points.dtype)
    if len(edge_index):
        edge_vertices = []
        if mode in ["circle", "ellipsoid"]:
            # For circles and ellipsoids, join centroid to centroid
            cent = [points[c].mean(axis=0) for c in clusts]
            for i, j in edge_index:
                edge_vertices.extend([cent[i], cent[j], [None, None, None]])

        elif mode in ["scatter", "hull"]:
            # For scatter and hull, join closest point to closest point
            for i, j in edge_index:
                vi, vj = points[clusts[i]], points[clusts[j]]
                i1, i2, _ = closest_pair(vi, vj, "recursive")
                edge_vertices.extend([vi[i1], vj[i2], [None, None, None]])

        else:
            # For cones, use the cone start points
            sts = []
            for trace in traces:
                start = [trace["x"][0], trace["y"][0], trace["z"][0]]
                sts.append(start)

            for i, j in edge_index:
                edge_vertices.extend([sts[i], sts[j], [None, None, None]])

        edge_vertices = np.vstack(edge_vertices)

    # Initialize the edge labels, if they are provided
    if edge_labels is not None:
        edge_labels = np.repeat(edge_labels, 3)

    # Add the edge trace
    traces += scatter_points(
        edge_vertices,
        color=edge_labels,
        line=line,
        linewidth=linewidth,
        mode="lines",
        name=edge_name,
    )

    # Return
    return traces


def network_schematic(
    clusts,
    edge_index,
    clust_labels,
    edge_labels=None,
    color=None,
    name=None,
    linewidth=2,
    **kwargs,
):
    """Network 2D schematic representation.

    This is to be used exclusevely with bipartite graphs where the nodes
    are either classified as primary or secondaries under clust_labels and
    connections only exist between primaries and secondaries.

    Parameters
    ----------
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    edge_index : np.ndarray
        (E, 2) List of connections between clusters
    clust_labels : np.ndarray
        (C) Whether a cluster is a primary or a secondary
    edge_labels : np.ndarray, optional
        (E) List of edge labels
    linewidth : float, default 2
        Width of the edge lines
    color : Union[str, np.ndarray], optional
        Color of clusters or (C) list of color of clusters
    name : str, optional
        Name of the network
    linewidth : float, default 2
        Width of the edge lines
    **kwargs : dict, optional
        List of additional arguments to pass to plotly

    Returns
    -------
    List[plotly.graph_objs.Scatter]
        Node and edge traces in the same list
    """
    # Check that color is not passed directly, ambiguous for a network
    assert (
        color is None
    ), "Use `clust_labels` instead of `color` to specify node colors."

    # Define the node size on the bases of the cluster size
    counts = np.array([len(c) for c in clusts])
    node_sizes = np.sqrt(counts)

    # Set the prefix to add to the trace names
    prefix = f"{name}" if name is not None else "Graph"
    node_name = name if edge_index is None else f"{prefix} nodes"
    edge_name = f"{prefix} edges"

    # Check that the labels are binary (0 or 1)
    assert len(clust_labels) == len(
        clusts
    ), "Must provide a primary label for each cluster."
    assert np.all(
        (clust_labels == 0) | (clust_labels == 1)
    ), "All cluster labels should be 0 or 1."

    # Define the hovertext attribute
    num_clusts = len(clusts)
    node_labels = []
    for i in range(num_clusts):
        node_labels.append(f"Cluster ID: {i:d}")
        node_labels[i] += f"<br>Primary: {clust_labels[i]:0.0f}"
        node_labels[i] += f"<br>Size: {counts[i]:d}"

    # Define the positions (primaries on the left, secondaries on the right)
    pos = np.array([[i, l] for i, l in enumerate(clust_labels)])

    # Define the trace associated with the graph nodes
    node_trace = scatter_points(
        pos,
        color=clust_labels,
        hovertext=node_labels,
        hoverinfo=["x", "y", "text"],
        markersize=node_sizes,
        name=node_name,
        **kwargs,
    )

    # Define the trace associated with the graph edges
    edge_vertices = np.empty((0, 2), dtype=pos.dtype)
    if len(edge_index):
        edge_vertices = []
        for i, j in edge_index:
            edge_vertices.extend([pos[i], pos[j], [None, None]])

        edge_vertices = np.vstack(edge_vertices)

    # Initialize the edge labels, if they are provided
    if edge_labels is not None:
        edge_labels = np.repeat(edge_labels, 3)

    # Add the edge trace
    edge_trace = []
    edge_trace = scatter_points(
        edge_vertices,
        color=edge_labels,
        linewidth=linewidth,
        mode="lines",
        name=edge_name,
    )

    return node_trace + edge_trace
