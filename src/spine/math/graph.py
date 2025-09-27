"""Numba JIT compiled implementation of graph routines.

In particular, this module supports the CSR data structure and derived methods,
which tremendously speeds up graph construction and computation in Numba.
"""

import numba as nb
import numpy as np

from .distance import METRICS, cdist, chebyshev, cityblock, minkowski, sqeuclidean

CSR_DTYPE = (
    ("num_nodes", nb.int64),
    ("neighbors", nb.int64[:]),
    ("offsets", nb.int64[:]),
)


@nb.experimental.jitclass(CSR_DTYPE)
class CSRGraph:
    """Numba-enabled compressed Sparse Row (CSR) representation of a sparse matrix.

    Attributes
    ----------
    neighbors : np.ndarray
        (E) List of node neighbors in a compressed array
    offsets : np.ndarray
        (N+1) Per-node slicing boundaries to query each node neighborhood
    num_nodes : int
        Number of nodes in the graph, N
    """

    def __init__(
        self, neighbors: nb.int64[:], offsets: nb.int64[:], num_nodes: nb.int64
    ):
        """Construct the Compressed Sparse Row (CSR) representation of a
        sparse matrix based on a list of nodes and edges.

        Parameters
        ----------
        neighbors : np.ndarray
            (E) List of node neighbors in a compressed array
        offsets : np.ndarray
            (N+1) Per-node slicing boundaries to query each node neighborhood
        num_nodes : int
            Number of nodes in the graph, N
        """
        self.neighbors = neighbors
        self.offsets = offsets
        self.num_nodes = num_nodes

    def __getitem__(self, node_id: nb.int64):
        """Get the list of neighbors associated with a node.

        Parameters
        ----------
        node_id : int
            Node index i

        Returns
        -------
        np.ndarray
            List of neighbors associated with node i
        """
        start, end = self.offsets[node_id], self.offsets[node_id + 1]
        return self.neighbors[start:end]

    def num_neighbors(self, node_id: nb.int64):
        """Returns the number of neighbors of a node.

        Parameters
        ----------
        node_id : int
            Node index i

        Returns
        -------
        int
            Number of neighbors of node i
        """
        start, end = self.offsets[node_id], self.offsets[node_id + 1]
        return end - start


@nb.njit
def csr_graph(
    edge_index: nb.int64[:, :], num_nodes: nb.int64, directed: nb.boolean = True
) -> CSR_DTYPE:
    """Construct the Compressed Sparse Row (CSR) representation of a sparse
    matrix based on a list of nodes and edges.

    Parameters
    ----------
    edge_index : np.ndarray
        (E, 2) List of active edge indices in the graph
    num_nodes : int
        Number of nodes in the graph, N
    directed : bool
        Whether the input graph is directed or not
    """
    # Count the number of connections per node
    counts = np.zeros(num_nodes, dtype=np.int64)
    for s, t in edge_index:
        counts[s] += 1
        if not directed:
            counts[t] += 1

    # Build the offsets array
    offsets = np.empty(num_nodes + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(num_nodes):
        offsets[i + 1] = offsets[i] + counts[i]

    # Build the neighbors array
    neighbors = np.empty(offsets[-1], dtype=np.int64)
    fill = np.zeros(num_nodes, dtype=np.int64)
    for s, t in edge_index:
        idx = offsets[s] + fill[s]
        neighbors[idx] = t
        fill[s] += 1
        if not directed:
            idx = offsets[t] + fill[t]
            neighbors[idx] = s
            fill[t] += 1

    # Initialize the CSR graph
    return CSRGraph(neighbors, offsets, num_nodes)


@nb.njit(cache=True)
def connected_components(
    edge_index: nb.int64[:, :],
    num_nodes: nb.int64,
    min_samples: nb.int64 = 1,
    directed: nb.boolean = True,
) -> nb.int64[:]:
    """Find connected components.

    Parameters
    ----------
    edge_index : np.ndarray
        (E, 2) List of active edge indices in the graph
    num_nodes : int
        Number of nodes in the graph, N
    directed : bool, default True
        Whether the input graph is directed or not

    Returns
    -------
    np.ndarray
        (N) Cluster label associated with each node
    """
    # Initialize the CSR data structure
    graph = csr_graph(edge_index, num_nodes, directed)

    # Initialize output
    labels = np.arange(graph.num_nodes)
    visited = np.zeros(graph.num_nodes, dtype=nb.boolean)
    component = np.empty(graph.num_nodes, dtype=nb.int64)
    comp_idx = np.empty(1, dtype=nb.int64)  # Acts as pointer

    # Loop through all nodes and start DFS from unvisited nodes
    label = 0
    min_neighbors = min_samples - 1
    for node in range(graph.num_nodes):
        if not visited[node]:
            if graph.num_neighbors(node) > min_neighbors:
                # Perform DFS and collect all nodes in this connected component
                comp_idx[0] = 0
                dfs_iterative(graph, visited, node, component, comp_idx)

                # Collect all nodes that belong to the same connected component
                for i in range(comp_idx[0]):
                    labels[component[i]] = label

            else:
                # Relabel solitary nodes to maintain ordering
                labels[node] = label

            # Increment label
            label += 1

    return labels


@nb.njit(cache=True)
def dfs(
    graph: CSR_DTYPE,
    visited: nb.boolean[:],
    node: nb.int64,
    component: nb.int64[:],
    comp_idx: nb.int64[:],
):
    """Does a depth-first search and builds a connected component.

    Parameters
    ----------
    graph : CSRGraph
        CSR representation of a graph
    visitied : np.ndarray
        (N) Boolean array which specified weather a node has been visited or not.
    node : int
        Current node index
    component : np.ndarray
        (N) Current component (padded)
    comp_idx : np.ndarray
        Current component index (pointer)

    Notes
    -----
    This implementation is recursive, which is the fastest implementation but
    silently throws segementation faults if the maximum recursion depth is
    reached. The :func:`dfs_iterative` function is safer, but slightly slower.
    """
    # Mark the node as visited, incremant pointer
    visited[node] = True
    component[comp_idx[0]] = node
    comp_idx[0] += 1

    # Traverse all the neighbors of the node
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(graph, visited, neighbor, component, comp_idx)


@nb.njit(cache=True)
def dfs_iterative(
    graph: CSR_DTYPE,
    visited: nb.boolean[:],
    start_node: nb.int64,
    component: nb.int64[:],
    comp_idx: nb.int64[:],
):
    """Does a depth-first search and builds a connected component.

    Parameters
    ----------
    graph : CSRGraph
        CSR representation of a graph
    visitied : np.ndarray
        (N) Boolean array which specified weather a node has been visited or not.
    node : int
        Current node index
    component : np.ndarray
        (N) Current component (padded)
    comp_idx : np.ndarray
        Current component index (pointer)

    Notes
    -----
    This implementation is iterative and does not suffer from the recursion
    depth maximum issue which affects the recursive version, at a small cost
    to the overall execution speed.
    """
    # Initialize a node stack (fixed size)
    stack = np.empty(graph.num_nodes, dtype=nb.int64)
    stack[0] = start_node
    stack_idx = 1

    visited[start_node] = True

    # Loop until there is no more node to visit
    while stack_idx > 0:
        stack_idx -= 1
        node = stack[stack_idx]

        component[comp_idx[0]] = node
        comp_idx[0] += 1

        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack[stack_idx] = neighbor
                stack_idx += 1


@nb.njit(cache=True)
def radius_graph(
    x: nb.float32[:, :],
    radius: nb.float32,
    metric_id: nb.int64 = METRICS["euclidean"],
    p: nb.float32 = 2.0,
) -> nb.int64[:, :]:
    """Builds an undirected radius graph.

    This function generates a list of edges in a graph which connects all nodes
    which live within some radius R of each other.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) array of node coordinates
    radius : float
        Radius within which to build connections in the graph
    metric_id : int, default 2 (Euclidean)
        Distance metric enumerator
    p : float, default 2.
        p-norm factor for the Minkowski metric, if used

    Returns
    -------
    np.ndarray
        (E, 2) array of edges in the radius graph
    """
    # Determine the distance function to use. If the metric is Euclidean, it
    # is cheaper to square the radius and use the squared Euclidean metric
    if metric_id == np.int64(0):
        return _radius_graph_minkowski(x, radius, p)
    elif metric_id == np.int64(1):
        return _radius_graph_cityblock(x, radius)
    elif metric_id == np.int64(2):
        radius = radius * radius
        return _radius_graph_sqeuclidean(x, radius)
    elif metric_id == np.int64(3):
        return _radius_graph_sqeuclidean(x, radius)
    elif metric_id == np.int64(4):
        return _radius_graph_chebyshev(x, radius)
    else:
        raise ValueError("Distance metric not recognized.")


@nb.njit(cache=True)
def _radius_graph_minkowski(
    x: nb.float32[:, :], radius: nb.float32, p: nb.float32
) -> nb.float32[:, :]:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, ass edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if minkowski(x[i], x[j], p) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def _radius_graph_cityblock(
    x: nb.float32[:, :], radius: nb.float32
) -> nb.float32[:, :]:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, ass edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cityblock(x[i], x[j]) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def _radius_graph_sqeuclidean(
    x: nb.float32[:, :], radius: nb.float32
) -> nb.float32[:, :]:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, ass edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sqeuclidean(x[i], x[j]) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def _radius_graph_chebyshev(
    x: nb.float32[:, :], radius: nb.float32
) -> nb.float32[:, :]:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, ass edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if chebyshev(x[i], x[j]) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def union_find(
    edge_index: nb.int64[:, :], count: nb.int64, return_inverse: bool = True
) -> nb.int64[:]:
    """Numba implementation of the Union-Find algorithm.

    This function assigns a group to each node in a graph, provided
    a set of edges connecting the nodes together.

    Parameters
    ----------
    edge_index : np.ndarray
        (E, 2) List of edges (sparse adjacency matrix)
    count : int
        Number of nodes in the graph, C
    return_inverse : bool, default True
        Make sure the group IDs range from 0 to N_groups-1

    Returns
    -------
    np.ndarray
        (C) Group assignments for each of the nodes in the graph
    Dict[int, np.ndarray]
        Dictionary which maps groups to indexes
    """
    labels = np.arange(count)
    groups = {i: np.array([i]) for i in labels}
    for e in edge_index:
        li, lj = labels[e[0]], labels[e[1]]
        if li != lj:
            labels[groups[lj]] = li
            groups[li] = np.concatenate((groups[li], groups[lj]))
            del groups[lj]

    if return_inverse:
        mask = np.zeros(count, dtype=np.bool_)
        mask[labels] = True
        mapping = np.empty(count, dtype=labels.dtype)
        mapping[mask] = np.arange(np.sum(mask))
        labels = mapping[labels]

    return labels, groups
