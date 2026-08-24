"""Numba-accelerated graph construction and traversal routines.

The public API provides a compact CSR representation, radius-based graph
builders, connected-component algorithms, multi-source shortest paths and
union-find grouping. Private helpers implement the allocation, spatial-hash
and traversal details used by those operations.
"""

from heapq import heappop, heappush

import numba as nb
import numpy as np

from .distance import (
    CHEBYSHEV,
    CITYBLOCK,
    EUCLIDEAN,
    METRICS,
    MINKOWSKI,
    SQEUCLIDEAN,
    chebyshev,
    cityblock,
    minkowski,
    sqeuclidean,
)

__all__ = [
    "CSRGraph",
    "bipartite_radius_graph",
    "connected_components",
    "csr_graph",
    "grouped_radius_graph",
    "radius_graph",
    "shortest_path",
    "union_find",
]


_CSR_DTYPE = (
    ("num_nodes", nb.int64),
    ("neighbors", nb.int64[:]),
    ("offsets", nb.int64[:]),
)

_CELL_OFFSETS = np.array(
    [
        (offset_x, offset_y, offset_z)
        for offset_x in range(-1, 2)
        for offset_y in range(-1, 2)
        for offset_z in range(-1, 2)
    ],
    dtype=np.int64,
)


@nb.experimental.jitclass(spec=_CSR_DTYPE)  # type: ignore[call-arg]
class CSRGraph:
    """Numba-compatible compressed sparse row representation of a graph.

    Instances are normally constructed with :func:`csr_graph`, which derives
    the neighbor and offset arrays from an edge list. The class remains public
    for callers that already own a valid CSR representation and want to avoid
    rebuilding it.

    Attributes
    ----------
    neighbors : np.ndarray
        ``(E,)`` flattened node-neighbor indexes.
    offsets : np.ndarray
        ``(N + 1,)`` boundaries of each node neighborhood in ``neighbors``.
    num_nodes : int
        Number of nodes, ``N``.
    """

    def __init__(self, neighbors: np.ndarray, offsets: np.ndarray, num_nodes: int):
        """Initialize a graph from precomputed CSR arrays.

        Parameters
        ----------
        neighbors : np.ndarray
            ``(E,)`` flattened node-neighbor indexes.
        offsets : np.ndarray
            ``(N + 1,)`` boundaries of each node neighborhood in ``neighbors``.
        num_nodes : int
            Number of nodes, ``N``.
        """
        self.neighbors = neighbors
        self.offsets = offsets
        self.num_nodes = num_nodes

    def __getitem__(self, node_id: int) -> np.ndarray:
        """Return the neighbors of one node.

        Parameters
        ----------
        node_id : int
            Node index.

        Returns
        -------
        np.ndarray
            View of the flattened neighbor array associated with ``node_id``.
        """
        start, end = self.offsets[node_id], self.offsets[node_id + 1]
        return self.neighbors[start:end]

    def num_neighbors(self, node_id: int) -> int:
        """Return the number of neighbors associated with one node.

        Parameters
        ----------
        node_id : int
            Node index.

        Returns
        -------
        int
            Number of stored neighbors.
        """
        start, end = self.offsets[node_id], self.offsets[node_id + 1]
        return end - start


@nb.njit
def csr_graph(
    edge_index: np.ndarray, num_nodes: int, directed: bool = True
) -> CSRGraph:
    """Construct a compressed sparse row graph from an edge list.

    Directed graphs store each input edge once. Undirected graphs store both
    orientations so that indexing the result returns the complete neighborhood
    of every node. Input ordering and duplicate edges are preserved.

    Parameters
    ----------
    edge_index : np.ndarray
        ``(E, 2)`` source and target node indexes.
    num_nodes : int
        Number of nodes, ``N``.
    directed : bool, default True
        If ``False``, insert the reverse orientation of every input edge.

    Returns
    -------
    CSRGraph
        Numba-compatible CSR graph.
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
def _weighted_csr(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int,
    directed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned CSR neighbor, weight and offset arrays."""
    counts = np.zeros(num_nodes, dtype=np.int64)
    for source, target in edge_index:
        counts[source] += 1
        if not directed:
            counts[target] += 1

    offsets = np.empty(num_nodes + 1, dtype=np.int64)
    offsets[0] = 0
    for node in range(num_nodes):
        offsets[node + 1] = offsets[node] + counts[node]

    neighbors = np.empty(offsets[-1], dtype=np.int64)
    weights = np.empty(offsets[-1], dtype=edge_weights.dtype)
    fill = np.zeros(num_nodes, dtype=np.int64)
    for edge_id, (source, target) in enumerate(edge_index):
        position = offsets[source] + fill[source]
        neighbors[position] = target
        weights[position] = edge_weights[edge_id]
        fill[source] += 1
        if not directed:
            position = offsets[target] + fill[target]
            neighbors[position] = source
            weights[position] = edge_weights[edge_id]
            fill[target] += 1

    return neighbors, weights, offsets


@nb.njit(cache=True)
def _shortest_path(
    adjacency: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_nodes: int,
    sources: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Run multi-source Dijkstra traversal over a weighted CSR graph."""
    neighbors, weights, offsets = adjacency

    # Initialize a heap with the best distance for each source node.
    distances = np.full(num_nodes, np.inf, dtype=np.float64)
    closest_source = np.full(num_nodes, -1, dtype=np.int64)
    queue = [(np.inf, -1)]
    heappop(queue)
    for index, node in enumerate(sources[0]):
        distance = np.float64(sources[1][index])
        if distance < distances[node]:
            distances[node] = distance
            closest_source[node] = sources[2][index]
            heappush(queue, (distance, node))

    # Relax graph edges in increasing-distance order.
    while queue:
        distance, node = heappop(queue)
        if distance > distances[node]:
            continue

        for position in range(offsets[node], offsets[node + 1]):
            neighbor = neighbors[position]
            candidate = np.float64(distance + weights[position])
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                closest_source[neighbor] = closest_source[node]
                heappush(queue, (candidate, neighbor))

    return distances, closest_source


@nb.njit(cache=True)
def shortest_path(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int,
    sources: tuple[np.ndarray, np.ndarray, np.ndarray],
    directed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted shortest paths from one or more source nodes.

    The routine uses Dijkstra's algorithm and therefore expects non-negative
    edge weights. Each source may provide a nonzero initial distance and an
    arbitrary external identifier. The identifier associated with the best
    path is propagated to every reachable node; unreachable nodes retain an
    infinite distance and identifier ``-1``.

    Parameters
    ----------
    edge_index : np.ndarray
        ``(E, 2)`` source and target node indexes.
    edge_weights : np.ndarray
        ``(E,)`` non-negative edge weights aligned with ``edge_index``.
    num_nodes : int
        Number of nodes, ``N``.
    sources : tuple[np.ndarray, np.ndarray, np.ndarray]
        Source-node indexes, initial distances and external source identifiers.
        All three arrays must have the same length.
    directed : bool, default True
        If ``False``, traverse each edge in both directions.

    Returns
    -------
    np.ndarray
        ``(N,)`` shortest distance from the source set.
    np.ndarray
        ``(N,)`` external source identifier associated with each best path.

    Raises
    ------
    ValueError
        If the edge arrays or source arrays have inconsistent lengths.
    """
    source_index, source_distances, source_ids = sources
    if len(edge_index) != len(edge_weights):
        raise ValueError("Each graph edge must have exactly one weight.")
    if len(source_index) != len(source_distances):
        raise ValueError("Each source node must have exactly one distance.")
    if len(source_index) != len(source_ids):
        raise ValueError("Each source node must have exactly one identifier.")

    adjacency = _weighted_csr(edge_index, edge_weights, num_nodes, directed)
    return _shortest_path(adjacency, num_nodes, sources)


@nb.njit(cache=True)
def connected_components(
    edge_index: np.ndarray,
    num_nodes: int,
    min_samples: int = 1,
    directed: bool = True,
) -> np.ndarray:
    """Assign a component identifier to every graph node.

    With the default ``min_samples=1``, this performs ordinary component
    labeling. Larger values require an unvisited node to have at least
    ``min_samples - 1`` stored neighbors before it can seed a traversal. Nodes
    reached from an accepted seed remain part of that component regardless of
    their own degree.

    For directed input, traversal follows outgoing edges and does not compute
    strongly connected components. Use ``directed=False`` for conventional
    undirected component labeling.

    Parameters
    ----------
    edge_index : np.ndarray
        ``(E, 2)`` source and target node indexes.
    num_nodes : int
        Number of nodes, ``N``.
    min_samples : int, default 1
        Minimum neighborhood size, including the seed itself, required to
        initiate a traversal.
    directed : bool, default True
        If ``False``, traverse each edge in both directions.

    Returns
    -------
    np.ndarray
        ``(N,)`` contiguous component identifiers ordered by first discovery.
    """
    # Initialize the CSR data structure
    graph = csr_graph(edge_index, num_nodes, directed)

    # Initialize output
    labels = np.arange(graph.num_nodes)
    visited = np.zeros(graph.num_nodes, dtype=np.bool_)
    component = np.empty(graph.num_nodes, dtype=np.int64)
    comp_idx = np.empty(1, dtype=np.int64)  # Acts as pointer

    # Loop through all nodes and start DFS from unvisited nodes
    label = 0
    min_neighbors = min_samples - 1
    for node in range(graph.num_nodes):
        if not visited[node]:
            if graph.num_neighbors(node) >= min_neighbors:
                # Perform DFS and collect all nodes in this connected component
                comp_idx[0] = 0
                _dfs_iterative(graph, visited, node, component, comp_idx)

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
def _dfs(
    graph: CSRGraph,
    visited: np.ndarray,
    node: int,
    component: np.ndarray,
    comp_idx: np.ndarray,
) -> None:
    """Build one connected component through recursive depth-first search.

    Parameters
    ----------
    graph : CSRGraph
        CSR representation of a graph
    visited : np.ndarray
        (N,) Boolean array which specifies whether a node has been visited.
    node : int
        Current node index
    component : np.ndarray
        (N,) Current component (padded)
    comp_idx : np.ndarray
        Current component index (pointer)

    Notes
    -----
    This implementation is recursive and can exhaust the native call stack on
    large components. Production component labeling uses
    :func:`_dfs_iterative` instead.
    """
    # Mark the node as visited, increment pointer
    visited[node] = True
    component[comp_idx[0]] = node
    comp_idx[0] += 1

    # Traverse all the neighbors of the node
    for neighbor in graph[node]:
        if not visited[neighbor]:
            _dfs(graph, visited, neighbor, component, comp_idx)


@nb.njit(cache=True)
def _dfs_iterative(
    graph: CSRGraph,
    visited: np.ndarray,
    start_node: int,
    component: np.ndarray,
    comp_idx: np.ndarray,
) -> None:
    """Build one connected component through iterative depth-first search.

    Parameters
    ----------
    graph : CSRGraph
        CSR representation of a graph
    visited : np.ndarray
        (N,) Boolean array which specifies whether a node has been visited.
    start_node : int
        Starting node index
    component : np.ndarray
        (N,) Current component (padded)
    comp_idx : np.ndarray
        Current component index (pointer)

    Notes
    -----
    A fixed-size node stack avoids the native recursion-depth limitation of
    :func:`_dfs`.
    """
    # Initialize a node stack (fixed size)
    stack = np.empty(graph.num_nodes, dtype=np.int64)
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
    x: np.ndarray,
    radius: float,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
    use_hash: bool = False,
) -> np.ndarray:
    """Build an undirected radius graph over one point set.

    Each qualifying node pair appears exactly once as ``(i, j)`` with
    ``i < j``; self-edges are excluded. By default, all pairs are examined.
    The spatial-hash backend partitions three-dimensional space into cells and
    is generally preferable only for sufficiently large, sparse point sets.

    The radius uses the native units of the selected metric. In particular,
    ``sqeuclidean`` expects a squared-distance threshold, whereas ``euclidean``
    expects an ordinary distance and avoids square roots internally.

    Parameters
    ----------
    x : np.ndarray
        ``(N, 3)`` node coordinates.
    radius : float
        Inclusive distance threshold used to connect nodes.
    metric_id : int, default 2 (Euclidean)
        Distance enumerator from :mod:`spine.math.distance`.
    p : float, default 2.
        Norm degree used only by the Minkowski metric.
    use_hash : bool, default False
        Use a uniform spatial hash instead of brute-force pair enumeration.

    Returns
    -------
    np.ndarray
        ``(E, 2)`` undirected edges represented once with ordered indexes.

    Raises
    ------
    ValueError
        If ``radius`` is negative or ``metric_id`` is not recognized.
    """
    params = _radius_parameters(radius, metric_id, p)
    if use_hash:
        return _radius_pairs(x, x, params, unique=True)
    return _radius_graph_brute_force(x, radius, metric_id, p)


@nb.njit(cache=True)
def grouped_radius_graph(
    x: np.ndarray,
    groups: np.ndarray,
    radius: float,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
) -> np.ndarray:
    """Build an undirected radius graph independently within each group.

    Nodes in different groups are never compared, even when they occupy the
    same position. Points are sorted by group internally, but returned edge
    indexes always refer to the original input order. Each qualifying pair is
    represented once with its smaller node index first.

    The implementation enumerates pairs independently within each group, so
    its pair complexity is the sum of ``N_g**2`` rather than ``N**2``. For
    workloads that already hold one index array per group, parallel independent
    calls to :func:`radius_graph` may nevertheless be faster.

    Parameters
    ----------
    x : np.ndarray
        ``(N, 3)`` array of node coordinates.
    groups : np.ndarray
        ``(N,)`` scalar group identifier associated with each node. Identifiers
        need not be contiguous or ordered.
    radius : float
        Inclusive distance threshold used to connect same-group nodes.
    metric_id : int, default 2 (Euclidean)
        Distance enumerator from :mod:`spine.math.distance`.
    p : float, default 2.
        Norm degree used only by the Minkowski metric.

    Returns
    -------
    np.ndarray
        ``(E, 2)`` same-group edges represented once with ordered indexes.

    Raises
    ------
    ValueError
        If ``groups`` is not one-dimensional, does not contain one identifier
        per point, ``radius`` is negative or ``metric_id`` is not recognized.
    """
    if groups.ndim != 1 or len(groups) != len(x):
        raise ValueError("Groups must provide one identifier per point.")

    params = _radius_parameters(radius, metric_id, p)
    return _grouped_radius_graph_brute_force(x, groups, params)


@nb.njit(cache=True)
def bipartite_radius_graph(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
) -> np.ndarray:
    """Build a radius graph between two distinct point sets.

    The first edge column indexes ``x`` and the second indexes ``y``. Unlike
    :func:`radius_graph`, the two index spaces are independent: equal numeric
    indexes and coincident coordinates are valid cross-set pairs. A spatial
    hash over ``y`` avoids examining every Cartesian-product pair.

    Parameters
    ----------
    x, y : np.ndarray
        ``(N, 3)`` and ``(M, 3)`` point coordinates.
    radius : float
        Inclusive distance threshold used to connect cross-set nodes.
    metric_id : int, default 2 (Euclidean)
        Distance enumerator from :mod:`spine.math.distance`.
    p : float, default 2.
        Norm degree used only by the Minkowski metric.

    Returns
    -------
    np.ndarray
        ``(E, 2)`` edges whose columns index ``x`` and ``y``, respectively.

    Raises
    ------
    ValueError
        If ``radius`` is negative or ``metric_id`` is not recognized.
    """
    params = _radius_parameters(radius, metric_id, p)
    return _radius_pairs(x, y, params, unique=False)


@nb.njit(cache=True)
def _radius_parameters(
    radius: float,
    metric_id: int,
    p: float,
) -> tuple[float, float, int, float]:
    """Normalize a metric threshold and its spatial-cell size."""
    if radius < 0.0:
        raise ValueError("Radius must be non-negative.")
    if metric_id == EUCLIDEAN:
        cell_size = radius if radius > 0.0 else 1.0
        return radius * radius, cell_size, SQEUCLIDEAN, p
    if metric_id == SQEUCLIDEAN:
        cell_size = np.sqrt(radius) if radius > 0.0 else 1.0
        return radius, cell_size, metric_id, p
    if metric_id in (MINKOWSKI, CITYBLOCK, CHEBYSHEV):
        cell_size = radius if radius > 0.0 else 1.0
        return radius, cell_size, metric_id, p
    raise ValueError("Distance metric not recognized.")


@nb.njit(cache=True)
def _cell_key(point: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    """Return the integer spatial-cell key of one 3D point."""
    return (
        int(np.floor(point[0] / cell_size)),
        int(np.floor(point[1] / cell_size)),
        int(np.floor(point[2] / cell_size)),
    )


@nb.njit(cache=True)
def _cell_index(
    points: np.ndarray,
    cell_size: float,
) -> tuple[dict[tuple[int, int, int], int], np.ndarray]:
    """Index points as linked lists headed by occupied spatial cells."""
    heads = {}
    linked = np.full(len(points), -1, dtype=np.int64)
    for index, point in enumerate(points):
        key = _cell_key(point, cell_size)
        if key in heads:
            linked[index] = heads[key]
        heads[key] = index

    return heads, linked


@nb.njit(cache=True)
def _within_radius(
    first: np.ndarray,
    second: np.ndarray,
    threshold: float,
    metric_id: int,
    p: float,
) -> bool:
    """Check whether two points satisfy a normalized radius threshold."""
    if metric_id == MINKOWSKI:
        return minkowski(first, second, p) <= threshold
    if metric_id == CITYBLOCK:
        return cityblock(first, second) <= threshold
    if metric_id == SQEUCLIDEAN:
        return sqeuclidean(first, second) <= threshold
    return chebyshev(first, second) <= threshold


@nb.njit(cache=True)
def _count_radius_pairs(
    first: np.ndarray,
    second: np.ndarray,
    cell_index: tuple[dict[tuple[int, int, int], int], np.ndarray],
    params: tuple[float, float, int, float],
    unique: bool,
) -> int:
    """Count radius pairs before allocating the exact edge array."""
    count = 0
    for first_index, point in enumerate(first):
        cell = _cell_key(point, params[1])
        for offset in _CELL_OFFSETS:
            key = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            second_index = cell_index[0][key] if key in cell_index[0] else -1
            while second_index >= 0:
                if (not unique or second_index > first_index) and _within_radius(
                    point,
                    second[second_index],
                    params[0],
                    params[2],
                    params[3],
                ):
                    count += 1
                second_index = cell_index[1][second_index]

    return count


@nb.njit(cache=True)
def _fill_radius_pairs(
    first: np.ndarray,
    second: np.ndarray,
    cell_index: tuple[dict[tuple[int, int, int], int], np.ndarray],
    params: tuple[float, float, int, float],
    options: tuple[bool, int],
) -> np.ndarray:
    """Fill a pre-sized radius edge array from spatial-cell candidates."""
    edges = np.empty((options[1], 2), dtype=np.int64)
    edge_index = 0
    for first_index, point in enumerate(first):
        cell = _cell_key(point, params[1])
        for offset in _CELL_OFFSETS:
            key = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            second_index = cell_index[0][key] if key in cell_index[0] else -1
            while second_index >= 0:
                if (not options[0] or second_index > first_index) and _within_radius(
                    point,
                    second[second_index],
                    params[0],
                    params[2],
                    params[3],
                ):
                    edges[edge_index] = first_index, second_index
                    edge_index += 1
                second_index = cell_index[1][second_index]

    return edges


@nb.njit(cache=True)
def _radius_pairs(
    first: np.ndarray,
    second: np.ndarray,
    params: tuple[float, float, int, float],
    unique: bool,
) -> np.ndarray:
    """Build radius pairs through a uniform 3D spatial-cell index."""
    cell_index = _cell_index(second, params[1])
    count = _count_radius_pairs(first, second, cell_index, params, unique)
    return _fill_radius_pairs(first, second, cell_index, params, (unique, count))


@nb.njit(cache=True)
def _grouped_max_edges(sorted_groups: np.ndarray) -> int:
    """Return the number of possible pairs within contiguous groups."""
    max_edges = 0
    start = 0
    while start < len(sorted_groups):
        end = start + 1
        while end < len(sorted_groups) and sorted_groups[end] == sorted_groups[start]:
            end += 1
        size = end - start
        max_edges += size * (size - 1) // 2
        start = end

    return max_edges


@nb.njit(cache=True)
def _grouped_radius_graph_brute_force(
    x: np.ndarray,
    groups: np.ndarray,
    params: tuple[float, float, int, float],
) -> np.ndarray:
    """Build a grouped radius graph by enumerating within-group pairs."""
    order = np.argsort(groups)
    sorted_points = x[order]
    sorted_groups = groups[order]

    # Allocate for within-group pairs only, never for the global N-squared set.
    edges = np.empty((_grouped_max_edges(sorted_groups), 2), dtype=np.int64)
    edge_count = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_groups[end] == sorted_groups[start]:
            end += 1

        # Emit original node indexes so callers need not reorder their data.
        for first_position in range(start, end):
            first_index = order[first_position]
            for second_position in range(first_position + 1, end):
                second_index = order[second_position]
                if _within_radius(
                    sorted_points[first_position],
                    sorted_points[second_position],
                    params[0],
                    params[2],
                    params[3],
                ):
                    if first_index < second_index:
                        edges[edge_count] = first_index, second_index
                    else:
                        edges[edge_count] = second_index, first_index
                    edge_count += 1
        start = end

    return edges[:edge_count]


@nb.njit(cache=True)
def _radius_graph_brute_force(
    x: np.ndarray,
    radius: float,
    metric_id: int = METRICS["euclidean"],
    p: float = 2.0,
) -> np.ndarray:
    """Build a radius graph by checking all pairs as a reference oracle."""
    if metric_id == MINKOWSKI:
        return _radius_graph_minkowski(x, radius, p)
    if metric_id == CITYBLOCK:
        return _radius_graph_cityblock(x, radius)
    if metric_id == EUCLIDEAN:
        return _radius_graph_sqeuclidean(x, radius * radius)
    if metric_id == SQEUCLIDEAN:
        return _radius_graph_sqeuclidean(x, radius)
    if metric_id == CHEBYSHEV:
        return _radius_graph_chebyshev(x, radius)
    raise ValueError("Distance metric not recognized.")


@nb.njit(cache=True)
def _radius_graph_minkowski(x: np.ndarray, radius: float, p: float) -> np.ndarray:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, add edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if minkowski(x[i], x[j], p) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def _radius_graph_cityblock(x: np.ndarray, radius: float) -> np.ndarray:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, add edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cityblock(x[i], x[j]) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def _radius_graph_sqeuclidean(x: np.ndarray, radius: float) -> np.ndarray:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, add edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sqeuclidean(x[i], x[j]) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def _radius_graph_chebyshev(x: np.ndarray, radius: float) -> np.ndarray:
    # Initialize a data structure to hold edges
    num_nodes = len(x)
    max_edges = num_nodes * (num_nodes - 1) // 2
    edge_index = np.empty((max_edges, 2), dtype=np.int64)

    # Loop over pairs of nodes, add edges if the distance fits the bill
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if chebyshev(x[i], x[j]) <= radius:
                edge_index[edge_count, 0], edge_index[edge_count, 1] = i, j
                edge_count += 1

    return edge_index[:edge_count]


@nb.njit(cache=True)
def _find_root(parents: np.ndarray, node: int) -> int:
    """Find the root parent of a node with path compression."""
    root = node
    while parents[root] != root:
        root = parents[root]

    while parents[node] != node:
        parent = parents[node]
        parents[node] = root
        node = parent

    return root


@nb.njit(cache=True)
def union_find(
    edge_index: np.ndarray, count: int, return_inverse: bool = True
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Group graph nodes with the union-find algorithm.

    Every input edge merges the sets containing its two endpoints. Edge
    orientation is ignored. Isolated nodes form singleton groups.

    By default, root identifiers are remapped to contiguous values in
    ``[0, num_groups)``. When ``return_inverse=False``, labels retain their raw
    root-node indexes. Dictionary keys always match the returned label space.

    Parameters
    ----------
    edge_index : np.ndarray
        ``(E, 2)`` pairs of node indexes to merge.
    count : int
        Number of nodes, ``N``.
    return_inverse : bool, default True
        Remap root-node indexes to contiguous group identifiers.

    Returns
    -------
    np.ndarray
        ``(N,)`` group identifier associated with each node.
    dict[int, np.ndarray]
        Mapping from each returned group identifier to its node indexes. Both
        the label array and dictionary are empty when ``count`` is zero.
    """
    if count == 0:
        labels = np.empty(0, dtype=np.int64)
        groups = {0: labels}
        del groups[0]
        return labels, groups

    parents = np.arange(count)
    for src, dst in edge_index:
        src_root = _find_root(parents, int(src))
        dst_root = _find_root(parents, int(dst))
        if src_root != dst_root:
            if src_root < dst_root:
                parents[dst_root] = src_root
            else:
                parents[src_root] = dst_root

    labels = np.empty(count, dtype=np.int64)
    for node in range(count):
        labels[node] = _find_root(parents, node)

    if return_inverse:
        mask = np.zeros(count, dtype=np.bool_)
        mask[labels] = True
        mapping = np.empty(count, dtype=labels.dtype)
        mapping[mask] = np.arange(np.sum(mask))
        labels = mapping[labels]

    groups = {labels[0]: np.array([0])}
    for node in range(1, count):
        label = labels[node]
        node_arr = np.array([node])
        if label in groups:
            groups[label] = np.concatenate((groups[label], node_arr))
        else:
            groups[label] = node_arr

    return labels, groups
