"""Module which contains fast CPU-based graph routines.

In particular, this module supports CSR data structure and derived methods,
which tremendously speeds up graph construction and computation in Numba.
"""

import numba as nb
import numpy as np

CSR_DTYPE = (
    ('neighbors', nb.int64[:]),
    ('offsets', nb.int64[:])
)

@nb.experimental.jitclass(CSR_DTYPE)
class CSRGraph:
    def __init__(self,
                 edge_index: nb.int64[:,:],
                 num_nodes: nb.int64):
        """Construct the Compressed Sparse Row (CSR) representation of a
        sparse matrix based on a list of nodes and edges.

        Parameters
        ----------
        edge_index : np.ndarray
            (E, 2) List of active edge indices in the graph
        num_nodes : int
            Number of nodes in the graph, N
        """
        # Count the number of connections per node
        counts = np.zeros(num_nodes, dtype=np.int64)
        for s, _ in edge_index:
            counts[s] += 1

        # Build the offsets array
        self.offsets = np.empty(num_nodes + 1, dtype=np.int64)
        self.offsets[0] = 0
        for i in range(num_nodes):
            self.offsets[i + 1] = self.offsets[i] + counts[i]

        # Build the neighbors array
        self.neighbors = np.empty(self.offsets[-1], dtype=np.int64)
        fill = np.zeros(num_nodes, dtype=np.int64)
        for s, t in edge_index:
            idx = self.offsets[s] + fill[s]
            self.neighbors[idx] = t
            fill[s] += 1

    def __getitem__(self,
                    node_id: nb.int64):
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

    def num_neighbors(self,
                      node_id: nb.int64):
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
        return self.offsets[node_id + 1] - self.offsets[node_id]


@nb.njit
def dfs(graph: CSR_DTYPE,
        visited: nb.bool_[:],
        node: nb.int64,
        component: nb.int64[:],
        comp_idx: nb.int64[:]):
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
    """
    # Mark the node as visited, incremant pointer
    visited[node] = True
    component[comp_idx[0]] = node
    comp_idx[0] += 1
    
    # Traverse all the neighbors of the node
    for neighbor in graph[node]:
        if not visited[neighbor]:        
            dfs(graph, visited, neighbor, component, comp_idx)


@nb.njit
def connected_components(edge_index, num_nodes):
    """Find connected components.

    Parameters
    ----------
    edge_index : np.ndarray
        (E, 2) List of active edge indices in the graph
    num_nodes : int
        Number of nodes in the graph, N
    
    Returns
    -------
    np.ndarray
        (N) Cluster label associated with each node
    """
    # Initialize a CSR graph data structure
    graph = CSRGraph(edge_index, num_nodes)
    
    # Initialize output
    visited = np.zeros(num_nodes, dtype=nb.bool_)
    component = np.empty(num_nodes, dtype=nb.int64)
    comp_idx = np.empty(1, dtype=nb.int64) # Acts as pointer
    labels = np.arange(num_nodes)
    
    # Loop through all nodes and start DFS from unvisited nodes
    for node in range(num_nodes):
        if not visited[node] and graph.num_neighbors(node) > 0:
            # Perform DFS and collect all nodes in this connected component
            comp_idx[0] = 0
            dfs(graph, visited, node, component, comp_idx)
            
            # Collect all nodes that belong to the same connected component
            for i in range(comp_idx[0]):
                labels[component[i]] = node
    
    return labels
