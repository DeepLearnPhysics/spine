"""Graph topology helpers used with clustered particle representations."""

# Public re-export lists intentionally overlap with the package facade.
# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Sequence

import numba as nb
import numpy as np

from spine.utils.jit import numbafy

__all__ = ["complete_graph", "filter_invalid_nodes", "get_fragment_edges"]


@numbafy(cast_args=["graph"])
def get_fragment_edges(graph: np.ndarray, clust_ids: np.ndarray) -> np.ndarray:
    """Convert edges between cluster IDs to edges between fragment positions.

    Parameters
    ----------
    graph : Union[np.ndarray, torch.Tensor]
        ``(E, 2)`` edges expressed using cluster IDs.
    clust_ids : np.ndarray
        ``(C,)`` cluster ID associated with each fragment position.

    Returns
    -------
    np.ndarray
        ``(E', 2)`` edges expressed using fragment positions. Edges whose
        cluster IDs are absent from ``clust_ids`` are omitted.
    """
    return _get_fragment_edges(graph, clust_ids)


@nb.njit(cache=True)
def _get_fragment_edges(graph: np.ndarray, clust_ids: np.ndarray) -> np.ndarray:
    """Numba implementation of cluster-ID to fragment-index conversion."""
    true_edges = np.empty((0, 2), dtype=np.int64)
    for edge in graph:
        node_1 = np.where(clust_ids == edge[0])[0]
        node_2 = np.where(clust_ids == edge[1])[0]
        if len(node_1) and len(node_2):
            true_edges = np.vstack(
                (
                    true_edges,
                    np.array([[node_1[0], node_2[0]]], dtype=np.int64),
                )
            )

    return true_edges


@nb.njit(cache=True)
def complete_graph(counts: np.ndarray) -> np.ndarray:
    """Build the upper-triangular edges of each complete graph in a batch.

    Parameters
    ----------
    counts : np.ndarray
        ``(B,)`` number of nodes in each batch entry.

    Returns
    -------
    np.ndarray
        ``(2, E)`` incidence matrix with no cross-entry edges.
    """
    num_edges = np.sum((counts * (counts - 1)) // 2)
    edge_index = np.empty((2, num_edges), dtype=np.int64)
    offset, index = 0, 0
    for count in counts:
        # Build local edges first, then shift them into the batched node space.
        adjacency = np.triu(np.ones((count, count)), k=1)
        edges = np.vstack(np.where(adjacency))
        entry_edges = edges.shape[1]

        edge_index[:, index : index + entry_edges] = offset + edges
        index += entry_edges
        offset += count

    return edge_index


@nb.njit(cache=True)
def filter_invalid_nodes(
    edge_index: np.ndarray, invalid_nodes: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Remove invalid tree nodes while bridging the gaps they leave.

    Leaf edges are dropped. For an internal node with one parent, its children
    are reassigned to that parent. Removing a root drops its outgoing edges.

    Parameters
    ----------
    edge_index : np.ndarray
        ``(E, 2)`` original parent-to-child incidence map.
    invalid_nodes : np.ndarray
        ``(N,)`` nodes to remove from the incidence map.

    Returns
    -------
    np.ndarray
        ``(E', 2)`` filtered incidence map.
    """
    edges = edge_index.copy()
    for node in invalid_nodes:
        children = np.where(edges[:, 0] == node)[0]
        if len(children) == 0:
            # A leaf has no descendants to reconnect.
            edges = edges[edges[:, 1] != node]
            continue

        parent = np.where(edges[:, 1] == node)[0]
        assert len(parent) <= 1, "Found a particle with multiple parents."
        if len(parent) == 1:
            # Bridge the removed internal node by adopting its children.
            parent_id = edges[parent][0][0]
            edges[:, 0][children] = parent_id
        else:
            # A removed root has no ancestor to which children can attach.
            edges = edges[edges[:, 0] != node]

        edges = edges[edges[:, 1] != node]

    return edges
