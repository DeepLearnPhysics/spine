"""Module used to label or evaluate GNNs.

It contains two classes of functions:
- Functions used in GNN losses
- Functions used to quantify the performance of GNNs
"""

import numba as nb
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

import spine.math as sm
from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.metrics import ami, ari, pur_eff, sbd

int_array = nb.int64[:]


def edge_assignment_batch(edge_index, group_ids):
    """Batched version of :func:`edge_assignment`.

    Parameters
    ----------
    edge_index: EdgeIndexBatch
        (2, E) Sparse incidence matrix
    group_ids : TensorBatch
        (C) Cluster group IDs

    Returns
    -------
    TensorBatch
        (E) Array specifying on/off edges
    """
    edge_assn = edge_assignment(edge_index.index.T, group_ids.tensor)

    return TensorBatch(edge_assn, edge_index.counts)


def edge_assignment_from_graph_batch(edge_index, true_edge_index, part_ids):
    """Batched version of :func:`edge_assignment_from_graph`.

    Parameters
    ----------
    edge_index : EdgeIndexBatch
        (2, E) Input sparse incidence matrix (on clusters)
    truth_edge_index : TensorBatch
        (2, E') Label sparse incidence matrix (on particles)
    part_ids : TensorBatch
        (C) Particle IDs of the clusters

    Returns
    -------
    TensorBatch
        (E) Array specifying on/off edges
    """
    # Convert the cluster IDs in the original edge index to particle IDs
    edge_index_part = TensorBatch(
        part_ids.tensor[edge_index.index].T, edge_index.counts
    )

    # Loop over the list of entries in the batch
    edge_assn = np.zeros(edge_index.index.shape[1], dtype=np.int64)
    for b in range(clust_label.batch_size):
        lower, upper = edge_index.edges[b], edge_index.edges[b + 1]
        edge_assn[lower:upper] = edge_assignment_from_graph(
            edge_index_part[b], true_index[b]
        )

    return TensorBatch(edge_assn, edge_index.counts)


def edge_assignment_forest_batch(edge_index, edge_pred, group_ids):
    """Batched version of :func:`edge_assignment_forest`.

    Parameters
    ----------
    edge_index : EdgeIndexBatch
        (2, E) Input sparse incidence matrix (on clusters)
    edge_pred : TensorBatch
        (E, 2) Logits associated with each edge
    group_ids : TensorBatch
        (C) Cluster group IDs

    Returns
    -------
    TensorBatch
        (E) Array specifying on/off edges
    TensorBatch
        (E) Array specifying edges to apply the loss to
    """
    # Loop over the list of entries in the batch
    edge_assn = np.empty(edge_index.index.shape[1], dtype=np.int64)
    valid_mask = np.empty(edge_index.index.shape[1], dtype=bool)
    for b in range(clust_label.batch_size):
        # Get the list of labels and the list of nodes to apply the loss to
        lower, upper = edge_index.edges[b], edge_index.edges[b + 1]
        edge_assn_b, edge_valid_b = edge_assignment_forest(
            edge_index[b], edge_pred[b], group_ids[b]
        )

        edge_assn[lower:upper] = edge_assn_b
        edge_valid[lower:upper] = edge_valid_b

    return (
        TensorBatch(edge_assn, counts=edge_index.counts),
        TensorBatch(edge_valid, counts=edge_valid.counts),
    )


def edge_assignment_score_batch(edge_index, edge_pred, clusts, track_node=None):
    """Batched version of :func:`edge_assignment_score`.

    Parameters
    ----------
    edge_index : EdgeIndexBatch
        (2, E) Sparse incidence matrix
    edge_pred : TensorBatch
        (E, 2) Logits associated with each edge
    clusts : IndexBatch
        (C) List of cluster indexes
    track_node : TensorBatch, optional
        (C) Whether a node is a track fragment/particle or not

    Returns
    -------
    np.ndarray
        (E', 2) Optimal incidence matrix
    np.ndarray
        (C) Optimal group ID for each node
    float
        Score of the optimal incidence matrix
    """
    edge_index_list = []
    group_ids = np.empty(len(clusts.index_list), dtype=np.int64)
    scores = np.empty(edge_index.batch_size, dtype=edge_pred.dtype)
    edge_counts = np.empty(edge_index.batch_size, dtype=np.int64)
    offset = 0
    for b in range(edge_index.batch_size):
        lower, upper = clusts.edges[b], clusts.edges[b + 1]
        track_node_b = track_node[b] if track_node is not None else None
        edge_index_b, group_ids_b, score_b = edge_assignment_score(
            edge_index[b], edge_pred[b], clusts.counts[b], track_node_b
        )

        edge_index_list.append(edge_index_b + edge_index.offsets[b])
        group_ids[lower:upper] = offset + group_ids_b
        scores[b] = score_b
        edge_counts[b] = len(edge_index_b)
        if upper - lower > 0:
            offset = np.max(group_ids[lower:upper]) + 1

    # Make a new EdgeIndexBatch out of the selected edges
    new_edge_index = EdgeIndexBatch(
        np.vstack(edge_index_list).T, edge_counts, edge_index.offsets, directed=True
    )

    return new_edge_index, TensorBatch(group_ids, clusts.counts), scores


def node_assignment_batch(edge_index, edge_pred, clusts):
    """Batched version of :func:`node_assignment`.

    Parameters
    ----------
    edge_index : EdgeIndexBatch
        (2, E) Sparse incidence matrix
    edge_pred : TensorBatch
        (E, 2) Logits associated with each edge
    clusts : IndexBatch
        (C) List of cluster indexes

    Returns
    -------
        np.ndarray: (C) List of group ids
    """
    # Loop over on edges, reset the group IDs of connected node
    group_ids = np.empty(len(clusts.index_list), dtype=np.int64)
    offset = 0
    for b in range(edge_index.batch_size):
        lower, upper = clusts.edges[b], clusts.edges[b + 1]
        if upper - lower > 0:
            group_ids[lower:upper] = offset + node_assignment(
                edge_index[b], edge_pred[b], clusts.counts[b]
            )
            offset = np.max(group_ids[lower:upper]) + 1

    return TensorBatch(group_ids, counts=clusts.counts)


def node_assignment_score_batch(edge_index, edge_pred, clusts, track_node=None):
    """Finds the graph that produces the lowest grouping score and use
    union-find to find group IDs for each of the nodes in the graph.

    Parameters
    ----------
    edge_index : EdgeIndexBatch
        (2, E) Sparse incidence matrix
    edge_pred : TensorBatch
        (E, 2) Logits associated with each edge
    clusts : IndexBatch
        (C) List of cluster indexes
    track_node : TensorBatch, optional
        (C) Whether a node is a track fragment/particle or not

    Returns
    -------
    np.ndarray
        (C) Optimal group ID for each node
    """
    return edge_assignment_score_batch(edge_index, edge_pred, clusts, track_node)[1]


def edge_purity_mask_batch(edge_index, part_ids, group_ids, primary_ids):
    """Batched version of :func:`edge_purity_mask`.

    Parameters
    ----------
    edge_index : EdgeIndexBatch
        (2, E) Sparse incidence matrix
    part_ids : TensorBatch
        (C) Array of cluster particle IDs
    group_ids : TensorBatch
        (C) Array of cluster group IDs
    primary_ids : TensorBatch
        (C) Array of cluster primary IDs

    Returns
    -------
    np.ndarray
        (E) High purity edge mask
    """
    # Loop over the entries in the batch
    valid_mask = np.empty(edge_index.index.shape[1], dtype=bool)
    for b in range(edge_index.batch_size):
        lower, upper = edge_index.edges[b], edge_index.edges[b + 1]
        valid_mask[lower:upper] = edge_purity_mask(
            edge_index[b], part_ids[b], group_ids[b], primary_ids[b]
        )

    return valid_mask


def node_purity_mask_batch(group_ids, primary_ids):
    """Batched version of :func:`node_purity_mask`.

    Parameters
    ----------
    group_ids : TensorBatch
        (C) Array of cluster group IDs
    primary_ids : TensorBatch
        (C) Cluster of cluster primary IDs

    Returns
    -------
    np.ndarray
        (C) High purity node mask
    """
    # Loop over the entries in the batch
    valid_mask = np.empty(len(group_ids.tensor), dtype=bool)
    for b in range(group_ids.batch_size):
        lower, upper = group_ids.edges[b], group_ids.edges[b + 1]
        valid_mask[lower:upper] = node_purity_mask(group_ids[b], primary_ids[b])

    return valid_mask


def primary_assignment_batch(node_pred, group_ids=None):
    """Batched version of :func:`primary_assignment`.

    Parameters
    ----------
    node_pred : TensorBatch
        (C, 2) Logits associated with each node
    group_ids : TensorBatch, optional
        (C) List of node group IDs

    Returns
    -------
    TensorBatch
        (C) Primary labels
    """
    if group_ids is None:
        primary_ids = primary_assignment(node_pred.tensor)
    else:
        primary_ids = np.empty(len(node_pred.tensor), dtype=bool)
        for b in range(node_pred.batch_size):
            lower, upper = node_pred.edges[b], node_pred.edges[b + 1]
            primary_ids[lower:upper] = primary_assignment(node_pred[b], group_ids[b])

    return TensorBatch(primary_ids, node_pred.counts)


def edge_assignment(edge_index, group_ids):
    """Determines which edges are turned on based on the group ID of the
    clusters they are connecting.

    Parameters
    ----------
    edge_index: np.ndarray
        (E, 2) Sparse incidence matrix
    group_ids : np.ndarray
        (C) Cluster group IDs

    Returns
    -------
    np.ndarray:
        (E) Array specifying on/off edges
    """
    # Set the edge as true if it connects two nodes that belong to the same
    # entry (free; no edges between entries) and the same group
    mask = group_ids[edge_index[:, 0]] == group_ids[edge_index[:, 1]]

    return mask.astype(np.int64)


def edge_assignment_from_graph(edge_index, true_edge_index, part_ids):
    """Determines which edges are turned on based on whether they appear in
    a reference list of true edges or not.

    Parameters
    ----------
    edge_index : EdgeIndexBatch
        (E, 2) Input sparse incidence matrix (on clusters)
    truth_edge_index : TensorBatch
        (E', 2) Label sparse incidence matrix (on particles)
    part_ids : np.ndarray
        (C) Particle IDs of the clusters

    Returns
    -------
    np.ndarray:
        (E) Array specifying on/off edges
    """
    # Convert the cluster IDs in the original edge index to particle IDs
    edge_index_part = part_ids[edge_index]

    # Compare with the reference sparse incidence matrix
    compare_index = lambda x, y: (x.T == y[..., None]).all(axis=1).any(axis=1)

    return compare_index(edge_index_part, true_edge_index)


def edge_assignment_forest(edge_index, edge_pred, group_ids):
    """Determines which edges must be turned on based on to form a
    minimum-spanning tree (MST) for each node group.

    For each group, find the most likely spanning tree, label the edges in the
    tree as 1. For all other edges, apply loss only if in separate groups. If
    undirected, also assign symmetric path. This method enforces that the
    network minmally forms a forest graph on the input nodes, with each tree
    in the forest spanning a target node group.

    Parameters
    ----------
    edge_index : np.ndarray
        (E, 2) Input sparse incidence matrix (on clusters)
    edge_pred : np.ndarray
        (E, 2) Logits associated with each edge
    group_ids : np.ndarray
        (C) Cluster group IDs

    Returns
    -------
    np.ndarray:
        (E) Array specifying on/off edges
    np.ndarray
        (E) Valid edge mask (edges to apply the loss to)
    """
    # If there are no edges, nothing to do here
    edge_assn = np.zeros(len(edge_index), dtype=np.int64)
    valid_mask = np.ones(len(edge_index), dtype=bool)
    if not len(edge_index):
        return edge_assn, valid_mask

    # Convert the sparse incidence matrix scores to a CSR matrix
    n = len(group_ids)
    off_scores = sm.softmax(edge_pred, axis=1)[:, 0]
    score_mat = csr_array((off_scores, edge_index.T), shape=(n, n))

    # Build the MST graph to minimize off scores
    mst_mat = minimum_spanning_tree(score_mat)
    mst_index = np.vstack(np.where(mst_mat.toarray() > 0.0))

    # Loop over the groups, turn edges on if they appear in the MST
    # TODO: understand the impact of having an undirected graph
    compare_index = lambda x, y: (x.T == y[..., None]).all(axis=1).any(axis=1)
    for g in np.unique(group_ids):
        group_index == np.where(group_ids == g)[0]
        edge_assn_g = compare_index(edge_index_b[group_index], tree_index)
        edge_assn[group_index[edge_assn_g]] = True
        edge_valid[group_index[~edge_assn_g]] = False

    return edge_assn, edge_valid


@nb.njit(cache=True)
def node_assignment(
    edge_index: nb.int64[:, :], edge_pred: nb.int64[:, :], num_nodes: nb.int64
) -> nb.int64[:]:
    """Assigns each node to a group, based on the edge assigment provided.

    This uses the locally-defined union find implementation.

    Parameters
    ----------
    edge_index : np.ndarray
        (2, E) Sparse incidence matrix
    edge_pred : np.ndarray
        (E, 2) Logits associated with each edge
    num_nodes : int
        Number of nodes in the graph, C

    Returns
    -------
    np.ndarray
        (C) Assigned node group IDs
    """
    # Loop over on edges, reset the group IDs of connected node
    on_edges = edge_index[np.where(edge_pred[:, 1] > edge_pred[:, 0])[0]]

    return sm.graph.union_find(on_edges, num_nodes, return_inverse=True)[0]


@nb.njit(cache=True)
def node_assignment_bipartite(
    edge_index: nb.int64[:, :],
    edge_label: nb.int64[:],
    primaries: nb.int64[:],
    num_nodes: nb.int64,
) -> nb.int64[:]:
    """Assigns each node to a group represented by a primary node.

    This function loops over secondaries and associates it to the primary with
    that is connected to it with the strongest edge.

    Parameters
    ----------
    edge_index : np.ndarray
        (2, E) Sparse incidence matrix
    edge_pred : np.ndarray
        (E, 2) Logits associated with each edge
    primaries : np.ndarray
        (P) List of primary IDs
    num_nodes : int
        Number of nodes in the graph, C

    Returns
    -------
    np.ndarray
        (C) Assigned node group IDs
    """
    group_ids = np.arange(num_nodes, dtype=np.int64)
    others = [i for i in range(num_nodes) if i not in primaries]
    for i in others:
        inds = edge_index[:, 1] == i
        if np.sum(inds) == 0:
            continue
        indmax = np.argmax(edge_label[inds])
        group_ids[i] = edge_index[inds, 0][indmax]

    return group_ids


@nb.njit(cache=True)
def primary_assignment(
    node_pred: nb.float32[:, :], group_ids: nb.int64[:] = None
) -> nb.boolean[:]:
    """Select shower primary fragments based on the node-score.

    If node groupings are provided, selects a single primary per node
    group: the one that is most likely.

    Parameters
    ----------
    node_pred : np.ndarray
        (C, 2) Logits associated with each node
    group_ids : np.ndarray, optional
        (C) List of node group IDs

    Returns
    -------
    np.ndarray
        (C) Primary labels
    """
    if group_ids is None:
        return sm.argmax(node_pred, axis=1).astype(np.bool_)

    primary_ids = np.zeros(len(node_pred), dtype=np.bool_)
    node_pred = sm.softmax(node_pred, axis=1)
    for g in np.unique(group_ids):
        mask = np.where(group_ids == g)[0]
        idx = np.argmax(node_pred[mask][:, 1])
        primary_ids[mask[idx]] = True

    return primary_ids


@nb.njit(cache=True)
def adjacency_matrix(edge_index: nb.int64[:, :], n: nb.int64) -> nb.boolean[:, :]:
    """Creates a dense adjacency matrix from a list of connected edges in a
    graph, i.e. densify the graph incidence matrix.

    Parameters
    ----------
    edge_index : np.ndarray
        (2, E) Sparse incidence matrix
    num_nodes : int
        Number of nodes in the graph, C

    Returns
    -------
    np.ndarray
        (C, C) Adjacency matrix
    """
    # Cannot use double array indexing to fill the matrix
    adj_mat = np.eye(n, dtype=np.bool_)
    for i, j in edge_index:
        adj_mat[i, j] = True

    return adj_mat


@nb.njit(cache=True)
def grouping_loss(
    pred_mat: nb.float32[:], target_mat: nb.boolean[:], loss: str = "ce"
) -> np.float32:
    """Defines the graph clustering score.

    Given a target adjacency matrix A and a predicted adjacency P, the score is
    evaluated the average CE, L1 or L2 distance between truth and prediction.

    Parameters
    ----------
    pred_mat : np.ndarray
        (C*C) Predicted adjacency matrix scores (flattened)
    target_mat : np.ndarray
        (C*C) Target adjacency matrix scores (flattened)
    loss : str, default 'ce'
        Loss mode used to compute the graph score

    Returns
    -------
    float
        Graph grouping loss
    """
    if loss == "ce":
        return sm.log_loss(target_mat, pred_mat)
    elif loss == "l1":
        return np.mean(np.absolute(pred_mat - target_mat))
    elif loss == "l2":
        return np.mean((pred_mat - target_mat) * (pred_mat - target_mat))
    else:
        raise ValueError("Loss type not recognized")


@nb.njit(cache=True)
def edge_assignment_score(
    edge_index: nb.int64[:, :],
    edge_pred: nb.float32[:, :],
    num_nodes: nb.int64,
    track_node: nb.boolean[:] = None,
) -> (nb.int64[:, :], nb.int64[:], nb.float32):
    """Finds the graph that produces the lowest grouping score by iteratively
    adding the next most likely edge, if it improves the the score. This method
    effectively builds a spanning tree.

    Parameters
    ----------
    edge_index : np.ndarray
        (2, E) Sparse incidence matrix
    edge_pred : np.ndarray
        (E, 2) Logits associated with each edge
    num_nodes : int
        Number of nodes in the graph, C
    track_node : np.ndarray, optional
        (C) Whether a node is a track fragment/particle or not

    Returns
    -------
    np.ndarray
        (E', 2) Optimal incidence matrix
    np.ndarray
        (C) Optimal group ID for each node
    float
        Score of the optimal incidence matrix
    """
    # If there is no edge, do not bother
    if not len(edge_index):
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(num_nodes, dtype=np.int64),
            0.0,
        )

    # Build an input adjacency matrix to constrain the edge selection to
    # the input graph
    adj_mat = adjacency_matrix(edge_index, num_nodes)

    # Interpret the softmax score as a dense adjacency matrix probability
    edge_scores = sm.softmax(edge_pred, axis=1)
    pred_adj = np.eye(num_nodes, dtype=edge_pred.dtype)
    for k, (i, j) in enumerate(edge_index):
        pred_adj[i, j] = edge_scores[k, 1]

    # Remove edges with a score < 0.5 and sort the remainder by increasing
    # order of OFF score
    on_mask = edge_scores[:, 1] >= 0.5
    args = np.argsort(edge_scores[on_mask, 0])
    ord_index = edge_index[on_mask][args]

    # Now iteratively identify the best edges, until the total score cannot
    # be improved any longer
    empty_adj = np.eye(num_nodes, dtype=np.bool_)
    best_ids = np.empty(0, dtype=np.int64)
    best_groups = np.arange(num_nodes, dtype=np.int64)
    track_used = np.zeros(num_nodes, dtype=np.bool_)
    best_loss = grouping_loss(pred_adj.flatten(), empty_adj.flatten())
    known_pairs = nb.typed.List.empty_list(nb.int64)
    for k, (a, b) in enumerate(ord_index):
        # If the edge connect two nodes already in the same group, proceed
        group_a, group_b = best_groups[a], best_groups[b]
        if group_a == group_b:
            continue

        # If the group pair has already been checked against, proceed
        pair = (min(group_a, group_b), max(group_a, group_b))
        pair_hash = pair[0] * num_nodes + pair[1]
        if pair_hash in known_pairs:
            continue

        # If requested, check whether there is already a track connection
        if track_node is not None:
            if track_used[a] or track_used[b]:
                continue
            if track_node[a] ^ track_node[b]:
                if not track_node[a]:
                    track_used[a] = True
                if not track_node[b]:
                    track_used[b] = True

        # Restrict the adjacency matrix and the predictions to the nodes in
        # the two candidate groups
        node_mask = np.where((best_groups == group_a) | (best_groups == group_b))[0]
        sub_pred = sm.linalg.submatrix(pred_adj, node_mask, node_mask).flatten()
        sub_adj = sm.linalg.submatrix(adj_mat, node_mask, node_mask).flatten()

        # Compute the current adjacency matrix between the two groups
        current_adj = (
            best_groups[node_mask] == best_groups[node_mask].reshape(-1, 1)
        ).flatten()

        # Join the two groups if it minimizes the loss
        current_loss = grouping_loss(sub_pred, sub_adj * current_adj)
        combined_loss = grouping_loss(sub_pred, sub_adj)
        if combined_loss < current_loss:
            best_groups[best_groups == pair[1]] = pair[0]
            best_loss += combined_loss - current_loss
            best_ids = np.append(best_ids, k)
            for pair_hash in list(known_pairs):
                pair_i = (pair_hash // num_nodes, pair_hash % num_nodes)
                if group_a in pair_i or group_b in pair_i:
                    known_pairs.remove(pair_hash)
        else:
            known_pairs.append(pair_hash)

    # Build the edge index
    best_index = ord_index[best_ids]

    return best_index, best_groups, best_loss


@nb.njit(cache=True)
def node_assignment_score(
    edge_index: nb.int64[:, :],
    edge_pred: nb.float32[:, :],
    num_nodes: nb.int64,
    track_node: nb.boolean[:] = None,
) -> nb.int64[:]:
    """Finds the graph that produces the lowest grouping score and use
    union-find to find group IDs for each of the nodes in the graph.

    Parameters
    ----------
    edge_index : np.ndarray
        (2, E) Sparse incidence matrix
    edge_pred : TensorBatch
        (E, 2) Logits associated with each edge
    num_nodes : int
        Number of nodes in the graph, C
    track_node : np.ndarray, optional
        (C) Whether a node is a track fragment/particle or not

    Returns
    -------
    np.ndarray
        (C) Optimal group ID for each node
    """
    return edge_assignment_score(edge_index, edge_pred, num_nodes, track_node)[1]


@nb.njit(cache=True)
def node_purity_mask(group_ids: nb.int64[:], primary_ids: nb.int64[:]) -> nb.boolean[:]:
    """Creates a mask that is `True` only for node which belong to a group
    with more exactly one primary.

    This is useful for shower clustering only, for which there can be no or
    multiple primaries in the group, making the primary identification
    ill-defined.

    Note: It is possible that the single true primary has been broken into
    several nodes. In that case, the primary is also ambiguous, skip.
    TODO: pick the most sensible primary in that case, too restrictive
    otherwise (complicated, though).

    Parameters
    ----------
    group_ids : np.ndarray
        (C) Array of cluster group IDs
    primary_ids : np.ndarray
        (C) Cluster of cluster primary IDs

    Returns
    -------
    np.ndarray
        (C) High purity node mask
    """
    purity_mask = np.ones(len(group_ids), dtype=np.bool_)
    for g in np.unique(group_ids):
        group_mask = np.where(group_ids == g)[0]
        if np.sum(primary_ids[group_mask] == 1) != 1:
            purity_mask[group_mask] = False

    return purity_mask


@nb.njit(cache=True)
def edge_purity_mask(
    edge_index: nb.int64[:, :],
    part_ids: nb.int64[:],
    group_ids: nb.int64[:],
    primary_ids: nb.int64[:],
) -> nb.boolean[:]:
    """Creates a mask that is `True` only for edges which connect two nodes
    that both belong to a common group which has a single clear primary.

    This is useful for shower clustering only, for which there can be no or
    multiple primaries in the group, making the the edge classification
    ill-defined (no primary typically indicates a shower which originates
    outside of the active volume).

    Note: It is possible that the single true primary has been broken into
    several nodes; that is acceptable.

    Parameters
    ----------
    edge_index : np.ndarray
        (2, E) Sparse incidence matrix
    part_ids : np.ndarray
        (C) Array of cluster particle IDs
    group_ids : np.ndarray
        (C) Array of cluster group IDs
    primary_ids : np.ndarray
        (C) Array of cluster primary IDs

    Returns
    -------
    np.ndarray
        (E) High purity edge mask
    """
    # Start by building a mask of valid nodes
    node_purity_mask = np.ones(len(group_ids), dtype=np.bool_)
    for g in np.unique(group_ids):
        group_mask = np.where(group_ids == g)[0]
        primary_ids_g = primary_ids[group_mask]
        part_ids_g = part_ids[group_mask]
        if len(np.unique(part_ids_g[primary_ids_g == 1])) != 1:
            # If there not exactly one primary particle ID, the group
            # is not valid
            node_purity_mask[group_mask] = False

    # Edges that connect two invalid nodes are invalid
    purity_mask = (
        node_purity_mask[edge_index[:, 0]] | node_purity_mask[edge_index[:, 1]]
    )

    return purity_mask


def clustering_metrics(clusts, node_assn, node_pred):
    """Computes several clustering metrics for a set of clusters.

    Parameters
    ----------
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    node_assn : np.ndarray
        (C) True node groups labels
    node_pred : np.ndarray
        (C) Predicted node group labels

    Returns
    -------
    float
        Adjusted Rand Index (ARI)
    float
        Adjusted Mutual information (AMI)
    float
        Symetric Best Dice (SBD)
    float
        Purity
    float
        Efficiency
    """
    pred_vox = cluster_to_voxel_label(clusts, node_pred)
    true_vox = cluster_to_voxel_label(clusts, node_assn)
    ari_val = ari(truth_vox, pred_vox)
    ami_val = ami(truth_vox, pred_vox)
    sbd_val = sbd(truth_vox, pred_vox)
    pur_val, eff_val = pur_eff(truth_vox, pred_vox)

    return ari_val, ami_val, sbd_val, pur_val, eff_val


def voxel_efficiency_bipartite(clusts, node_assn, node_pred, primaries):
    """Computes the fraction of secondary voxels that are associated to the
    correct primary.

    Parameters
    ----------
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    node_assn : np.ndarray
        (C) True node groups labels
    node_pred : np.ndarray
        (C) Predicted node group labels
    node_pred : np.ndarray
        (P) List of primary IDs

    Returns
    -------
    float
        Fraction of correctly assigned secondary voxels
    """
    others = [i for i in range(n) if i not in primaries]
    tot_vox = np.sum([len(clusts[i]) for i in others])
    int_vox = np.sum([len(clusts[i]) for i in others if node_pred[i] == node_assn[i]])

    return int_vox / tot_vox


@nb.njit(cache=True)
def cluster_to_voxel_label(
    clusts: nb.types.List(nb.int64[:]), node_labels: nb.int64[:]
) -> nb.int64[:]:
    """Turns a list of labels on clusters to an array of labels on voxels.

    Parameters
    ----------
    clusts : List[np.ndarray]
        (C) List of cluster indexes
    node_labels : np.ndarray
        (C) Node labels

    Returns
    -------
    np.ndarray
        (N) Voxel labels
    """
    counts = [len(c) for c in clusts]

    return np.repeat(node_labels, counts)
