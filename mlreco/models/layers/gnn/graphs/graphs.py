"""Graph construction classes for GNNs."""

import numpy as np
import numba as nb
import torch
from typing import List, Union

from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree

import mlreco.utils.numba_local as nbl
from mlreco.utils.globals import COORD_COLS
from mlreco.utils.data_structures import TensorBatch, IndexBatch

from .base import GraphBase

__all__ = ['LoopGraph', 'CompleteGraph', 'DelaunayGraph', 'MSTGraph',
           'KNNGraph', 'BipartiteGraph']


class LoopGraph(GraphBase):
    """Generates loop-only graphs.

    Connects every node in the graph with itself but nothing else.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """
    name = 'loop'

    def generate(self, clusts, **kwargs):
        """Generate a loop-graph on a set of N nodes.

        Parameters
        ----------
        clusts : IndexBatch
            (C) Cluster indexes
        **kwargs : dict, optional
            Unused graph generation arguments

        Returns
        -------
        np.ndarray
            (2, E) Tensor of edges
        np.ndarray
            (B) Number of edges in each entry of the batch
        """
        # There is exactly one edge per cluster
        edge_counts = clusts.list_counts

        # Define the loop graph
        num_nodes = np.sum(edge_counts)
        edge_index = np.repeat(np.arange(num_nodes)[None, :], 2, axis=0)

        return edge_index, edge_counts


class CompleteGraph(GraphBase):
    """Generates graphs that connect each node with every other node.

    If two nodes belong to separate batches, they cannot be connected.

    See :class:`GraphBase` for attributes/methods shared
    across all graph constructors.
    """
    name = 'complete'

    def generate(self, clusts, **kwargs):
        """Generates a complete graph on a set of batched nodes.

        Parameters
        ----------
        clusts : IndexBatch
            (C) Cluster indexes
        **kwargs : dict, optional
            Unused graph generation arguments

        Returns
        -------
        np.ndarray
            (2, E) Tensor of edges
        np.ndarray
            (B) Number of edges in each entry of the batch
        """
        return self._generate(clusts.list_counts)

    @staticmethod
    @nb.njit(cache=True)
    def _generate(counts: nb.int64[:]) -> (nb.int64[:,:], nb.int64[:]):
        # Loop over the batches, define the adjacency matrix for each
        edge_list, offset = [], 0
        edge_counts = np.zeros(len(counts), dtype=counts.dtype)
        for b in range(len(counts)):
            # Build a list of edges
            c = counts[b]
            adj_mat = np.triu(np.ones((c, c)), k=1)
            edges = np.vstack(np.where(adj_mat))

            edge_list.append(offset + edges)
            edge_counts[b] = edges.shape[-1]
            offset += c

        # Merge the blocks together
        num_edges = np.sum(edge_counts)
        edge_index = np.empty((2, num_edges), dtype=np.int64)
        offset = 0
        for edges in edge_list:
            num_block = edges.shape[1]
            edge_index[:, offset:offset + num_block] = edges
            offset += num_block

        return edge_index, edge_counts


class DelaunayGraph(GraphBase):
    """Generates graphs based on the Delaunay triangulation of the input
    node locations.

    Triangulates the input, converts the triangles into a list of valid edges.
    """
    name = 'delaunay'

    def generate(self, data, clusts, batch_ids):
        """Generates an incidence matrix that connects nodes
        that share an edge in their corresponding Euclidean Delaunay graph.

        Parameters
        ----------
        data : np.ndarray
            (N, 1 + D) Array of point batches and coordinates
        clusts : np.ndarray
            (C) List of arrays of voxel IDs in each cluster
        batch_ids : np.ndarray
            (C) Batch ID associated with each cluster

        Returns
        -------
        np.ndarray
            (2, E) Tensor of edges
        """
        return self._generate(data, clusts, batch_ids, self.directed)

    @staticmethod
    @nb.njit(cache=True)
    def _generate(data: nb.float64[:,:],
                  clusts: nb.types.List(nb.int64[:]),
                  batch_ids: nb.int64[:],
                  directed: bool = False) -> nb.int64[:,:]:
        # For each batch, find the list of edges, append it
        edge_list, offset = [], 0
        edge_counts = np.zeros(len(counts), dtype=counts.dtype)
        for b in np.unique(batch_ids):
            # Combine the cluster masks into one
            clust_ids = np.where(batch_ids == b)[0]
            limits = np.array([0] + [len(clusts[i]) for i in clust_ids])
            limits = np.cumsum(limits)
            mask = np.zeros(limits[-1], dtype=np.int64)
            labels = np.zeros(limits[-1], dtype=np.int64)
            for i in range(len(clust_ids)):
                l, u = limits[i], limits[i+1]
                mask[l:u] = clusts[clust_ids[i]]
                labels[l:u] = i

            # Run Delaunay triangulation in object mode because it relies on an
            # external package. Only way to speed this up would be to implement
            # Delaunay triangulation in Numba (tall order)
            with nb.objmode(tri = 'int32[:,:]'): 
                # Run Delaunay triangulation in joggled mode, this
                # guarantees simplical faces (no ambiguities)
                tri = Delaunay(data[mask][:, COORD_COLS], 
                               qhull_options='QJ').simplices

            # Create an adjanceny matrix from the simplex list
            adj_mat = np.zeros((len(clust_ids),len(clust_ids)), dtype=np.bool_)
            for s in tri:
                for i in s:
                    for j in s:
                        if labels[j] > labels[i]:
                            adj_mat[labels[i],labels[j]] = True

            # Convert the adjancency matrix to a list of edges, store
            edges = np.where(adj_mat)
            edges = np.vstack((clust_ids[edges[0]], clust_ids[edges[1]]))

            # Add reciprocal edges if the graph is undirected
            if not directed:
                edge_index = np.hstack((edge_index, edge_index[::-1]))

            edge_list.append(offset + edge_index.T)
            edge_counts[b] = edge_index.shape[-1]
            offset += c

        # Merge the blocks together
        offset = 0
        result = np.empty((2, num_edges), dtype=np.int64)
        for edges in edge_list:
            num_block = edges.shape[-1]
            result[:, offset:offset+num_block] = edges
            offset += num_block

        # Add the reciprocal edges for undericted graph
        if not self.directed:
            result = np.hstack((result, result[::-1]))

        return result


class MSTGraph(GraphBase):
    """Generates graphs based on the minimum-spanning tree (MST) of the input
    node locations.

    Makes an edge for each branch in the minimum-spanning tree.
    """
    name = 'mst'

    def generate(self, data, clusts, batch_ids):
        pass

@nb.njit(cache=True)
def mst_graph(batch_ids: nb.int64[:],
              dist_mat: nb.float64[:,:],
              directed: bool = False) -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix that connects nodes
    that share an edge in their corresponding Euclidean Minimum Spanning Tree (MST).

    Parameters
    ----------
        batch_ids (np.ndarray): (C) List of batch ids
        dist_mat (np.ndarray) : (C,C) Tensor of pair-wise cluster distances
        directed (bool)       : If directed, only keep edges [i,j] for which j>=i
    Returns
    -------
        np.ndarray: (2,E) Tensor of edges
    """
    # For each batch, find the list of edges, append it
    edge_list = []
    num_edges = 0
    ret = np.empty((0, 2), dtype=np.int64)
    for b in np.unique(batch_ids):
        clust_ids = np.where(batch_ids == b)[0]
        if len(clust_ids) > 1:
            submat = np.triu(nbl.submatrix(dist_mat, clust_ids, clust_ids))
            with nb.objmode(mst_mat = 'float32[:,:]'): # Suboptimal. Ideally want to reimplement in Numba, but tall order...
                mst_mat = minimum_spanning_tree(submat).toarray().astype(np.float32)
            edges = np.where(mst_mat > 0.)
            edges = np.vstack((clust_ids[edges[0]],clust_ids[edges[1]])).T
            ret   = np.vstack((ret, edges))

    # Add the reciprocal edges as to create an undirected graph, if requested
    if not directed:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T

class KNNGraph(GraphBase):
    """Generates graphs based on the k nearest-neighbor (kNN) graph of the
    input node locations.

    Makes an edge for each nearest neighbor connection.
    """
    name = 'knn'

    def generate(self, data, clusts, batch_ids):
        pass


@nb.njit(cache=True)
def knn_graph(batch_ids: nb.int64[:],
              k: nb.int64,
              dist_mat: nb.float64[:,:],
              directed: bool = False) -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix that connects nodes
    that are k nearest neighbors. Sorts the distance matrix.

    Parameters
    ----------
        batch_ids (np.ndarray): (C) List of batch ids
        k (int)               : Number of connected neighbors for each node
        dist_mat (np.ndarray) : (C,C) Tensor of pair-wise cluster distances
        directed (bool)       : If directed, only keep edges [i,j] for which j>=i
    Returns
    -------
        np.ndarray: (2,E) Tensor of edges
    """
    # Use the available distance matrix to build a kNN graph
    ret = np.empty((0, 2), dtype=np.int64)
    for b in np.unique(batch_ids):
        clust_ids = np.where(batch_ids == b)[0]
        if len(clust_ids) > 1:
            subk = min(k+1, len(clust_ids))
            submat = nbl.submatrix(dist_mat, clust_ids, clust_ids)
            for i in range(len(submat)):
                idxs = np.argsort(submat[i])[1:subk]
                edges = np.empty((subk-1,2), dtype=np.int64)
                for j, idx in enumerate(np.sort(idxs)):
                    edges[j] = [clust_ids[i], clust_ids[idx]]
                if len(edges):
                    ret = np.vstack((ret, edges))

    # Add the reciprocal edges as to create an undirected graph, if requested
    if not directed:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T

class BipartiteGraph(GraphBase):
    """Generates graphs that connect primary nodes to secondary nodes. """
    name = 'bipartite'

    def generate(self, data, clusts, batch_ids):
        pass


@nb.njit(cache=True)
def bipartite_graph(batch_ids: nb.int64[:],
                    primaries: nb.boolean[:],
                    directed: nb.boolean = True,
                    directed_to: str = 'secondary') -> nb.int64[:,:]:
    """
    Function that returns an incidence matrix of the bipartite graph
    between primary nodes and non-primary nodes.

    Parameters
    ----------
        batch_ids (np.ndarray): (C) List of batch ids
        primaries (np.ndarray): (C) Primary mask (True if primary)
        directed (bool)       : True if edges only exist in one direction
        directed_to (str)     : Whether to point the edges to the primaries or the secondaries
    Returns
    -------
        np.ndarray: (2,E) Tensor of edges
    """
    # Create the incidence matrix
    ret = np.empty((0,2), dtype=np.int64)
    for i in np.where(primaries)[0]:
        for j in np.where(~primaries)[0]:
            if batch_ids[i] ==  batch_ids[j]:
                ret = np.vstack((ret, np.array([[i,j]])))

    # Handle directedness, by default graph is directed towards secondaries
    if directed:
        if directed_to == 'primary':
            ret = ret[:,::-1]
        elif directed_to != 'secondary':
            raise ValueError('Graph orientation not recognized')
    else:
        ret = np.vstack((ret, ret[:,::-1]))

    return ret.T


