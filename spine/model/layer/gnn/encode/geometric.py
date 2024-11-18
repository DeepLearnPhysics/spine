import numpy as np
import torch

from spine.data import TensorBatch

from spine.utils.torch_local import local_cdist
from spine.utils.globals import COORD_COLS, VALUE_COL, SHAPE_COL
from spine.utils.gnn.cluster import (
        get_cluster_features_batch, get_cluster_points_label_batch,
        get_cluster_directions_batch, get_cluster_dedxs_batch)
from spine.utils.gnn.network import get_cluster_edge_features_batch

__all__ = ['ClustGeoNodeEncoder', 'ClustGeoEdgeEncoder']


class ClustGeoNodeEncoder(torch.nn.Module):
    """Produces cluster node features using hand-engineered quantities.

    The basic 16 geometric features are composed of:
    - Center (3)
    - Covariance matrix (9)
    - Principal axis (3)
    - Voxel count (1)

    The flag `add_value` adds the following 2 features:
    - Mean energy (1)
    - RMS energy (1)

    The flag `add_shape` adds the particle shape information:
    - Semantic type (1), i.e. most represented type in cluster

    The flag `add_points` adds the particle end points information
    - Start point (3)
    - End point (3)

    The flag `add_directions` adds the particle direction information
    - Start direction (3)
    - End direction (3)

    The flag `add_local_dedxs` adds the local dEdx estimate at each endpoint
    - Start dEdx (1)
    - End dEdx (1)
    """

    # Name of the node encoder (as specified in the configuration)
    name = 'geometric'

    # Alternative allowed names of the node encoder
    aliases = ('geo',)

    def __init__(self, use_numpy=True, add_value=False, add_shape=False,
                 add_points=False, add_local_dirs=False, dir_max_dist=5.,
                 add_local_dedxs=False, dedx_max_dist=5.):
        """Initializes the geometric-based node encoder.

        Parameters
        ----------
        use_numpy : bool, default True
            Generate the features on CPU
        add_value : bool, default False
            Add mean and RMS value of pixels in the cluster
        add_shape : bool, default False
            Add the particle semantic type
        add_points : bool, default False
            Add the start/end points of the particles
        add_local_dirs : bool, default False
            Add the local direction estimates at the start and end points
        dir_max_dist : float, default 5.
            Radius around the end points included to estimate the directions
        add_local_dedxs : boo, default False
            Add the local dE/dx estimates at the start and end points
        dedx_max_dist : float, default 5.
            Readius around the end points incldued to estimate the dE/dx
        """
        # Initialize the parent class
        super().__init__()

        # Store the paramters
        self.use_numpy = use_numpy
        self.add_value = add_value
        self.add_shape = add_shape
        self.add_points = add_points
        self.add_local_dirs = add_local_dirs
        self.dir_max_dist = dir_max_dist
        self.add_local_dedxs = add_local_dedxs
        self.dedx_max_dist = dedx_max_dist

        # If the maximum distance is specified as `optimize`, optimize it
        self.opt_dir_max_dist = False
        if isinstance(self.dir_max_dist, str):
            assert self.dir_max_dist == 'optimize', (
                    "If specified as a string, `dir_max_dist` should "
                    "only take the value 'optimize'")
            self.opt_dir_max_dist = True

        # Sanity check
        assert (self.add_points or
                (not self.add_local_dirs and not self.add_local_dedxs)), (
                "If directions or dE/dx is requested, must also add points")

    def forward(self, data, clusts, coord_label=None,
                points=None, extra=None, **kwargs):
        """Generate geometric cluster node features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            (C) Indexes that make up each cluster
        coord_label : TensorBatch
            (P, 1 + D + 8) Label start, end, time and shape for each point
        points : TensorBatch
            (C, 6) Set of start/end points for each input cluster
        extra : TensorBatch
            (C, 1/2/3) Set of mean/rms values in the cluster and/or shape
        **kwargs : dict, optional
            Additional objects no used by this encoder

        Returns
        -------
        TensorBatch
           (C, N_c) Set of N_c features per cluster
        """
        # If features are provided directly, must ensure that the corresponding
        # flags in the configuration are as expected.
        assert points is None or self.add_points, (
                "If end points are provided, `add_points` should be `True`.")
        assert extra is None or (self.add_value or self.add_shape), (
                "If extra features are provided, either `add_value` or "
                "`add_shape` should be `True`.")
        assert (not self.add_points or
                ((coord_label is not None) ^ (points is not None))), (
                "Must provide either `coord_label` or `points` to add points, "
                "not both.")

        # Update the flags depending what is provided
        add_value, add_shape = self.add_value, self.add_shape
        if extra is not None:
            if self.add_value and self.add_shape:
                assert extra.shape[1] == 3
            elif self.add_value and not self.add_shape:
                assert extra.shape[1] == 2
            elif not self.add_value and self.add_shape:
                assert extra.shape[1] == 1
            add_value, add_shape = False, False

        # Extract the base geometric features
        if self.use_numpy:
            # If numpy is to be used, pass it through the Numba function
            feats = get_cluster_features_batch(
                    data, clusts, add_value, add_shape).tensor
        else:
            # Otherwise, use the local torch method
            feats = self.get_base_features(
                    data, clusts, add_value, add_shape).tensor

        # Add the extra features if they were provided independantly
        if extra is not None:
            feats = torch.cat((feats, extra.tensor), dim=1)

        # Add the points
        if self.add_points:
            if points is None:
                points = get_cluster_points_label_batch(
                        data, coord_label, clusts)

            feats = torch.cat((feats, points.tensor), dim=1)

        # Add the local directions
        if self.add_local_dirs:
            for cols in np.arange(points.tensor.shape[1]).reshape(-1, 3):
                starts = TensorBatch(points.tensor[:, cols], points.counts)
                dirs = get_cluster_directions_batch(
                        data, starts, clusts, self.dir_max_dist,
                        self.opt_dir_max_dist)
                feats = torch.cat((feats, dirs.tensor), dim=1)

        # Add the local dE/dx information
        if self.add_local_dedxs:
            for cols in np.arange(points.tensor.shape[1]).reshape(-1, 3):
                starts = TensorBatch(points.tensor[:, cols], points.counts)
                dedxs = get_cluster_dedxs_batch(
                        data, starts, clusts, self.dedx_max_dist)
                feats = torch.cat((feats, dedxs.tensor[:, None]), dim=1)

        feats = TensorBatch(feats, clusts.counts)

        # Return
        if self.add_points:
            return feats, points

        return feats

    def get_base_features(self, data, clusts, add_value, add_shape):
        """Generate base geometric cluster node features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            (C) Indexes that make up each cluster
        add_value : bool, default False
            Add mean and RMS value of pixels in the cluster
        add_shape : bool, default False
            Add the particle semantic type
        """
        # Get the value & semantic types
        voxels = data.tensor[:, COORD_COLS]
        values = data.tensor[:, VALUE_COL]
        if add_value:
            sem_types = data.tensor[:, SHAPE_COL]

        # Below is a torch-based implementation of cluster_features
        feats = []
        dtype, device = voxels.dtype, voxels.device
        zeros = lambda x: torch.zeros(x, dtype=dtype, device=device)
        full = lambda x, y: torch.full(x, y, dtype=dtype, device=device)
        for c in clusts.index_list:
            # Get list of voxels in the cluster
            x = voxels[c]
            size = full([1], len(c))

            # Give default values to size-1 clusters
            if len(c) < 2:
                feats_v = torch.cat((x.flatten(), zeros(12), size), dim=1)
                if add_value:
                    vals = zeros(2)
                    vals[0] = values[c[0]]
                    feats_v = torch.cat((feats_v, vals), dim=1)
                if add_shape:
                    shape = full([1], sem_types[c[0]])
                    feats_v = torch.cat((feats_v, shape), dim=1)

                feats.append(feats_v)
                continue

            # Center data
            center = x.mean(dim=0)
            x = x - center

            # Get orientation matrix
            A = x.t().mm(x)

            # Get eigenvectors, normalize orientation matrix and
            # eigenvalues to largest. This step assumes points are not
            # superimposed, i.e. that largest eigenvalue != 0
            #w, v = torch.symeig(A, eigenvectors=True)
            w, v = torch.linalg.eigh(A, UPLO='U')
            dirwt = 1.0 - w[1] / w[2]
            B = A / w[2]

            # Get the principal direction
            v0 = v[:,2]

            # Projection all points along the principal axis
            x0 = x.mv(v0)

            # Evaluate the distance from the points to the principal axis
            xp0 = x - torch.ger(x0, v0)
            np0 = torch.norm(xp0, dim=1)

            # Flip the principal direction if it is not pointing
            # towards the maximum spread
            sc = torch.dot(x0, np0)
            if sc < 0:
                v0 = -v0

            # Weight direction
            v0 = dirwt * v0

            # Append (center, B.flatten(), v0, size)
            feats_v = torch.cat((center, B.flatten(), v0, size))
            if add_value:
                vals = zeros(2)
                vals[0] = values[c].mean()
                vals[1] = values[c].std()
                feats_v = torch.cat((feats_v, vals), dim=1)
            if add_shape:
                shape = full([1], sem_types[c].mode())
                feats_v = torch.cat((feats_v, shape), dim=1)

            feats.append(feats_v)

        # Return
        if len(feats):
            return TensorBatch(torch.stack(feats, dim=0), clusts.counts)
        else:
            return TensorBatch(zeros((0, 16)), clusts.counts)


class ClustGeoEdgeEncoder(torch.nn.Module):
    """Produces cluster edge features using hand-engineered quantities.

    The basic 19 geometric features are composed of:
    - Position of the voxel in the first cluster closest to the second (3)
    - Position of the voxel in the second cluster closest to the first (3)
    - Displacement vector from the first to the second point defined above (3)
    - Length of the displacement vector (1)
    - Outer product of the displacement vector (9)
    """

    # Name of the edge encoder (as specified in the configuration)
    name = 'geometric'

    # Alternative allowed names of the edge encoder
    aliases = ('geo',)

    def __init__(self, use_numpy=True):
        """Initializes the geometric-based node encoder.

        Parameters
        ----------
        use_numpy : bool, default True
            Generate the features on CPU
        """
        # Initialize the parent class
        super().__init__()

        # Store the paramters
        self.use_numpy = use_numpy

    def forward(self, data, clusts, edge_index, closest_index=None, **kwargs):
        """Generate geometric cluster edge features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            (C) Indexes that make up each cluster
        edge_index : EdgeIndexBatch
            Incidence map between clusters
        closest_index : Union[np.ndarray, torch.Tensor], optional
            (C, C) : Combined index of the closest pair of voxels per
            pair of clusters
        **kwargs : dict, optional
            Additional objects no used by this encoder

        Returns
        -------
        TensorBatch
           (C, N_e) Set of N_e features per edge
        """
        # Extract the base geometric features
        if self.use_numpy:
            # If numpy is to be used, pass it through the Numba function
            feats = get_cluster_edge_features_batch(
                    data, clusts, edge_index,
                    closest_index=closest_index).tensor
        else:
            # Otherwise, use the local torch method
            feats = self.get_base_features(
                    data, clusts, edge_index, closest_index)

        # If the graph is undirected, infer reciprocal features
        if not edge_index.directed:
            # Create the feature tensor of reciprocal edges
            feats_flip = feats.clone()
            feats_flip[:,  :3] =  feats[:, 3:6]
            feats_flip[:, 3:6] =  feats[:,  :3]
            feats_flip[:, 6:9] = -feats[:, 6:9]

            # Create the full feature tensor
            full_feats = torch.empty(
                    (2*feats.shape[0], feats.shape[1]),
                    dtype=feats.dtype, device=feats.device)
            full_feats[::2] = feats
            full_feats[1::2] = feats_flip

            feats = full_feats

        return TensorBatch(feats, edge_index.counts)

    def get_base_features(self, data, clusts, edge_index, closest_index=None):
        """Generate base geometric cluster node features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            (C) Indexes that make up each cluster
        edge_index : EdgeIndexBatch
            Incidence map between clusters
        closest_index : Union[np.ndarray, torch.Tensor], optional
            (C, C) : Combined index of the closest pair of voxels per
            pair of clusters
        """
        # Get the voxel set
        voxels = data.tensor[:, COORD_COLS]

        # Here is a torch-based implementation of cluster_edge_features
        feats = []
        for e in edge_index.directed_index_t:

            # Get the voxels in the clusters connected by the edge
            x1 = voxels[clusts.index_list[e[0]]]
            x2 = voxels[clusts.index_list[e[1]]]

            # Find the closest set point in each cluster
            if closest_index is None:
                d12 = local_cdist(x1, x2)
                imin = torch.argmin(d12)
            else:
                imin = closest_index[e[0], e[1]]

            i1, i2 = imin//len(x2), imin%len(x2)
            v1 = x1[i1,:] # closest point in c1
            v2 = x2[i2,:] # closest point in c2

            # Displacement
            disp = v1 - v2

            # Distance
            lend = torch.norm(disp)
            if lend > 0:
                disp = disp / lend

            # Outer product
            B = torch.ger(disp, disp).flatten()

            feats.append(torch.cat([v1, v2, disp, lend.reshape(1), B]))

        if len(feats):
            return torch.stack(feats, dim=0)
        else:
            return torch.zeros((0, 19))
