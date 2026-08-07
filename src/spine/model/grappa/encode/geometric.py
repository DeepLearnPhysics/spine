from __future__ import annotations

import numpy as np
import torch

from spine.data import ClusterLabelBatch, EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.gnn.cluster import (
    get_cluster_dedxs_batch,
    get_cluster_directions_batch,
    get_cluster_features_batch,
    get_cluster_points_label_batch,
)
from spine.utils.gnn.network import get_cluster_edge_features_batch
from spine.utils.torch.scripts import cdist_fast

__all__ = ["ClustGeoNodeEncoder", "ClustGeoEdgeEncoder"]


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
    name = "geometric"

    # Alternative allowed names of the node encoder
    aliases = ("geo",)

    def __init__(
        self,
        use_numpy: bool = True,
        add_value: bool = False,
        add_shape: bool = False,
        add_points: bool = False,
        random_order: bool = True,
        add_local_dirs: bool = False,
        dir_max_dist: float | str = 5.0,
        add_local_dedxs: bool = False,
        dedx_max_dist: float = 5.0,
    ) -> None:
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
        random_order : bool, default True
            If `True`, randomize the order of the start/end points fetched
            from labels
        add_local_dirs : bool, default False
            Add the local direction estimates at the start and end points
        dir_max_dist : float, default 5.
            Radius around the end points included to estimate the directions
        add_local_dedxs : boo, default False
            Add the local dE/dx estimates at the start and end points
        dedx_max_dist : float, default 5.
            Radius around the end points included to estimate the dE/dx
        """
        # Initialize the parent class
        super().__init__()

        # Store the parameters
        self.use_numpy = use_numpy
        self.add_value = add_value
        self.add_shape = add_shape
        self.add_points = add_points
        self.random_order = random_order
        self.add_local_dirs = add_local_dirs
        self.add_local_dedxs = add_local_dedxs
        self.dedx_max_dist = dedx_max_dist
        self.feature_size = (
            16
            + 2 * int(add_value)
            + int(add_shape)
            + 6 * int(add_points)
            + 6 * int(add_local_dirs)
            + 2 * int(add_local_dedxs)
        )

        # If the maximum distance is specified as `optimize`, optimize it
        self.opt_dir_max_dist = isinstance(dir_max_dist, str)
        if isinstance(dir_max_dist, str):
            if dir_max_dist != "optimize":
                raise ValueError(
                    "If specified as a string, `dir_max_dist` should "
                    "only take the value 'optimize'"
                )
            self.dir_max_dist = -1.0
        else:
            self.dir_max_dist = dir_max_dist

        # Sanity check
        if not self.add_points and (self.add_local_dirs or self.add_local_dedxs):
            raise ValueError(
                "If directions or dE/dx is requested, must also add points"
            )

    def forward(
        self,
        data: ClusterLabelBatch | TensorBatch,
        clusts: IndexBatch,
        coord_label: TensorBatch | None = None,
        points: TensorBatch | None = None,
        extra: TensorBatch | None = None,
        **kwargs: object,
    ) -> TensorBatch | tuple[TensorBatch, TensorBatch]:
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
            Additional objects not used by this encoder

        Returns
        -------
        TensorBatch
           (C, N_c) Set of N_c features per cluster
        """
        # If features are provided directly, must ensure that the corresponding
        # flags in the configuration are as expected.
        if points is not None and not self.add_points:
            raise ValueError(
                "If end points are provided, `add_points` should be `True`."
            )
        if extra is not None and not self.add_value and not self.add_shape:
            raise ValueError(
                "If extra features are provided, either `add_value` or "
                "`add_shape` should be `True`."
            )
        if self.add_points and (coord_label is None) == (points is None):
            raise ValueError(
                "Must provide either `coord_label` or `points` to add points, "
                "not both."
            )

        # Update the flags depending what is provided
        add_value, add_shape = self.add_value, self.add_shape
        if extra is not None:
            if self.add_value and self.add_shape:
                if extra.shape[1] != 3:
                    raise ValueError("Expected `extra.shape[1] == 3`.")
            elif self.add_value and not self.add_shape:
                if extra.shape[1] != 2:
                    raise ValueError("Expected `extra.shape[1] == 2`.")
            elif not self.add_value and self.add_shape:
                if extra.shape[1] != 1:
                    raise ValueError("Expected `extra.shape[1] == 1`.")
            add_value, add_shape = False, False

        # Extract the base geometric features
        if self.use_numpy:
            # If numpy is to be used, pass it through the Numba function
            feats = get_cluster_features_batch(
                data, clusts, add_value, add_shape
            ).torch_tensor()
        else:
            # Otherwise, use the local torch method
            feats = self.get_base_features(
                data, clusts, add_value, add_shape
            ).torch_tensor()

        # Add the extra features if they were provided independently
        if extra is not None:
            feats = torch.cat((feats, extra.torch_tensor()), dim=1)

        # Add the points
        if self.add_points:
            if points is None:
                if coord_label is None:  # pragma: no cover - validated above
                    raise RuntimeError("Point labels were validated but are missing.")
                if not isinstance(data, ClusterLabelBatch):
                    raise TypeError(
                        "Deriving labeled cluster points requires structured "
                        "cluster labels or an explicit `points` tensor."
                    )
                points = get_cluster_points_label_batch(
                    data, coord_label, clusts, random_order=self.random_order
                )

            feats = torch.cat((feats, points.torch_tensor()), dim=1)

        # Add the local directions
        if self.add_local_dirs:
            if points is None:  # pragma: no cover - constructed above
                raise RuntimeError("Local directions require endpoint data.")
            point_tensor = points.torch_tensor()
            for cols in np.arange(point_tensor.shape[1]).reshape(-1, 3):
                starts = TensorBatch(point_tensor[:, cols], points.counts)
                dirs = get_cluster_directions_batch(
                    data, starts, clusts, self.dir_max_dist, self.opt_dir_max_dist
                )
                feats = torch.cat((feats, dirs.torch_tensor()), dim=1)

        # Add the local dE/dx information
        if self.add_local_dedxs:
            if points is None:  # pragma: no cover - constructed above
                raise RuntimeError("Local dE/dx requires endpoint data.")
            point_tensor = points.torch_tensor()
            for cols in np.arange(point_tensor.shape[1]).reshape(-1, 3):
                starts = TensorBatch(point_tensor[:, cols], points.counts)
                dedxs = get_cluster_dedxs_batch(
                    data, starts, clusts, self.dedx_max_dist
                )
                feats = torch.cat((feats, dedxs.torch_tensor()[:, None]), dim=1)

        feats = TensorBatch(feats, clusts.counts)

        # Return
        if self.add_points:
            if points is None:  # pragma: no cover - constructed above
                raise RuntimeError("Endpoint data was not constructed.")
            return feats, points

        return feats

    def get_base_features(
        self,
        data: ClusterLabelBatch | TensorBatch,
        clusts: IndexBatch,
        add_value: bool,
        add_shape: bool,
    ) -> TensorBatch:
        """Generate base geometric cluster node features for one batch of data.

        Parameters
        ----------
        data : ClusterLabelBatch or TensorBatch
            Structured labels or a batch of sparse tensors
        clusts : IndexBatch
            (C) Indexes that make up each cluster
        add_value : bool, default False
            Add mean and RMS value of pixels in the cluster
        add_shape : bool, default False
            Add the particle semantic type
        """
        # Get the value & semantic types
        voxels = data.coords.torch_tensor()
        # Structured labels expose one canonical value, while reconstruction
        # tensors may carry multiple features alongside their primary charge.
        value_data = (
            data.values if isinstance(data, ClusterLabelBatch) else data.feature(0)
        )
        values = value_data.torch_tensor()
        sem_types = None
        if add_shape:
            if not isinstance(data, ClusterLabelBatch):
                raise TypeError(
                    "Semantic geometric features require structured cluster "
                    "labels or an explicit `extra` feature tensor."
                )
            sem_types = data.shapes.torch_tensor()

        # Below is a torch-based implementation of cluster_features
        feats = []
        dtype, device = voxels.dtype, voxels.device
        zeros = lambda x: torch.zeros(x, dtype=dtype, device=device)
        full = lambda x, y: torch.full(x, y, dtype=dtype, device=device)
        for cluster in clusts.index_list:
            # Get list of voxels in the cluster
            cluster_voxels = voxels[cluster]
            size = full([1], len(cluster))

            # Give default values to size-1 clusters
            if len(cluster) < 2:
                cluster_features = torch.cat(
                    (cluster_voxels.flatten(), zeros(12), size)
                )
                if add_value:
                    value_features = zeros(2)
                    value_features[0] = values[cluster[0]]
                    cluster_features = torch.cat((cluster_features, value_features))
                if add_shape and sem_types is not None:
                    shape = full([1], sem_types[cluster[0]])
                    cluster_features = torch.cat((cluster_features, shape))

                feats.append(cluster_features)
                continue

            # Center data
            center = cluster_voxels.mean(dim=0)
            centered_voxels = cluster_voxels - center

            # Get orientation matrix
            orientation = centered_voxels.t().mm(centered_voxels)

            # Get eigenvectors, normalize orientation matrix and
            # eigenvalues to largest. This step assumes points are not
            # superimposed, i.e. that largest eigenvalue != 0
            eigenvalues, eigenvectors = torch.linalg.eigh(
                orientation,
                UPLO="U",
            )
            largest_eigenvalue = eigenvalues[2]
            if largest_eigenvalue <= torch.finfo(eigenvalues.dtype).eps:
                normalized_orientation = torch.zeros_like(orientation)
                direction_weight = eigenvalues.new_zeros(())
            else:
                normalized_orientation = orientation / largest_eigenvalue
                direction_weight = 1.0 - eigenvalues[1] / largest_eigenvalue

            # Get the principal direction
            principal_direction = eigenvectors[:, 2]

            # Projection all points along the principal axis
            longitudinal_projection = centered_voxels.mv(principal_direction)

            # Evaluate the distance from the points to the principal axis
            transverse_projection = centered_voxels - torch.outer(
                longitudinal_projection,
                principal_direction,
            )
            transverse_norm = torch.norm(transverse_projection, dim=1)

            # Flip the principal direction if it is not pointing
            # towards the maximum spread
            orientation_score = torch.dot(
                longitudinal_projection,
                transverse_norm,
            )
            if orientation_score < 0:
                principal_direction = -principal_direction

            # Weight direction
            principal_direction = direction_weight * principal_direction

            # Append (center, B.flatten(), v0, size)
            cluster_features = torch.cat(
                (
                    center,
                    normalized_orientation.flatten(),
                    principal_direction,
                    size,
                )
            )
            if add_value:
                value_features = zeros(2)
                value_features[0] = values[cluster].mean()
                value_features[1] = values[cluster].std()
                cluster_features = torch.cat((cluster_features, value_features))
            if add_shape and sem_types is not None:
                shape = sem_types[cluster].mode().values.reshape(1).to(dtype=dtype)
                cluster_features = torch.cat((cluster_features, shape))

            feats.append(cluster_features)

        # Return
        if len(feats) > 0:
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
    name = "geometric"

    # Alternative allowed names of the edge encoder
    aliases = ("geo",)

    def __init__(
        self,
        use_numpy: bool = True,
        use_legacy_distance: bool = False,
    ) -> None:
        """Initializes the geometric-based node encoder.

        Parameters
        ----------
        use_numpy : bool, default True
            Generate the features on CPU
        use_legacy_distance : bool, default False
            Preserve the historical iterative closest-pair behavior if edge
            features are computed without precomputed closest indexes
        """
        # Initialize the parent class
        super().__init__()

        # Store the parameters
        self.use_numpy = use_numpy
        self.use_legacy_distance = use_legacy_distance
        self.feature_size = 19

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        closest_index: np.ndarray | torch.Tensor | None = None,
        **kwargs: object,
    ) -> TensorBatch:
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
            Additional objects not used by this encoder

        Returns
        -------
        TensorBatch
           (C, N_e) Set of N_e features per edge
        """
        # Extract the base geometric features
        if self.use_numpy:
            # If numpy is to be used, pass it through the Numba function
            feats = get_cluster_edge_features_batch(
                data,
                clusts,
                edge_index,
                closest_index=closest_index,
                use_legacy_distance=self.use_legacy_distance,
            ).torch_tensor()
        else:
            # Otherwise, use the local torch method
            feats = self.get_base_features(data, clusts, edge_index, closest_index)

        # If the graph is undirected, infer reciprocal features
        if not edge_index.directed:
            # Create the feature tensor of reciprocal edges
            feats_flip = feats.clone()
            feats_flip[:, :3] = feats[:, 3:6]
            feats_flip[:, 3:6] = feats[:, :3]
            feats_flip[:, 6:9] = -feats[:, 6:9]

            # Create the full feature tensor
            full_feats = torch.empty(
                (2 * feats.shape[0], feats.shape[1]),
                dtype=feats.dtype,
                device=feats.device,
            )
            full_feats[::2] = feats
            full_feats[1::2] = feats_flip

            feats = full_feats

        return TensorBatch(feats, edge_index.counts)

    def get_base_features(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        closest_index: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        voxels = data.coords.torch_tensor()

        # Here is a torch-based implementation of cluster_edge_features
        feats = []
        for edge in edge_index.directed_index_t:

            # Get the voxels in the clusters connected by the edge
            first_voxels = voxels[clusts.index_list[edge[0]]]
            second_voxels = voxels[clusts.index_list[edge[1]]]

            # Find the closest set point in each cluster
            if closest_index is None:
                distance_matrix = cdist_fast(first_voxels, second_voxels)
                if distance_matrix is None:
                    raise RuntimeError("Failed to compute inter-cluster distances.")
                closest_flat_index = torch.argmin(distance_matrix)
            else:
                closest_flat_index = closest_index[edge[0], edge[1]]

            first_index = closest_flat_index // len(second_voxels)
            second_index = closest_flat_index % len(second_voxels)
            first_point = first_voxels[first_index, :]
            second_point = second_voxels[second_index, :]

            # Displacement
            displacement = first_point - second_point

            # Distance
            length = torch.norm(displacement)
            if length > 0:
                displacement = displacement / length

            # Outer product
            outer_product = torch.outer(
                displacement,
                displacement,
            ).flatten()

            feats.append(
                torch.cat(
                    (
                        first_point,
                        second_point,
                        displacement,
                        length.reshape(1),
                        outer_product,
                    )
                )
            )

        if len(feats) > 0:
            return torch.stack(feats, dim=0)
        else:
            return voxels.new_zeros((0, self.feature_size))
