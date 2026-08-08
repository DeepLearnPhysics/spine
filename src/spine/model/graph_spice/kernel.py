"""Edge kernels that produce graph logits from pairs of node features."""

from __future__ import annotations

import torch

from spine.model.common.mlp import MLP, MLPConfig

__all__ = ["DefaultKernel", "BilinearKernel", "MLPKernel"]


class DefaultKernel(torch.nn.Module):
    """Kernel producing edge score based on feature L2 similarity.

    This Kernel assumes that the upstream embedder produces a set of spatial
    and embedding coordinates and computes the L2 similarity between the two
    node feature vectors. It scales the L2 distance by the covariance and
    penalizes for cluster size dissimilarity.
    """

    name = "default"

    def __init__(self, num_features: int, eps: float = 1e-3) -> None:
        """Initialize the kernel.

        Parameters
        ----------
        num_features : int
            Number of dimensions in feature embedding space
        eps : float
            Features regularization factor
        """
        # Initialize the parent class
        super().__init__()

        # Store the parameters
        self.num_features = num_features
        self.eps = eps
        if self.num_features < 1:
            raise ValueError("`num_features` must be positive.")
        if self.eps <= 0.0:
            raise ValueError("`eps` must be positive.")

    def compute_edge_weight(
        self,
        spatial_embedding_1: torch.Tensor,
        spatial_embedding_2: torch.Tensor,
        feature_embedding_1: torch.Tensor,
        feature_embedding_2: torch.Tensor,
        covariance_1: torch.Tensor,
        covariance_2: torch.Tensor,
        occupancy_1: torch.Tensor,
        occupancy_2: torch.Tensor,
    ) -> torch.Tensor:
        """Converts the output of the embedder into an edge score.

        Parameters
        ----------
        spatial_embedding_1 : torch.Tensor
            (E, 3) Spatial embeddings of the source nodes
        spatial_embedding_2 : torch.Tensor
            (E, 3) Spatial embeddings of the target nodes
        feature_embedding_1 : torch.Tensor
            (E, N_f) Feature embeddings of the source nodes
        feature_embedding_2 : torch.Tensor
            (E, N_f) Feature embeddings of the target nodes
        covariance_1 : torch.Tensor
            (E, 2) Spatial extent of the source node's cluster
        covariance_2 : torch.Tensor
            (E, 2) Spatial extent of the target node's cluster
        occupancy_1 : torch.Tensor
            (E, 1) Multiplicity of the source node's cluster
        occupancy_2 : torch.Tensor
            (E, 1) Multiplicity of the target node's cluster

        Returns
        -------
        torch.Tensor
            (E,) Edge probabilities.
        """
        # Measure spatial distance between nodes, weighted by cluster covariance
        spatial_covariance_1 = covariance_1[:, 0]
        spatial_covariance_2 = covariance_2[:, 0]
        spatial_distance = ((spatial_embedding_1 - spatial_embedding_2) ** 2).sum(dim=1)
        spatial_distance_1 = spatial_distance / (spatial_covariance_1**2 + self.eps)
        spatial_distance_2 = spatial_distance / (spatial_covariance_2**2 + self.eps)

        # Measure feature distance between nodes, weighted by cluster covariance
        feature_covariance_1 = covariance_1[:, 1]
        feature_covariance_2 = covariance_2[:, 1]
        feature_distance = ((feature_embedding_1 - feature_embedding_2) ** 2).sum(dim=1)
        feature_distance_1 = feature_distance / (feature_covariance_1**2 + self.eps)
        feature_distance_2 = feature_distance / (feature_covariance_2**2 + self.eps)

        # Convert the L2 distances to a probability measure (Gaussian kernel)
        probability_12 = torch.exp(-spatial_distance_1 - feature_distance_1)
        probability_21 = torch.exp(-spatial_distance_2 - feature_distance_2)

        probability = torch.clamp(
            probability_12 + probability_21 - probability_12 * probability_21,
            min=0,
            max=1,
        )

        # Scale down the probability if the cluster sizes are highly different
        occupancy_1 = occupancy_1.flatten()
        occupancy_2 = occupancy_2.flatten()
        occupancy_ratio = torch.maximum(
            (occupancy_2 + self.eps) / (occupancy_1 + self.eps),
            (occupancy_1 + self.eps) / (occupancy_2 + self.eps),
        )
        probability /= occupancy_ratio

        return probability

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute kernel logits for all node pairs in a graph.

        This kernel expects a set of (3 + N_f + 2 + 1) features per node:

        - 3 spatial embedding features
        - N_f feature embedding features
        - 2 covariance features
        - 1 occupancy feature

        Parameters
        ----------
        x1 : torch.Tensor
            (E, 3 + N_f + 2 + 1) Features of the source nodes
        x2 : torch.Tensor
            (E, 3 + N_f + 2 + 1) Features of the target nodes

        Returns
        -------
        torch.Tensor
            (E,) Edge logits.
        """
        # Decompose the two feature sets into their constituents
        expected_features = 6 + self.num_features
        if (
            x1.ndim != 2
            or x2.ndim != 2
            or x1.shape != x2.shape
            or x1.shape[1] != expected_features
        ):
            raise ValueError(
                "Expected matching endpoint tensors with shape "
                f"(E, {expected_features}), got {tuple(x1.shape)} and "
                f"{tuple(x2.shape)}."
            )

        num_features = self.num_features
        splits = [3, num_features + 3, num_features + 5]
        spatial_1, feature_1, covariance_1, occupancy_1 = torch.tensor_split(
            x1, splits, dim=1
        )
        spatial_2, feature_2, covariance_2, occupancy_2 = torch.tensor_split(
            x2, splits, dim=1
        )

        # Compute the edge weight, make sure it's between 0 and 1
        weight = self.compute_edge_weight(
            spatial_1,
            spatial_2,
            feature_1,
            feature_2,
            covariance_1,
            covariance_2,
            occupancy_1,
            occupancy_2,
        )
        weight = torch.clamp(weight, min=1e-6, max=1 - 1e-6)

        # Convert probability to a logit, return
        result = torch.logit(weight)

        return result


class BilinearKernel(torch.nn.Module):
    """Kernel producing edges scores based on a learnable bilinear layer."""

    name = "bilinear"

    def __init__(self, num_features: int, bias: bool = False) -> None:
        """Initialize the kernel.

        Parameters
        ----------
        num_features : int
            Number of dimensions in feature embedding space
        bias : bool, default False
            If `True`, allows for an overall bias in the bilinear layer
        """
        # Initialize the parent class
        super().__init__()

        if num_features < 1:
            raise ValueError("`num_features` must be positive.")

        # Initialize the bilinear layer
        self.num_features = num_features
        self.bilin = torch.nn.Bilinear(num_features, num_features, 1, bias=bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Computes the kernel edge score of all node pairs in the graph.

        Parameters
        ----------
        x1 : torch.Tensor
            (E, N_f) Features of the source nodes
        x2 : torch.Tensor
            (E, N_f) Features of the target nodes

        Returns
        -------
        torch.Tensor
            (E, 1) Edge logits.
        """
        # Check on input size, pass through the bilinear layer
        if (
            x1.ndim != 2
            or x2.ndim != 2
            or x1.shape != x2.shape
            or x1.shape[1] != self.num_features
        ):
            raise ValueError(
                "Expected matching endpoint tensors with shape "
                f"(E, {self.num_features}), got {tuple(x1.shape)} and "
                f"{tuple(x2.shape)}."
            )

        return self.bilin(x1, x2)


class MLPKernel(torch.nn.Module):
    """Kernel producing edges scores based on an MLP and a linear layer."""

    name = "mlp"

    def __init__(
        self,
        num_features: int,
        bias: bool = False,
        mlp: MLPConfig | None = None,
    ) -> None:
        """Initialize the kernel.

        Parameters
        ----------
        num_features : int
            Number of dimensions in feature embedding space
        bias : bool, default False
            If `True`, allows for an overall bias in the bilinear layer
        mlp : dict, optional
            MLP architecture configuration, see :class:`MLP`
        """
        # Initialize the parent class
        super().__init__()
        if num_features < 1:
            raise ValueError("`num_features` must be positive.")

        # Initialize the underlying MLP
        if mlp is None:
            mlp = {
                "depth": 2,
                "width": 32,
                "activation": "elu",
                "normalization": "batch_norm",
            }

        self.mlp = MLP(in_channels=num_features, **mlp)

        # Initialize the final linear layer
        self.lin = torch.nn.Linear(2 * self.mlp.feature_size, 1, bias=bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Computes the kernel edge score of all node pairs in the graph.

        Parameters
        ----------
        x1 : torch.Tensor
            (E, N_f) Features of the source nodes
        x2 : torch.Tensor
            (E, N_f) Features of the target nodes

        Returns
        -------
        torch.Tensor
            (E, 1) Edge logits.
        """
        if (
            x1.ndim != 2
            or x2.ndim != 2
            or x1.shape != x2.shape
            or x1.shape[1] != self.mlp.in_channels
        ):
            raise ValueError(
                "Expected matching endpoint tensors with shape "
                f"(E, {self.mlp.in_channels}), got {tuple(x1.shape)} and "
                f"{tuple(x2.shape)}."
            )

        # Pass the node features through the MLP
        endpoint_features_1 = self.mlp(x1)
        endpoint_features_2 = self.mlp(x2)

        # Concatenate them and pass them through the linear layer, return
        result = self.lin(torch.cat([endpoint_features_1, endpoint_features_2], dim=1))

        return result
