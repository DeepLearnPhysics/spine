"""Feature embedding for pixel supervised connected-component clustering."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from spine.data import TensorBatch
from spine.model import sparse
from spine.model.cnn.uresnet_layers import UResNet

__all__ = ["GraphSPICEEmbedder", "GraphSPICEEmbeddingHead"]


class _GraphSPICEEmbeddingBase(sparse.Network):
    """Project aligned voxel features into Graph-SPICE embedding attributes.

    This module owns the output projections but no feature-extraction backbone.
    Standalone Graph-SPICE and shared UResNet reconstruction use the same
    spatial, feature, covariance and occupancy parameterization.
    """

    def __init__(
        self,
        num_features: int,
        spatial_size: int,
        dimension: int = 3,
        **base: Any,
    ) -> None:
        """Initialize projections over externally produced voxel features.

        Parameters
        ----------
        num_features : int
            Number of channels in the supplied decoder feature matrix.
        spatial_size : int
            Positive detector coordinate range used for spatial normalization.
        dimension : int, default 3
            Number of coordinate dimensions.
        **base : dict, optional
            Embedding projection options accepted by ``process_model_config``.

        Raises
        ------
        ValueError
            If a positive spatial normalization range is not provided.
        """
        # Initialize the sparse network metadata and feature specification
        super().__init__(dimension)
        if spatial_size is None or spatial_size <= 0:
            raise ValueError("Must provide a positive spatial size.")
        self.num_filters = num_features
        self.spatial_size = spatial_size

        # Declare attributes populated by the configuration helper. Keeping
        # these declarations in ``__init__`` makes their lifetime explicit to
        # static analyzers.
        self.predict_semantics: bool
        self.num_classes: int | None
        self.coord_conv: bool
        self.covariance_mode: str
        self.occupancy_mode: str
        self.feature_embedding_dim: int
        self.spatial_embedding_dim: int
        self.use_raw_features: bool
        self.hyper_dimension: int
        self.cov_func: Callable[[torch.Tensor], torch.Tensor]
        self.occ_func: Callable[[torch.Tensor], torch.Tensor]

        # Process the rest of the configuration
        self.process_model_config(**base)

        # Define output layers, if there is a need for them
        self.out_spatial: torch.nn.Module
        self.out_feature: torch.nn.Module
        self.out_cov: torch.nn.Module
        self.out_occupancy: torch.nn.Module
        if not self.use_raw_features:
            self.out_spatial = torch.nn.Sequential(
                torch.nn.Linear(self.num_filters, self.spatial_embedding_dim),
                torch.nn.Tanh(),
            )
            self.out_feature = torch.nn.Linear(
                self.num_filters, self.feature_embedding_dim
            )
            self.out_cov = torch.nn.Linear(self.num_filters, 2)
            self.out_occupancy = torch.nn.Linear(self.num_filters, 1)

        self.out_seg: torch.nn.Module
        if self.predict_semantics:
            if self.num_classes is None:
                raise ValueError(
                    "Must specify the number of classes predicting semantics."
                )
            self.out_seg = torch.nn.Linear(self.num_filters, self.num_classes)

    def process_model_config(
        self,
        predict_semantics: bool = False,
        num_classes: int | None = None,
        coord_conv: bool = True,
        covariance_mode: str = "softplus",
        occupancy_mode: str = "softplus",
        feature_embedding_dim: int = 16,
        spatial_embedding_dim: int = 3,
        use_raw_features: bool = False,
    ) -> None:
        """Process the embedding parameters.

        Parameters
        ----------
        predict_semantics : bool, default False
            If `True`, the embedder will output semantic predictions
        num_classes : int, optional
            Number of classes to classify the voxels as
        coord_conv : bool, default True
            If `True`, include the normalized pixel coordinates as a set of
            input features to the backbone UResNet
        covariance_mode : str, default 'softplus'
            Activation used to predict cluster covariance (spatial extent)
        occupancy_mode : str, default 'softplus'
            Activation used to predict cluster occupancy (pixel count)
        feature_embedding_dim : int, default 16
            Number of features per pixel in embedding space
        spatial_embedding_dim : int, default 3
            Number of spatial features per pixel in embedding space
        use_raw_features : bool, default False
            Use the list of embedder features as is, without the output layers
        """
        # Store basic properties
        self.num_classes = num_classes
        self.coord_conv = coord_conv
        self.predict_semantics = predict_semantics
        self.use_raw_features = use_raw_features
        self.covariance_mode = covariance_mode
        self.occupancy_mode = occupancy_mode

        self.feature_embedding_dim = feature_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        if self.feature_embedding_dim < 1:
            raise ValueError("`feature_embedding_dim` must be positive.")
        if self.spatial_embedding_dim < 1:
            raise ValueError("`spatial_embedding_dim` must be positive.")
        if not self.use_raw_features and self.spatial_embedding_dim != self.dimension:
            raise ValueError(
                "`spatial_embedding_dim` must match the input dimension "
                f"({self.dimension}), got {self.spatial_embedding_dim}."
            )
        self.hyper_dimension = (
            self.spatial_embedding_dim + self.feature_embedding_dim + 3
        )

        # Initialize covariance activation function
        if self.covariance_mode == "exp":
            self.cov_func = torch.exp
        elif self.covariance_mode == "softplus":
            self.cov_func = torch.nn.Softplus()
        else:
            raise ValueError(f"Covariance mode not recognized: {self.covariance_mode}")

        # Initialize occupancy activation function
        if self.occupancy_mode == "exp":
            self.occ_func = torch.exp
        elif self.occupancy_mode == "softplus":
            self.occ_func = torch.nn.Softplus()
        else:
            raise ValueError(f"Occupancy mode not recognized: {self.occupancy_mode}")

    def project(
        self, data: TensorBatch, features: TensorBatch
    ) -> dict[str, TensorBatch]:
        """Compute the embeddings for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel
        features : TensorBatch
            Aligned final decoder features with ``num_features`` channels.

        Returns
        -------
        dict
            Dictionary of outputs
        """
        if features.shape[0] != data.shape[0] or list(features.counts) != list(
            data.counts
        ):
            raise ValueError(
                "Shared features must match the data rows and batch counts."
            )
        output_features = features.torch_tensor()
        if output_features.shape[1] != self.num_filters:
            raise ValueError("Shared feature width does not match the embedding head.")
        coordinates = data.batch_coordinates
        half_size = self.spatial_size / 2
        normalized_coords = (coordinates[:, 1:] - half_size) / half_size

        # Convert the output to tensor batches
        coordinate_batch = TensorBatch(
            coordinates,
            data.counts,
            has_batch_col=True,
            coord_cols=tuple(range(1, 1 + self.dimension)),
        )
        feature_batch = TensorBatch(output_features, data.counts)

        # Initialize the result
        result = {
            "coordinates": coordinate_batch,
            "features": feature_batch,
        }

        # If requested, pass the raw output features through final layers
        if not self.use_raw_features:
            # Spatial Embeddings (offset by the normalized coordinates)
            spatial_embeddings = self.out_spatial(output_features) + normalized_coords

            # Feature Embeddings
            feature_embeddings = self.out_feature(output_features)

            # Covariance
            covariance_logits = self.out_cov(output_features)
            covariance = self.cov_func(covariance_logits)

            # Occupancy
            occupancy_logits = self.out_occupancy(output_features)
            occupancy = self.occ_func(occupancy_logits)

            # Bundle the features together
            hypergraph_features = torch.cat(
                [spatial_embeddings, feature_embeddings, covariance, occupancy], dim=1
            )

            # Convert the output to tensor batches
            spatial_embeddings = TensorBatch(spatial_embeddings, data.counts)
            feature_embeddings = TensorBatch(feature_embeddings, data.counts)
            covariance = TensorBatch(covariance, data.counts)
            occupancy = TensorBatch(occupancy, data.counts)
            hypergraph_features = TensorBatch(hypergraph_features, data.counts)

            # Append results
            result.update(
                {
                    "spatial_embeddings": spatial_embeddings,
                    "feature_embeddings": feature_embeddings,
                    "covariance": covariance,
                    "occupancy": occupancy,
                    "hypergraph_features": hypergraph_features,
                }
            )

        # If requested, add a semantic prediction to the output
        if self.predict_semantics:
            # Segmentation layer
            segmentation = self.out_seg(output_features)

            # Append results
            result["segmentation"] = TensorBatch(segmentation, data.counts)

        return result


class GraphSPICEEmbeddingHead(_GraphSPICEEmbeddingBase):
    """Apply Graph-SPICE projections to externally supplied decoder features.

    This head and the standalone embedder share projection parameters and
    implementation through their common base. Each exposes the forward inputs
    appropriate to its source of voxel features.
    """

    def forward(
        self, data: TensorBatch, features: TensorBatch
    ) -> dict[str, TensorBatch]:
        """Project shared decoder features into graph embedding attributes.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + F) Input voxel coordinates and values.
        features : TensorBatch
            (N, C) Decoder features aligned to the input rows and event counts.

        Returns
        -------
        dict
            Aligned features, coordinates and configured embedding attributes.
        """
        return self.project(data, features)


class GraphSPICEEmbedder(_GraphSPICEEmbeddingBase):
    """Standalone embedding head with its own UResNet backbone.

    Parameter paths are unchanged: ``backbone`` and output projections remain
    direct children, allowing existing standalone checkpoints to load.
    """

    def __init__(self, uresnet: dict[str, Any], **base: Any) -> None:
        """Initialize a standalone backbone and its embedding projections.

        Parameters
        ----------
        uresnet : dict
            Backbone UResNet configuration, including its spatial range.
        **base : dict, optional
            Embedding projection and coordinate-input options.
        """
        # Build the backbone first to preserve existing parameter initialization
        backbone = UResNet(uresnet)
        super().__init__(
            backbone.num_filters,
            backbone.spatial_size,
            int(uresnet.get("data_dim", 3)),
            **base,
        )
        self.backbone = backbone

    def forward(self, data: TensorBatch) -> dict[str, TensorBatch]:
        """Compute backbone features and apply the embedding projections.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + F) Voxel coordinates and input features. The standalone
            embedder consumes the primary value channel and optionally appends
            normalized coordinates to the backbone inputs.

        Returns
        -------
        dict
            Aligned features, coordinates and configured embedding attributes.
        """
        # Build an input feature tensor
        coordinates = data.batch_coordinates

        # Graph-SPICE historically consumes the primary charge/value channel
        # (VALUE_COL), even when the shared point tensor carries auxiliary
        # features for other models.
        input_features = data.values.torch_tensor().view(-1, 1)

        # If requested, append the normalized coordinates to the feature tensor
        half_size = self.spatial_size / 2
        points = coordinates[:, 1:]
        normalized_coords = (points - half_size) / half_size
        if self.coord_conv:
            input_features = torch.cat([normalized_coords, input_features], dim=1)

        # Pass it through the backbone UResNet, extract output features
        backbone_data = torch.cat((coordinates, input_features), dim=1)
        result_backbone = self.backbone(backbone_data, batch_size=data.batch_size)
        output_features = result_backbone["decoder_tensors"][-1].aligned_features()

        return self.project(data, TensorBatch(output_features, data.counts))
