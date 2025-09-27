"""Feature embedding for pixel supervised connected-component clustering."""

import MinkowskiEngine as ME
import torch
import torch.nn as nn

from spine.data import TensorBatch
from spine.model.layer.cnn.uresnet_layers import UResNet
from spine.utils.globals import COORD_COLS, VALUE_COL

__all__ = ["GraphSPICEEmbedder"]


class GraphSPICEEmbedder(nn.Module):
    """Model which produces embeddings of an input sparse point cloud."""

    MODULES = ["uresnet"]

    def __init__(self, uresnet, **base):
        """Initialize the embedding model.

        Parameters
        ----------
        uresnet : dict
            Backbone UResNet configuration
        **base : dict, optional
            Basic parameters
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the uresnet backbone
        self.backbone = UResNet(uresnet)
        self.num_filters = self.backbone.num_filters
        self.spatial_size = self.backbone.spatial_size
        assert (
            self.spatial_size is not None
        ), "Must provide a spatial size to compute normalized coordinates."

        # Process the rest of the configuration
        self.process_model_config(**base)

        # Define output layers, if there is a need for them
        if not self.use_raw_features:
            self.out_spatial = nn.Sequential(
                nn.Linear(self.num_filters, self.spatial_embedding_dim), nn.Tanh()
            )
            self.out_feature = nn.Linear(self.num_filters, self.feature_embedding_dim)
            self.out_cov = nn.Linear(self.num_filters, 2)
            self.out_occupancy = nn.Linear(self.num_filters, 1)

        if self.predict_semantics:
            assert (
                self.num_classes is not None
            ), "Must specify the number of classes predicting semantics."
            self.out_seg = nn.Linear(self.num_filters, self.num_classes)

    def process_model_config(
        self,
        predict_semantics=False,
        num_classes=None,
        coord_conv=True,
        covariance_mode="softplus",
        occupancy_mode="softplus",
        feature_embedding_dim=16,
        spatial_embedding_dim=3,
        use_raw_features=False,
    ):
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
        spatial_embedding_space : int, default 3
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
        self.hyper_dimension = (
            self.spatial_embedding_dim + self.feature_embedding_dim + 3
        )

        # Initialize covariance activation function
        if self.covariance_mode == "exp":
            self.cov_func = torch.exp
        elif self.covariance_mode == "softplus":
            self.cov_func = nn.Softplus()
        else:
            raise ValueError(f"Covariance mode not recognized: {self.covariance_mode}")

        # Initialize occupancy activation function
        if self.occupancy_mode == "exp":
            self.occ_func = torch.exp
        elif self.occupancy_mode == "softplus":
            self.occ_func = nn.Softplus()
        else:
            raise ValueError(f"Occupancy mode not recognized: {self.covariance_mode}")

    def forward(self, data):
        """Compute the embeddings for one batch of data.

        Inputs
        ------
        data: TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel

        Returns
        -------
        dict
            Dictionary of outputs
        """
        # Build an input feature tensor
        coords = data.tensor[:, :VALUE_COL]
        features = data.tensor[:, VALUE_COL].view(-1, 1)

        # If requested, append the normalized coordinates to the feature tensor
        half_size = self.spatial_size / 2
        points = coords[:, 1:]
        normalized_coords = (points - half_size) / half_size
        if self.coord_conv:
            features = torch.cat([normalized_coords, features], dim=1)

        # Pass it through the backbone UResNet, extract output features
        backbone_data = torch.cat((coords, features), dim=1)
        result_backbone = self.backbone(backbone_data)
        output_features = result_backbone["decoder_tensors"][-1].F

        # Convert the output to tensor batches
        coords = TensorBatch(coords, data.counts, coord_cols=COORD_COLS)
        features = TensorBatch(output_features, data.counts)

        # Intialize the result
        result = {"coordinates": coords, "features": features}

        # If requested, pass the raw output features through final layers
        if not self.use_raw_features:
            # Spatial Embeddings (offset by the normalized coordinates)
            spatial_embeddings = self.out_spatial(output_features)

            # Feature Embeddings
            feature_embeddings = self.out_feature(output_features)

            # Covariance
            out = self.out_cov(output_features)
            covariance = self.cov_func(out)

            # Occupancy
            out = self.out_occupancy(output_features)
            occupancy = self.occ_func(out)

            # Bundle the features together
            hypergraph_features = torch.cat(
                [spatial_embeddings, feature_embeddings, covariance, occupancy], dim=1
            )

            # Convert the output to tensor batches
            spatial_embeddings = TensorBatch(
                spatial_embeddings + normalized_coords, data.counts
            )
            feature_embeddings = TensorBatch(feature_embeddings, data.counts)
            covariance = TensorBatch(covariance, data.counts)
            occupancy = TensorBatch(occupancy, data.counts)
            hypergraphneeded_features = TensorBatch(hypergraph_features, data.counts)

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
