import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.models.layers.cnn.uresnet_layers import UResNet
from mlreco.utils.globals import *

from mlreco.utils.data_structures import TensorBatch

class GraphSPICEEmbedder(UResNet):

    MODULES = ['network_base', 'uresnet', 'graph_spice_embedder']

    RETURNS = {
        'spatial_embeddings': ['tensor', 'coordinates'],
        'covariance': ['tensor', 'coordinates'],
        'feature_embeddings': ['tensor', 'coordinates'],
        'occupancy': ['tensor', 'coordinates'],
        'features': ['tensor', 'coordinates'],
        'hypergraph_features': ['tensor', 'coordinates'],
        'segmentation': ['tensor', 'coordinates']
    }

    def __init__(self, cfg):
        """Initialize the GraphSPICEEmbedder model.

        Parameters
        ----------
        graph_spice_embedder : dict
            Model configuration
        """
        
        uresnet = cfg['uresnet']
        graph_spice_embedder = cfg['graph_spice_embedder']
        super(GraphSPICEEmbedder, self).__init__(uresnet)
        
        self.process_model_config(**graph_spice_embedder)

        # Define outputlayers

        self.outputSpatialEmbeddings = nn.Linear(self.num_filters,
                                                 self.spatial_embedding_dim)

        self.outputFeatureEmbeddings = nn.Linear(self.num_filters,
                                                 self.feature_embedding_dim)

        if self.predict_semantics:
            self.outputSegmentation = nn.Linear(self.num_filters,
                                               self.num_classes)

        self.outputCovariance = nn.Linear(self.num_filters, 2)

        self.outputOccupancy = nn.Linear(self.num_filters, 1)

        self.hyper_dimension = self.spatial_embedding_dim \
                             + self.feature_embedding_dim + 3

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def process_model_config(self, num_classes=5, coordConv=True, 
                             predict_semantics=False, 
                             covariance_mode='softplus', 
                             occupancy_mode='softplus',
                             feature_embedding_dim=16,
                             spatial_embedding_dim=3,
                             dimension=3):
        """Initialize the GraphSPICEEmbedder model.

        Parameters
        ----------
        num_classes : int, default 5
            Number of semantic classes, if predict_semantics is True.
        coordConv : bool, default True
            Whether to append spatial coordinates to the input features
        predict_semantics : bool, default False
            Whether to predict semantic labels
        covariance_mode : str, default 'softplus'
            Covariance layer output function
        occupancy_mode : str, default 'softplus'
            Occupancy layer output function
        feature_embedding_dim : int, default 16
            Dimension of the feature embeddings
        spatial_embedding_dim : int, default 3
            Dimension of the spatial embeddings
        """
        
        self.num_classes = num_classes
        self.coordConv = coordConv
        self.predict_semantics = predict_semantics
        self.covariance_mode = covariance_mode
        self.occupancy_mode = occupancy_mode
        self.feature_embedding_dim = feature_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.D = dimension
        
        if self.covariance_mode == 'exp':
            self.cov_func = torch.exp
        elif self.covariance_mode == 'softplus':
            self.cov_func = nn.Softplus()
        else:
            raise ValueError("Covariance mode not recognized")
        
        if self.occupancy_mode == 'exp':
            self.occ_func = torch.exp
        elif self.occupancy_mode == 'softplus':
            self.occ_func = nn.Softplus()
        else:
            raise ValueError("Occupancy mode not recognized")
        

    def get_embeddings(self, tensorbatch):
        '''
        Forward function for computing GraphSPICE embeddings.
        
        Inputs
        ------
        tensorbatch : TensorBatch
            Input TensorBatch object
            
        Returns
        -------
        res : dict
            Dict[TensorBatch] of output embeddings
        '''
        
        coords = tensorbatch.tensor[:, BATCH_COL:COORD_COLS[-1]+1].int()
        features = tensorbatch.tensor[:, VALUE_COL].float().view(-1, 1)
        
        counts = tensorbatch.counts

        normalized_coords = (coords[:, 1:self.D+1] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = ME.SparseTensor(features, coordinates=coords)

        encoder_res = self.encoder(x)
        encoder_tensors = encoder_res['encoder_tensors']
        final_tensor = encoder_res['final_tensor']
        decoder_tensors = self.decoder(final_tensor, encoder_tensors)

        output_features = decoder_tensors[-1].F

        # Spatial Embeddings
        out = self.outputSpatialEmbeddings(output_features)
        spatial_embeddings = self.tanh(out)

        # Covariance
        out = self.outputCovariance(output_features)
        covariance = self.cov_func(out)

        # Feature Embeddings
        feature_embeddings = self.outputFeatureEmbeddings(output_features)

        # Occupancy
        out = self.outputOccupancy(output_features)
        occupancy = self.occ_func(out)

        # Segmentation
        if self.predict_semantics:
            segmentation = self.outputSegmentation(output_features)

        hypergraph_features = torch.cat([
            spatial_embeddings,
            feature_embeddings,
            covariance,
            occupancy], dim=1)

        res = {
            "spatial_embeddings": TensorBatch(spatial_embeddings + normalized_coords, counts=counts, 
                                              batch_col=None),
            "covariance": TensorBatch(covariance, counts=counts,
                                      batch_col=None),
            "feature_embeddings": TensorBatch(feature_embeddings, counts=counts,
                                              batch_col=None),
            "occupancy": TensorBatch(occupancy, counts=counts,
                                     batch_col=None),
            "features": TensorBatch(output_features, counts=counts,
                                    batch_col=None),
            "hypergraph_features": TensorBatch(hypergraph_features, counts=counts,
                                               batch_col=None)
        }
        if self.predict_semantics:
            res["segmentation"] = TensorBatch(segmentation, counts=counts, batch_col=None)

        return res

    def forward(self, tensorbatch):
        '''
        Train time forward
        '''
        out = self.get_embeddings(tensorbatch)

        return out
