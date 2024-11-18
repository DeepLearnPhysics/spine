"""Analysis script used to evaluate the point proposal accuracy."""

from warnings import warn

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist

from spine.ana.base import AnaBase

from spine.utils.globals import (
        LOWES_SHP, COORD_COLS, PPN_LTYPE_COL, PPN_LENDP_COL,
        PPN_SHAPE_COL, PPN_END_COLS)

__all__ = ['PointProposalAna']


class PointProposalAna(AnaBase):
    """Class which computes and stores the necessary data to evaluate the
    point proposal accuracy.

    It evaluates the following metrics:
    - Distance from true to closest predicted point (efficiency)
    - Distance from predicted to closest true point (purity)
    - Point type classification accuracy
    - Point end classification accuracy
    """

    # Name of the analysis script (as specified in the configuration)
    name = 'point_eval'

    # Set of data keys needed for this analysis script to operate
    _keys = (('ppn_pred', True),)

    def __init__(self, num_classes=LOWES_SHP, label_key='ppn_label',
                 endpoints=False, **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        num_classes : int, default 4
            Number of pixel classses, excluding the ghost class
        label_key : str, default 'seg_label'
            Name of the tensor which contains the segmentation labels
        endpoints : bool, default False
            Evaluate the accuracy of end point classification
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Store the basic parameters
        self.num_classes = num_classes
        self.label_key = label_key
        self.endpoints = endpoints

        # Append other required key
        self.update_keys({self.label_key: True})

        # Initialize the output
        self.initialize_writer('truth_to_reco')
        self.initialize_writer('reco_to_truth')

        # Initialize a dummy dictionary to return when there is no match
        self.dummy_dict = {
            'dist': -1., 'shape': -1, 'closest_shape': -1}
        for s in range(self.num_classes):
            self.dummy_dict[f'dist_{s}'] = -1.
        if endpoints:
            self.dummy_dict['end'] = -1
            self.dummy_dict['closest_end'] = -1

    def process(self, data):
        """Store the semantic segmentation metrics for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Initialize dictionaries to compare
        points, types, ends = {}, {}, {}

        # Fetch the list of label points and their characteristics
        points['truth'] = data[self.label_key][:, COORD_COLS]
        types['truth'] = data[self.label_key][:, PPN_LTYPE_COL].astype(int)
        if self.endpoints:
            ends['truth'] = data[self.label_key][:, PPN_LENDP_COL].astype(int)

        # Fetch the list of predicted points and their characteristics
        points['reco'] = data['ppn_pred'][:, COORD_COLS]
        types['reco'] = data['ppn_pred'][:, PPN_SHAPE_COL].astype(int)
        if self.endpoints:
            ends['reco'] = np.argmax(data['ppn_pred'][:, PPN_END_COLS], axis=1)

        # Compute the pair-wise distances between label and predicted points
        dist_mat = {}
        dist_mat['reco'] = cdist(points['reco'], points['truth'])
        dist_mat['truth'] = dist_mat['reco'].T

        # Store one row per predicted point
        matches = (('truth', 'reco'), ('reco', 'truth'))
        for k, (source, target) in enumerate(matches):
            # If there is no source points, nothing to do
            if not len(points[source]):
                continue

            # If there are no target points, record no match
            if not len(points[target]):
                # Append dummy values
                dummy = {**self.dummy_dict}
                for i in range(len(points[source])):
                    dummy['shape'] = types[source][i]
                    if self.endpoints:
                        dummy['end'] = ends[source][i]
                    self.append(f'{source}_to_{target}', **dummy)

                # Proceed
                continue

            # Otherwise, use closest point as reference
            dists = dist_mat[source]
            closest_index = np.argmin(dists, axis=1)
            masks = [np.where(types[target] == s)[0] for s in range(self.num_classes)]
            for i in range(len(points[source])):
                point_dict = {**self.dummy_dict}
                point_dict['dist'] = dists[i, closest_index[i]]
                point_dict['shape'] = types[source][i]
                point_dict['closest_shape'] = types[target][closest_index[i]]
                if self.endpoints:
                    point_dict['end'] = ends[source][i]
                    point_dict['closest_end'] = ends[target][closest_index[i]]
                for s in range(self.num_classes):
                    if len(masks[s]) > 0:
                        point_dict[f'dist_{s}'] = np.min(dists[i, masks[s]])

                self.append(f'{source}_to_{target}', **point_dict)
