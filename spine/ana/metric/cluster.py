"""Analysis script used to evaluate the clustering accuracy."""

from warnings import warn

import numpy as np
from scipy.special import softmax

import spine.utils.metrics
from spine.utils.globals import (
        SHAPE_COL, LOWES_SHP, CLUST_COL, GROUP_COL, INTER_COL)

from spine.ana.base import AnaBase

__all__ = ['ClusterAna']


class ClusterAna(AnaBase):
    """Class which computes and stores the necessary data to evaluate
    clustering metrics at different aggregation stages:
    - fragments
    - particles
    - interactions
    """
    name = 'cluster_eval'

    # Label column to use for each clustering target
    _label_cols = {
            'fragment': CLUST_COL, 'particle': GROUP_COL,
            'interaction': INTER_COL
    }

    def __init__(self, obj_type, use_objects=False, per_shape=True,
                 metrics=('pur', 'eff', 'ari'), label_key='clust_label_adapt',
                 **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        obj_type : Union[str, List[str]]
            Name or list of names of the object types to process
        use_objects : bool, default False
            If `True`, rebuild the clustering labels for truth and reco
            from the set of truth and reco particles
        per_shape : bool, default True
            Evaluate the clustering accuracy for each object shape (not
            relevant in the case of interactions)
        metrics : Tuple[str], default ('pur', 'eff', 'ari')
            List of clustering metrics to evaluate
        label_key : str, default 'clust_label_adapt'
            Name of the tensor which contains the cluster labels, when
            using the raw reconstruction output
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(obj_type, 'both', **kwargs)
        if not use_objects:
            for key in self.obj_keys:
                del self.keys[key]
        
        # Store the basic parameters
        self.use_objects = use_objects
        self.per_shape = per_shape
        self.label_key = label_key

        # Convert metric strings to functions
        self.metrics = {m: getattr(spine.utils.metrics, m) for m in metrics}

        # List the necessary data products
        if not self.use_objects:
            # Store the labels and the clusters output by the reco chain
            self.keys[label_key] = True
            for obj in self.obj_type:
                self.keys[f'{obj}_clusts'] = True

        else:
            self.keys['points'] = True

        # Initialize the output
        for obj in self.obj_type:
            self.initialize_writer(obj)

    def process(self, data):
        """Store the clustering metrics for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the different object types
        for obj_type in self.obj_type:
            # Build the cluster labels for this object type
            if not self.use_objects:
                # Fetch the right label column
                num_points = len(data[self.label_key])
                labels = data[self.label_key][:, self._label_cols[obj_type]]
                shapes = data[self.label_key][:, SHAPE_COL]
            else:
                # Rebuild the labels
                num_points = len(data['points'])
                labels = -np.ones(num_points)
                for i, obj in enumerate(data[f'truth_{obj_type}s']):
                    labels[obj.index] = i

            # Build the cluster predictions for this object type
            preds = -np.ones(num_points)
            if not self.use_objects:
                # Use clusters directly from the full chain output
                for i, index in enumerate(data[f'{obj_type}_clusts']):
                    preds[index] = i
            else:
                # Use clusters from the object indexes
                shapes = -np.full(num_points, LOWES_SHP)
                for i, obj in enumerate(data[f'reco_{obj_type}s']):
                    preds[obj.index] = i
                    if obj_type != 'interaction':
                        shapes[obj.index] = obj.shape

            # Evaluate clustering metrics
            row_dict = {}
            for metric, func in self.metrics.items():
                valid_index = np.where(preds > -1)[0]
                row_dict[metric] = func(labels[valid_index], preds[valid_index])
                if self.per_shape and obj_type != 'interaction':
                    for shape in range(LOWES_SHP):
                        shape_index = np.where(shapes == shape)[0]
                        row_dict[f'{metric}_{shape}'] = func(
                                labels[shape_index], preds[shape_index])

            self.append(obj_type, **row_dict)
