"""Analysis script used to evaluate the clustering accuracy."""

from warnings import warn

import numpy as np
from scipy.special import softmax

import spine.utils.metrics
from spine.utils.enums import enum_factory
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

    # Name of the analysis script (as specified in the configuration)
    name = 'cluster_eval'

    # Label column to use for each clustering label_col
    _label_cols = (
            ('fragment', CLUST_COL),
            ('particle', GROUP_COL),
            ('interaction', INTER_COL)
    )

    def __init__(self, obj_type=None, use_objects=False, per_object=True,
                 per_shape=True, metrics=('pur', 'eff', 'ari'),
                 label_key='clust_label_adapt', label_col=None, time_window=None, **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        obj_type : Union[str, List[str]], optional
            Name or list of names of the object types to process
        use_objects : bool, default False
            If `True`, rebuild the clustering assignments for truth and reco
            from the set of truth and reco particles
        per_object : bool, default True
            Evaluate the clustering accuracy for each object type (not relevant
            if running GrapPA standalone)
        per_shape : bool, default True
            Evaluate the clustering accuracy for each object shape (not
            relevant in the case of interactions)
        metrics : Tuple[str], default ('pur', 'eff', 'ari')
            List of clustering metrics to evaluate
        label_key : str, default 'clust_label_adapt'
            Name of the tensor which contains the cluster labels, when
            using the raw reconstruction output
        label_col : str, optional
            Column name in the label tensor specifying the aggregation label_col
        time_window : List[float], optional
            Time window within which to include objects. If provided, must be a list
            of two values [t_min, t_max]. Objects outside this window will be excluded.
            For reconstructed objects, filtering is based on their matched truth objects' times.
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Check parameters
        assert obj_type is not None or not per_object, (
                "If evaluating clustering metrics per object, provide a list "
                "of object types to evaluate the clustering for.")
        assert per_object or label_col is not None, (
                "If evaluating clustering standalone (not per object), must "
                "provide the name of the target clustering label column.")
        assert per_object or not use_objects, (
                "If evaluating clustering standalone (not per object), cannot "
                "use objects to evaluate it.")
        assert time_window is None or len(time_window) == 2, (
                "Time window must be specified as a list of two values [t_min, t_max].")

        # Initialize the parent class
        super().__init__(obj_type, 'both', **kwargs)

        # Store the time window
        self.time_window = time_window
        print(f'Time window: {self.time_window}')

        # If the clustering is not done per object, fix target
        if not per_object:
            self.obj_type = [label_col]

        # Store the basic parameters
        self.use_objects = use_objects
        self.per_object = per_object
        self.per_shape = per_shape
        self.label_key = label_key

        # Parse the label_col column, if necessary
        self.label_col = None
        if label_col is not None:
            self.label_col = enum_factory('cluster', label_col)

        # Convert metric strings to functions
        self.metrics = {m: getattr(spine.utils.metrics, m) for m in metrics}

        # If objects are not used, remove them from the required keys
        keys = self.keys
        if not use_objects:
            for key in self.obj_keys:
                del keys[key]

        # List other necessary data products
        if self.per_object:
            if not self.use_objects:
                # Store the labels and the clusters output by the reco chain
                keys[label_key] = True
                for obj in self.obj_type:
                    keys[f'{obj}_clusts'] = True
                    if obj != 'interaction':
                        keys[f'{obj}_shapes'] = True
                    if self.time_window is not None:
                        # Need truth objects for time filtering
                        keys[f'truth_{obj}s'] = True

            else:
                keys['points'] = True

        else:
            keys[label_key] = True
            keys['clusts'] = True
            keys['group_pred'] = True

        self.keys = keys

        # Initialize the output
        for obj in self.obj_type:
            self.initialize_writer(obj)

    @property
    def label_cols(self):
        """Dictionary of (key, column_id) pairs which determine which column
        in the label tensor corresponds to a specific clustering target.

        Returns
        -------
        Dict[str, int]
            Dictionary of (key, column_id) mapping from name to label column
        """
        return dict(self._label_cols)

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
                label_col = self.label_col or self.label_cols[obj_type]
                num_points = len(data[self.label_key])
                labels = data[self.label_key][:, label_col]
                if obj_type != 'interaction':
                    shapes = data[self.label_key][:, SHAPE_COL]
                
                # Get truth objects for time filtering
                truth_objects = data[f'truth_{obj_type}s']
                truth_times = np.array([obj.t for obj in truth_objects])
                
                # Create a mapping from truth object index to label
                truth_to_label = {}
                for i, obj in enumerate(truth_objects):
                    #print(f'{obj.index_adapt} {len(labels)}')
                    if (obj.index_adapt < len(labels)).all():
                        truth_to_label[i] = labels[obj.index_adapt]
                
                # Filter truth objects by time window if specified
                if self.time_window is not None:
                    valid_truth_mask = (truth_times >= self.time_window[0]) & (truth_times <= self.time_window[1])
                    valid_truth_indices = np.where(valid_truth_mask)[0]
                    
                    # Create a mask for valid labels
                    valid_labels = np.zeros_like(labels, dtype=bool)
                    for idx in valid_truth_indices:
                        if idx in truth_to_label:
                            valid_labels[truth_objects[idx].index_adapt] = True
                    
                    num_truth = len(np.unique(labels[labels > -1]))
                    # Set invalid labels to -1
                    labels[~valid_labels] = -1
                else:
                    valid_truth_indices = np.arange(len(truth_objects))
                    num_truth = len(np.unique(labels[labels > -1]))
                

            else:
                # Rebuild the labels
                num_points = len(data['points'])
                labels = -np.ones(num_points)
                
                # First pass: collect truth objects and their times
                truth_objects = data[f'truth_{obj_type}s']
                truth_times = np.array([obj.t for obj in truth_objects])
                
                # Filter truth objects by time window if specified
                if self.time_window is not None:
                    valid_truth_mask = (truth_times >= self.time_window[0]) & (truth_times <= self.time_window[1])
                    valid_truth_indices = np.where(valid_truth_mask)[0]
                else:
                    valid_truth_indices = np.arange(len(truth_objects))
                
                # Build labels only for valid truth objects
                for i, idx in enumerate(valid_truth_indices):
                    obj = truth_objects[idx]
                    labels[obj.index_adapt] = i
                
                num_truth = len(truth_objects)

            # Build the cluster predictions for this object type
            preds = -np.ones(num_points)
            if self.per_object:
                shapes = -np.full(num_points, LOWES_SHP)
                if not self.use_objects:
                    # Use clusters directly from the full chain output
                    num_reco = len(data[f'{obj_type}_clusts'])
                    for i, index in enumerate(data[f'{obj_type}_clusts']):
                        preds[index] = i
                        if obj_type != 'interaction':
                            shapes[index] = data[f'{obj_type}_shapes'][i]

                else:
                    # Use clusters from the object indexes
                    reco_objects = data[f'reco_{obj_type}s']
                    truth_objects = data[f'truth_{obj_type}s']
                    
                    # Filter reconstructed objects based on their matched truth objects' times
                    valid_reco_indices = []
                    for i, reco_obj in enumerate(reco_objects):
                        # Skip if no matches
                        if not len(reco_obj.match_ids):
                            continue
                            
                        # Get the matched truth object
                        truth_idx = reco_obj.match_ids[0]
                        if truth_idx >= len(truth_objects):
                            continue
                            
                        truth_obj = truth_objects[truth_idx]
                        
                        # Apply time window filter if specified
                        if self.time_window is not None:
                            if truth_obj.t < self.time_window[0] or truth_obj.t > self.time_window[1]:
                                continue
                                
                        valid_reco_indices.append(i)
                    
                    # Build predictions only for valid reconstructed objects
                    for i, reco_idx in enumerate(valid_reco_indices):
                        reco_obj = reco_objects[reco_idx]
                        preds[reco_obj.index] = i
                        if obj_type != 'interaction':
                            shapes[reco_obj.index] = reco_obj.shape
                            
                    num_reco = len(valid_reco_indices)

            else:
                num_reco = len(data['clusts'])
                for i, index in enumerate(data['clusts']):
                    preds[index] = data['group_pred'][i]

            # Evaluate clustering metrics
            row_dict = {'num_points': num_points, 'num_truth': num_truth,
                        'num_reco': num_reco}
            for metric, func in self.metrics.items():
                valid_index = np.where((preds > -1) & (labels > -1))[0]
                row_dict[metric] = func(labels[valid_index], preds[valid_index])
                if self.per_shape and obj_type != 'interaction':
                    for shape in range(LOWES_SHP):
                        shape_index = np.where(
                                (shapes == shape) & (labels > -1))[0]
                        row_dict[f'{metric}_{shape}'] = func(
                                labels[shape_index], preds[shape_index])

            self.append(obj_type, **row_dict)
