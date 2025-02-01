"""Contains implementations of data collation classes.

Collate classes are a middleware between parsers and datasets. They are given
to :class:`torch.utils.data.DataLoader` as the `collate_fn` argumement.
"""

import numpy as np

from spine.data import TensorBatch, IndexBatch, EdgeIndexBatch

from spine.utils.geo import Geometry

__all__ = ['CollateAll']


class CollateAll:
    """General collate function for all data types coming from the parsers.

    Provide it with a list of dictionaries, each of which maps keys to one of:
    1. Tuple of (voxel tensor, feature tensor, metadata) which get merged
       into a single tensor with rows [batch_id, *coords, *features]
    2. Simple feature tensor which gets merged into a single tensor with
       rows [batch_id, *features]
    3. Scalars/list/objects which simply get put in a single list
    """
    name = 'all'

    def __init__(self, split=False, target_id=0, detector=None,
                 geometry_file=None, overlay=None, source=None):
        """Initialize the collation parameters.

        Parameters
        ----------
        split : bool, default False
            Whether to split the input by module ID (each module gets its
            own batch ID, multiplies the number of batches by `num_modules`)
        target_id : int, default 0
            If split is `True`, specifies where to relocate the points
        detector : str, optional
            Name of a recognized detector to the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        overlay : dict, optional
            Image overlay configuration
        source : dict, optional
            Dictionary which maps keys to their corresponding sources. This can
            be used to split tensors without having to check the geometry
        """
        # Initialize the geometry, if required
        self.split = split
        self.source = None
        if split:
            assert (detector is not None) or (geoemtry_file is not None), (
                    "If splitting the input per module, must provide detector")

            self.target_id = target_id
            self.geo = Geometry(detector, geometry_file)
            self.source = source

        if overlay is not None:
            self.process_overlay_config(**overlay)

    def process_overlay_config(self, mode='const', size=2):
        """Process the image overlay configuration

        Parameters
        ----------
        mode : str, default 'const'
            Method used to from overlay indexes ('const' or 'poisson')
        size : int, default 2
            Number of images to merge to produce each new image
        """
        # TODO: start with merge_batch under self.utils.gnn.data
        # Initialize the batch merger here...
        raise NotImplementedError("Work in progress...")

    def __call__(self, batch):
        """Takes a list of parsed information, one per event in a batch, and
        collates them into a single object per entry in the batch.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.

        Returns
        -------
        Dict
            Dictionary that matches one data key to one batch-worth of data

        Notes
        -----
        Assumptions:
        - The input batch is a tuple of length >= 1. Length 0 tuple
          will fail (IndexError)
        - The dictionaries in the input batch tuple are assumed to have
          an identical list of keys
        """
        # Loop over the data keys, merge all events in a batch
        batch_size = len(batch)
        data = {}
        for key in batch[0].keys():
            # Fetch the reference object
            ref_obj = batch[0][key]

            # Fetch the source object, if it exists
            sources = None
            if self.source is not None and key in self.source:
                source_key = self.source[key]
                sources = [batch[i][source_key] for i in range(len(batch))]

            # Dispatch
            if isinstance(ref_obj, tuple) and len(ref_obj) == 3:
                # Case where a coordinates tensor and a feature tensor
                # are provided, along with the metadata information
                if not self.split:
                    # If not split, simply stack everything
                    voxels    = np.vstack([sample[key][0] for sample in batch])
                    features  = np.vstack([sample[key][1] for sample in batch])
                    counts    = [len(sample[key][0]) for sample in batch]
                    batch_ids = np.repeat(
                            np.arange(batch_size, dtype=voxels.dtype), counts)

                else:
                    # If split, must shift the voxel coordinates and create
                    # one batch ID per [batch, volume] pair
                    voxels_v, features_v, batch_ids_v = [], [], []
                    counts = np.empty(
                            batch_size*self.geo.tpc.num_modules, dtype=np.int64)
                    for s, sample in enumerate(batch):
                        # Identify which point belongs to which module
                        voxels, features, meta = sample[key]
                        voxels_wrapped, module_indexes = self.geo.split(
                                voxels.reshape(-1, 3),
                                self.target_id, meta=meta)
                        voxels = voxels_wrapped.reshape(-1, voxels.shape[1])

                        # If there are more than one point per row and they
                        # are in separate volumes, the choice is arbitrary
                        if voxels.shape[1] > 3:
                            num_points = voxels.shape[1]//3
                            free = np.ones(len(voxels), dtype=bool)
                            for m, module_index in enumerate(module_indexes):
                                mask = np.zeros(len(voxels_wrapped), dtype=bool)
                                mask[module_index] = True
                                mask = mask.reshape(-1, num_points).any(axis=1)
                                module_indexes[m] = np.where(free & mask)[0]
                                free[module_indexes[m]] = False

                        # Assign a different batch ID to each volume
                        for m, module_index in enumerate(module_indexes):
                            voxels_v.append(voxels[module_index])
                            features_v.append(features[module_index])
                            idx = self.geo.tpc.num_modules * s + m
                            batch_ids_v.append(np.full(len(module_index),
                                               idx, dtype=voxels.dtype))
                            counts[idx] = len(module_index)

                    voxels = np.vstack(voxels_v)
                    features = np.vstack(features_v)
                    batch_ids = np.concatenate(batch_ids_v)

                # Stack the coordinates with the features
                tensor = np.hstack([batch_ids[:, None], voxels, features])
                coord_cols = np.arange(1, 1+voxels.shape[1])
                data[key] = TensorBatch(
                        tensor.astype(features.dtype), counts,
                        has_batch_col=True, coord_cols=coord_cols)

            elif isinstance(ref_obj, tuple) and len(ref_obj) == 2:
                # Case where an index and an offset is provided per entry.
                # Stack the indexes, do not add a batch column
                tensor  = np.concatenate(
                        [sample[key][0] for sample in batch], axis=1)
                counts  = [sample[key][0].shape[-1] for sample in batch]
                offsets = [sample[key][1] for sample in batch]

                if len(tensor.shape) == 1:
                    data[key] = IndexBatch(tensor, counts, offsets)
                else:
                    data[key] = EdgeIndexBatch(
                            tensor, counts, offsets, directed=True)

            elif isinstance(ref_obj, np.ndarray):
                # Case where there is a simple feature tensor returned per
                # entry. Stack the features, do not add a batch column
                if not self.split or sources is None:
                    tensor = np.concatenate([sample[key] for sample in batch])
                    counts = [len(sample[key]) for sample in batch]

                else:
                    features_v = []
                    counts = np.empty(
                            batch_size*self.geo.tpc.num_modules, dtype=np.int64)
                    for s, sample in enumerate(batch):
                        features = sample[key]
                        for m in range(self.geo.tpc.num_modules):
                            module_index = np.where(sources[s][:, 0] == m)[0]
                            features_v.append(features[module_index])
                            idx = self.geo.tpc.num_modules * s + m
                            counts[idx] = len(module_index)

                    tensor = np.vstack(features_v)

                data[key] = TensorBatch(tensor, counts)

            else:
                # In all other cases, just make a list of size batch_size
                data[key] = [sample[key] for sample in batch]

        return data
