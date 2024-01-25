import numpy as np

from mlreco.utils.geometry import Geometry


class CollateSparse:
    '''
    Collates data from each event in the batch into a single object
    '''
    def __init__(self, split = False, target_id = 0,
            detector = None, boundary = None):
        '''
        Parameters
        ----------
        split : bool, default False
            Whether to split the input by module ID
        target_id : int, default 0
            If a geometry i
        detector : str, optional
            Name of a recognized detector to the geometry from
        boundary : str, optional
            Path to a `.npy` boundary file to load the boundaries from
        '''
        # Initialize the geometry, if required
        self.split = split
        if split:
            assert (detector is not None) or (boundary is not None), \
                    'If splitting the input per module, must provide detector'

            self.target_id = target_id
            self.geo = Geometry(detector, boundary)

    def __call__(self, batch):
        '''
        Takes a list of parsed information, one per event in a batch, and
        collates them into a single object per batch

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
        '''
        # Loop over the data keys, merge all events in a batch
        num_batches = len(batch)
        data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], tuple) \
                    and isinstance(batch[0][key][0], np.ndarray) \
                    and len(batch[0][key][0].shape) == 2:
                # Case where a coordinates tensor and a features tensor
                # are provided, along with the metadata information
                assert len(batch[0][key]) == 3, 'Expecting a voxel tensor, ' \
                        'a feature tensor and a Meta object'

                # Stack the voxel and feature tensors in the batch
                if not self.split:
                    # If not split, simply stack everything
                    voxels    = np.vstack([sample[key][0] for sample in batch])
                    features  = np.vstack([sample[key][1] for sample in batch])
                    counts    = [len(sample[key][0]) for sample in batch]
                    batch_ids = np.repeat(np.arange(num_batches), counts)
                else:
                    # If split, must shift the voxel coordinates and create
                    # one batch ID per [batch, volume] pair
                    voxels_v, features_v, batch_ids_v = [], [], []
                    for s, sample in enumerate(batch):
                        voxels, features, meta = sample[key]
                        voxels, module_indexes = self.geo.split(voxels,
                                self.target_id, meta = meta)
                        for m, module_index in enumerate(module_indexes):
                            voxels_v.append(voxels[module_index])
                            features_v.append(features[module_index])
                            batch_ids_v.append(np.full(len(module_index),
                                self.geo.num_modules * s + m,
                                dtype = np.int32))

                    voxels = np.vstack(voxels_v)
                    features = np.vstack(features_v)
                    batch_ids = np.concatenate(batch_ids_v)

                # Concatenate voxel coordinates with their features
                data[key] = np.hstack([batch_ids[:, None], voxels, features])

            elif isinstance(batch[0][key],np.ndarray):
                # Case where the output of the parser is a single np.ndarray
                # Stack the tensors vertically, create and append batch column
                if batch[0][key].shape < 2:
                    tensor = np.concatenate([sample[key] for sample in batch])
                    tensor = tensor[:, None]
                else:
                    tensor = np.vstack([sample[key] for sample in batch])

                counts    = [len(sample[key]) for sample in batch]
                batch_ids = np.repeat(np.arange(num_batches), counts)
                data[key] = np.hstack([batch_ids[:, None], tensor]) 

            else:
                # In all other cases, just stick in a list of size batch_size
                data[key] = [sample[key] for sample in batch]

        return data
