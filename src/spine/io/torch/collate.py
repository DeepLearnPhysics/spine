"""Contains implementations of data collation classes.

Collate classes are a middleware between parsers and datasets. They are given
to :class:`torch.utils.data.DataLoader` as the `collate_fn` argumement.
"""

import numpy as np

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.geo import Geometry

from .overlay import Overlayer

__all__ = ["CollateAll"]


class CollateAll:
    """General collate function for all data types coming from the parsers.

    Provide it with a list of dictionaries, each of which maps keys to one of:
    1. ParserTensor with (coord tensor, feature tensor, meta data) which get
       merged into a single tensor with rows [batch_id, *coords, *features]
    2. ParserTensor with simple feature tensor which gets merged into a single
       tensor with rows [batch_id, *features]
    3. ParserTensor with a list of indexes and offsets which gets merged into
       a single index with the appropriate offsets applied
    4. Scalars/list/objects which simply get put in a single list
    """

    name = "all"

    def __init__(
        self,
        data_types,
        split=False,
        target_id=0,
        detector=None,
        geometry_file=None,
        source=None,
        overlay=None,
        overlay_methods=None,
    ):
        """Initialize the collation parameters.

        Parameters
        ----------
        data_types : dict
            Dictionary of data types returned by the parsers
        split : bool, default False
            Whether to split the input by module ID (each module gets its
            own batch ID, multiplies the number of batches by `num_modules`)
        target_id : int, default 0
            If split is `True`, specifies where to relocate the points
        detector : str, optional
            Name of a recognized detector to the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        source : dict, optional
            Dictionary which maps keys to their corresponding sources. This can
            be used to split tensors without having to check the geometry
        overlay : dict, optional
            Image overlay configuration
        overlay_methods : dict
            Dictionary of overlay methods
        """
        # Store the data types of each parser output
        self.data_types = data_types

        # Initialize the geometry, if required
        self.split = split
        self.source = None
        if split:
            assert (detector is not None) or (
                geometry_file is not None
            ), "If splitting the input per module, must provide detector."
            assert (
                overlay is None
            ), "Cannot overlay and split at the same time, for now."

            self.target_id = target_id
            self.geo = Geometry(detector, geometry_file)
            self.num_modules = self.geo.tpc.num_modules
            self.source = source

        # Initialize the overlayer, if required
        self.overlayer = None
        if overlay is not None:
            self.overlayer = Overlayer(
                **overlay, data_types=data_types, methods=overlay_methods
            )

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
        """
        # Overlay data (modify batch), if needed
        if self.overlayer is not None:
            batch = self.overlayer(batch)

        # Loop over the data keys, merge all events in a batch
        data = {}
        for key, data_type in self.data_types.items():
            # Dispatch
            ref_data = batch[0][key]
            if data_type == "tensor":
                if ref_data.coords is not None and not ref_data.feats_only:
                    # Case where a coordinates tensor and a feature tensor
                    # are provided, along with the metadata information
                    data[key] = self.stack_coord_tensors(batch, key)

                elif ref_data.global_shift is not None:
                    # Case where an index and a count is provided per entry
                    data[key] = self.stack_index_tensors(batch, key)

                else:
                    # Case where there is a feature tensor provided per entry
                    data[key] = self.stack_feat_tensors(batch, key)

            else:
                # In all other cases, just make a list
                data[key] = [sample[key] for sample in batch]

        return data

    def stack_coord_tensors(self, batch, key):
        """Stack coordinate tensors together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Data product key

        Returns
        -------
        TensorBatch
            Batched coordinate tensor
        """
        # Dispatch
        batch_size = len(batch)
        if not self.split:
            # If not split, simply stack everything
            coords = np.vstack([sample[key].coords for sample in batch])
            features = np.vstack([sample[key].features for sample in batch])
            counts = [len(sample[key].coords) for sample in batch]
            batch_ids = np.repeat(np.arange(batch_size, dtype=coords.dtype), counts)

        else:
            # If split, must shift the voxel coordinates and create
            # one batch ID per [batch, volume] pair
            coords_v, features_v, batch_ids_v = [], [], []
            counts = np.empty(batch_size * self.num_modules, dtype=np.int64)
            for s, sample in enumerate(batch):
                # Identify which point belongs to which module
                coords = sample[key].coords
                features = sample[key].features
                meta = sample[key].meta
                coords_wrapped, module_indexes = self.geo.split(
                    coords.reshape(-1, 3), self.target_id, meta=meta
                )
                coords = coords_wrapped.reshape(-1, coords.shape[1])

                # If there are more than one point per row and they
                # are in separate volumes, the choice is arbitrary
                if coords.shape[1] > 3:
                    num_points = coords.shape[1] // 3
                    free = np.ones(len(coords), dtype=bool)
                    for m, module_index in enumerate(module_indexes):
                        mask = np.zeros(len(coords_wrapped), dtype=bool)
                        mask[module_index] = True
                        mask = mask.reshape(-1, num_points).any(axis=1)
                        module_indexes[m] = np.where(free & mask)[0]
                        free[module_indexes[m]] = False

                # Assign a different batch ID to each volume
                for m, module_index in enumerate(module_indexes):
                    coords_v.append(coords[module_index])
                    features_v.append(features[module_index])
                    idx = self.num_modules * s + m
                    batch_ids_v.append(
                        np.full(len(module_index), idx, dtype=coords.dtype)
                    )
                    counts[idx] = len(module_index)

            coords = np.vstack(coords_v)
            features = np.vstack(features_v)
            batch_ids = np.concatenate(batch_ids_v)

        # Stack the coordinates with the features
        tensor = np.hstack([batch_ids[:, None], coords, features])
        coord_cols = np.arange(1, 1 + coords.shape[1])

        return TensorBatch(
            tensor.astype(features.dtype),
            counts,
            has_batch_col=True,
            coord_cols=coord_cols,
        )

    def stack_index_tensors(self, batch, key):
        """Stack index tensors together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Data product key

        Returns
        -------
        Union[IndexBatch, EdgeIndexBatch]
            Batched index tensor
        """
        # Start by computing the necessary node ID offsets to apply
        total_counts = [sample[key].global_shift for sample in batch]
        offsets = np.zeros(len(total_counts), dtype=int)
        offsets[1:] = np.cumsum(total_counts)[:-1]

        # Stack the indexes, do not add a batch column
        index_list = []
        for i, sample in enumerate(batch):
            index_list.append(sample[key].features + offsets[i])
        index = np.concatenate(index_list, axis=1)
        counts = [sample[key].features.shape[-1] for sample in batch]

        if len(index.shape) == 1:
            return IndexBatch(index, counts, offsets)
        else:
            return EdgeIndexBatch(index, counts, offsets, directed=True)

    def stack_feat_tensors(self, batch, key):
        """Stack feature tensors together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Data product key

        Returns
        -------
        TensorBatch
            Batched feature tensor
        """
        # Fetch the source object, if it exists
        sources = None
        if self.split and self.source is not None and key in self.source:
            source_key = self.source[key]
            sources = [batch[i][source_key].features for i in range(len(batch))]

        # Dispatch
        if not self.split or sources is None:
            tensor = np.concatenate([sample[key].features for sample in batch])
            counts = [len(sample[key].features) for sample in batch]

        else:
            batch_size = len(batch)
            features_v = []
            counts = np.empty(batch_size * self.num_modules, dtype=np.int64)
            for s, sample in enumerate(batch):
                features = sample[key].features
                for m in range(self.num_modules):
                    module_index = np.where(sources[s][:, 0] == m)[0]
                    features_v.append(features[module_index])
                    idx = self.num_modules * s + m
                    counts[idx] = len(module_index)

            tensor = np.vstack(features_v)

        return TensorBatch(tensor, counts)
