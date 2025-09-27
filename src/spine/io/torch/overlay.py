"""Module with methods to overlay multiple events."""

from dataclasses import dataclass
from warnings import warn

import numpy as np

from ..core.parse.clean_data import clean_sparse_data
from ..core.parse.data import ParserObjectList, ParserTensor

__all__ = ["Overlayer"]


class Overlayer:
    """Generic class to produce data overlays.

    This class supports 3 image overlay modes:
    - 'constant' will produce overlays with a constant multiplicity;
    - 'uniform' will produce overlays with multiplicities, M_i, sampled from a
      uniform distribution such that, for a batch size B, \Sum_i M_i = B;
    - 'poisson' will produce overlays with multiplicities, M_i, sampled from a
      Poisson distribution of mean set by 'multiplicity'. For a batch size B,
      the multiplicities are set such that \Sum_i M_i = B.
    """

    # List of recognized overlay modes
    _modes = ("constant", "uniform", "poisson")

    def __init__(self, data_types, methods, multiplicity, mode="constant"):
        """Store the overlay parameters.

        Parameters
        ----------
        data_types : Dict[str, str]
            Types of data returned by the upstream parsers
        methods : Dict[str, str]
            Maps data products onto overlay methods
        multiplicity : int
            Number of images to stack in the overlay
        mode : str, default 'constant'
            Overlay mode (one of 'constant', 'uniform' or 'poisson')
        """
        # Check that the overlay mode is recognized
        assert mode in self._modes, (
            f"Overlay mode not recognized: {mode}. Must be one of " f"{self._modes}."
        )
        self.mode = mode

        # Check that multiplicity is sensible
        assert (
            multiplicity > 0
        ), "Overlay multiplicity should be a non-zero positive integer."
        self.multiplicity = multiplicity

        # Store the data types and methods
        self.data_types = data_types
        self.methods = methods

    def __call__(self, batch):
        """Given a batch of data, provides an overlay batching and modifies
        the data in place to avoid indexing conflicts.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.

        Returns
        -------
        List[Dict]
            Overlayed list of dictionaries of parsed information, one per overlay.
        """
        # Fetch the batch size, build an overlap map
        batch_size = len(batch)
        overlay_ids = self.get_assignments(batch_size)

        # Loop over the unique overlay indexes
        overlay_batch = []
        _, splits = np.unique(overlay_ids, return_index=True)
        indexes = np.split(np.arange(batch_size), splits[1:])
        for overlay_id, index in enumerate(indexes):
            # If there is only a single index in the overlay, nothing to do
            if len(index) < 2:
                overlay_batch.append(batch[index[0]])
                continue

            # Loop over the keys to overlay
            overlay = {}
            for key, data_type in self.data_types.items():
                # Dispatch and fill the overlay
                if data_type == "scalar":
                    # Check whether scalars can be harmonized
                    overlay[key] = self.merge_scalars(batch, key, index)

                elif data_type == "object":
                    # Check that objects are compatible when overlaying
                    overlay[key] = self.merge_objects(batch, key, index)

                elif data_type == "object_list":
                    # Offset object list index attributes if needed
                    overlay[key] = self.cat_objects(batch, key, index)

                elif data_type == "tensor":
                    # Stack tensors, offset index columns if needed
                    overlay[key] = self.stack_tensors(batch, key, index)

            # Add overlay to the batch
            overlay_batch.append(overlay)

        return overlay_batch

    def get_assignments(self, batch_size):
        """Given a data product count, produce batch assignments.

        Parameters
        ----------
        batch_size : int
            Number of entries in the batch

        Returns
        -------
        np.ndarray
            Overlay ID assignments
        """
        # Dispatch
        if self.mode == "constant":
            # Uniform multiplicity of overlays
            if batch_size % self.multiplicity != 0:
                warn(
                    f"The overlay multiplicity ({self.multiplicity}) is not a "
                    "divider of the batch size ({batch_size}). The overlay "
                    "multiplicity will not be uniform."
                )

            overlay_ids = np.arange(batch_size, dtype=int) // self.multiplicity

        elif self.mode in ["poisson", "uniform"]:
            # Sample from a Poisson distribution until it adds up to the batch size
            overlay_ids = np.empty(batch_size, dtype=int)
            idx, total = 0, 0
            while total < batch_size:
                # Sample distribution
                if self.mode == "poisson":
                    sample = np.random.poisson(self.multiplicity)
                else:
                    sample = np.random.randint(1, self.multiplicity + 1)

                # Assign overlay indices
                if sample > 0:
                    overlay_ids[total : total + sample] = idx
                    idx += 1
                    total += sample

        # Return
        return overlay_ids

    def merge_scalars(self, batch, key, index):
        """Merge scalars into one per overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Scalar data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        object
            Single scalar for the batch
        """
        scalars = np.array([batch[idx][key] for idx in index])
        if self.methods[key] in ["first", "match"]:
            # Make sure that all scalars match within the overlay, if needed
            if self.methods[key] == "match":
                if not np.all(scalars[1:] == scalars[0]):
                    raise ValueError(
                        f"The scalar values to overlay do not match for {key}."
                    )

            return scalars[0]

        elif self.methods[key] == "sum":
            # Sum the values within each overlay
            return np.sum(scalars)

        elif self.methods[key] == "cat":
            # Concatenate the scalars in a single array (type change)
            return scalars

        else:
            if self.methods[key] is None:
                raise ValueError(f"Scalar overlay method not specified for {key}.")

            raise ValueError(
                f"Scalar overlay method not recognized: {self.methods[key]}. "
                "Must be one of 'first', 'match' or 'sum'."
            )

    def merge_objects(self, batch, key, index):
        """Merge objects into one per overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Object data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        object
            Single object for the batch
        """
        objects = [batch[idx][key] for idx in index]
        if self.methods[key] in ["first", "match"]:
            # Make sure that all objects match within the overlay, if needed
            if self.methods[key] == "match":
                if not np.all([obj == objects[0] for obj in objects]):
                    raise ValueError(f"The objects to overlay do not match for {key}.")

            return objects[0]

        elif self.methods[key] == "cat":
            # Concatenate the objects in a single list (type change)
            return ParserObjectList(objects, default=objects[0])

        else:
            if self.methods[key] is None:
                raise ValueError(f"Object overlay method not specified for {key}.")

            raise ValueError(
                f"Object overlay method not recognized: {self.methods[key]}. "
                "Must be one of 'first' or 'match'."
            )

    def cat_objects(self, batch, key, index):
        """Concatenate object lists into one, offset index attributes if needed.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Object list data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        ObjList
            Concatenated obejct list
        """
        # If the objects in the lists contain indexes, must offset them
        ref_list = batch[index[0]][key]
        shifts = None
        if len(ref_list.default.index_attrs) > 0:
            shifts = ref_list.index_shifts
            for idx in index[1:]:
                # Shift indexes in the objects
                obj_list = batch[idx][key]
                for obj in obj_list:
                    obj.shift_indexes(shifts)

                # Increment shifts
                if not isinstance(shifts, dict):
                    shifts += obj_list.index_shifts
                else:
                    for attr in shifts:
                        shifts[attr] += obj_list.index_shifts[attr]

        # Concatenate and return
        obj_list = []
        for idx in index:
            obj_list.extend(batch[idx][key])

        return ParserObjectList(obj_list, ref_list.default, shifts)

    def stack_tensors(self, batch, key, index):
        """Stack tensors together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Tensor data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        ParserTensor
            Overlayed tensor
        """
        # Define a reference tensor
        ref_data = batch[index[0]][key]

        # Stack coordinates, if present
        coords = None
        if ref_data.coords is not None:
            # Check that the meta data matches between all images (it must)
            if not np.all([batch[idx][key].meta == ref_data.meta for idx in index]):
                raise ValueError("The metadata must match across all overlayed tensor.")
            coords = np.vstack([batch[idx][key].coords for idx in index])

        # If required, offset indexes in the feature tensor
        global_shift, index_shifts = None, None
        if ref_data.global_shift is not None:
            # Shift the whole feature tensor (index tensor)
            global_shift = ref_data.global_shift
            for idx in index[1:]:
                mask = batch[idx][key].features > -1
                batch[idx][key].features[mask] += global_shift
                global_shift += batch[idx][key].global_shift

        elif ref_data.feat_index_cols is not None:
            # Apply offsets to the relevant columns only (mixed features)
            index_shifts = ref_data.index_shifts.copy()
            for idx in index[1:]:
                for i, col in enumerate(ref_data.feat_index_cols):
                    mask = batch[idx][key].features[:, col] > -1
                    batch[idx][key].features[mask, col] += index_shifts[i]
                index_shifts += batch[idx][key].index_shifts

        # Stack features
        if ref_data.global_shift is None:
            features = np.vstack([batch[idx][key].features for idx in index])
        else:
            features = np.concatenate(
                [batch[idx][key].features for idx in index], axis=-1
            )

        # If requested, remove rows corresponding to duplicate coordinates
        if ref_data.remove_duplicates:
            # Check that we have coordinates to make the check
            assert (
                ref_data.coords is not None
            ), "Must provide coordinates to filter duplicates."

            # Filter out duplicates, aggregate features
            coords, features = clean_sparse_data(
                coords,
                features,
                sum_cols=ref_data.feat_sum_cols,
                prec_col=ref_data.feat_prec_col,
                precedence=ref_data.precedence,
            )

        # Returns
        return ParserTensor(
            coords=coords,
            features=features,
            meta=ref_data.meta,
            global_shift=global_shift,
            index_shifts=index_shifts,
            index_cols=ref_data.index_cols,
            sum_cols=ref_data.sum_cols,
            prec_col=ref_data.prec_col,
            precedence=ref_data.precedence,
            feats_only=ref_data.feats_only,
        )
