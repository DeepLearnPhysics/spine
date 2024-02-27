"""Module with the classes/functions needed to unwrap batched data."""

import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from .globals import BATCH_COL, COORD_COLS
from .data_structures import TensorBatch
from .geometry import Geometry


class Unwrapper:
    """Unwraps batched data to its constituent entries.

    Class used to break down the batched input and output dictionaries into
    individual events. When passed through the model, the input is concatenated
    into single tensors/arrays for faster processing; this class breaks the
    output down event-wise to be human-readable.
    """
    def __init__(self, batch_size, rules={}, geometry=None,
                 remove_batch_col=False):
        """Translate rule arrays and boundaries into instructions.

        Parameters
        ----------
        batch_size : int
             Number of events in the batch
        rules : dict
             Dictionary which contains a set of unwrapping rules for each
             output key of the reconstruction chain. If there is no rule
             associated with a key, the list is concatenated.
        geometry : Geometry
             Detector geometry (needed if the input was split)
        remove_batch_col : bool
             Remove column which specifies batch ID from the unwrapped tensors
        """
        self.batch_size = batch_size
        self.remove_batch_col = remove_batch_col
        self.geo = geometry
        self.num_volumes = self.geo.num_modules if self.geo else 1
        self.rules = self._process_rules(rules)

    def __call__(self, data_dict, result_dict):
        """Main unwrapping function.
        
        Loops over the data and result keys and applies the unwrapping rules.
        Returns the unwrapped versions of the two dictionaries.

        Parameters
        ----------
        data_dict : dict
            Dictionary of input data (key, batched input)
        result_dict : dict
            Dictionary of output of trainval.forward (key, batched output)

        Returns
        -------
        dict
            Dictionary of unwrapped input data (key, [batch_size])
        dict
            Dictionary of unwrapped output data (key, [batch_size])
        """
        data_unwrapped, result_unwrapped = {}, {}
        for key, value in data_dict.items():
            data_unwrapped[key] = self._unwrap(key, value)
        for key, value in result_dict.items():
            result_unwrapped[key] = self._unwrap(key, value)

        return data_unwrapped, result_unwrapped

    @dataclass
    class Rule:
        """Simple dataclass which stores the relevant unwrapping rule
        attributes for a speicific data product human-readable names.

        Attributes
        ----------
        method : str
            Unwrapping scheme
        ref_key : str, optional
            Key of the data product that supplies the batch mapping
        done : bool, default False
            True if the unwrapping is done by the model internally
        translate : bool, default False
            True if the coordinates of the tensor need to be shifted
            from voxel indexes to detector coordinates
        default : object
            Default object to base the unwrapping on if the data product
            for a specific event is empty.
        """
        method    : str
        ref_key   : str = None
        done      : bool = False
        translate : bool = False
        default   : object = None

    def _process_rules(self, rules):
        """Check that the ruls provided are valid.

        Parameters
        ----------
        rules : dict
             Dictionary which contains a set of unwrapping rules for each
             output key of the reconstruction chain. If there is no rule
             associated with a key, the list is concatenated.
        """
        valid_methods = ['scalar', 'list', 'tensor', 'tensor_list',
                         'edge_tensor', 'index_tensor', 'index_list']
        for key, rule in rules.items():
            if not rules[key].ref_key:
                rules[key].ref_key = key

            assert rules[key].method in valid_methods, (
                    f"Unwrapping method {rules[key].method} "
                    f"for {key} not recognized. Should be one of "
                    f"{valid_methods}")

        return rules

    def _unwrap(self, key, data):
        """Routes set of data to the appropriate unwrapping scheme.

        Parameters
        ----------
        key : str
            Name of the data product to unwrap
        data : list
            Data product
        """
        # Check that unwrapping rules exist for this key
        assert key in self.rules, f"Must provide unwrapping rule for {key}"
        method = self.rules[key].method

        # Dispatch to the correct unwrapping scheme
        if method == 'scalar':
            # If the data is a scalar, return the same number for each entry
            return np.full(self.batch_size, data)

        elif method == 'list':
            # If the data is a list, check that it is the right length, return
            assert len(data) == self.batch_size, (
                    f"The `{key}` list is not the same length as the batch "
                    "size. Got {len(data)} != {self.batch_size}")
            if len(data) and np.isscalar(data[0]):
                return np.asarray(data)
            else:
                return data

        elif method == 'tensor':
            # If the data is a tensor, split it between its consistuents
            return self._unwrap_tensor(key, data)

        elif method == 'tensor_list':
            # If the data is a tensor list, split each between its constituents
            data_split = [self._unwrap_tensor(key, t) for t in data]
            tensor_lists = []
            for b in range(self.batch_size):
                tensor_lists.append([l[b] for l in data_split])

        elif method == 'edge_tensor':
            # If the data is an edge tensor, split and offset
            # TODO: must fix
            ref_edge, ref_node = ref_key
            masks = self.masks[ref_edge]
            offsets = self.offsets[ref_node]
            tensors = []
            for v in range(self.num_volumes):
                idx = b * self.num_volumes + v
                if not self.rules[key].done:
                    tensor = data[masks[idx]]
                    offset = (key == ref_edge) * offsets[idx]
                else:
                    tensor = data[idx]
                    offset = (key == ref_edge) * (offsets[idx] \
                            - offsets[b * self.num_volumes])
                tensors.append(tensor + offset)
            unwrapped.append(np.concatenate(tensors))

        elif method == 'index_tensor':
            # If the data is an index tensor, split and offset
            # TODO: must fix
            masks = self.masks[ref_key]
            offsets = self.offsets[ref_key]
            tensors = []
            for v in range(self.num_volumes):
                idx = b * self.num_volumes + v
                if not self.rules[key].done:
                    tensors.append(data[masks[idx]] - offsets[idx])
                else:
                    offset = offsets[idx] \
                            - offsets[b * self.num_volumes]
                    tensors.append(data[idx] + offset)
            unwrapped.append(np.concatenate(tensors))

        elif method == 'index_list':
            # If the data is an index list, split and offset
            # TODO: must fix
            ref_tensor, ref_index = ref_key
            offsets = self.offsets[ref_tensor]
            masks = self.masks[ref_index]
            index_list = []
            for v in range(self.num_volumes):
                idx = b * self.num_volumes + v
                if not self.rules[key].done:
                    for i in masks[idx]:
                        index_list.append(data[i] - offsets[idx])
                else:
                    offset = offsets[idx] \
                            - offsets[b * self.num_volumes]
                    for index in data[idx]:
                        index_list.append(index + offset)

            index_list_nb    = np.empty(len(index_list), dtype=object)
            index_list_nb[:] = index_list
            unwrapped.append(index_list_nb)

        else:
            raise ValueError(
                    f"Unwrapping method `{method}` not recognized for {key}")

    def _unwrap_tensor(self, key, data):
        """Unwrap a tensor into its consitituents.

        Parameters
        ----------
        key : str
            Name of the tensor product to unwrap
        data : TensorBatch
            Tensor batch product
        """
        eff_batch_size = self.batch_size * self.num_volumes
        assert data.batch_size == eff_batch_size, (
                "The `{key}` tensor batch is not the same length as the "
                "batch size. Got {data.batch_size} != {eff_batch_size}")

        tensors = []
        for b in range(self.batch_size):
            for v in range(self.num_volumes):
                idx = b * self.num_volumes + v
                tensor = data[idx]
                if v > 0:
                    tensor[:, COORDS_COLS] = self.geo.translate(
                            tensor[:, COORD_COLS], 0, v)
                if self.remove_batch_col:
                    tensor = np.hstack(
                            [tensor[:, :BATCH_COL], tensor[:, BATCH_COL:]])
                tensors.append(tensor)

        return tensors


def prefix_unwrapper_rules(rules, prefix):
    """Modifies the default rules of a module to account for
    a prefix being added to its standard set of output names.

    Parameters
    ----------
    rules : dict
        Dictionary which contains a set of unwrapping rules for each
        output key of a given module in the reconstruction chain.
    prefix : str
        Prefix to add in front of all output names

    Returns
    -------
    dict
        Dictionary of rules containing the appropriate names
    """
    prules = {}
    for key, value in rules.items():
        pkey = f'{prefix}_{key}'
        prules[pkey] = deepcopy(rules[key])
        if len(value) > 1:
            if isinstance(value[1], str):
                prules[pkey][1] = f'{prefix}_{value[1]}'
            else:
                for i in range(len(value[1])):
                    prules[pkey][1][i] = f'{prefix}_{value[1][i]}'

    return prules
