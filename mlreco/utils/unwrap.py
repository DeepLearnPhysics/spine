import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from .globals import *
from .volumes import VolumeBoundaries


class Unwrapper:
    '''
    Class used to break down the batched input and output dictionaries into
    individual events. When passed through the model, the input is concatenated
    into single tensors/arrays for faster processing; this class breaks the
    output down event-wise to be human-readable.
    '''
    def __init__(self, batch_size, rules = {}, boundaries = None, 
            remove_batch_col = False):
        '''
        Translate rule arrays and boundaries into instructions.

        Parameters
        ----------
        batch_size : int
             Number of events in the batch
        rules : dict
             Dictionary which contains a set of unwrapping rules for each
             output key of the reconstruction chain. If there is no rule
             associated with a key, the list is concatenated.
        boundaries : list
             List of detector volume boundaries
        remove_batch_col : bool
             Remove column which specifies batch ID from the unwrapped tensors
        '''
        self.batch_size = batch_size
        self.remove_batch_col = remove_batch_col
        self.merger = VolumeBoundaries(boundaries) if boundaries else None
        self.num_volumes = self.merger.num_volumes() if self.merger else 1
        self.rules = self._parse_rules(rules)

    def __call__(self, data_dict, result_dict):
        '''
        Main unwrapping function. Loops over the data and result keys
        and applies the unwrapping rules. Returns the unwrapped versions
        of the two dictionaries

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
        '''
        self._build_batch_masks(data_dict, result_dict)
        data_unwrapped, result_unwrapped = {}, {}
        for key, value in data_dict.items():
            data_unwrapped[key] = self._unwrap(key, value)
        for key, value in result_dict.items():
            result_unwrapped[key] = self._unwrap(key, value)

        return data_unwrapped, result_unwrapped

    @dataclass
    class Rule:
        '''
        Simple dataclass which stores the relevant
        unwrapping rule attributes for a speicific
        data product human-readable names.

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
        '''
        method    : str
        ref_key   : str = None
        done      : bool = False
        translate : bool = False
        default   : object = None


    def _parse_rules(self, rules):
        '''
        Translate rule arrays into Rule objects. Do the
        necessary checks to ensure rule sanity.

        Parameters
        ----------
        rules : dict
             Dictionary which contains a set of unwrapping rules for each
             output key of the reconstruction chain. If there is no rule
             associated with a key, the list is concatenated.
        '''
        valid_methods = [None, 'scalar', 'list', 'tensor',
                'tensor_list', 'edge_tensor', 'index_tensor', 'index_list']
        parsed_rules = {}
        for key, rule in rules.items():
            parsed_rules[key] = self.Rule(*rule)
            if not parsed_rules[key].ref_key:
                parsed_rules[key].ref_key = key

            assert parsed_rules[key].method in valid_methods, 'Unwrapping ' \
                    f'method {parsed_rules[key].method} for {key} not valid'

        return parsed_rules

    def _build_batch_masks(self, data_dict, result_dict):
        '''
        For all the returned data objects that require a batch mask:
        build it and store it. Also store the index offsets within that
        batch, wherever necessary to unwrap.

        Parameters
        ----------
        data_dict : dict
            Dictionary of input data (key, batched input)
        result_dict : dict
            Dictionary of output of trainval.forward (key, batched output)
        '''
        comb_dict = dict(data_dict, **result_dict)
        self.masks, self.offsets = {}, {}
        for key in comb_dict.keys():
            # Skip outputs with no rule
            if key not in self.rules:
                continue

            # For tensors and tensor list, build one mask per reference tensor
            if not self.rules[key].done \
                    and self.rules[key].method in ['tensor', 'tensor_list']:
                ref_key = self.rules[key].ref_key
                if ref_key not in self.masks:
                    assert ref_key in comb_dict, f'Must provide reference ' \
                            f'tensor ({ref_key}) to unwrap {key}'
                    assert self.rules[key].method \
                            == self.rules[ref_key].method, 'Reference ' \
                            f'({ref_key}) must be of same type as {key}'
                    if self.rules[key].method == 'tensor':
                        self.masks[ref_key] = \
                                self._batch_masks(comb_dict[ref_key])
                    elif self.rules[key].method == 'tensor_list':
                        self.masks[ref_key] = [self._batch_masks(v) \
                                for v in comb_dict[ref_key]]

            # For edge tensors, build one mask from each tensor (must
            # figure out batch IDs of edges)
            elif self.rules[key].method == 'edge_tensor':
                assert len(self.rules[key].ref_key) == 2, 'Must provide a ' \
                        'reference to the edge_index and the node batch ids'
                for ref_key in self.rules[key].ref_key:
                    assert ref_key in comb_dict, 'Must provide reference ' \
                            f'tensor ({ref_key}) to unwrap {key}'
                ref_edge, ref_node = self.rules[key].ref_key
                edge_index, batch_ids = comb_dict[ref_edge], comb_dict[ref_node]
                if not self.rules[key].done and ref_edge not in self.masks:
                    self.masks[ref_edge] = \
                            self._batch_masks(batch_ids[edge_index[:,0]])
                if ref_node not in self.offsets:
                    self.offsets[ref_node] = self._batch_offsets(batch_ids)

            # For an index tensor, only need to record the batch offsets
            # within the wrapped tensor
            elif self.rules[key].method == 'index_tensor':
                ref_key = self.rules[key].ref_key
                assert ref_key in comb_dict, f'Must provide reference ' \
                        f'tensor ({ref_key}) to unwrap {key}'
                if not self.rules[key].done and ref_key not in self.masks:
                    self.masks[ref_key] = self._batch_masks(comb_dict[ref_key])
                if ref_key not in self.offsets:
                    self.offsets[ref_key] = \
                            self._batch_offsets(comb_dict[ref_key])

            # For lists of tensor indices, only need to record the offsets
            # within the wrapped tensor
            elif self.rules[key].method == 'index_list':
                assert len(self.rules[key].ref_key) == 2, 'Must provide a ' \
                        'reference to indexed tensor and the index batch ids'
                for ref_key in self.rules[key].ref_key:
                    assert ref_key in comb_dict, 'Must provide reference ' \
                            f'tensor ({ref_key}) to unwrap {key}'
                ref_tensor, ref_index = self.rules[key].ref_key
                if not self.rules[key].done and ref_index not in self.masks:
                    self.masks[ref_index] = \
                            self._batch_masks(comb_dict[ref_index])
                if ref_tensor not in self.offsets:
                    self.offsets[ref_tensor] = \
                            self._batch_offsets(comb_dict[ref_tensor])

    def _batch_masks(self, tensor):
        '''
        Makes a list of masks for each batch entry, for a specific tensor.

        Parameters
        ----------
        tensor : np.ndarray
            Tensor with a batch ID column

        Returns
        -------
        list
            List of batch masks
        '''
        # Create batch masks
        masks = []
        for b in range(self.batch_size * self.num_volumes):
            if len(tensor.shape) == 1:
                masks.append(np.where(tensor == b)[0])
            else:
                masks.append(np.where(tensor[:, BATCH_COL] == b)[0])

        return masks

    def _batch_offsets(self, tensor):
        '''
        Computes the index of the first element in a tensor
        for each entry in the batch.

        Parameters
        ----------
        tensor : np.ndarray
            Tensor with a batch ID column

        Returns
        -------
        np.ndarray
            Array of batch offsets
        '''
        # Compute batch offsets
        offsets = np.zeros(self.batch_size * self.num_volumes, np.int64)
        for b in range(1, self.batch_size * self.num_volumes):
            if len(tensor.shape) == 1:
                offsets[b] = offsets[b-1] + np.sum(tensor == b-1)
            else:
                offsets[b] = offsets[b-1] + np.sum(tensor[:, BATCH_COL] == b-1)

        return offsets

    def _unwrap(self, key, data):
        '''
        Routes set of data to the appropriate unwrapping scheme

        Parameters
        ----------
        key : str
            Name of the data product to unwrap
        data : list
            Data product
        '''
        # Scalars and lists are trivial to unwrap
        if key not in self.rules \
                or self.rules[key].method in [None, 'scalar', 'list']:
            unwrapped = self._concatenate(data)
        else:
            ref_key = self.rules[key].ref_key
            unwrapped = []
            for b in range(self.batch_size):
                # Tensor unwrapping
                if self.rules[key].method == 'tensor':
                    masks = self.masks[ref_key]
                    tensors = []
                    for v in range(self.num_volumes):
                        idx = b * self.num_volumes + v
                        if not self.rules[key].done:
                            tensor = data[masks[idx]]
                            if key == ref_key:
                                if len(tensor.shape) == 2:
                                    tensor[:, BATCH_COL] = v
                                else:
                                    tensor[:] = v
                            if self.rules[key].translate:
                                if v > 0:
                                    tensor[:, COORD_COLS] = \
                                            self.merger.translate(
                                                    tensor[:,COORD_COLS], v)
                            tensors.append(tensor)
                        else:
                            tensors.append(data[idx])
                    unwrapped.append(np.concatenate(tensors))

                # Tensor list unwrapping
                elif self.rules[key].method == 'tensor_list':
                    masks = self.masks[ref_key]
                    tensors = []
                    for i, d in enumerate(data):
                        subtensors = []
                        for v in range(self.num_volumes):
                            idx = b * self.num_volumes + v
                            subtensor = d[masks[i][idx]]
                            if key == ref_key:
                                if len(subtensor.shape) == 2:
                                    subtensor[:, BATCH_COL] = v
                                else:
                                    subtensor[:] = v
                            if self.rules[key].translate:
                                if v > 0:
                                    subtensor[:, COORD_COLS] = \
                                            self.merger.translate(
                                                    subtensor[:,COORD_COLS], v)
                            subtensors.append(subtensor)
                        tensors.append(np.concatenate(subtensors))
                    unwrapped.append(tensors)

                # Edge tensor unwrapping
                elif self.rules[key].method == 'edge_tensor':
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

                # Index tensor unwrapping
                elif self.rules[key].method == 'index_tensor':
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

                # Index list unwrapping
                elif self.rules[key].method == 'index_list':
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

        return unwrapped

    def _concatenate(self, data):
        '''
        Simply concatenates the lists coming from each GPU

        Parameters
        ----------
        key : str
            Name of the data product to unwrap
        data : list
            Data product
        '''
        if np.isscalar(data):
            return np.full(self.batch_size, data)
        else:
            if len(data) != self.batch_size:
                raise ValueError('Only accept scalars or arrays of length ' \
                        f'batch_size: {len(data)} != {self.batch_size}')
            if np.isscalar(data[0]):
                return np.asarray(data)
            elif isinstance(data[0], list):
                concat_data = []
                for d in data:
                    concat_data += d
                return concat_data
            elif isinstance(data[0], np.ndarray):
                return np.concatenate(data)
            else:
                raise TypeError('Unexpected data type', type(data[0]))


def prefix_unwrapper_rules(rules, prefix):
    '''
    Modifies the default rules of a module to account for
    a prefix being added to its standard set of outputs

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
    '''
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
