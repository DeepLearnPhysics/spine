"""Module that contains all parsers related to LArCV sparse data.

Contains the following parsers:
- :class:`Sparse2DParser`
- :class:`Sparse3DParser`
- :class:`Sparse3DGhostParser`
- :class:`Sparse3DChargeRescaledParser`
"""

import numpy as np
from warnings import warn
from larcv import larcv

from mlreco.utils.globals import GHOST_SHP
from mlreco.utils.data_structures import Meta

from .parser import Parser

__all__ = ['Sparse2DParser', 'Sparse3DParser', 'Sparse3DGhostParser',
           'Sparse3DChargeRescaledParser']


class Sparse2DParser(Parser):
    """Class that retrieves and parses a 2D sparse tensor.

    .. code-block. yaml

        schema:
          input_data:
            parser: parse_sparse2d
            sparse_event_list:
              - sparse2d_pcluster_0
              - sparse2d_pcluster_1
              - ...
            projection_id: 0
    """
    name = 'parse_sparse2d'

    def __init__(self, sparse_event=None, sparse_event_list=None,
                 projection_id=None):
        """Initialize the parser.

        Parameters
        ----------
        sparse_event: larcv.EventSparseTensor2D, optional
            Sparse tensor to get the voxel/features from
        sparse_event_list: List[larcv.EventSparseTensor2D], optional
            List of sparse tensors to get the voxel/features from
        projection_id : int, optional
            Projection ID to get the 2D images from
        """
        # Initialize the parent class
        super().__init__(sparse_event=sparse_event,
                         sparse_event_list=sparse_event_list)

        # Store the revelant attributes
        self.projection_id = projection_id

        # Get the number of features in the output tensor
        assert (sparse_event is not None) ^ (sparse_event_list is not None), (
                "Must provide either `sparse_event` or `sparse_event_list`")
        assert sparse_event_list is None or len(sparse_event_list), (
                "Must provide as least 1 sparse_event in the list")

        self.num_features = 1
        if sparse_event_list is not None:
            self.num_features = len(sparse_event_list)

    def process(self, sparse_event=None, sparse_event_list=None):
        """Fetches one or a list of tensors, concatenate their feature vectors.

        Parameters
        -------------
        sparse_event: larcv.EventSparseTensor2D, optional
            Sparse tensor to get the voxel/features from
        sparse_event_list: List[larcv.EventSparseTensor2D], optional
            List of sparse tensors to get the voxel/features from

        Returns
        -------
        np_voxels : np.ndarray
            (N, 2) array of [x, y] coordinates
        np_features : np.ndarray
            (N, C) array of [pixel value 0, pixel value 1, ...]
        meta : Meta
            Metadata of the parsed images
        """
        # Parse input into a list
        if sparse_event_list is None:
            sparse_event_list = [sparse_event]

        # Loop over the list of sparse events
        np_voxels, meta, num_points = None, None, None
        np_features = []
        for sparse_event in sparse_event_list:
            # Get the tensor
            tensor = sparse_event.sparse_tensor_2d(self.projection_id)

            # Get the shared information
            if meta is None:
                meta = tensor.meta()
                num_points = tensor.as_vector().size()
                np_voxels = np.empty((num_points, 2), dtype=np.int32)
                larcv.fill_2d_voxels(tensor, np_voxels)
            else:
                assert meta == tensor.meta(), (
                        "The metadata must match between tensors")
                assert num_points == tensor.as_vector().size(), (
                        "The number of pixels must match between tensors")

            # Get the feature vector for this tensor
            np_data = np.empty((num_points, 1), dtype=np.float32)
            larcv.fill_2d_pcloud(tensor, np_data)
            np_features.append(np_data)

        return np_voxels, np.hstack(np_features), Meta.from_larcv(meta)


class Sparse3DParser(Parser):
    """Class that retrieves and parses a 3D sparse tensor.

    .. code-block. yaml

        schema:
          input_data:
            parser: parse_sparse3d
            sparse_event_list:
              - sparse3d_pcluster_0
              - sparse3d_pcluster_1
              - ...
    """
    name = 'parse_sparse3d'

    def __init__(self, sparse_event=None, sparse_event_list=None,
                 num_features=None, features=None, hit_keys=None,
                 nhits_idx=None, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        sparse_event: larcv.EventSparseTensor3D, optional
            Sparse tensor to get the voxel/features from
        sparse_event_list: List[larcv.EventSparseTensor3D], optional
            List of sparse tensors to get the voxel/features from
        num_features : int, optional
            If a positive integer is specified, the sparse_event_list will be
            split in equal lists of length `features`. Each list will be
            concatenated along the feature dimension separately. Then all
            lists are concatenated along the first dimension (voxels). For
            example, this lets you work with distinct detector volumes whose
            input data is stored in separate TTrees.`features` is required to
            be a divider of the `sparse_event_list` length.
        hit_keys : list of int, optional
            Indices among the input features of the `_hit_key_` TTrees that can
            be used to infer the `nhits` quantity (doublet vs triplet point).
        nhits_idx : int, optional
            Index among the input features where the `nhits` feature
            (doublet vs triplet) should be inserted.
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(sparse_event=sparse_event,
                         sparse_event_list=sparse_event_list)

        # Store the revelant attributes
        assert (num_features is None) or (features is None), (
                "Do not specify both `features` and `num_features`.")
        self.num_features = num_features
        if features is not None:
            warn("Parameter `features` is deprecated, use `num_features` "
                 "instead", DeprecationWarning, stacklevel=1)
            self.num_features = features
        self.hit_keys = hit_keys
        self.nhits_idx = nhits_idx

        # Check on the parameters
        self.compute_nhits = hit_keys is not None
        if self.compute_nhits and nhits_idx is None:
            raise ValueError("The argument nhits_idx needs to be specified if "
                             "you want to compute the nhits feature.")

        # Get the number of features in the output tensor
        assert (sparse_event is not None) ^ (sparse_event_list is not None), (
                "Must provide either `sparse_event` or `sparse_event_list`")
        assert sparse_event_list is None or len(sparse_event_list), (
                "Must provide as least 1 sparse_event in the list")

        num_tensors = 1 if sparse_event is not None else len(sparse_event_list)
        if self.num_features is not None:
            if num_tensors % self.num_features != 0:
                raise ValueError(
                        "The `num_features` number in parse_sparse3d should "
                        "be a divider of the `sparse_event_list` length.")
        else:
            self.num_features = num_tensors

    def process(self, sparse_event=None, sparse_event_list=None):
        """Fetches one or a list of tensors, concatenate their feature vectors.

        Parameters
        ----------
        sparse_event: larcv.EventSparseTensor3D, optional
            Sparse tensor to get the voxel/features from
        sparse_event_list: List[larcv.EventSparseTensor3D], optional
            List of sparse tensors to get the voxel/features from

        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, C) array of [pixel value 0, pixel value 1, ...]
        meta : Meta
            Metadata of the parsed images
        """
        # Parse input into a list
        if sparse_event_list is None:
            sparse_event_list = [sparse_event]

        # If requested, split the input list into multiple lists
        split_sparse_event_list = [sparse_event_list]
        if self.num_features != len(sparse_event_list):
            num_groups = len(sparse_event_list) // self.num_features
            split_sparse_event_list = np.split(
                    np.array(sparse_event_list), num_groups)

        # Loop over the individual lists, load the voxels/features
        all_voxels, all_features = [], []
        meta = None
        for sparse_event_list in split_sparse_event_list:
            np_voxels, num_points = None, None 
            np_features = []
            hit_key_array = []
            for idx, sparse_event in enumerate(sparse_event_list):
                # Get the shared information
                if meta is None:
                    meta = sparse_event.meta()
                else:
                    assert meta == sparse_event.meta(), (
                            "The metadata must match between tensors")

                if num_points is None:
                    num_points = sparse_event.as_vector().size()
                    np_voxels = np.empty((num_points, 3), dtype=np.int32)
                    larcv.fill_3d_voxels(sparse_event, np_voxels)
                else:
                    assert num_points == sparse_event.as_vector().size(), (
                            "The number of pixels must match between tensors")

                # Get the feature vector for this tensor
                np_data = np.empty((num_points, 1), dtype=np.float32)
                larcv.fill_3d_pcloud(sparse_event, np_data)
                np_features.append(np_data)

                # If the number of hits is to be computed, keep track of the
                # required information to do so downstream
                if self.compute_nhits:
                    if idx in self.hit_keys:
                        hit_key_array.append(np_data)

            # If requested, add a feature related to the number of planes
            if self.compute_nhits:
                hit_key_array = np.hstack(hit_key_array)
                nhits = np.sum(hit_key_array >= 0., axis=1)[:, -1]
                if nhits_idx < 0 or nhits_idx > self.num_features:
                    raise ValueError(
                            f"nhits_idx ({nhits_idx}) is out of range")
                np_features.insert(nhits_idx, nhits)

            # Append to the global list of voxel/features
            all_voxels.append(np_voxels)
            all_features.append(np.hstack(np_features))

        return (np.vstack(all_voxels), np.vstack(all_features),
                Meta.from_larcv(meta))


class Sparse3DGhostParser(Sparse3DParser):
    """Class that convert a tensor containing semantics to binary ghost labels.

    .. code-block. yaml

        schema:
          ghost_label:
            parser: parse_sparse3d
            sparse_event_semantics: sparse3d_semantics
    """
    name = 'parse_sparse3d_ghost'
    aliases = []

    def process(self, sparse_event):
        """Fetches one or a list of tensors, concatenate their feature vectors.

        Parameters
        -------------
        sparse_event: larcv.EventSparseTensor3D, optional
            Sparse tensor to get the voxel/features from

        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 1) array of ghost labels (1 for ghosts, 0 otherwise)
        meta : Meta
            Metadata of the parsed image
        """
        # Convert the semantics feature to a ghost feature
        np_voxels, np_data, meta = super().process(sparse_event)
        np_ghosts = np_data == GHOST_SHP

        return np_voxels, np_ghosts, meta


class Sparse3DChargeRescaledParser(Sparse3DParser):
    """Class that convert a tensor containing semantics to binary ghost labels.

    .. code-block. yaml

        schema:
          input_rescaled:
            parser: parse_sparse3d_charge_rescaled
            sparse_event_semantics: sparse3d_semantics
    """
    name = 'parse_sparse3d_rescale_charge'
    aliases = ['parse_sparse3d_charge_rescaled']

    def __init__(self, collection_only=False, collection_id=2, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        collection_only : bool, default False
            If True, only uses the collection plane charge
        collection_id : int, default 2
            Index of the collection plane
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.collection_only = collection_only
        self.collection_id = collection_id

    def process(self, sparse_event_list):
        """Fetches one or a list of tensors, concatenate their feature vectors.

        Parameters
        -------------
        sparse_event: larcv.EventSparseTensor3D, optional
            Sparse tensor to get the voxel/features from

        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 1) array of ghost labels (1 for ghosts, 0 otherwise)
        meta : Meta
            Metadata of the parsed image
        """
        np_voxels, np_data, meta = super().process(
                sparse_event_list=sparse_event_list)

        deghost_mask = np.where(output[:, -1] < GHOST_SHP)[0]
        charges = compute_rescaled_charge(
                np_data[:, :-1], deghost_mask, last_index=0,
                collection_only=self.collection_only, 
                collection_id=self.collection_id, use_batch=False)

        return np_voxels[deghost_mask], charges[:, None], meta
