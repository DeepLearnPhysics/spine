"""Module that contains all parsers related to LArCV sparse data.

Contains the following parsers:
- :class:`Sparse2DParser`
- :class:`Sparse3DParser`
- :class:`Sparse3DGhostParser`
- :class:`Sparse3DChargeRescaledParser`
"""

import numpy as np

from spine.data import Meta

from spine.utils.globals import GHOST_SHP
from spine.utils.ghost import compute_rescaled_charge
from spine.utils.conditional import larcv

from .base import ParserBase

__all__ = ['Sparse2DParser', 'Sparse3DParser', 'Sparse3DGhostParser',
           'Sparse3DChargeRescaledParser']


class Sparse2DParser(ParserBase):
    """Class that retrieves and parses a 2D sparse tensor.

    .. code-block. yaml

        schema:
          input_data:
            parser: sparse2d
            sparse_event_list:
              - sparse2d_pcluster_0
              - sparse2d_pcluster_1
              - ...
            projection_id: 0
    """

    # Name of the parser (as specified in the configuration)
    name = 'parse_sparse2d'

    def __init__(self, dtype, projection_id, sparse_event=None,
                 sparse_event_list=None):
        """Initialize the parser.

        Parameters
        ----------
        projection_id : int
            Projection ID to get the 2D images from
        sparse_event: larcv.EventSparseTensor2D, optional
            Sparse tensor to get the voxel/features from
        sparse_event_list: List[larcv.EventSparseTensor2D], optional
            List of sparse tensors to get the voxel/features from
        """
        # Initialize the parent class
        super().__init__(
                dtype, sparse_event=sparse_event,
                sparse_event_list=sparse_event_list)

        # Store the revelant attributes
        self.projection_id = projection_id

        # Get the number of features in the output tensor
        assert (sparse_event is not None) ^ (sparse_event_list is not None), (
                "Must provide either `sparse_event` or `sparse_event_list`.")
        assert sparse_event_list is None or len(sparse_event_list), (
                "Must provide as least 1 sparse_event in the list.")

        self.num_features = 1
        if sparse_event_list is not None:
            self.num_features = len(sparse_event_list)

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
            # Get the tensor from the appropriate projection
            tensor = sparse_event.sparse_tensor_2d(self.projection_id)

            # Get the shared information
            if meta is None:
                meta = tensor.meta()
                num_points = tensor.as_vector().size()
                np_voxels = np.empty((num_points, 2), dtype=self.itype)
                larcv.fill_2d_voxels(tensor, np_voxels)
            else:
                assert meta == tensor.meta(), (
                        "The metadata must match between tensors.")
                assert num_points == tensor.as_vector().size(), (
                        "The number of pixels must match between tensors.")

            # Get the feature vector for this tensor
            np_data = np.empty((num_points, 1), dtype=self.ftype)
            larcv.fill_2d_pcloud(tensor, np_data)
            np_features.append(np_data)

        return np_voxels, np.hstack(np_features), Meta.from_larcv(meta)


class Sparse3DParser(ParserBase):
    """Class that retrieves and parses a 3D sparse tensor.

    .. code-block. yaml

        schema:
          input_data:
            parser: sparse3d
            sparse_event_list:
              - sparse3d_pcluster_0
              - sparse3d_pcluster_1
              - ...
    """

    # Name of the parser (as specified in the configuration)
    name = 'sparse3d'

    def __init__(self, dtype, sparse_event=None, sparse_event_list=None,
                 num_features=None, hit_keys=None, nhits_idx=None,
                 feature_only=False):
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
            input data is stored in separate TTrees. `num_features` is required
            to be a divider of the `sparse_event_list` length.
        hit_keys : list of int, optional
            Indices among the input features of the `_hit_key_` TTrees that can
            be used to infer the `nhits` quantity (doublet vs triplet point).
        nhits_idx : int, optional
            Index among the input features where the `nhits` feature
            (doublet vs triplet) should be inserted.
        feature_only : bool, default False
            If `True`, only return the feature vector without the coordinates
        """
        # Initialize the parent class
        super().__init__(
                dtype, sparse_event=sparse_event,
                sparse_event_list=sparse_event_list)

        # Store the revelant attributes
        self.num_features = num_features
        self.hit_keys = hit_keys
        self.nhits_idx = nhits_idx
        self.feature_only = feature_only

        # Check on the parameters
        self.compute_nhits = hit_keys is not None
        if self.compute_nhits and nhits_idx is None:
            raise ValueError("The argument nhits_idx needs to be specified if "
                             "you want to compute the nhits feature.")

        # Get the number of features in the output tensor
        assert (sparse_event is not None) ^ (sparse_event_list is not None), (
                "Must provide either `sparse_event` or `sparse_event_list`.")
        assert sparse_event_list is None or len(sparse_event_list), (
                "Must provide as least 1 sparse_event in the list.")

        num_tensors = 1 if sparse_event is not None else len(sparse_event_list)
        if self.num_features is not None:
            if num_tensors % self.num_features != 0:
                raise ValueError(
                        "The `num_features` number in Sparse3DParser should "
                        "be a divider of the `sparse_event_list` length.")
        else:
            self.num_features = num_tensors

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
                            "The metadata must match between tensors.")

                if num_points is None:
                    num_points = sparse_event.as_vector().size()
                    if not self.feature_only:
                        np_voxels = np.empty((num_points, 3), dtype=self.itype)
                        larcv.fill_3d_voxels(sparse_event, np_voxels)
                else:
                    assert num_points == sparse_event.as_vector().size(), (
                            "The number of pixels must match between tensors.")

                # Get the feature vector for this tensor
                np_data = np.empty((num_points, 1), dtype=self.ftype)
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
                if self.nhits_idx < 0 or self.nhits_idx > self.num_features:
                    raise ValueError(
                            f"`nhits_idx` ({self.nhits_idx}) is out of range.")
                np_features.insert(self.nhits_idx, nhits)

            # Append to the global list of voxel/features
            if not self.feature_only:
                all_voxels.append(np_voxels)
            all_features.append(np.hstack(np_features))

        if self.feature_only:
            return np.vstack(all_features)
        else:
            return (np.vstack(all_voxels), np.vstack(all_features),
                    Meta.from_larcv(meta))


class Sparse3DGhostParser(Sparse3DParser):
    """Class that convert a tensor containing semantics to binary ghost labels.

    .. code-block. yaml

        schema:
          ghost_label:
            parser: sparse3d
            sparse_event_semantics: sparse3d_semantics
    """

    # Name of the parser (as specified in the configuration)
    name = 'sparse3d_ghost'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process_ghost(**self.get_input_data(trees))

    def process_ghost(self, sparse_event):
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
        np_voxels, np_data, meta = self.process(sparse_event)
        np_ghosts = (np_data == GHOST_SHP).astype(np_data.dtype)

        return np_voxels, np_ghosts, meta


class Sparse3DChargeRescaledParser(Sparse3DParser):
    """Class that convert a tensor containing semantics to binary ghost labels.

    .. code-block. yaml

        schema:
          input_rescaled:
            parser: sparse3d_charge_rescaled
            sparse_event_semantics: sparse3d_semantics
    """

    # Name of the parser (as specified in the configuration)
    name = 'parse_sparse3d_rescale_charge'

    # Alternative allowed names of the parser
    aliases = ('parse_sparse3d_charge_rescaled',)

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

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process_rescale(**self.get_input_data(trees))

    def process_rescale(self, sparse_event_list):
        """Fetches one or a list of tensors, concatenate their feature vectors.

        Parameters
        -------------
        sparse_event_list: List[larcv.EventSparseTensor3D]
            (7) List of sparse tensors used to compute the rescaled charge
            - Charge value of each of the contributing planes (3)
            - Index of the plane hit contributing to the space point (3)
            - Semantic labels (1)

        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 1) array of ghost labels (1 for ghosts, 0 otherwise)
        meta : Meta
            Metadata of the parsed image
        """
        np_voxels, np_data, meta = self.process(
                sparse_event_list=sparse_event_list)

        deghost_mask = np.where(np_data[:, -1] < GHOST_SHP)[0]
        charges = compute_rescaled_charge(
                np_data[deghost_mask, :-1],
                collection_only=self.collection_only,
                collection_id=self.collection_id)

        return np_voxels[deghost_mask], charges[:, None], meta
