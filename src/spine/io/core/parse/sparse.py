"""Module that contains all parsers related to LArCV sparse data.

Contains the following parsers:
- :class:`Sparse2DParser`
- :class:`Sparse3DParser`
- :class:`Sparse3DAggregateParser`
- :class:`Sparse3DChargeRescaledParser`
- :class:`Sparse3DGhostParser`
"""

import numpy as np

from spine.data import Meta
from spine.utils.conditional import larcv
from spine.utils.ghost import ChargeRescaler
from spine.utils.globals import GHOST_SHP, SHAPE_PREC

from .base import ParserBase
from .data import ParserTensor

__all__ = [
    "Sparse2DParser",
    "Sparse3DParser",
    "Sparse3DAggregateParser",
    "Sparse3DChargeRescaledParser",
    "Sparse3DGhostParser",
]


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
    name = "parse_sparse2d"

    # Type of object(s) returned by the parser
    returns = "tensor"

    def __init__(self, dtype, projection_id, sparse_event=None, sparse_event_list=None):
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
            dtype, sparse_event=sparse_event, sparse_event_list=sparse_event_list
        )

        # Store the revelant attributes
        self.projection_id = projection_id

        # Get the number of features in the output tensor
        assert (sparse_event is not None) ^ (
            sparse_event_list is not None
        ), "Must provide either `sparse_event` or `sparse_event_list`."
        assert sparse_event_list is None or len(
            sparse_event_list
        ), "Must provide as least 1 sparse_event in the list."

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
        ParserTensor
            coords : np.ndarray
                (N, 2) array of [x, y] coordinates
            features : np.ndarray
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
                np_voxels = np.empty((num_points, 2), dtype=np.int32)
                larcv.fill_2d_voxels(tensor, np_voxels)
                np_voxels = np_voxels.astype(self.itype)
            else:
                assert meta == tensor.meta(), "The metadata must match between tensors."
                assert (
                    num_points == tensor.as_vector().size()
                ), "The number of pixels must match between tensors."

            # Get the feature vector for this tensor
            np_data = np.empty((num_points, 1), dtype=np.float32)
            larcv.fill_2d_pcloud(tensor, np_data)
            np_data = np_data.astype(self.ftype)
            np_features.append(np_data)

        return ParserTensor(
            coords=np_voxels,
            features=np.hstack(np_features),
            meta=Meta.from_larcv(meta),
        )


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
    name = "sparse3d"

    # Type of object(s) returned by the parser
    returns = "tensor"

    def __init__(
        self,
        dtype,
        sparse_event=None,
        sparse_event_list=None,
        num_features=None,
        hit_keys=None,
        nhits_idx=None,
        feature_only=False,
        lexsort=False,
        index_cols=None,
        sum_cols=None,
        prec_col=None,
        precedence=SHAPE_PREC,
    ):
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
        lexsort : bool, default False
            When merging points from multiple sources (num_features is not
            `None`), this allows to lexicographically sort coordinates
        index_cols : np.ndarray, optional
            (C) Columns which contain indexes
        sum_cols : np.ndarray, optional
            (S) Columns which should be summed when removing duplicates
        prec_col : int, optional
            Column to be used as a precedence source when removing duplicates
        precedence : np.ndarray, default SHAPE_PREC
            Order of precedence among the classes in prec_col
        """
        # Initialize the parent class
        super().__init__(
            dtype, sparse_event=sparse_event, sparse_event_list=sparse_event_list
        )

        # Store the revelant attributes
        self.num_features = num_features
        self.hit_keys = hit_keys
        self.nhits_idx = nhits_idx
        self.feature_only = feature_only

        # Only lexsort when needed and if there is more than one sparse3d source
        self.lexsort = lexsort
        if self.num_features is None and lexsort:
            raise ValueError(
                "No need to lexsort if there is only one coordinate source."
            )

        # Check on the parameters
        self.compute_nhits = hit_keys is not None
        if self.compute_nhits and nhits_idx is None:
            raise ValueError(
                "The argument nhits_idx needs to be specified if "
                "you want to compute the nhits feature."
            )

        # Get the number of features in the output tensor
        assert (sparse_event is not None) ^ (
            sparse_event_list is not None
        ), "Must provide either `sparse_event` or `sparse_event_list`."
        assert sparse_event_list is None or len(
            sparse_event_list
        ), "Must provide as least 1 sparse_event in the list."

        num_tensors = 1 if sparse_event is not None else len(sparse_event_list)
        if self.num_features is not None:
            if num_tensors % self.num_features != 0:
                raise ValueError(
                    "The `num_features` number in Sparse3DParser should "
                    "be a divider of the `sparse_event_list` length."
                )
        else:
            self.num_features = num_tensors

        # Define the overlay strategy parameters
        self.index_cols = None
        if index_cols is not None:
            self.index_cols = np.asarray(index_cols)
        self.sum_cols = None
        if sum_cols is not None:
            self.sum_cols = np.asarray(sum_cols)
        self.prec_col = prec_col
        self.precedence = precedence

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
        ParserTensor
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
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
            split_sparse_event_list = np.split(np.array(sparse_event_list), num_groups)

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
                    assert (
                        meta == sparse_event.meta()
                    ), "The metadata must match between tensors."

                if num_points is None:
                    num_points = sparse_event.as_vector().size()
                    np_voxels = np.empty((num_points, 3), dtype=np.int32)
                    larcv.fill_3d_voxels(sparse_event, np_voxels)
                    np_voxels = np_voxels.astype(self.itype)
                else:
                    assert (
                        num_points == sparse_event.as_vector().size()
                    ), "The number of pixels must match between tensors."

                # Get the feature vector for this tensor
                np_data = np.empty((num_points, 1), dtype=np.float32)
                larcv.fill_3d_pcloud(sparse_event, np_data)
                np_data = np_data.astype(self.ftype)
                np_features.append(np_data)

                # If the number of hits is to be computed, keep track of the
                # required information to do so downstream
                if self.compute_nhits:
                    if idx in self.hit_keys:
                        hit_key_array.append(np_data)

            # If requested, add a feature related to the number of planes
            if self.compute_nhits:
                hit_key_array = np.hstack(hit_key_array)
                nhits = np.sum(hit_key_array >= 0.0, axis=1)[:, -1]
                if self.nhits_idx < 0 or self.nhits_idx > self.num_features:
                    raise ValueError(f"`nhits_idx` ({self.nhits_idx}) is out of range.")
                np_features.insert(self.nhits_idx, nhits)

            # Append to the global list of voxel/features
            all_voxels.append(np_voxels)
            all_features.append(np.hstack(np_features))

        # Stack coordinates/features
        all_voxels = np.vstack(all_voxels)
        all_features = np.vstack(all_features)

        # Lexicographically sort coordinates/features, if requested
        if self.lexsort:
            perm = np.lexsort(all_voxels.T)
            all_voxels = all_voxels[perm]
            all_features = all_features[perm]

        # Return
        return ParserTensor(
            coords=all_voxels,
            features=all_features,
            meta=Meta.from_larcv(meta),
            remove_duplicates=True,
            index_cols=self.index_cols,
            sum_cols=self.sum_cols,
            prec_col=self.prec_col,
            precedence=self.precedence,
            feats_only=self.feature_only,
        )


class Sparse3DAggregateParser(Sparse3DParser):
    """Class that aggregates features from multiple sparse tensors

    .. code-block. yaml

        schema:
          charge_label:
            parser: sparse3d_aggr
            aggr: sum
            sparse_event_list:
              - sparse3d_reco_cryoE_rescaled
              - sparse3d_reco_cryoW_rescaled
    """

    # Name of the parser (as specified in the configuration)
    name = "sparse3d_aggr"

    def __init__(self, dtype, aggr, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        aggr : str
            Aggregation function to apply ('sum', 'mean', 'max', etc.)
        """
        # Initialize the parent class
        super().__init__(dtype, **kwargs)

        # Store the revelant attributes
        self.aggr_fn = getattr(np, aggr)

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process_aggr(**self.get_input_data(trees))

    def process_aggr(self, sparse_event_list):
        """Fetches a list of tensors, aggregate their feature vectors.

        Parameters
        -------------
        sparse_event_list: List[larcv.EventSparseTensor3D]
            Sparse tensor list to get the voxel/features from

        Returns
        -------
        ParserTensor
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                (N, 1) array of aggregated features
            meta : Meta
                Metadata of the parsed image
        """
        # Fetch the list of features using the standard parser
        tensor = self.process(sparse_event_list=sparse_event_list)

        # Combine them into a single feature using the aggregator function
        tensor.features = self.aggr_fn(tensor.features, axis=1)[:, None]

        return tensor


class Sparse3DChargeRescaledParser(Sparse3DParser):
    """Class that convert a tensor containing semantics to binary ghost labels.

    .. code-block. yaml

        schema:
          input_rescaled:
            parser: sparse3d_charge_rescaled
            sparse_event_semantics: sparse3d_semantics
    """

    # Name of the parser (as specified in the configuration)
    name = "parse_sparse3d_rescale_charge"

    # Alternative allowed names of the parser
    aliases = ("parse_sparse3d_charge_rescaled",)

    def __init__(self, dtype, collection_only=False, collection_id=2, **kwargs):
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
        super().__init__(dtype, **kwargs)

        # Initialize the charge rescaler
        self.rescaler = ChargeRescaler(collection_only, collection_id)

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
        ParserTensor
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                (N, 1) array of rescaled charge values
            meta : Meta
                Metadata of the parsed image
        """
        # Fetch the list of features using the standard parser
        tensor = self.process(sparse_event_list=sparse_event_list)

        # Use individual hit informations to compute a rescaled charge
        deghost_mask = np.where(tensor.features[:, -1] < GHOST_SHP)[0]
        charges = self.rescaler.process_single(tensor.features[deghost_mask, :-1])

        tensor.features = charges[:, None]

        return tensor


class Sparse3DGhostParser(Sparse3DParser):
    """Class that convert a tensor containing semantics to binary ghost labels.

    .. code-block. yaml

        schema:
          ghost_label:
            parser: sparse3d_ghost
            sparse_event_semantics: sparse3d_semantics
    """

    # Name of the parser (as specified in the configuration)
    name = "sparse3d_ghost"

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
        sparse_event: larcv.EventSparseTensor3D
            Sparse tensor to get the semantic labels

        Returns
        -------
        ParserTensor
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                (N, 1) array of ghost labels (1 for ghosts, 0 otherwise)
            meta : Meta
                Metadata of the parsed image
        """
        # Fetch the list of features using the standard parser
        tensor = self.process(sparse_event)

        # Convert the semantics feature to a ghost feature
        tensor.features = (tensor.features == GHOST_SHP).astype(tensor.features.dtype)

        return tensor
