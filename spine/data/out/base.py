"""Module with classes for all reconstructed and true objects."""

from dataclasses import dataclass, field

import numpy as np

from spine.utils.decorators import inherit_docstring

from spine.data.base import PosDataBase


@dataclass(eq=False)
class OutBase(PosDataBase):
    """Base data structure shared among all output classes.

    Attributes
    ----------
    id : int
        Unique index of the object within the object list
    index : np.ndarray
        (N) Voxel indexes corresponding to this object in the input tensor
    size : int
        Number of points, N, that make up this object
    points : np.ndarray
        (N, 3) Set of voxel coordinates that make up this object
    depositions : np.ndarray
        (N) Array of charge deposition values for each voxel
    depositions_sum : float
        Total amount of depositions
    sources : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs
    module_ids : np.ndarray
        (M) List of module indexes that make up this object
    is_contained : bool
        Whether this object is fully contained within the detector
    is_matched: bool
        True if a true object match was found
    match_ids : np.ndarray
        List of true object IDs this object is matched to
    match_overlaps : np.ndarray
        List of match overlaps (IoU) between the object and its matches
    is_cathode_crosser : bool
        True if the particle crossed a cathode, i.e. if it is made up
        of space points coming from > 1 TPC in one module
    cathode_offset : float
        If the particle is a cathode crosser, this corresponds to the offset
        to apply to the particle to match its components at the cathode
    is_truth: bool
        Whether this object contains truth information or not
    units : str
        Units in which coordinates are expressed
    """
    id: int = -1
    index: np.ndarray = None
    size: int = None
    points: np.ndarray = None
    depositions: np.ndarray = None
    depositions_sum: float = None
    sources: np.ndarray = None
    module_ids: np.ndarray = None
    is_contained: bool = False
    is_matched: bool = False
    match_ids: np.ndarray = None
    match_overlaps: np.ndarray = None
    is_cathode_crosser: bool = False
    cathode_offset: float = -np.inf
    is_truth: bool = None
    units: str = 'cm'

    # Variable-length attribtues
    _var_length_attrs = (
            ('index', np.int64), ('depositions', np.float32),
            ('match_ids', np.int64), ('match_overlaps', np.float32),
            ('points', (3, np.float32)), ('sources', (2, np.int64)),
            ('module_ids', np.int64)
    )

    # Boolean attributes
    _bool_attrs = (
            'is_contained', 'is_matched', 'is_cathode_crosser', 'is_truth'
    )

    # Attributes to concatenate when merging objects
    _cat_attrs = ('index', 'points', 'depositions', 'sources')

    # Attributes that must never be stored to file
    _skip_attrs = ('points', 'depositions', 'sources')

    # Attributes that must not be stored to file when storing lite files
    _lite_skip_attrs = ('index',)

    @property
    def size(self):
        """Total number of voxels that make up the object.

        Returns
        -------
        int
            Total number of voxels in the object
        """
        return len(self.index)

    @size.setter
    def size(self, size):
        pass

    @property
    def depositions_sum(self):
        """Total deposition value for the entire object.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions)

    @depositions_sum.setter
    def depositions_sum(self, depositions_sum):
        pass

    @property
    def module_ids(self):
        """List of modules that contribute to this object.

        Returns
        -------
        np.ndarray
            List of unique modules contributing to this object.
        """
        return np.unique(self.sources[:, 0])

    @module_ids.setter
    def module_ids(self, module_ids):
        pass


@dataclass(eq=False)
@inherit_docstring(OutBase)
class RecoBase(OutBase):
    """Base data structure shared among all reconstructed output classes."""
    is_truth: bool = False


@dataclass(eq=False)
@inherit_docstring(OutBase)
class TruthBase(OutBase):
    """Base data structure shared among all truth output classes.

    Attributes
    ----------
    orig_id : int
        If matched to an MC truth instance, ID of the original instance
    depositions_q : np.ndarray
        (N) Array of values for each voxel in the same units as the input image
    depositions_q_sum : float
        Total amount of depositions in the same units as the input image
    index_adapt: np.ndarray
        (N') Voxel indexes corresponding to this object in the adapted cluster
        label tensor
    size_adapt : int
        Number of points, N', that make up this object in the adapted cluster
        label tensor
    points_adapt : np.ndarray
        (N', 3) Set of voxel coordinates using adapted cluster labels
    sources_adapt : np.ndarray
        (N', 2) Set of voxel sources as (Module ID, TPC ID) pairs, adapted
    depositions_adapt : np.ndarray
        (N') Array of values for each voxel in the adapted cluster label tensor
    depositions_adapt_sum : float
        Total amount of depositions in adapted cluster label tensor
    depositions_adapt_q : np.ndarray
        (N) Array of values for each voxel in the same units as the input image
    depositions_adapt_q_sum : float
        Total amount of depositions in adapted cluster label tensor in the same
        units as the input image
    sources_adapt : np.ndarray
        (N, 2) Set of voxel sources as (Module ID, TPC ID) pairs, adapted
    index_g4: np.ndarray
        (N'') Fragment voxel indexes in the true Geant4 energy deposition tensor
    size_g4 : int
        Number of points, N'', that make up this object in the true Geant4
        energy deposition tensor
    points_g4 : np.ndarray
        (N'', 3) Set of voxel coordinates of true Geant4 energy depositions
    depositions_g4 : np.ndarray
        (N'') Array of true Geant4 energy depositions per voxel
    depositions_g4_sum : float
        Total amount of true Geant4 depositions
    """
    orig_id: int = -1
    depositions_q: np.ndarray = None
    depositions_q_sum: float = None
    index_adapt: np.ndarray = None
    size_adapt: int = None
    size_g4: int = None
    points_adapt: np.ndarray = None
    depositions_adapt: np.ndarray = None
    depositions_adapt_sum: float = None
    depositions_adapt_q: np.ndarray = None
    depositions_adapt_q_sum: float = None
    sources_adapt: np.ndarray = None
    index_g4: np.ndarray = None
    points_g4: np.ndarray = None
    depositions_g4: np.ndarray = None
    depositions_g4_sum: float = None
    is_truth: bool = True

    # Variable-length attribtues
    _var_length_attrs = (
            ('depositions_q', np.float32), ('index_adapt', np.int64),
            ('depositions_adapt', np.float32), ('depositions_adapt_q', np.float32),
            ('index_g4', np.int64), ('depositions_g4', np.int64),
            ('points_adapt', (3, np.float32)), ('sources_adapt', (2, np.int64)),
            ('points_g4', (3, np.float32)), *OutBase._var_length_attrs
    )

    # Attributes to concatenate when merging objects
    _cat_attrs = (
            'depositions_q', 'index_adapt', 'points_adapt', 'depositions_adapt',
            'depositions_adapt_q', 'sources_adapt', 'index_g4', 'points_g4',
            'depositions_g4', *OutBase._cat_attrs
    )

    # Attributes that must never be stored to file
    _skip_attrs = (
            'depositions_q', 'points_adapt', 'depositions_adapt',
            'depositions_adapt_q', 'sources_adapt', 'depositions_g4',
            'points_g4', 'depositions_g4', *OutBase._skip_attrs
    )

    # Attributes that must not be stored to file when storing lite files
    _lite_skip_attrs = ('index_adapt', 'index_g4', *OutBase._lite_skip_attrs)

    @property
    def size_adapt(self):
        """Total number of voxels that make up the object in the adapted tensor.

        Returns
        -------
        int
            Total number of voxels in the object
        """
        return len(self.index_adapt)

    @size_adapt.setter
    def size_adapt(self, size_adapt):
        pass

    @property
    def size_g4(self):
        """Total number of voxels that make up the object in the Geant4 tensor.

        Returns
        -------
        int
            Total number of voxels in the object
        """
        return len(self.index_g4)

    @size_g4.setter
    def size_g4(self, size_g4):
        pass

    @property
    def depositions_q_sum(self):
        """Total deposition value for the entire object in the original units.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_q)

    @depositions_q_sum.setter
    def depositions_q_sum(self, depositions_q_sum):
        pass

    @property
    def depositions_adapt_sum(self):
        """Total deposition value for the entire object in the adapted tensor.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_adapt)

    @depositions_adapt_sum.setter
    def depositions_adapt_sum(self, depositions_adapt_sum):
        pass

    @property
    def depositions_adapt_q_sum(self):
        """Total deposition value for the entire object in the adapted tensor
        and in the original units.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_adapt_q)

    @depositions_adapt_q_sum.setter
    def depositions_adapt_q_sum(self, depositions_adapt_q_sum):
        pass

    @property
    def depositions_g4_sum(self):
        """Total deposition value for the entire object in the Geant4 tensor.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_g4)

    @depositions_g4_sum.setter
    def depositions_g4_sum(self, depositions_g4_sum):
        pass
