"""Module with classes for all reconstructed and true objects."""

from dataclasses import dataclass, field

import numpy as np

from spine.data.base import PosDataBase
from spine.data.decorator import stored_property
from spine.data.field import FieldMetadata


@dataclass(eq=False)
class OutBase(PosDataBase):
    """Base data structure shared among all output classes.

    Attributes
    ----------
    id : int
        Unique index of the object within the object list
    index : np.ndarray
        (N) Voxel indexes corresponding to this object in the input tensor
    orig_index: np.ndarray
        (N) Original voxel indexes corresponding to this object in the original
        point cloud (before any filtering or adaptation)
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
        (M) List of unique module indexes that make up this object
    is_contained : bool
        Whether this object is fully contained within the detector
    is_time_contained : bool
        Whether this object's points are within the expected readout window
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
        If the particle is a cathode crosser, this is how far in cm one needs to
        move its points along the drift direction to reconcile at the cathode.
        This is directly proportional to time through time=offset/vdrift
    units : str
        Units in which coordinates are expressed
    """

    # Index attributes
    id: int = field(default=-1, metadata=FieldMetadata(index=True))

    # Scalar attributes
    is_contained: bool = False
    is_time_contained: bool = False
    is_cathode_crosser: bool = False
    is_matched: bool = False

    cathode_offset: float = field(default=np.nan, metadata=FieldMetadata(units="cm"))

    units: str = "cm"

    # Vector attributes
    index: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64, cat=True, lite_skip=True),
    )

    orig_index: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64, cat=True, lite_skip=True),
    )

    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        metadata=FieldMetadata(
            dtype=np.float32,
            position=True,
            cat=True,
            skip=True,
            units="instance",
        ),
    )

    depositions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32, cat=True, skip=True),
    )

    sources: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64, cat=True, skip=True),
    )

    match_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64),
    )

    match_overlaps: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32),
    )

    def reset_match(self) -> None:
        """Resets the reco/truth matching information for the object."""
        self.is_matched = False
        self.match_ids = np.empty(0, dtype=np.int64)
        self.match_overlaps = np.empty(0, dtype=np.float32)

    def reset_cathode_crosser(self) -> None:
        """Resets the cathode crossing information for the object."""
        self.is_cathode_crosser = False
        self.cathode_offset = np.nan

    @property
    @stored_property
    def size(self) -> int:
        """Total number of voxels that make up the object.

        Returns
        -------
        int
            Total number of voxels in the object
        """
        return len(self.index)

    @property
    @stored_property
    def depositions_sum(self) -> float:
        """Total deposition value for the entire object.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions).item()

    @property
    @stored_property
    def module_ids(self) -> np.ndarray:
        """List of modules that contribute to this object.

        Returns
        -------
        np.ndarray
            List of unique modules contributing to this object.
        """
        return np.unique(self.sources[:, 0])


@dataclass(eq=False)
class RecoBase:
    """Mixin class for reconstructed output objects.

    This is a mixin that adds reconstructed-specific attributes. It should be
    used with a primary base class (e.g., FragmentBase, ParticleBase,
    InteractionBase) that inherits from OutBase.

    This eliminates diamond inheritance since OutBase is only inherited once
    through the primary base class.

    Attributes
    ----------
    is_truth : bool
        Whether this is a truth object (always False for reconstructed)
    """

    # Scalar attributes
    is_truth: bool = False


@dataclass(eq=False)
class TruthBase:
    """Mixin class for truth output objects.

    This is a mixin that adds truth-specific attributes. It should be used
    with a primary base class (e.g., FragmentBase, ParticleBase,
    InteractionBase) that inherits from OutBase.

    This eliminates diamond inheritance since OutBase is only inherited once
    through the primary base class.

    Base data structure shared among all truth output classes.

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

    # Scalar attributes
    orig_id: int = -1

    is_truth: bool = True

    # Vector attributes
    index_adapt: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64, cat=True, lite_skip=True),
    )
    index_g4: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64, cat=True, lite_skip=True),
    )

    points_adapt: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        metadata=FieldMetadata(
            dtype=np.float32,
            position=True,
            cat=True,
            skip=True,
            units="instance",
        ),
    )
    points_g4: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32),
        metadata=FieldMetadata(
            dtype=np.float32,
            position=True,
            cat=True,
            skip=True,
            units="instance",
        ),
    )

    depositions_q: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32, cat=True, skip=True),
    )
    depositions_adapt: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32, cat=True, skip=True),
    )
    depositions_adapt_q: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32, cat=True, skip=True),
    )
    depositions_g4: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32, cat=True, skip=True),
    )

    sources_adapt: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64),
        metadata=FieldMetadata(dtype=np.int64, cat=True, skip=True),
    )

    @property
    @stored_property
    def size_adapt(self) -> int:
        """Total number of voxels that make up the object in the adapted tensor.

        Returns
        -------
        int
            Total number of voxels in the object
        """
        return len(self.index_adapt)

    @property
    @stored_property
    def size_g4(self) -> int:
        """Total number of voxels that make up the object in the Geant4 tensor.

        Returns
        -------
        int
            Total number of voxels in the object
        """
        return len(self.index_g4)

    @property
    @stored_property
    def depositions_q_sum(self) -> float:
        """Total deposition value for the entire object in the original units.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_q).item()

    @property
    @stored_property
    def depositions_adapt_sum(self) -> float:
        """Total deposition value for the entire object in the adapted tensor.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_adapt).item()

    @property
    @stored_property
    def depositions_adapt_q_sum(self) -> float:
        """Total deposition value for the entire object in the adapted tensor
        and in the original units.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_adapt_q).item()

    @property
    @stored_property
    def depositions_g4_sum(self) -> float:
        """Total deposition value for the entire object in the Geant4 tensor.

        Returns
        -------
        float
            Sum of all depositions that make up the object
        """
        return np.sum(self.depositions_g4).item()
