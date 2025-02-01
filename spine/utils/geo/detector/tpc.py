"""TPC detector geometry classes."""

from typing import List
from dataclasses import dataclass

import numpy as np

from .base import Box

__all__ = ['TPCDetector']


@dataclass
class Chamber(Box):
    """Class which holds all properties of an individual time-projection
    chamber (TPC).

    Attributes
    ----------
    drift_dir : np.ndarray
        (3) TPC drift direction vector (normalized)
    drift_axis : int
        Axis along which the electrons drift (0, 1 or 2)
    """
    drift_dir: np.ndarray
    drift_axis: int

    def __init__(self, position, dimensions, drift_dir):
        """Initialize the TPC object.

        Parameters
        ----------
        position : np.ndarray
            (3) Position of the center of the TPC
        dimensions : np.ndarray
            (3) Dimension of the TPC
        drift_dir : np.ndarray
            (3) Drift direction vector
        """
        # Initialize the underlying box object
        lower = position - dimensions/2
        upper = position + dimensions/2
        super().__init__(lower, upper)

        # Make sure that the drift axis only points in one direction
        nonzero_axes = np.where(drift_dir)[0]
        assert len(nonzero_axes) == 1, (
                "The drift direction must be aligned with a base axis.")

        # Store drift information
        self.drift_dir = drift_dir
        self.drift_axis = nonzero_axes[0]

    @property
    def drift_sign(self):
        """Sign of drift w.r.t. to the drift axis orientation.

        Returns
        -------
        int
            Returns the sign of the drift vector w.r.t. to the drift axis
        """
        return int(self.drift_dir[self.drift_axis])

    @property
    def anode_side(self):
        """Returns whether the anode is on the lower or upper boundary of
        the TPC along the drift axis (0 for lower, 1 for upper).

        Returns
        -------
        int
            Anode side of the TPC
        """
        return (self.drift_sign + 1)//2

    @property
    def cathode_side(self):
        """Returns whether the cathode is on the lower or upper boundary of
        the TPC along the drift axis (0 for lower, 1 for upper).

        Returns
        -------
        int
            Cathode side of the TPC
        """
        return 1 - self.anode_side

    @property
    def anode_pos(self):
        """Position of the anode along the drift direction.

        Returns
        -------
        float
            Anode position along the drift direction
        """
        return self.boundaries[self.drift_axis, self.anode_side]

    @property
    def cathode_pos(self):
        """Position of the cathode along the drift direction.

        Returns
        -------
        float
            Cathode position along the drift direction
        """
        return self.boundaries[self.drift_axis, self.cathode_side]


@dataclass
class Module(Box):
    """Class which holds all properties of a TPC module.

    A module can hold either one chamber or two chambers with a shared cathode.

    Attributes
    ----------
    chambers : List[Chamber]
        List of individual TPCs that make up the module
    """
    chambers: List[Chamber]

    def __init__(self, positions, dimensions, drift_dirs=None):
        """Intialize the TPC module.

        Parameters
        ----------
        positions : np.ndarray
            (N_t) List of TPC center positions, one per TPC
        dimensions : np.ndarray
            (3) Dimensions of one TPC
        drift_dirs : np.ndarray, optional
            (N_t, 3) List of drift directions. If this is not provided, it is
            inferred from the module configuration, provided that the module
            is composed of two TPCs with a shared cathode.
        """
        # Sanity checks
        assert len(positions) in [1, 2], (
                "A TPC module must contain exactly one or two TPCs.")
        assert (drift_dirs is not None) ^ (len(positions) == 2), (
                "For TPC modules with one TPC, the drift direction cannot be "
                "inferred and must be provided explicitely. For modules with "
                "two TPCs, must not set the drift direction arbitrarily.")

        # Build TPCs
        self.chambers = []
        for t in range(len(positions)):
            # Fetch the drift axis. If not provided, join the two TPC centers
            if drift_dirs is not None:
                drift_dir = drift_dirs[t]
            else:
                drift_dir = positions[t] - positions[1 - t]
                drift_dir /= np.linalg.norm(drift_dir)

            # Instantiate TPC
            self.chambers.append(Chamber(positions[t], dimensions, drift_dir))

        # Initialize the underlying box object
        lower = np.min(np.vstack([c.lower for c in self.chambers]), axis=0)
        upper = np.max(np.vstack([c.upper for c in self.chambers]), axis=0)
        super().__init__(lower, upper)

    @property
    def num_chambers(self):
        """Number of individual TPCs that make up this module.

        Returns
        -------
        int
            Number of TPCs in the module
        """
        return len(self.chambers)

    @property
    def drift_axis(self):
        """Drift axis for the module (shared between chambers).

        Returns
        -------
        int
            Axis along which electrons drift in this module (0, 1 or 2)
        """
        return self.chambers[0].drift_axis

    @property
    def cathode_pos(self):
        """Location of the cathode plane along the drift axis.

        Returns
        -------
        float
            Location of the cathode plane along the drift axis
        """
        return np.mean([c.cathode_pos for c in self.chambers])

    def __len__(self):
        """Returns the number of TPCs in the module.

        Returns
        -------
        int
            Number of TPCs in the module
        """
        return self.num_chambers

    def __getitem__(self, idx):
        """Returns an underlying TPC of index idx.

        Parameters
        ----------
        idx : int
            Index of the TPC within the module

        Returns
        -------
        Chamber
            Chamber object
        """
        return self.chambers[idx]

    def __iter__(self):
        """Resets an iterator counter, return self.

        Returns
        -------
        Module
            The module itself
        """
        self._counter = 0
        return self

    def __next__(self):
        """Defines how to process the next TPC in the module.

        Returns
        -------
        Chamber
            Next Chamber instance in the list
        """
        # If there are more TPCs to go through, return it
        if self._counter < len(self):
            tpc = self.chambers[self._counter]
            self._counter += 1

            return tpc

        raise StopIteration


@dataclass
class TPCDetector(Box):
    """Handles all geometry queries for a set of time-projection chambers.

    Attributes
    ----------
    modules : List[Module]
        (N_m) List of TPC modules associated with this detector
    chambers : List[Chamber]
        (N_t) List of individual TPC associated with this detector
    det_ids : np.ndarray, optional
        (N_c) Map between logical and physical TPC index
    """
    modules : List[Module]
    chambers: List[Chamber]
    det_ids : np.ndarray = None

    def __init__(self, dimensions, positions, module_ids, det_ids=None,
                 drift_dirs=None):
        """Parse the detector boundary configuration.

        Parameters
        ----------
        dimensions : List[float]
            (3) Dimensions of one TPC
        positions : List[List[float]]
            (N_t) List of TPC center positions, one per TPC
        module_ids : List[int]
            (N_t) List of the module IDs each TPC belongs to
        det_ids : List[int], optional
            (N_c) Index of the physical detector which corresponds to each
            logical ID. This is needed if a TPC is divided into multiple logical
            IDs. If this is not specified, it assumed that there is a one-to-one
            correspondance between logical and physical.
        drift_dirs : List[List[float]], optional
            (N_t) List of drift direction vectors. If this is not provided, it
            is inferred from the module configuration, provided that modules
            are composed of two TPCs (with a shared cathode)
        """
        # Check the sanity of the configuration
        assert len(dimensions) == 3, (
                "Should provide the TPC dimension along 3 dimensions.")
        assert np.all([len(pos) == 3 for pos in positions]), (
                "Must provide the TPC position along 3 dimensions.")
        assert len(module_ids) == len(positions), (
                "Must provide one module ID for each TPC.")

        # Cast the dimensions, positions, ids to arrays
        dimensions = np.asarray(dimensions)
        positions = np.asarray(positions)
        module_ids = np.asarray(module_ids, dtype=int)

        # Construct TPC chambers, organized by module
        self.modules = []
        self.chambers = []
        for m in np.unique(module_ids):
            # Narrow down the set of TPCs to those in this module
            module_index = np.where(module_ids == m)[0]
            module_positions = positions[module_index]
            module_drift_dirs = None
            if drift_dirs is not None:
                module_drift_dirs = drift_dirs[module_index]

            # Initialize the module, store
            module = Module(module_positions, dimensions, module_drift_dirs)
            self.modules.append(module)
            self.chambers.extend(module.chambers)

        # Check that if detector IDs are provided, they are comprehensive
        if det_ids is not None:
            self.det_ids = np.asarray(det_ids, dtype=int)
            assert len(np.unique(det_ids)) == self.num_chambers_per_module, (
                "All physical TPCs must be associated with at least one "
                "logical TPC.")

        # Initialize the underlying all-encompasing box object
        lower = np.min(np.vstack([m.lower for m in self.modules]), axis=0)
        upper = np.max(np.vstack([m.upper for m in self.modules]), axis=0)
        super().__init__(lower, upper)

    @property
    def num_chambers(self):
        """Number of individual TPC voulmes.

        Returns
        -------
        int
            Number of TPC volumes, N_t
        """
        return len(self.chambers)

    @property
    def num_modules(self):
        """Number of detector modules.

        Returns
        -------
        int
            Number of detector modules, N_m
        """
        return len(self.modules)

    @property
    def num_chambers_per_module(self):
        """Number of TPC volumes per module.

        Returns
        -------
        int
            Number of TPC volumes per module, N_t
        """
        return len(self.modules[0])

    def __len__(self):
        """Returns the number of modules in the detector.

        Returns
        -------
        int
            Number of modules in the detector
        """
        return self.num_modules

    def __getitem__(self, idx):
        """Returns an underlying module or TPC, depending on the index type.

        If the index is specified as a simple integer, a module is returned. If
        the index is specified with two integers, a specific chamber within a
        module is returned instead.

        Parameters
        ----------
        idx : Uniont[int, List[int]]
            Module index or pair of [module ID, chamber ID]

        Returns
        -------
        Union[Module, Chamber]
            Module or Chamber object
        """
        if np.isscalar(idx):
            return self.modules[idx]

        else:
            module_id, chamber_id = idx
            return self.modules[module_id].chambers[chamber_id]

    def __iter__(self):
        """Resets an iterator counter, return self.

        Returns
        -------
        TPCDetector
            The module itself
        """
        self._counter = 0
        return self

    def __next__(self):
        """Defines how to process the next Module in the detector.

        Returns
        -------
        Module
            Next Module instance in the list
        """
        # If there are more TPCs to go through, return it
        if self._counter < len(self):
            module = self.modules[self._counter]
            self._counter += 1

            return module

        raise StopIteration
