"""Optical detector geometry classes."""

from typing import List
from dataclasses import dataclass

import numpy as np

__all__ = ['OptDetector']


@dataclass
class OptDetector:
    """Handles all geometry queries for a set of optical detectors.

    Attributes
    ----------
    volume : str
        The boundaries of each optical volume ('tpc' or 'module'), as defined
        by the the set of PMTs in each volume
    positions : np.ndarray
        (N_v, N_o, 3) Location of the center of each of the optical detectors
        - N_v is the number of optical volumes
        - N_o is the number of optical detectors in each volume
    shape : List[str]
        (N_d) Optical detector shape(s), combination of 'ellipsoid' and 'box'
        - N_d is the number of detector types
    dimensions : np.ndarray
        (N_d, 3) Dimensions of each of the optical detector types
        - N_d is the number of detector types
    shape_ids : np.ndarray, optional
        (N_o) Type of each of the optical detectors
        - N_o is the number of optical detectors
    det_ids : np.ndarray, optional
        (N_c) Mapping between the optical channel and its corresponding detector
        - N_c is the number of optical channels (this number can be larger
        than the number of detectors if e.g. multiple SiPMs are used per
        optical detector)
    """
    volume: str
    positions: np.ndarray
    shape: list
    dimensions: np.ndarray
    shape_ids: np.ndarray = None
    det_ids: np.ndarray = None

    def __init__(self, volume, volume_offsets, shape, dimensions, positions,
                 shape_ids=None, det_ids=None, global_index=False, mirror=False):
        """Parse the optical detector configuration.

        Parameters
        ----------
        volume : str
            Optical decteor segmentation (per 'tpc' or per 'module')
        volume_offsets : np.ndarray
            Offsets of the optical volumes w.r.t. to the origin
        shape : Union[str, List[str]]
            Optical detector geomtry (combination of 'ellipsoid' and/or 'box')
        dimensions : Union[List[float], List[List[float]]]]
            (N_o, 3) List of optical detector dimensions along each axis
            (either the ellipsoid axis lenghts or the box edge lengths)
        positions : List[List[float]]
            (N_o, 3) Positions of each of the optical detectors
        shape_ids : List[int], optional
            (N_o) If there is different types of optical detectors, specify
            which type each of the index corresponds to
        det_ids : List[int], optional
            (N_c) If there are multiple readout channels which contribute to each
            physical optical detector, map each channel onto a physical detector
        global_index : bool, default False
            If `True`, the flash objects have a `pe_per_ch` attribute which refers
            to the entire index of optical detectors, rather than one volume
        mirror : bool, default False
            If True, mirror the z positons of the optical modules in the second
            TPC of each module
        """
        # Parse the detector shape(s) and its mapping, store is as a list
        assert (shape in ['ellipsoid', 'box'] or
                np.all([s in ['ellipsoid', 'box'] for s in shape])), (
                "The shape of optical detectors must be represented as either "
                "an 'ellipsoid' or a 'box', or a list of them.")
        assert isinstance(shape, str) or shape_ids is not None, (
                "If mutiple shapes are provided, must provide a shape map.")
        assert shape_ids is None or len(shape_ids) == len(positions), (
                "Must provide a shape index for each optical channel.")

        self.shape = shape
        if isinstance(shape, str):
            self.shape = [shape]
        self.shape_ids = shape_ids
        if shape_ids is not None:
            self.shape_ids = np.asarray(shape_ids, dtype=int)

        # Parse the detector dimensions, store as a 2D array
        self.dimensions = np.asarray(dimensions).reshape(-1, 3)
        assert len(self.dimensions) == len(self.shape), (
                "Must provide optical detector dimensions for each shape.")

        # Store remaining optical detector parameters
        self.volume = volume
        self.det_ids = det_ids
        if det_ids is not None:
            self.det_ids = np.asarray(det_ids, dtype=int)

        # Parse the relative optical detector posiitons
        rel_positions = np.asarray(positions)
        if mirror:
            rel_positions_m = np.copy(rel_positions)
            rel_positions_m[:, -1] = -rel_positions_m[:, -1]

        # Store the optical detector positions in each optical volume
        count = len(positions)
        offsets = np.asarray(volume_offsets)
        self.positions = np.empty((len(offsets), count, 3))
        for v in range(len(offsets)):
            if mirror and v%2 != 0:
                self.positions[v] = rel_positions_m + offsets[v]
            else:
                self.positions[v] = rel_positions + offsets[v]

        # Store if the flash points to the entire index of optical detectors
        self.global_index = global_index

    @property
    def num_volumes(self):
        """Returns the number of optical volumes.

        Returns
        -------
        int
            Number of optical volumes, N_v
        """
        return self.positions.shape[0]

    @property
    def num_detectors_per_volume(self):
        """Returns the number of optical detectors in each optical volume.

        Returns
        -------
        int
            Number of optical detectors in each volume, N_o
        """
        return self.positions.shape[1]

    @property
    def num_detectors(self):
        """Number of optical detectors.

        Returns
        -------
        int
            Total number of optical detector, N_v*N_o
        """
        return self.num_volumes*self.num_detectors_per_volume

    def volume_index(self, volume_id):
        """Returns an index which corresponds to detectors in a certain volume.

        Parameters
        ----------
        volume_id : int
            ID of the volume to return the index for

        Returns
        -------
        np.ndarray
            Index of the detectors which belong to the requested volume ID
        """
        # If using a global index, all volumes point to the same index
        if self.global_index:
            return np.arange(self.num_detectors)

        return (volume_id*self.num_detectors_per_volume +
                np.arange(self.num_detectors_per_volume))
