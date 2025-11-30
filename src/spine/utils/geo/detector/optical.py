"""Optical detector geometry classes."""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from .base import Box

__all__ = ["OptDetector"]


@dataclass
class OpticalVolume(Box):
    """Class which holds all properties of an individual optical volume.

    Attributes
    ----------
    centroid : np.ndarray
        (3,) Position of the centroid of the optical volume
    positions : np.ndarray
        (N_o, 3) Location of the center of each of the optical detectors
        - N_o is the number of optical detectors in the volume
    shape : Union[str, List[str]]
        (N_d) Optical detector shape(s), combination of 'ellipsoid', 'box' and 'disk'
        - N_d is the number of detector types
    sizes : np.ndarray
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

    centroid: np.ndarray
    positions: np.ndarray
    sizes: np.ndarray
    shape: Union[str, List[str]]
    shape_ids: Optional[np.ndarray] = None
    det_ids: Optional[np.ndarray] = None

    def __init__(
        self,
        centroid,
        positions,
        sizes,
        shape,
        shape_ids=None,
        det_ids=None,
    ):
        """Initialize the optical volume.

        Parameters
        ----------
        centroid : np.ndarray
            (3,) Position of the centroid of the optical volume
        positions : np.ndarray
            (N_o, 3) Location of the center of each of the optical detectors
            - N_o is the number of optical detectors in the volume
        sizes : np.ndarray
            (N_d, 3) Dimensions of each of the optical detector types
            - N_d is the number of detector types
        shape : Union[str, List[str]]
            Optical detector shape(s), combination of 'ellipsoid', 'box' and 'disk'
            - N_d is the number of detector types
        shape_ids : np.ndarray, optional
            (N_o) Type of each of the optical detectors
            - N_o is the number of optical detectors
        det_ids : np.ndarray, optional
            (N_c) Mapping between the optical channel and its corresponding detector
            - N_c is the number of optical channels (this number can be larger
        """
        # Store the optical volume properties
        self.centroid = np.asarray(centroid)
        self.positions = np.asarray(positions)
        self.sizes = np.asarray(sizes)
        self.shape = shape
        if shape_ids is not None:
            self.shape_ids = np.asarray(shape_ids, dtype=int)
        if det_ids is not None:
            self.det_ids = np.asarray(det_ids, dtype=int)

        # Setup the overall boundaries of the optical volume
        shape_ids = (
            self.shape_ids
            if self.shape_ids is not None
            else np.zeros(self.num_detectors, dtype=int)
        )
        lower = np.min(
            np.array(
                [
                    position - self.sizes[shape_id] / 2
                    for position, shape_id in zip(self.positions, shape_ids)
                ]
            ),
            axis=0,
        )
        upper = np.max(
            np.array(
                [
                    position + self.sizes[shape_id] / 2
                    for position, shape_id in zip(self.positions, shape_ids)
                ]
            ),
            axis=0,
        )

        # Initialize the base Box object
        super().__init__(lower=lower, upper=upper)

    @property
    def num_detectors(self):
        """Number of optical detectors.

        Returns
        -------
        int
            Total number of optical detector, N_o
        """
        return self.positions.shape[0]


@dataclass
class OptDetector(Box):
    """Handles all geometry queries for a set of optical detectors.

    Attributes
    ----------
    volumes: List[OpticalVolume]
        List of optical volumes in the detector
    segmentation : str
        The level of optical detector segmentation ('tpc' or 'module')
    global_index : bool
        If `True`, the flash objects have a `pe_per_ch` attribute which refers
        to the entire index of optical detectors, rather than one volume
    """

    volumes: List[OpticalVolume]
    segmentation: str
    global_index: bool

    def __init__(
        self,
        volume,
        volume_offsets,
        shape,
        dimensions,
        positions,
        shape_ids=None,
        det_ids=None,
        global_index=False,
        mirror=False,
    ):
        """Parse the optical detector configuration.

        Parameters
        ----------
        volume : str
            Optical decteor segmentation (per 'tpc' or per 'module')
        volume_offsets : np.ndarray
            Offsets of the optical volumes w.r.t. to the origin
        shape : Union[str, List[str]]
            Optical detector geomtry (combination of 'ellipsoid', 'box' and/or 'disk')
        dimensions : Union[List[float], List[List[float]]]]
            (N_o, 3) List of optical detector dimensions along each axis
            (either the ellipsoid axis lenghts, box edge lengths or disk lengths)
        positions : List[List[float]]
            (N_o, 3) Relative positions of each of the optical detectors with respect
            to the optical volume centroid
        shape_ids : List[int], optional
            (N_o) If there are different types of optical detectors, specify
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
        # Store the detector optical segmentation
        assert volume in (
            "tpc",
            "module",
        ), "The optical volume segmentation must be either 'tpc' or 'module'."
        self.segmentation = volume

        # Parse the detector shape(s) and its mapping
        assert shape in ("ellipsoid", "box", "disk") or np.all(
            [s in ("ellipsoid", "box", "disk") for s in shape]
        ), (
            "The shape of optical detectors must be represented as either "
            "an 'ellipsoid', a 'box', a 'disk', or a list of them."
        )
        assert (
            isinstance(shape, str) or shape_ids is not None
        ), "If mutiple shapes are provided, must provide a shape map."
        assert shape_ids is None or len(shape_ids) == len(
            positions
        ), "Must provide a shape index for each optical channel."

        if isinstance(shape, str):
            shape = [shape]
        if shape_ids is not None:
            shape_ids = np.asarray(shape_ids, dtype=int)

        # Parse the detector dimensions, store as a 2D array
        sizes = np.asarray(dimensions).reshape(-1, 3)
        assert len(sizes) == len(
            shape
        ), "Must provide optical detector sizes for each shape."

        # Store remaining optical detector parameter
        if det_ids is not None:
            assert len(det_ids) == len(
                positions
            ), "If provided, must provide a detector ID for each optical channel."
            det_ids = np.asarray(det_ids, dtype=int)

        # Parse the relative optical detector posiitons
        rel_positions = np.asarray(positions)
        rel_positions_m = np.copy(rel_positions)
        if mirror:
            rel_positions_m[:, -1] = -rel_positions_m[:, -1]

        # Create an list of optical volumes
        offsets = np.asarray(volume_offsets)
        self.volumes = []
        for v, offset in enumerate(offsets):
            # Extract positions for this volume
            if mirror and v % 2 != 0:
                positions = rel_positions_m + offset
            else:
                positions = rel_positions + offset

            # Insert volume
            self.volumes.append(
                OpticalVolume(
                    centroid=offset,
                    positions=positions,
                    sizes=sizes,
                    shape=shape,
                    shape_ids=shape_ids,
                    det_ids=det_ids,
                )
            )

        # Store if the flash points to the entire index of optical detectors
        self.global_index = global_index

        # Initialize the base Box object
        lower = np.min(np.array([volume.lower for volume in self.volumes]), axis=0)
        upper = np.max(np.array([volume.upper for volume in self.volumes]), axis=0)

        super().__init__(lower=lower, upper=upper)

    @property
    def num_volumes(self):
        """Returns the number of optical volumes.

        Returns
        -------
        int
            Number of optical volumes, N_v
        """
        return len(self.volumes)

    @property
    def num_detectors_per_volume(self):
        """Returns the number of optical detectors in each optical volume.

        Returns
        -------
        int
            Number of optical detectors in each volume, N_o
        """
        return self.volumes[0].num_detectors

    @property
    def num_detectors(self):
        """Number of optical detectors.

        Returns
        -------
        int
            Total number of optical detector, N_v*N_o
        """
        return self.num_volumes * self.num_detectors_per_volume

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

        return volume_id * self.num_detectors_per_volume + np.arange(
            self.num_detectors_per_volume
        )

    @property
    def positions(self):
        """Returns the positions of all optical detectors.

        Returns
        -------
        np.ndarray
            (N_v, N_o, 3) Positions of all optical detectors in each volume
            - N_v is the number of optical volumes
            - N_o is the number of optical detectors per volume
        """
        positions = []
        for volume in self.volumes:
            positions.append(volume.positions)

        return np.vstack(positions)

    @property
    def sizes(self):
        """Returns the sizes of all optical detectors.

        Returns
        -------
        np.ndarray
            (N_d, 3) Sizes of each of the optical detector types
            - N_d is the number of detector types
        """
        return self.volumes[0].sizes

    @property
    def shape(self):
        """Returns the shape of all optical detectors.

        Returns
        -------
        Union[str, List[str]]
            Shape(s) of the optical detectors
        """
        return self.volumes[0].shape

    @property
    def shape_ids(self):
        """Returns the shape IDs of all optical detectors.

        Returns
        -------
        Optional[np.ndarray]
            (N_v, N_o) Shape IDs of all optical detectors in each volume
            - N_v is the number of optical volumes
            - N_o is the number of optical detectors per volume
        """
        # If all volumes have the same shape IDs, return None
        if self.volumes[0].shape_ids is None:
            return None

        # Otherwise, stack the shape IDs from each volume
        shape_ids = []
        for volume in self.volumes:
            shape_ids.append(volume.shape_ids)

        return np.concatenate(shape_ids)

    @property
    def det_ids(self):
        """Returns the detector IDs of all optical detectors.

        Returns
        -------
        Optional[np.ndarray]
            (N_c) Mapping between the optical channel and its corresponding detector
            - N_c is the number of optical channels
        """
        # If theere is only one volume, return its det IDs
        if self.num_volumes == 1 or self.volumes[0].det_ids is None:
            return self.volumes[0].det_ids

        # Otherwise, offset the det IDs from each volume by the number of detectors
        det_ids = []
        n_detectors = self.num_detectors_per_volume
        for v, volume in enumerate(self.volumes):
            det_ids.append(volume.det_ids + v * n_detectors)

        return np.concatenate(det_ids)
