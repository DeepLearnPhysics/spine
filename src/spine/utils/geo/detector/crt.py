"""CRT detector geometry classes."""

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import Box

__all__ = ["CRTDetector"]


@dataclass
class CRTPlane(Box):
    """Class which holds all properties of an individual CRT plane.

    Attributes
    ----------
    normal : np.ndarray
        (3) Vector normal to the CRT plane
    normal_axis : int
        Axis along which the normal vector is pointing
    """

    normal: np.ndarray
    normal_axis: int

    def __init__(self, position, dimensions, normal_axis):
        """Initialize the CRT plane object.

        Parameters
        ----------
        position : np.ndarray
            (3) Position of the center of the TPC
        dimensions : np.ndarray
            (3) Dimension of the TPC
        normal_axis : int
            Axis along which the normal vector of this CRT plane is pointing
        """
        # Initialize the underlying box object
        lower = position - dimensions / 2
        upper = position + dimensions / 2
        super().__init__(lower, upper)

        # Store the normal vector
        self.normal = np.zeros(3, dtype=position.dtype)
        self.normal[normal_axis] = 1.0
        self.normal_axis = normal_axis


@dataclass
class CRTDetector(Box):
    """Handles all geometry queries for a set of cosmic-ray taggers.

    Attributes
    ----------
    planes : List[CRTPlane]
        (N_c) List of CRT planes associated with this detector
    det_ids : Dict[int, int], optional
        Mapping between the CRT channel and its corresponding detector
    """

    planes: List[CRTPlane]
    det_ids: dict = None

    def __init__(self, dimensions, positions, normals, logical_ids=None):
        """Parse the CRT detector configuration.

        The assumption here is that the CRT detectors collectively cover the
        entire detector, regardless of TPC/optical modularization.

        Parameters
        ----------
        dimensions : List[List[float]]
            (N_c, 3) Dimensions of each of the CRT plane
        positions : List[List[float]]
            (N_c, 3) Positions of each of the CRT plane
        normals : List[int]
            (N_c, 3) Axes along which the normal vectors of each CRT plane is pointing
        logical_ids : List[int], optional
            (N_c) Logical index corresponding to each CRT channel. If not
            specified, it is assumed that there is a one-to-one correspondance
            between physical and logical CRT planes.
        """
        # Check the sanity of the configuration
        assert len(positions) == len(dimensions), (
            "Must provide the dimensions of each of the CRT element. "
            f"Got {len(dimensions)}, but expected {len(positions)}."
        )
        assert logical_ids is None or len(logical_ids) == len(positions), (
            "Must provide the logical ID of each of the CRT element. "
            f"Got {len(logical_ids)}, but expected {len(positions)}."
        )

        # Cast the dimensions, positions, normal to arrays
        positions = np.asarray(positions)
        dimensions = np.asarray(dimensions)
        normals = np.asarray(normals, dtype=int)

        # Construct the CRT planes
        self.planes = []
        for i in range(len(positions)):
            plane = CRTPlane(positions[i], dimensions[i], normals[i])
            self.planes.append(plane)

        # Store CRT detector parameters
        if logical_ids is not None:
            self.det_ids = {idx: i for i, idx in enumerate(logical_ids)}
        else:
            self.det_ids = np.arange(len(positions), dtype=int)

        # Initialize the underlying all-encompasing box object
        lower = np.min(np.vstack([p.lower for p in self.planes]), axis=0)
        upper = np.max(np.vstack([p.upper for p in self.planes]), axis=0)
        super().__init__(lower, upper)

    @property
    def num_planes(self):
        """Returns the number of CRT planes around the detector.

        Returns
        -------
        int
            Number of CRT planes, N_c
        """
        return len(self.planes)

    def __len__(self):
        """Returns the number of CRT planes in the detector.

        Returns
        -------
        int
            Number of CRT planes, N_c
        """
        return self.num_planes

    def __getitem__(self, idx):
        """Returns an underlying CRT plane.

        Parameters
        ----------
        idx : int
            CRT plane index

        Returns
        -------
        CRTPlane
            CRTPlane object at that index
        """
        return self.planes[idx]

    def __iter__(self):
        """Resets an iterator counter, return self.

        Returns
        -------
        CRTDetector
            The detector itself
        """
        self._counter = 0
        return self

    def __next__(self):
        """Defines how to process the next CRTPlane in the detector.

        Returns
        -------
        CRTPlane
            Next CRTPlane instance in the list
        """
        # If there are more TPCs to go through, return it
        if self._counter < len(self):
            plane = self.planes[self._counter]
            self._counter += 1

            return plane

        raise StopIteration
