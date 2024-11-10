"""CRT detector geometry classes."""

from typing import List
from dataclasses import dataclass

import numpy as np

__all__ = ['CRTDetector']


@dataclass
class CRTDetector:
    """Handles all geometry queries for a set of cosmic-ray taggers.

    Attributes
    ----------
    positions : np.ndarray
        (N_c, 3) Location of the center of each of the CRT planes
        - N_c is the number of CRT planes
    dimensions : np.ndarray
        (N_c, 3) Dimensions of each of the CRT planes
        - N_c is the number of CRT planes
    norms : np.ndarray
        (N_c) Axis aligned with the norm of each of the CRT planes
        - N_c is the number of CRT planes
    det_ids : Dict[int, int], optional
        Mapping between the CRT channel and its corresponding detector
    """
    positions : np.ndarray
    dimensions : np.ndarray
    norms : np.ndarray
    det_ids : dict = None

    def __init__(self, dimensions, positions, norms, logical_ids=None):
        """Parse the CRT detector configuration.

        The assumption here is that the CRT detectors collectively cover the
        entire detector, regardless of TPC/optical modularization.

        Parameters
        ----------
        dimensions : List[List[float]]
            Dimensions of each of the CRT plane
        positions : List[List[float]]
            Positions of each of the CRT plane
        norms : List[int]
            Axis along which each of the CRT plane norm is pointing
        logical_ids : List[int], optional
            Logical index corresponding to each CRT channel. If not specified,
            it is assumed that there is a one-to-one correspondance between
            physical and logical CRT planes.
        """
        # Check the sanity of the configuration
        assert len(positions) == len(dimensions), (
                 "Must provide the dimensions of each of the CRT element. "
                f"Got {len(dimensions)}, but expected {len(positions)}.")
        assert logical_ids is None or len(logical_ids) == len(positions), (
                 "Must provide the logical ID of each of the CRT element. "
                f"Got {len(logical_ids)}, but expected {len(positions)}.")

        # Store CRT detector parameters
        self.positions = np.asarray(positions)
        self.dimensions = np.asarray(dimensions)
        self.norms = np.asarray(norms, dtype=int)
        if logical_ids is not None:
            self.det_ids = {idx: i for i, idx in enumerate(logical_ids)}

    @property
    def num_detectors(self):
        """Returns the number of CRT planes around the detector.

        Returns
        -------
        int
            Number of CRT planes, N_c
        """
        return self.positions.shape[0]
