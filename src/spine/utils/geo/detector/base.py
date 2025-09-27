"""Basic detector components shared across multiple subsystems.

This currently handles:
- :class:`Box` which corresponds to box-shaped detector modules.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Box", "Plane"]


@dataclass
class Box:
    """Class which holds all methods associated with a box-shaped component.

    Attributes
    ----------
    boundaries : np.ndarray
        (3, 2) Box boundaries
        - 3 is the number of dimensions
        - 2 corresponds to the lower/upper boundaries along each axis
    """

    boundaries: np.ndarray

    def __init__(self, lower, upper):
        """Initialize the box object.

        Parameters
        ----------
        lower : np.ndarray
            (3) Lower bounds of the box
        upper : np.ndarray
            (3) Upper bounds of the box
        """
        # Store lower and upper boundaries in one array
        self.boundaries = np.vstack((lower, upper)).T

    @property
    def center(self):
        """Center of the box.

        Returns
        -------
        np.ndarray
            Center of the box
        """
        return np.mean(self.boundaries, axis=1)

    @property
    def lower(self):
        """Lower bounds of the box.

        Returns
        -------
        np.ndarray
            Lower bounds of the box
        """
        return self.boundaries[:, 0]

    @property
    def upper(self):
        """Upper bounds of the box.

        Returns
        -------
        np.ndarray
            Upper bounds of the box
        """
        return self.boundaries[:, 1]

    @property
    def dimensions(self):
        """Dimensions of the box.

        Returns
        -------
        np.ndarray
            Box dimensions
        """
        return self.boundaries[:, 1] - self.boundaries[:, 0]


@dataclass
class Plane:
    """Class which holds all methods associated with a plane.

    Attributes
    ----------
    intercept : np.ndarray
        (3) Coordinates of a point which belongs to the plane
    norm : np.ndarray
        (3) Vector perpendicular to the plane
    boundary : float
        Dot product between the interecept and the norm
    """

    intercept: np.ndarray
    norm: np.ndarray
    boundary: float

    def __init__(self, intercept, norm):
        """Initialize the box object.

        Parameters
        ----------
        intercept : np.ndarray
            (3) Coordinates of a point which belongs to the plane
        norm : np.ndarray
            (3) Vector perpendicular to the plane
        """
        # Store the intercept and norm
        self.intercept = np.asarray(intercept)
        self.norm = np.asarray(norm)

        # Compute their dot product (projection boundary)
        self.boundary = np.dot(intercept, norm)
