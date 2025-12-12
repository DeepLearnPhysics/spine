"""Basic detector components shared across multiple subsystems.

This currently handles:
- :class:`Box` which corresponds to box-shaped detector modules.
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np

__all__ = ["Plane", "Box"]


@dataclass
class Plane:
    """Class which holds all methods associated with a plane.

    Attributes
    ----------
    intercept : np.ndarray
        (3,) Coordinates of a point which belongs to the plane
    norm : np.ndarray
        (3,) Vector perpendicular to the plane
    boundary : float
        Dot product between the interecept and the norm
    """

    intercept: np.ndarray
    norm: np.ndarray
    boundary: float

    def __init__(self, intercept: np.ndarray, norm: np.ndarray):
        """Initialize the box object.

        Parameters
        ----------
        intercept : np.ndarray
            (3,) Coordinates of a point which belongs to the plane
        norm : np.ndarray
            (3,) Vector perpendicular to the plane
        """
        # Store the intercept and norm
        self.intercept = intercept
        self.norm = norm

        # Compute their dot product (projection boundary)
        self.boundary = np.dot(intercept, norm)

    def distance(self, point: np.ndarray) -> float:
        """Computes the perpendicular distance from a point to the plane.

        Parameters
        ----------
        point : np.ndarray
            (3,) Coordinates of the point to compute the distance to

        Returns
        -------
        float
            Perpendicular distance from the point to the plane
        """
        return (np.dot(point, self.norm) - self.boundary) / np.linalg.norm(self.norm)


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

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        """Initialize the box object.

        Parameters
        ----------
        lower : np.ndarray
            (3,) Lower bounds of the box
        upper : np.ndarray
            (3,) Upper bounds of the box
        """
        # Store lower and upper boundaries in one array
        self.boundaries = np.vstack((lower, upper)).T

    @property
    def center(self) -> np.ndarray:
        """Center of the box.

        Returns
        -------
        np.ndarray
            (3,) Center of the box
        """
        return np.mean(self.boundaries, axis=1)

    @property
    def lower(self) -> np.ndarray:
        """Lower bounds of the box.

        Returns
        -------
        np.ndarray
            (3,) Lower bounds of the box
        """
        return self.boundaries[:, 0]

    @property
    def upper(self) -> np.ndarray:
        """Upper bounds of the box.

        Returns
        -------
        np.ndarray
            (3,) Upper bounds of the box
        """
        return self.boundaries[:, 1]

    @property
    def dimensions(self) -> np.ndarray:
        """Dimensions of the box.

        Returns
        -------
        np.ndarray
            (3,) Box dimensions
        """
        return self.boundaries[:, 1] - self.boundaries[:, 0]

    @property
    def volume(self) -> float:
        """Volume of the box.

        Returns
        -------
        float
            Box volume
        """
        return float(np.prod(self.dimensions))

    @property
    def faces(self) -> List[Plane]:
        """Returns the 6 faces of the box as Plane objects.

        Returns
        -------
        List[Plane]
            List of 6 Plane objects corresponding to the box faces
        """
        faces = []
        # Loop over each axis
        for axis in range(3):
            # Create normal vector for the face
            norm = np.zeros(3)
            norm[axis] = 1.0

            # Lower face
            intercept_lower = self.lower.copy()
            faces.append(Plane(intercept_lower, norm))

            # Upper face
            intercept_upper = self.upper.copy()
            faces.append(Plane(intercept_upper, -norm))

        return faces

    def distance(self, points: np.ndarray) -> Union[float, np.ndarray]:
        """Computes the minimum distance from a set of points to the box.

        If the point(s) is(are) inside the box, the distance is 0.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Coordinates of the points to compute the distance to

        Returns
        -------
        np.ndarray
            (N,) Minimum distance from each point to the box
        """
        # Distance from each point to the lower and upper boundaries
        diff_lower = self.lower - points  # positive if point is below lower bound
        diff_upper = points - self.upper  # positive if point is above upper bound

        # For each coord, if inside the interval, contribution is 0;
        # if outside, take the amount by which it's outside.
        delta = np.maximum(0.0, np.maximum(diff_lower, diff_upper))  # shape (N, 3)

        # Euclidean distance
        if len(delta.shape) == 1:
            distances = float(np.linalg.norm(delta))
        else:
            distances = np.linalg.norm(delta, axis=1)

        return distances
