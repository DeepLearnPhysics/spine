"""Rotation augmentation module."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spine.data import Meta

from .base import AugmentBase


class RotateAugment(AugmentBase):
    """Generic class to handle right-angle image rotations."""

    name = "rotate"

    def __init__(
        self,
        axes: Tuple[int, int] = (0, 1),
        k: Optional[int] = None,
    ) -> None:
        """Initialize the rotater.

        Parameters
        ----------
        axes : Tuple[int, int], default (0, 1)
            Pair of axes defining the plane in which to rotate
        k : int, optional
            Number of 90-degree turns to apply. If not provided, sample
            uniformly from 0 to 3 at call time

        Returns
        -------
        None
            This method does not return anything
        """
        if len(axes) != 2:
            raise ValueError("Must provide exactly two rotation axes.")
        if axes[0] == axes[1]:
            raise ValueError("Rotation axes must be different.")
        if np.any(np.asarray(axes) < 0) or np.any(np.asarray(axes) > 2):
            raise ValueError("Rotation axes must be in the range [0, 2].")
        if k is not None and not isinstance(k, (int, np.integer)):
            raise ValueError("Rotation `k` must be an integer number of quarter turns.")

        self.axes = tuple(axes)
        self.k = None if k is None else int(k) % 4

    def apply(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Meta]:
        """Rotate the image by quarter turns in the requested plane.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to augment
        meta : Meta
            Shared image metadata
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        context : dict
            Shared augmentation context

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and rotated metadata
        """
        k = self.sample_k()
        if k == 0:
            return data, meta

        rot_meta = self.generate_meta(meta, k)
        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = rot_meta
                continue

            coords = data[key].coords.copy()
            coords = self.rotate_coords(coords, meta.count, k).astype(
                data[key].coords.dtype
            )
            data[key].coords = coords
            data[key].meta = rot_meta

        return data, rot_meta

    def sample_k(self) -> int:
        """Sample the number of quarter turns to apply.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of 90-degree turns to apply
        """
        if self.k is not None:
            return self.k

        return int(np.random.randint(4))

    def rotate_coords(
        self, coords: np.ndarray, count: np.ndarray, k: int
    ) -> np.ndarray:
        """Rotate voxel coordinates by quarter turns.

        Parameters
        ----------
        coords : np.ndarray
            Voxel coordinates to rotate
        count : np.ndarray
            Original voxel counts along each axis
        k : int
            Number of 90-degree turns to apply

        Returns
        -------
        np.ndarray
            Rotated voxel coordinates
        """
        rot_coords = coords.copy()
        axis_a, axis_b = self.axes
        count_a = int(count[axis_a])
        count_b = int(count[axis_b])

        if k == 1:
            rot_coords[:, axis_a] = count_b - 1 - coords[:, axis_b]
            rot_coords[:, axis_b] = coords[:, axis_a]
        elif k == 2:
            rot_coords[:, axis_a] = count_a - 1 - coords[:, axis_a]
            rot_coords[:, axis_b] = count_b - 1 - coords[:, axis_b]
        elif k == 3:
            rot_coords[:, axis_a] = coords[:, axis_b]
            rot_coords[:, axis_b] = count_a - 1 - coords[:, axis_a]

        return rot_coords

    def generate_meta(self, meta: Meta, k: int) -> Meta:
        """Generate the metadata for the rotated image.

        Parameters
        ----------
        meta : Meta
            Metadata of the image before rotation
        k : int
            Number of 90-degree turns to apply

        Returns
        -------
        Meta
            Metadata of the rotated image
        """
        count = meta.count.copy()
        size = meta.size.copy()
        if k % 2:
            axis_a, axis_b = self.axes
            count[[axis_a, axis_b]] = count[[axis_b, axis_a]]
            size[[axis_a, axis_b]] = size[[axis_b, axis_a]]

        lower = meta.lower.copy()
        upper = lower + size * count
        return Meta(lower=lower, upper=upper, size=size, count=count)
