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
        center: Optional[np.ndarray] = None,
        use_geo_center: bool = False,
        keep_meta: bool = True,
    ) -> None:
        """Initialize the rotater.

        Parameters
        ----------
        axes : Tuple[int, int], default (0, 1)
            Pair of axes defining the plane in which to rotate
        k : int, optional
            Number of 90-degree turns to apply. If not provided, sample
            uniformly from 0 to 3 at call time
        center : np.ndarray, optional
            Explicit rotation center in detector coordinates (cm). If not
            provided, the historical image-frame rotation behavior is used.
        use_geo_center : bool, default False
            If ``True``, rotate about the detector TPC center
        keep_meta : bool, default True
            If ``True``, keep the detector frame fixed and drop points that
            rotate outside the current metadata bounds. If ``False``, rotate
            the image volume together with the points.

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
        self.center = None if center is None else np.asarray(center, dtype=np.float32)
        self.use_geo_center = use_geo_center
        self.keep_meta = keep_meta

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

        if self.center is None and not self.use_geo_center:
            return self.apply_image_frame_rotation(data, meta, keys, k)

        pivot = self.resolve_center(meta, self.center, self.use_geo_center)
        rot_meta = (
            meta if self.keep_meta else self.generate_centered_meta(meta, pivot, k)
        )

        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = rot_meta
                continue

            coords_cm = self.voxel_to_cm(data[key].coords, meta)
            rot_cm = self.rotate_points(coords_cm, pivot, k)
            if self.keep_meta:
                keep_mask = rot_meta.inner_mask(rot_cm)
                rot_cm = rot_cm[keep_mask]
                data[key].features = data[key].features[keep_mask]
            coords = self.cm_to_voxel(rot_cm, rot_meta, data[key].coords.dtype)
            data[key].coords = coords
            data[key].meta = rot_meta

        return data, rot_meta

    def generate_centered_meta(self, meta: Meta, pivot: np.ndarray, k: int) -> Meta:
        """Generate metadata for a rotation about an explicit pivot.

        Parameters
        ----------
        meta : Meta
            Metadata of the image before rotation
        pivot : np.ndarray
            ``(3,)`` Rotation center in detector coordinates (cm)
        k : int
            Number of 90-degree turns to apply

        Returns
        -------
        Meta
            Metadata of the rotated image volume
        """
        count = meta.count.copy()
        size = meta.size.copy()
        if k % 2:
            axis_a, axis_b = self.axes
            count[[axis_a, axis_b]] = count[[axis_b, axis_a]]
            size[[axis_a, axis_b]] = size[[axis_b, axis_a]]

        dimensions = size * count
        meta_center = ((meta.lower + meta.upper) / 2.0).reshape(1, -1)
        new_center = self.rotate_points(meta_center, pivot, k)[0]
        lower = new_center - dimensions / 2.0
        upper = lower + dimensions
        return Meta(
            lower=lower.astype(meta.lower.dtype),
            upper=upper.astype(meta.upper.dtype),
            size=size.astype(meta.size.dtype),
            count=count.astype(meta.count.dtype),
        )

    def apply_image_frame_rotation(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        k: int,
    ) -> Tuple[Dict[str, Any], Meta]:
        """Apply the historical image-frame rotation behavior.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to rotate
        meta : Meta
            Shared image metadata before rotation
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        k : int
            Number of 90-degree turns to apply

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and rotated metadata
        """

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

    def rotate_points(
        self, points: np.ndarray, pivot: np.ndarray, k: int
    ) -> np.ndarray:
        """Rotate detector coordinates by quarter turns around a pivot.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 3)`` Detector coordinates in cm
        pivot : np.ndarray
            ``(3,)`` Rotation center in detector coordinates (cm)
        k : int
            Number of 90-degree turns to apply

        Returns
        -------
        np.ndarray
            ``(N, 3)`` Rotated detector coordinates in cm
        """
        rot_points = points.copy()
        axis_a, axis_b = self.axes
        rel_a = points[:, axis_a] - pivot[axis_a]
        rel_b = points[:, axis_b] - pivot[axis_b]

        if k == 1:
            rot_points[:, axis_a] = pivot[axis_a] - rel_b
            rot_points[:, axis_b] = pivot[axis_b] + rel_a
        elif k == 2:
            rot_points[:, axis_a] = pivot[axis_a] - rel_a
            rot_points[:, axis_b] = pivot[axis_b] - rel_b
        elif k == 3:
            rot_points[:, axis_a] = pivot[axis_a] + rel_b
            rot_points[:, axis_b] = pivot[axis_b] - rel_a

        return rot_points

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
