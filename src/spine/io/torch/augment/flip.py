"""Reflection augmentation module."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spine.data import Meta

from .base import AugmentBase


class FlipAugment(AugmentBase):
    """Reflect voxel coordinates across a plane normal to one detector axis."""

    name = "flip"

    def __init__(
        self,
        axis: int,
        center: Optional[np.ndarray] = None,
        use_geo_center: bool = False,
        keep_meta: bool = True,
        p: float = 1.0,
    ) -> None:
        """Initialize the flipper.

        Parameters
        ----------
        axis : int
            Axis normal to the reflection plane
        center : np.ndarray, optional
            Explicit point on the reflection plane in detector coordinates (cm)
        use_geo_center : bool, default False
            If ``True``, use the detector TPC center as the plane center
        keep_meta : bool, default True
            If ``True``, keep the detector frame fixed and drop points that
            reflect outside the current metadata bounds. If ``False``, reflect
            the image volume together with the points.
        p : float, default 1.0
            Probability of applying the flip to an event. Values less than 1
            randomly leave some events unchanged.
        """
        if not isinstance(axis, (int, np.integer)) or axis < 0 or axis > 2:
            raise ValueError("Flip axis must be an integer in the range [0, 2].")
        p = float(p)
        if not np.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError("Flip probability must be in the range [0, 1].")

        self.axis = int(axis)
        self.center = None if center is None else np.asarray(center, dtype=np.float32)
        self.use_geo_center = use_geo_center
        self.keep_meta = keep_meta
        self.p = p

    def apply(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Meta]:
        """Reflect coordinates across the requested plane.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to reflect
        meta : Meta
            Shared image metadata before reflection
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        context : dict
            Shared augmentation context

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and reflected metadata
        """
        if np.random.rand() >= self.p:
            return data, meta

        pivot = self.resolve_center(meta, self.center, self.use_geo_center)
        flip_meta = meta if self.keep_meta else self.generate_meta(meta, pivot)

        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = flip_meta
                continue

            coords_cm = self.voxel_to_cm(data[key].coords, meta)
            flip_cm = self.flip_points(coords_cm, pivot)
            if self.keep_meta:
                keep_mask = flip_meta.inner_mask(flip_cm)
                flip_cm = flip_cm[keep_mask]
                data[key].features = data[key].features[keep_mask]
            data[key].coords = self.cm_to_voxel(
                flip_cm, flip_meta, data[key].coords.dtype
            )
            data[key].meta = flip_meta

        return data, flip_meta

    def flip_points(self, points: np.ndarray, pivot: np.ndarray) -> np.ndarray:
        """Reflect detector coordinates across the configured plane.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 3)`` Detector coordinates in cm
        pivot : np.ndarray
            ``(3,)`` Point on the reflection plane in detector coordinates (cm)

        Returns
        -------
        np.ndarray
            ``(N, 3)`` Reflected detector coordinates in cm
        """
        reflected = points.copy()
        reflected[:, self.axis] = 2.0 * pivot[self.axis] - points[:, self.axis]
        return reflected

    def generate_meta(self, meta: Meta, pivot: np.ndarray) -> Meta:
        """Generate metadata for the reflected image.

        Parameters
        ----------
        meta : Meta
            Metadata of the image before reflection
        pivot : np.ndarray
            ``(3,)`` Point on the reflection plane in detector coordinates (cm)

        Returns
        -------
        Meta
            Metadata of the reflected image volume
        """
        dimensions = meta.size * meta.count
        meta_center = (meta.lower + meta.upper) / 2.0
        refl_center = meta_center.copy()
        refl_center[self.axis] = 2.0 * pivot[self.axis] - meta_center[self.axis]

        lower = refl_center - dimensions / 2.0
        return self.make_snapped_meta(
            meta,
            meta.size.copy(),
            meta.count.copy(),
            lower,
        )
