"""Cropping augmentation module."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spine.data import Meta
from spine.geo import GeoManager

from .base import AugmentBase


class CropAugment(AugmentBase):
    """Generic class to handle cropping images."""

    name = "crop"

    def __init__(
        self,
        min_dimensions: np.ndarray,
        max_dimensions: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        use_geo: bool = False,
    ) -> None:
        """Initialize the cropper.

        Parameters
        ----------
        min_dimensions : np.ndarray
            Minimum cropping dimensions in cm for each axis
        max_dimensions : np.ndarray
            Maximum cropping dimensions in cm for each axis
        lower : np.ndarray, optional
            Lower bounds for cropping in cm for each axis
        upper : np.ndarray, optional
            Upper bounds for cropping in cm for each axis
        use_geo : bool, optional
            Whether to use geometry-based cropping bounds

        Returns
        -------
        None
            This method does not return anything
        """
        if not len(min_dimensions) == len(max_dimensions) == 3:
            raise ValueError("Must provide dimensions for each axis.")
        if lower is not None and not len(lower) == 3:
            raise ValueError("Must provide lower bounds for each axis.")
        if upper is not None and not len(upper) == 3:
            raise ValueError("Must provide upper bounds for each axis.")

        self.min_dimensions = np.asarray(min_dimensions)
        self.max_dimensions = np.asarray(max_dimensions)
        if np.any(self.min_dimensions <= 0) or np.any(self.max_dimensions <= 0):
            raise ValueError("Cropping dimensions must be positive.")
        if np.any(self.min_dimensions > self.max_dimensions):
            raise ValueError("Minimum cropping dimensions must be less than maximum.")

        self.range = self.max_dimensions - self.min_dimensions

        self.lower = np.asarray(lower) if lower is not None else None
        self.upper = np.asarray(upper) if upper is not None else None
        if (
            self.lower is not None
            and self.upper is not None
            and np.any(self.lower > self.upper)
        ):
            raise ValueError("Lower bounds must be less than upper bounds.")

        if use_geo:
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    "Cannot use geometry if custom cropping bounds are provided."
                )
            geo = GeoManager.get_instance()
            self.lower = geo.tpc.lower
            self.upper = geo.tpc.upper

    def apply(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Meta]:
        """Randomly crop the image within the pre-defined range.

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
            Updated data dictionary and cropped metadata
        """
        crop_meta = self.generate_crop(meta)

        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = crop_meta
                continue

            voxels, features = data[key].coords, data[key].features
            voxels_cm = meta.to_cm(voxels, center=True)
            mask = crop_meta.inner_mask(voxels_cm)
            index = np.where(mask)[0]

            voxels_cm, features = voxels_cm[index], features[index]
            voxels = crop_meta.to_px(voxels_cm, floor=True).astype(voxels.dtype)

            data[key].coords = voxels
            data[key].features = features
            data[key].meta = crop_meta

        return data, crop_meta

    def generate_crop(self, meta: Meta) -> Meta:
        """Generate crop box metadata to apply to voxel index sets.

        Parameters
        ----------
        meta : Meta
            Metadata of the original image

        Returns
        -------
        Meta
            Metadata describing the cropped image volume
        """
        lower = self.lower if self.lower is not None else meta.lower
        upper = self.upper if self.upper is not None else meta.upper
        if np.any(self.range > (upper - lower)):
            raise ValueError(
                "The cropping range is larger than the allowed cropping bounds."
            )

        dimensions = self.min_dimensions + np.random.rand(3) * self.range
        count = np.ceil(dimensions / meta.size).astype(int)
        dimensions = count * meta.size

        crop_lower = lower + np.random.rand(3) * (upper - lower - dimensions)
        crop_upper = crop_lower + dimensions

        return Meta(lower=crop_lower, upper=crop_upper, size=meta.size, count=count)
