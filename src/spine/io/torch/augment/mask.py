"""Masking augmentation module."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spine.data import Meta
from spine.geo import GeoManager

from .base import AugmentBase


class MaskAugment(AugmentBase):
    """Generic class to handle cutting out regions of an image."""

    name = "mask"

    def __init__(
        self,
        min_dimensions: np.ndarray,
        max_dimensions: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        use_geo: bool = False,
    ) -> None:
        """Initialize the masker.

        Parameters
        ----------
        min_dimensions : np.ndarray
            Minimum masking dimensions in cm for each axis
        max_dimensions : np.ndarray
            Maximum masking dimensions in cm for each axis
        lower : np.ndarray, optional
            Lower bounds for masking in cm for each axis
        upper : np.ndarray, optional
            Upper bounds for masking in cm for each axis
        use_geo : bool, optional
            Whether to use geometry-based masking bounds

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
            raise ValueError("Masking dimensions must be positive.")
        if np.any(self.min_dimensions > self.max_dimensions):
            raise ValueError("Minimum masking dimensions must be less than maximum.")

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
                    "Cannot use geometry if custom masking bounds are provided."
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
        """Randomly mask a portion of the image.

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
            Updated data dictionary and unchanged metadata
        """
        mask_meta = self.generate_mask(meta)

        for key in keys:
            if isinstance(data[key], Meta):
                continue

            voxels, features = data[key].coords, data[key].features
            voxels_cm = meta.to_cm(voxels, center=True)
            mask = mask_meta.inner_mask(voxels_cm)
            index = np.where(~mask)[0]

            voxels, features = voxels[index], features[index]

            data[key].coords = voxels
            data[key].features = features

        return data, meta

    def generate_mask(self, meta: Meta) -> Meta:
        """Generate a masking box metadata to apply to voxel index sets.

        Parameters
        ----------
        meta : Meta
            Metadata of the original image

        Returns
        -------
        Meta
            Metadata describing the masked box
        """
        lower = self.lower if self.lower is not None else meta.lower
        upper = self.upper if self.upper is not None else meta.upper
        if np.any(self.range > (upper - lower)):
            raise ValueError(
                "The masking range is larger than the allowed masking bounds."
            )

        dimensions = self.min_dimensions + np.random.rand(3) * self.range
        count = np.ceil(dimensions / meta.size).astype(int)
        dimensions = count * meta.size

        mask_lower = lower + np.random.rand(3) * (upper - lower - dimensions)
        mask_upper = mask_lower + dimensions

        return Meta(lower=mask_lower, upper=mask_upper, size=meta.size, count=count)
