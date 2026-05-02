"""Translation augmentation module."""

from typing import Any

import numpy as np

from spine.data import Meta
from spine.geo import GeoManager

from .base import AugmentBase


class TranslateAugment(AugmentBase):
    """Generic class to handle moving images around."""

    name = "translate"

    def __init__(
        self,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        use_geo: bool = False,
    ) -> None:
        """Initialize the translater.

        Parameters
        ----------
        lower : np.ndarray, optional
            Lower bounds of the translation volume in cm
        upper : np.ndarray, optional
            Upper bounds of the translation volume in cm
        use_geo : bool, optional
            Whether to use detector geometry bounds for translation

        Returns
        -------
        None
            This method does not return anything
        """
        lower = np.asarray(lower, dtype=np.float32) if lower is not None else None
        upper = np.asarray(upper, dtype=np.float32) if upper is not None else None
        if (lower is None) != (upper is None):
            raise ValueError("Must provide both lower and upper bounds, or neither.")
        if lower is not None and upper is not None:
            if not len(lower) == len(upper) == 3:
                raise ValueError("Must provide bounds for each axis.")
            if np.any(lower > upper):
                raise ValueError("Lower bounds must be less than upper bounds.")

        self.lower = lower
        self.upper = upper

        if use_geo:
            if lower is not None or upper is not None:
                raise ValueError(
                    "Cannot use geometry if custom cropping bounds are provided."
                )

            geo = GeoManager.get_instance()
            self.lower = geo.tpc.lower.astype(np.float32)
            self.upper = geo.tpc.upper.astype(np.float32)

    def apply(
        self,
        data: dict[str, Any],
        meta: Meta,
        keys: list[str],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], Meta]:
        """Move an image around within the pre-defined volume.

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
            Updated data dictionary and translated metadata
        """
        target_meta = self.get_target_meta(meta, context.get("original_meta"))

        offset = self.generate_offset(meta, target_meta)

        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = target_meta
                continue

            voxels = data[key].coords
            width = voxels.shape[1]
            voxels = (voxels.reshape(-1, 3) + offset).reshape(-1, width)

            data[key].coords = voxels
            data[key].meta = target_meta

        return data, target_meta

    def get_target_meta(self, meta: Meta, original_meta: Meta | None = None) -> Meta:
        """Resolve the target translation volume metadata.

        Parameters
        ----------
        meta : Meta
            Current image metadata
        original_meta : Meta, optional
            Original pre-augmentation metadata

        Returns
        -------
        Meta
            Metadata describing the translation target volume
        """
        if self.lower is not None and self.upper is not None:
            size = (
                meta.size.copy() if original_meta is None else original_meta.size.copy()
            )
            count = np.ceil((self.upper - self.lower) / size).astype(int)
            upper = self.lower + size * count
            return Meta(
                lower=self.lower.copy(),
                upper=upper,
                size=size,
                count=count,
            )

        source_meta = original_meta if original_meta is not None else meta
        return Meta(
            lower=source_meta.lower.copy(),
            upper=source_meta.upper.copy(),
            size=source_meta.size.copy(),
            count=source_meta.count.copy(),
        )

    def generate_offset(self, meta: Meta, target_meta: Meta) -> np.ndarray:
        """Generate a voxel offset within the target bounding box.

        Parameters
        ----------
        meta : Meta
            Metadata of the image to translate
        target_meta : Meta
            Metadata of the translation target volume

        Returns
        -------
        np.ndarray
            Integer voxel offset to apply along each axis
        """
        if np.any(meta.size != target_meta.size):
            raise ValueError(
                "The pixel pitch of the original image must match that of the target volume."
            )

        return np.random.randint((target_meta.count - meta.count) + 1)
