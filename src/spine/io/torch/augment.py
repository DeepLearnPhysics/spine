"""Module with methods to augment the input data."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spine.data import Meta
from spine.geo import GeoManager
from spine.io.core.parse.data import ParserTensor

__all__ = ["Augmenter"]


class Augmenter:
    """Generic class to handle data augmentation."""

    def __init__(
        self,
        crop: Optional[Dict[str, Any]] = None,
        translate: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the augmenter.

        Parameters
        ----------
        crop : dict, optional
            Cropping configuration (crop input image to a smaller size)
        translate : dict, optional
            Translation configuration (move input image around)
        """
        # Make sure at least one augmentation scheme is requested
        if crop is None and translate is None:
            raise ValueError(
                "Must provide `crop` or `translate` block minimally to do any augmentation."
            )

        # Parse the cropping configuration
        self.cropper = None
        if crop is not None:
            self.cropper = Cropper(**crop)

        # Parse the translation configuration
        self.translater = None
        if translate is not None:
            self.translater = Translater(**translate)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the data products in one event.

        Parameters
        ----------
        data : dict
           Data product dictionary
        """
        # Get the list of keys to augment and the shared metadata
        augment_keys = []
        meta = None
        for key, value in data.items():
            if isinstance(value, ParserTensor) and value.coords is not None:
                augment_keys.append(key)
                if meta is None:
                    meta = value.meta
                elif meta != value.meta:
                    raise ValueError("Metadata should be shared by all data products.")

            elif isinstance(value, Meta):
                augment_keys.append(key)
                meta = value

        # If there are no sparse tensors in the input data, nothing to do
        if meta is None:
            return data

        # Crop
        if self.cropper is not None:
            data, meta = self.cropper(data, meta, augment_keys)

        # Translate
        if self.translater is not None:
            data = self.translater(data, meta, augment_keys)

        return data


class Cropper:
    """Generic class to handle cropping images."""

    def __init__(
        self,
        min_dimensions: np.ndarray,
        max_dimensions: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        use_geo: bool = False,
    ):
        """Initialize the cropper.

        This defines a way to crop the input image to a smaller size within
        a user-defined range. The cropping is done by randomly selecting a box of
        the appropriate size within the original image and cropping to it.

        Parameters
        ----------
        min_dimensions : np.ndarray
            Minimum dimensions of the cropping box
        max_dimensions : np.ndarray
            Maximum dimensions of the cropping box
        lower : np.ndarray, optional
            Lower bounds of the box in which to crop the image in
        upper : np.ndarray, optional
            Upper bounds of the box in which to crop the image in
        use_geo : bool, default False
            If `True`, the cropping box is defined within the outer bounds
            of the overall TPC detector geometry.
        """
        # Sanity check
        if not len(min_dimensions) == len(max_dimensions) == 3:
            raise ValueError("Must provide dimensions for each axis.")
        if lower is not None and not len(lower) == 3:
            raise ValueError("Must provide lower bounds for each axis.")
        if upper is not None and not len(upper) == 3:
            raise ValueError("Must provide upper bounds for each axis.")

        # Store the cropping dimension range
        self.min_dimensions = np.asarray(min_dimensions)
        self.max_dimensions = np.asarray(max_dimensions)
        if np.any(self.min_dimensions <= 0) or np.any(self.max_dimensions <= 0):
            raise ValueError("Cropping dimensions must be positive.")
        if np.any(self.min_dimensions > self.max_dimensions):
            raise ValueError("Minimum cropping dimensions must be less than maximum.")

        # Store the cropping dimension range
        self.range = self.max_dimensions - self.min_dimensions

        # Store the cropping location bounds, if provided
        self.lower = np.asarray(lower) if lower is not None else None
        self.upper = np.asarray(upper) if upper is not None else None
        if (
            self.lower is not None
            and self.upper is not None
            and np.any(self.lower > self.upper)
        ):
            raise ValueError("Lower bounds must be less than upper bounds.")

        # If using the geometry, set the cropping location bounds to the TPC boundaries
        if use_geo:
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    "Cannot use geometry if custom cropping bounds are provided."
                )
            geo = GeoManager.get_instance()
            self.lower = geo.tpc.lower
            self.upper = geo.tpc.upper

    def __call__(
        self, data: Dict[str, Any], meta: Meta, keys: List[str]
    ) -> Tuple[Dict[str, Any], Meta]:
        """Randomly crop the image to a smaller size within the pre-defined range.

        Parameters
        ----------
        data : dict
            Dictionary of data products to offset
        meta : Meta
            Shared image metadata
        keys : List[str]
            List of keys with coordinates to offset

        Returns
        -------
        np.ndarray
            (N, 3) Masked points
        Meta
            Metadata of the cropped image
        """
        # Generate a crop box metadata
        crop_meta = self.generate_crop(meta)

        # Crop the coordinates by masking out those outside the cropping box
        for key in keys:
            # If the key is the metadata, modify and continue
            if isinstance(data[key], Meta):
                data[key] = crop_meta
                continue

            # Fetch attributes to modify
            voxels, features = data[key].coords, data[key].features

            # Compute a mask of which voxels are within the cropping box
            voxels_cm = meta.to_cm(voxels, center=True)
            mask = crop_meta.inner_mask(voxels_cm)
            index = np.where(mask)[0]

            # Crop voxels and redifine the pixel index with respect to the cropping box
            voxels_cm, features = voxels_cm[index], features[index]
            voxels = crop_meta.to_px(voxels_cm, floor=True).astype(voxels.dtype)

            # Update
            data[key].coords = voxels
            data[key].features = features
            data[key].meta = crop_meta

        return data, crop_meta

    def generate_crop(self, meta: Meta) -> Meta:
        """Generate a crop box metadata to apply to the voxel index sets.

        This crop box is such that the voxels will be randomly masked
        out outside the target bounding box.

        Parameters
        ----------
        meta : Meta
            Metadata of the original image, used to determine the cropping bounds

        Returns
        -------
        Meta
            Metadata of the cropping box
        """
        # Determine upper/lower limits of the cropping box range
        lower = self.lower if self.lower is not None else meta.lower
        upper = self.upper if self.upper is not None else meta.upper
        if np.any(self.range > (upper - lower)):
            raise ValueError(
                "The cropping range is larger than the allowed cropping bounds."
            )

        # Pick a random cropping dimension within the range
        dimensions = self.min_dimensions + np.random.rand(3) * self.range

        # Adjust dimensions to be a multiple of the pixel pitch,
        # so that the cropping box is aligned with the pixel grid
        count = np.ceil(dimensions / meta.size).astype(int)
        dimensions = count * meta.size

        # Pick a random cropping box within the allowed bounds
        crop_lower = lower + np.random.rand(3) * (upper - lower - dimensions)
        crop_upper = crop_lower + dimensions

        return Meta(lower=crop_lower, upper=crop_upper, size=meta.size, count=count)


class Translater:
    """Generic class to handle moving images around."""

    def __init__(
        self,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        use_geo: bool = False,
    ):
        """Initialize the translater.

        This defines a way to move the image around within a volume greater
        than that define by the image metadata. The box must be larger than
        the image itself.

        Parameters
        ----------
        lower : np.ndarray, optional
            Lower bounds of the box in which to move the image around
        upper : np.ndarray, optional
            Upper bounds of the box in which to move the image around
        use_geo : bool, default False
            If `True`, the box in which to move the image around is defined
            within the outer bounds of the overall TPC detector geometry.
        """
        # Sanity check
        lower = np.asarray(lower) if lower is not None else None
        upper = np.asarray(upper) if upper is not None else None
        if (lower is None) != (upper is None):
            raise ValueError("Must provide both lower and upper bounds, or neither.")
        if lower is not None and upper is not None:
            if not len(lower) == len(upper) == 3:
                raise ValueError("Must provide bounds for each axis.")
            if np.any(lower > upper):
                raise ValueError("Lower bounds must be less than upper bounds.")

            self.meta = Meta(lower, upper)

        # If using the geometry, set the cropping location bounds to the TPC boundaries
        if use_geo:
            if lower is not None or upper is not None:
                raise ValueError(
                    "Cannot use geometry if custom cropping bounds are provided."
                )

            geo = GeoManager.get_instance()
            self.meta = Meta(lower=geo.tpc.lower, upper=geo.tpc.upper)

    def __call__(
        self, data: Dict[str, Any], meta: Meta, keys: List[str]
    ) -> Dict[str, Any]:
        """Move an image around within the the pre-defined volume.

        Parameters
        ----------
        data : dict
            Dictionary of data products to offset
        meta : Meta
            Shared image metadata
        keys : List[str]
            List of keys with coordinates to offset

        Returns
        -------
        np.ndarray
            (N, 3) Translated points
        """
        # Set the target volume pixel pitch to match that of the original image
        if np.all(self.meta.size < 0.0):
            self.meta.size = meta.size
            self.meta.count = np.ceil(
                (self.meta.upper - self.meta.lower) / meta.size
            ).astype(int)

        # Generate an offset
        offset = self.generate_offset(meta)

        # Offset all coordinates
        for key in keys:
            # If the key is the metadata, modify and continue
            if isinstance(data[key], Meta):
                data[key] = self.meta
                continue

            # Fetch attributes to modify
            voxels = data[key].coords

            # Translate
            width = voxels.shape[1]
            voxels = (voxels.reshape(-1, 3) + offset).reshape(-1, width)

            # Update
            data[key].coords = voxels
            data[key].meta = self.meta

        return data

    def generate_offset(self, meta: Meta) -> np.ndarray:
        """Generate an offset to apply to all the voxel index sets.

        This offset is such that the the voxels will be randomly shifted
        within the target bounding box.

        Parameters
        ----------
        meta : Meta
            Metadata of the original image

        Returns
        -------
        np.ndarray
            Value by which to shift the pixels by in integer voxel units
        """
        # Check that the original metadata is compatible with the target volume
        if np.any(meta.size != self.meta.size):
            raise ValueError(
                "The pixel pitch of the original image must match that of the target volume."
            )

        # Generate an offset with respect to the voxel indices
        offset = np.random.randint((self.meta.count - meta.count) + 1)

        return offset
