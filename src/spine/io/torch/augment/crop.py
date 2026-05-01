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
        min_dimensions: Optional[np.ndarray] = None,
        max_dimensions: Optional[np.ndarray] = None,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        use_geo_boundaries: bool = False,
        center_mode: str = "uniform",
        center_spread: Optional[np.ndarray] = None,
        center_feature_index: int = 0,
        active_volume: bool = False,
        keep_meta: bool = True,
    ) -> None:
        """Initialize the cropper.

        Parameters
        ----------
        min_dimensions : np.ndarray, optional
            Minimum cropping dimensions in cm for each axis. If omitted together
            with ``max_dimensions``, disable box cropping.
        max_dimensions : np.ndarray, optional
            Maximum cropping dimensions in cm for each axis. If omitted together
            with ``min_dimensions``, disable box cropping.
        lower : np.ndarray, optional
            Lower bounds for cropping in cm for each axis
        upper : np.ndarray, optional
            Upper bounds for cropping in cm for each axis
        use_geo_boundaries : bool, default False
            Whether to use detector TPC boundaries as the allowed cropping
            region
        center_mode : str, default "uniform"
            Box-center sampling strategy. Supported values are ``"uniform"``,
            ``"activity"`` and ``"weighted_activity"``.
        center_spread : np.ndarray, optional
            Standard deviation of the Gaussian box-center proposal in cm when
            using an activity-based center mode. Scalar values are broadcast.
        center_feature_index : int, default 0
            Feature column to use when ``center_mode="weighted_activity"``
        active_volume : bool, default False
            If ``True``, drop points outside the union of detector module
            active volumes defined by ``geo.tpc.modules``
        keep_meta : bool, default True
            If ``True``, preserve the original metadata and voxel indices while
            removing points outside the sampled crop or active volume.

        Returns
        -------
        None
            This method does not return anything
        """
        if (min_dimensions is None) != (max_dimensions is None):
            raise ValueError(
                "Must provide both `min_dimensions` and `max_dimensions`, or neither."
            )
        if min_dimensions is None and not active_volume:
            raise ValueError(
                "Cropping requires either box dimensions or `active_volume=True`."
            )
        if min_dimensions is not None:
            assert max_dimensions is not None
            if not len(min_dimensions) == len(max_dimensions) == 3:
                raise ValueError("Must provide dimensions for each axis.")
        if lower is not None and not len(lower) == 3:
            raise ValueError("Must provide lower bounds for each axis.")
        if upper is not None and not len(upper) == 3:
            raise ValueError("Must provide upper bounds for each axis.")

        self.has_box_crop = min_dimensions is not None
        if self.has_box_crop:
            self.min_dimensions = np.asarray(min_dimensions)
            self.max_dimensions = np.asarray(max_dimensions)
            if np.any(self.min_dimensions <= 0) or np.any(self.max_dimensions <= 0):
                raise ValueError("Cropping dimensions must be positive.")
            if np.any(self.min_dimensions > self.max_dimensions):
                raise ValueError(
                    "Minimum cropping dimensions must be less than maximum."
                )

            self.range = self.max_dimensions - self.min_dimensions
        else:
            self.min_dimensions = None
            self.max_dimensions = None
            self.range = None

        self.lower = np.asarray(lower) if lower is not None else None
        self.upper = np.asarray(upper) if upper is not None else None
        if (
            self.lower is not None
            and self.upper is not None
            and np.any(self.lower > self.upper)
        ):
            raise ValueError("Lower bounds must be less than upper bounds.")

        if use_geo_boundaries:
            if not self.has_box_crop:
                raise ValueError(
                    "`use_geo_boundaries` requires box cropping dimensions."
                )
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    "Cannot use geometry if custom cropping bounds are provided."
                )
            geo = GeoManager.get_instance()
            self.lower = geo.tpc.lower
            self.upper = geo.tpc.upper

        if center_mode not in ("uniform", "activity", "weighted_activity"):
            raise ValueError(
                "Cropping center mode must be one of ('uniform', 'activity', 'weighted_activity')."
            )
        if center_feature_index < 0:
            raise ValueError("Cropping center_feature_index must be non-negative.")

        self.center_mode = center_mode
        self.center_spread = self.parse_optional_vector(center_spread, "center_spread")
        self.center_feature_index = int(center_feature_index)
        self.active_volume = active_volume
        self.keep_meta = keep_meta

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
        crop_meta = self.generate_crop(data, meta, keys) if self.has_box_crop else None
        active_meta = (
            self.generate_active_volume_meta(meta) if self.active_volume else None
        )
        output_meta = (
            meta
            if self.keep_meta
            else crop_meta if crop_meta is not None else active_meta
        )

        if output_meta is None:
            raise ValueError("Crop augmenter must define an output metadata volume.")

        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = output_meta
                continue

            voxels, features = data[key].coords, data[key].features
            voxels_cm = meta.to_cm(voxels, center=True)
            keep_mask = np.ones(len(voxels), dtype=bool)
            if crop_meta is not None:
                keep_mask &= crop_meta.inner_mask(voxels_cm)
            if self.active_volume:
                keep_mask &= self.active_volume_mask(voxels_cm)
            if active_meta is not None:
                keep_mask &= active_meta.inner_mask(voxels_cm)

            index = np.where(keep_mask)[0]

            voxels_cm, features = voxels_cm[index], features[index]
            if self.keep_meta:
                voxels = voxels[index]
            else:
                voxels = output_meta.to_px(voxels_cm, floor=True).astype(voxels.dtype)

            data[key].coords = voxels
            data[key].features = features
            data[key].meta = output_meta

        return data, output_meta

    def active_volume_mask(self, coords_cm: np.ndarray) -> np.ndarray:
        """Check which detector coordinates lie inside any module active volume.

        Parameters
        ----------
        coords_cm : np.ndarray
            ``(N, 3)`` Detector coordinates in cm

        Returns
        -------
        np.ndarray
            Boolean mask selecting coordinates inside at least one module box
        """
        geo = GeoManager.get_instance()
        mask = np.zeros(len(coords_cm), dtype=bool)
        for module in geo.tpc.modules:
            bounds = module.boundaries
            inside = np.all(
                (coords_cm >= bounds[:, 0]) & (coords_cm < bounds[:, 1]), axis=1
            )
            mask |= inside

        return mask

    def generate_active_volume_meta(self, meta: Meta) -> Meta:
        """Generate metadata aligned to the detector active-volume envelope.

        Parameters
        ----------
        meta : Meta
            Metadata of the current image volume

        Returns
        -------
        Meta
            Metadata covering the overlap between the current image grid and the
            detector active-volume envelope
        """
        geo = GeoManager.get_instance()
        lower_bound = np.maximum(meta.lower, geo.tpc.lower)
        upper_bound = np.minimum(meta.upper, geo.tpc.upper)

        start = np.ceil((lower_bound - meta.lower) / meta.size - 0.5).astype(np.int64)
        stop = np.ceil((upper_bound - meta.lower) / meta.size - 0.5).astype(np.int64)
        start = np.clip(start, 0, meta.count)
        stop = np.clip(stop, start, meta.count)

        lower = meta.lower + start * meta.size
        count = stop - start
        upper = lower + count * meta.size
        return Meta(lower=lower, upper=upper, size=meta.size.copy(), count=count)

    def generate_crop(self, data: Dict[str, Any], meta: Meta, keys: List[str]) -> Meta:
        """Generate crop box metadata to apply to voxel index sets.

        Parameters
        ----------
        data : dict
            Dictionary of event data products used to estimate an activity
            center when activity-biased sampling is enabled
        meta : Meta
            Metadata of the original image
        keys : List[str]
            Keys corresponding to data products that carry coordinates

        Returns
        -------
        Meta
            Metadata describing the cropped image volume
        """
        if self.min_dimensions is None or self.range is None:
            raise ValueError("Box cropping dimensions are not configured.")

        lower = self.lower if self.lower is not None else meta.lower
        upper = self.upper if self.upper is not None else meta.upper
        if np.any(self.range > (upper - lower)):
            raise ValueError(
                "The cropping range is larger than the allowed cropping bounds."
            )

        dimensions = self.min_dimensions + np.random.rand(3) * self.range
        count = np.ceil(dimensions / meta.size).astype(int)
        dimensions = count * meta.size

        center = None
        spread = self.center_spread
        if self.center_mode != "uniform":
            center, activity_spread = self.resolve_activity_stats(
                data,
                keys,
                meta,
                weighted=self.center_mode == "weighted_activity",
                feature_index=self.center_feature_index,
            )
            if spread is None:
                spread = activity_spread

        crop_lower = self.sample_box_lower(
            lower, upper, dimensions, anchor=center, spread=spread
        )
        return self.make_grid_aligned_meta(meta, lower, upper, count, crop_lower)
