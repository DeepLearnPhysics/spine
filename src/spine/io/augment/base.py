"""Base interfaces for data augmentation modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from spine.data import Meta
from spine.geo import GeoManager
from spine.io.parse.data import ParserTensor


class AugmentBase(ABC):
    """Base class for augmentation modules."""

    name = ""

    def __call__(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Meta]:
        """Apply an augmentation module.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to augment
        meta : Meta
            Shared image metadata
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        context : dict
            Shared augmentation context built by the manager

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and shared metadata
        """
        return self.apply(data, meta, keys, context)

    @abstractmethod
    def apply(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Meta]:
        """Apply an augmentation to one event.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to augment
        meta : Meta
            Shared image metadata
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        context : dict
            Shared augmentation context built by the manager

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and shared metadata
        """

    @staticmethod
    def resolve_center(
        meta: Meta,
        center: np.ndarray | None = None,
        use_geo_center: bool = False,
    ) -> np.ndarray:
        """Resolve the pivot center for a geometric transform.

        Parameters
        ----------
        meta : Meta
            Current image metadata
        center : np.ndarray, optional
            Explicit center in detector coordinates (cm)
        use_geo_center : bool, default False
            If ``True``, use the detector TPC center from the geometry manager

        Returns
        -------
        np.ndarray
            (3,) Pivot center in detector coordinates (cm)
        """
        if center is not None and use_geo_center:
            raise ValueError("Cannot provide both `center` and `use_geo_center`.")

        if center is not None:
            center = np.asarray(center, dtype=np.float32)
            if center.shape != (3,):
                raise ValueError("Transform center must be a 3D point in cm.")
            return center

        if use_geo_center:
            return GeoManager.get_instance().tpc.center.astype(np.float32)

        return ((meta.lower + meta.upper) / 2.0).astype(np.float32)

    @staticmethod
    def voxel_to_cm(coords: np.ndarray, meta: Meta) -> np.ndarray:
        """Convert voxel indices to detector coordinates at voxel centers.

        Parameters
        ----------
        coords : np.ndarray
            ``(N, 3)`` Array of voxel indices
        meta : Meta
            Metadata used to convert voxel indices to detector coordinates

        Returns
        -------
        np.ndarray
            ``(N, 3)`` Detector coordinates in cm at voxel centers
        """
        return meta.to_cm(coords, center=True)

    @staticmethod
    def cm_to_voxel(coords_cm: np.ndarray, meta: Meta, dtype: np.dtype) -> np.ndarray:
        """Convert detector coordinates at voxel centers back to indices.

        Parameters
        ----------
        coords_cm : np.ndarray
            ``(N, 3)`` Detector coordinates in cm at voxel centers
        meta : Meta
            Metadata used to convert detector coordinates back to pixel space
        dtype : np.dtype
            Output dtype to use for the returned voxel indices

        Returns
        -------
        np.ndarray
            ``(N, 3)`` Array of voxel indices
        """
        return np.rint(meta.to_px(coords_cm) - 0.5).astype(dtype)

    @staticmethod
    def parse_optional_vector(
        value: float | List[float] | Tuple[float, ...] | np.ndarray | None,
        name: str,
    ) -> np.ndarray | None:
        """Parse an optional scalar-or-vector parameter into a length-3 array.

        Parameters
        ----------
        value : float or sequence or np.ndarray, optional
            Input value to parse. Scalars are broadcast to all three axes.
        name : str
            Parameter name used in validation error messages.

        Returns
        -------
        np.ndarray or None
            Length-3 vector if a value is provided, otherwise ``None``
        """
        if value is None:
            return None

        if np.isscalar(value):
            scalar = float(np.asarray(value, dtype=np.float32).item())
            array = np.full(3, scalar, dtype=np.float32)
        else:
            array = np.asarray(value, dtype=np.float32)

        if array.shape != (3,):
            raise ValueError(f"{name} must be a scalar or a length-3 vector.")

        return array

    @staticmethod
    def resolve_activity_center(
        data: Dict[str, Any],
        keys: List[str],
        meta: Meta,
        weighted: bool = False,
        feature_index: int = 0,
    ) -> np.ndarray:
        """Estimate an activity center from all coordinate-carrying tensors.

        Parameters
        ----------
        data : dict
            Dictionary of event data products
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        meta : Meta
            Shared image metadata
        weighted : bool, default False
            If ``True``, weight the center by the absolute feature value in the
            requested feature column
        feature_index : int, default 0
            Feature column to use when ``weighted=True``

        Returns
        -------
        np.ndarray
            (3,) Activity center in detector coordinates (cm)
        """
        center, _ = AugmentBase.resolve_activity_stats(
            data,
            keys,
            meta,
            weighted=weighted,
            feature_index=feature_index,
        )
        return center

    @staticmethod
    def resolve_activity_stats(
        data: Dict[str, Any],
        keys: List[str],
        meta: Meta,
        weighted: bool = False,
        feature_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Estimate activity center and spread from coordinate-carrying tensors.

        Parameters
        ----------
        data : dict
            Dictionary of event data products
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        meta : Meta
            Shared image metadata
        weighted : bool, default False
            If ``True``, weight the center and spread by the absolute feature
            value in the requested feature column
        feature_index : int, default 0
            Feature column to use when ``weighted=True``

        Returns
        -------
        Tuple[np.ndarray, np.ndarray or None]
            ``(3,)`` Activity center and standard deviation in detector
            coordinates (cm). If no activity is available, the center falls
            back to the metadata center and the spread is ``None``.
        """
        coords_list = []
        weights_list = []
        for key in keys:
            value = data.get(key)
            if not isinstance(value, ParserTensor) or value.coords is None:
                continue
            if len(value.coords) == 0:
                continue

            coords_cm = meta.to_cm(value.coords, center=True)
            coords_list.append(coords_cm)

            if weighted:
                features = np.asarray(value.features)
                if features.ndim == 1:
                    weights = np.abs(features)
                else:
                    column = min(feature_index, features.shape[1] - 1)
                    weights = np.abs(features[:, column])
                weights_list.append(weights)

        if not coords_list:
            center = ((meta.lower + meta.upper) / 2.0).astype(np.float32)
            return center, None

        coords = np.vstack(coords_list)
        if not weighted:
            center = np.mean(coords, axis=0)
            spread = np.std(coords, axis=0)
            return center.astype(np.float32), spread.astype(np.float32)

        weights = np.concatenate(weights_list).astype(np.float64)
        if np.allclose(weights.sum(), 0.0):
            center = np.mean(coords, axis=0)
            spread = np.std(coords, axis=0)
            return center.astype(np.float32), spread.astype(np.float32)

        center = np.average(coords, axis=0, weights=weights)
        variance = np.average((coords - center) ** 2, axis=0, weights=weights)
        spread = np.sqrt(variance)
        return center.astype(np.float32), spread.astype(np.float32)

    @staticmethod
    def sample_box_lower(
        lower: np.ndarray,
        upper: np.ndarray,
        dimensions: np.ndarray,
        anchor: np.ndarray | None = None,
        spread: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample the lower corner of a crop/mask box.

        Parameters
        ----------
        lower : np.ndarray
            Lower detector bounds of the allowed sampling region in cm
        upper : np.ndarray
            Upper detector bounds of the allowed sampling region in cm
        dimensions : np.ndarray
            Requested crop or mask box dimensions in cm
        anchor : np.ndarray, optional
            Preferred box center in cm. If provided, sampling is biased around
            this center.
        spread : np.ndarray, optional
            Standard deviation of the Gaussian proposal in cm when sampling
            around an anchor. If not provided, a fraction of the available
            range is used.

        Returns
        -------
        np.ndarray
            Lower detector corner of the sampled box in cm

        If an anchor is provided, sample the box center around it with a normal
        distribution and clamp to the valid range. Otherwise use a uniform draw.
        """
        max_lower = upper - dimensions
        if anchor is None:
            return lower + np.random.rand(3) * (max_lower - lower)

        center_lower = lower + dimensions / 2.0
        center_upper = upper - dimensions / 2.0
        anchor = np.clip(anchor, center_lower, center_upper)

        if spread is None:
            spread = 0.25 * (center_upper - center_lower)

        sampled_center = np.random.normal(loc=anchor, scale=spread)
        sampled_center = np.clip(sampled_center, center_lower, center_upper)
        return sampled_center - dimensions / 2.0

    @staticmethod
    def make_grid_aligned_meta(
        meta: Meta,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        count: np.ndarray,
        sampled_lower: np.ndarray,
    ) -> Meta:
        """Build a box metadata object snapped to the source voxel grid.

        Parameters
        ----------
        meta : Meta
            Reference metadata that defines the source voxel grid
        lower_bound : np.ndarray
            Minimum allowed lower edge in detector coordinates (cm)
        upper_bound : np.ndarray
            Maximum allowed upper edge in detector coordinates (cm)
        count : np.ndarray
            Requested number of voxels along each axis
        sampled_lower : np.ndarray
            Proposed lower edge in detector coordinates (cm)

        Returns
        -------
        Meta
            Grid-aligned metadata for the sampled box
        """
        count = np.asarray(count, dtype=meta.count.dtype)
        epsilon = 1.0e-6

        start_min = np.ceil((lower_bound - meta.lower) / meta.size - epsilon).astype(
            meta.count.dtype
        )
        stop_max = np.floor((upper_bound - meta.lower) / meta.size + epsilon).astype(
            meta.count.dtype
        )
        start_max = stop_max - count
        if np.any(start_max < start_min):
            raise ValueError(
                "The sampled box cannot fit within the allowed bounds on the source grid."
            )

        sampled_start = np.rint((sampled_lower - meta.lower) / meta.size).astype(
            meta.count.dtype
        )
        start = np.clip(sampled_start, start_min, start_max)

        lower = np.asarray(meta.lower + start * meta.size, dtype=meta.lower.dtype)
        upper = np.asarray(lower + count * meta.size, dtype=meta.upper.dtype)
        return Meta(
            lower=lower,
            upper=upper,
            size=meta.size.copy(),
            count=count,
        )

    @staticmethod
    def make_snapped_meta(
        meta: Meta,
        size: np.ndarray,
        count: np.ndarray,
        lower: np.ndarray,
    ) -> Meta:
        """Build metadata snapped to the source grid from a proposed lower edge.

        Parameters
        ----------
        meta : Meta
            Reference metadata that defines the source voxel grid
        size : np.ndarray
            Pixel size for the transformed metadata
        count : np.ndarray
            Pixel counts for the transformed metadata
        lower : np.ndarray
            Proposed lower edge in detector coordinates (cm)

        Returns
        -------
        Meta
            Grid-aligned metadata for the transformed image volume
        """
        size = np.asarray(size, dtype=meta.size.dtype)
        count = np.asarray(count, dtype=meta.count.dtype)
        start = np.rint((lower - meta.lower) / size).astype(meta.count.dtype)
        snapped_lower = np.asarray(meta.lower + start * size, dtype=meta.lower.dtype)
        upper = np.asarray(snapped_lower + size * count, dtype=meta.upper.dtype)
        return Meta(
            lower=snapped_lower,
            upper=upper,
            size=size,
            count=count,
        )
