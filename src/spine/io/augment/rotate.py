"""Rotation augmentation module."""

from typing import Any

import numpy as np

from spine.data import Meta

from .base import AugmentBase
from .spatial import (
    SpatialAdapter,
    field_from_cm,
    field_from_px,
    field_to_cm,
    field_to_px,
)


class RotateAugment(AugmentBase):
    """Generic class to handle right-angle image rotations."""

    name = "rotate"

    def __init__(
        self,
        axes: tuple[int, int] = (0, 1),
        k: int | list[int] | tuple[int, ...] | None = None,
        center: np.ndarray | None = None,
        use_geo_center: bool = False,
        keep_meta: bool = True,
        p: float = 1.0,
    ) -> None:
        """Initialize the rotater.

        Parameters
        ----------
        axes : Tuple[int, int], default (0, 1)
            Pair of axes defining the plane in which to rotate
        k : int or sequence of int, optional
            Number of 90-degree turns to apply. A sequence is sampled
            uniformly at call time. If omitted, sample uniformly from 0 to 3.
        center : np.ndarray, optional
            Explicit rotation center in detector coordinates (cm). If not
            provided, the historical image-frame rotation behavior is used.
        use_geo_center : bool, default False
            If ``True``, rotate about the detector TPC center
        keep_meta : bool, default True
            If ``True``, keep the detector frame fixed and drop points that
            rotate outside the current metadata bounds. If ``False``, rotate
            the image volume together with the points.
        p : float, default 1.0
            Event-level probability of applying the rotation. A skipped event
            is returned unchanged without sampling a quarter-turn count.

        Returns
        -------
        None
            This method does not return anything
        """
        # Validate the rotation plane and quarter-turn sampling policy
        if len(axes) != 2:
            raise ValueError("Must provide exactly two rotation axes.")
        if axes[0] == axes[1]:
            raise ValueError("Rotation axes must be different.")
        if np.any(np.asarray(axes) < 0) or np.any(np.asarray(axes) > 2):
            raise ValueError("Rotation axes must be in the range [0, 2].")
        k_value = None
        k_choices = None
        if isinstance(k, (int, np.integer)):
            k_value = int(k) % 4
        elif k is not None:
            if not isinstance(k, (list, tuple)):
                raise ValueError("Rotation `k` must be an integer or sequence.")
            if len(k) == 0:
                raise ValueError("Rotation `k` choices cannot be empty.")
            if any(not isinstance(choice, (int, np.integer)) for choice in k):
                raise ValueError("Rotation `k` choices must contain only integers.")
            k_choices = tuple(int(choice) % 4 for choice in k)

        p = float(p)
        if not np.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError("Rotation probability must be in the range [0, 1].")

        self.axes = tuple(axes)
        self.k = k_value
        self.k_choices = k_choices
        self.center = None if center is None else np.asarray(center, dtype=np.float32)
        self.use_geo_center = use_geo_center
        self.keep_meta = keep_meta
        self.p = p

    def apply(
        self,
        data: dict[str, Any],
        meta: Meta,
        keys: list[str],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], Meta]:
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
        # Avoid an extra random draw in the default path, preserving the
        # historical seeded quarter-turn sequence when ``p`` is one.
        if self.p == 0.0 or (self.p < 1.0 and np.random.rand() >= self.p):
            return data, meta

        k = self.sample_k()
        if k == 0:
            return data, meta

        # Preserve the historical image-frame path when no pivot is requested
        if self.center is None and not self.use_geo_center:
            return self.apply_image_frame_rotation(
                data, meta, keys, k, context["spatial"]
            )

        # Resolve the physical pivot and corresponding output image frame
        pivot = self.resolve_center(meta, self.center, self.use_geo_center)
        rot_meta = (
            meta if self.keep_meta else self.generate_centered_meta(meta, pivot, k)
        )

        spatial = context["spatial"]

        # Rotate every declared point field around the shared pivot
        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = rot_meta
                continue

            adapter = spatial[key]
            transformed = {}
            keep_masks = []
            for field in adapter.fields:
                coords_cm = field_to_cm(field, meta)
                rot_cm = self.rotate_points(coords_cm, pivot, k)
                if self.keep_meta and field.primary:
                    keep_masks.append(rot_meta.inner_mask(rot_cm))
                transformed[field.name] = field_from_cm(field, rot_cm, rot_meta)

            keep_mask = None
            if keep_masks:
                keep_mask = np.logical_and.reduce(keep_masks)
                for field in adapter.primary_fields:
                    transformed[field.name] = transformed[field.name][keep_mask]
                adapter.select_rows(keep_mask)
            for name, values in transformed.items():
                adapter.set_field(name, values)
            adapter.set_meta(rot_meta)

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
        # Odd quarter turns exchange grid dimensions within the rotation plane
        count = meta.count.copy()
        size = meta.size.copy()
        if k % 2:
            axis_a, axis_b = self.axes
            count[[axis_a, axis_b]] = count[[axis_b, axis_a]]
            size[[axis_a, axis_b]] = size[[axis_b, axis_a]]

        # Rotate the image center to determine the new physical lower bound
        dimensions = size * count
        meta_center = ((meta.lower + meta.upper) / 2.0).reshape(1, -1)
        new_center = self.rotate_points(meta_center, pivot, k)[0]
        lower = new_center - dimensions / 2.0

        return self.make_snapped_meta(
            meta,
            size.astype(meta.size.dtype),
            count.astype(meta.count.dtype),
            lower,
        )

    def apply_image_frame_rotation(
        self,
        data: dict[str, Any],
        meta: Meta,
        keys: list[str],
        k: int,
        spatial: dict[str, SpatialAdapter],
    ) -> tuple[dict[str, Any], Meta]:
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

        # Rotate the image grid and each declared point field in its voxel frame
        rot_meta = self.generate_meta(meta, k)
        for key in keys:
            if isinstance(data[key], Meta):
                data[key] = rot_meta
                continue

            adapter = spatial[key]
            for field in adapter.fields:
                field_px = field_to_px(field, meta)
                coords = self.rotate_coords(
                    field_px, meta.count, k, discrete=field.discrete
                )
                adapter.set_field(field.name, field_from_px(field, coords, rot_meta))
            adapter.set_meta(rot_meta)

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
        if self.k_choices is not None:
            index = int(np.random.randint(len(self.k_choices)))
            return self.k_choices[index]

        return int(np.random.randint(4))

    def rotate_coords(
        self,
        coords: np.ndarray,
        count: np.ndarray,
        k: int,
        discrete: bool | None = None,
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
        # Cache source coordinates and axis sizes because assignments overlap
        rot_coords = coords.copy()
        axis_a, axis_b = self.axes
        count_a = int(count[axis_a])
        count_b = int(count[axis_b])

        # Integer coordinates identify cell centers and therefore reflect about
        # ``count - 1``. Floating coordinates locate continuous points relative
        # to image edges and reflect about ``count`` without quantization.
        if discrete is None:
            discrete = np.issubdtype(coords.dtype, np.integer)
        offset = int(discrete)
        if k == 1:
            rot_coords[:, axis_a] = count_b - offset - coords[:, axis_b]
            rot_coords[:, axis_b] = coords[:, axis_a]
        elif k == 2:
            rot_coords[:, axis_a] = count_a - offset - coords[:, axis_a]
            rot_coords[:, axis_b] = count_b - offset - coords[:, axis_b]
        elif k == 3:
            rot_coords[:, axis_a] = coords[:, axis_b]
            rot_coords[:, axis_b] = count_a - offset - coords[:, axis_a]

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
        # Express the two rotation-plane coordinates relative to the pivot
        rot_points = points.copy()
        axis_a, axis_b = self.axes
        rel_a = points[:, axis_a] - pivot[axis_a]
        rel_b = points[:, axis_b] - pivot[axis_b]

        # Apply the matching right-angle transform in physical coordinates
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
        # Swap voxel geometry for odd turns while retaining the original origin
        count = meta.count.copy()
        size = meta.size.copy()
        if k % 2:
            axis_a, axis_b = self.axes
            count[[axis_a, axis_b]] = count[[axis_b, axis_a]]
            size[[axis_a, axis_b]] = size[[axis_b, axis_a]]

        lower = meta.lower.copy()
        upper = lower + size * count

        return Meta(lower=lower, upper=upper, size=size, count=count)
