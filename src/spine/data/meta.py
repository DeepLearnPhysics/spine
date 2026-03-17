"""Module with a data class object which represents rasterized images metadata.

This copies the internal structure of either :class:`larcv.ImageMeta` for 2D
images or :class:`larcv.Voxel3DMeta` for 3D images.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from .base import DataBase
from .field import FieldMetadata

__all__ = ["Meta"]


@dataclass(eq=False)
class Meta(DataBase):
    """Meta information about a rasterized image.

    Attributes
    ----------
    lower : np.ndarray
        (2/3) Array of image lower bounds in detector coordinates (cm)
    upper : np.ndarray
        (2/3) Array of image upper bounds in detector coordinates (cm)
    size : np.ndarray
        (2/3) Array of pixel size in each dimension (cm)
    count : np.ndarray
        (2/3) Array of pixel count in each dimension
    """

    # Vector attributes
    lower: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, category="vector", units="cm"
        ),
    )
    upper: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, category="vector", units="cm"
        ),
    )
    size: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, category="vector", units="cm"
        ),
    )
    count: np.ndarray = field(
        default_factory=lambda: np.full(3, -1, dtype=np.int64),
        metadata=FieldMetadata(length=3, dtype=np.int64, category="vector"),
    )

    @property
    def dimension(self) -> int:
        """Number of dimensions in the image.

        Returns
        -------
        int
            Number of dimensions in the image
        """
        return len(self.lower)

    @property
    def num_elements(self) -> int:
        """Total number of pixel in the image.

        Returns
        -------
        int
            Total number of pixel in the image.
        """
        return int(np.prod(self.count))

    def index(self, coords) -> np.ndarray:
        """Unique pix associated with individual axis indexes.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input pixel indices

        Returns
        -------
        np.ndarray
            (N) Unique pixel index per input pixel
        """
        mult = np.empty(self.dimension, dtype=self.count.dtype)
        for i in range(self.dimension):
            mult[i] = np.prod(self.count[i + 1 :])

        return np.dot(coords, mult).astype(np.int64)

    def to_cm(self, coords: np.ndarray, center: bool = False) -> np.ndarray:
        """Converts pixel coordinates to detector coordinates in cm.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input pixel coordinates
        center : bool, default False
            If `True`, offset the input coordinates by half a pixel size. This
            makes sense to provide unbiased coordinates when converting indexes.

        Returns
        -------
        np.ndarray
            Detector coordinates in cm
        """
        return self.lower + (coords + 0.5 * center) * self.size

    def to_px(self, coords: np.ndarray, floor: bool = False) -> np.ndarray:
        """Converts detector coordinates in cm to pixel coordinates.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input detector coordinates
        floor : bool, default False
            If `True`, converts pixel coordinates to indexes (floor function)

        Returns
        -------
        np.ndarray
            Pixel coordinates
        """
        if floor:
            return np.floor((coords - self.lower) / self.size)

        return (coords - self.lower) / self.size

    def inner_mask(self, coords: np.ndarray) -> np.ndarray:
        """Computes a boolean mask of which coordinates are within the image bounds.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input coordinates in cm

        Returns
        -------
        np.ndarray
            (N) Boolean mask of which coordinates are within the image bounds
        """
        return np.all((coords >= self.lower) & (coords < self.upper), axis=1)

    @classmethod
    def from_larcv(cls, meta) -> Self:
        """Builds and returns a Meta object from a LArCV 2D metadata object.

        Parameters
        ----------
        meta : Union[larcv.ImageMeta, larcv.Voxel3DMeta]
            LArCV-format 2D metadata

        Returns
        -------
        Meta
            Metadata object
        """
        if hasattr(meta, "pos_z"):
            lower = np.array(
                [meta.min_x(), meta.min_y(), meta.min_z()], dtype=np.float32
            )
            upper = np.array(
                [meta.max_x(), meta.max_y(), meta.max_z()], dtype=np.float32
            )
            size = np.array(
                [meta.size_voxel_x(), meta.size_voxel_y(), meta.size_voxel_z()],
                dtype=np.float32,
            )
            count = np.array(
                [meta.num_voxel_x(), meta.num_voxel_y(), meta.num_voxel_z()],
                dtype=np.int64,
            )

        else:
            lower = np.array([meta.min_x(), meta.min_y()], dtype=np.float32)
            upper = np.array([meta.max_x(), meta.max_y()], dtype=np.float32)
            size = np.array([meta.pixel_height(), meta.pixel_width()], dtype=np.float32)
            count = np.array([meta.rows(), meta.cols()], dtype=np.int64)

        return cls(lower=lower, upper=upper, size=size, count=count)
