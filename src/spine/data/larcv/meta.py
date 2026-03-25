"""Module with a data class object which represents rasterized images metadata.

This copies the internal structure of either :class:`larcv.ImageMeta` for 2D
images or :class:`larcv.Voxel3DMeta` for 3D images.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Self

import numpy as np

from spine.data.base import DataBase
from spine.data.field import FieldMetadata

__all__ = ["ImageMeta2D", "ImageMeta3D", "Meta"]


@dataclass(eq=False)
class ImageMetaBase(DataBase):
    """Base class for rasterized image metadata.

    Provides common functionality for both 2D and 3D image metadata,
    including coordinate transformations and pixel indexing.

    This class should not be instantiated directly. Use ImageMeta2D
    or ImageMeta3D instead.

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

    if TYPE_CHECKING:  # pragma: no cover
        # Type hints for fields defined in subclasses (only for type checking)
        lower: np.ndarray
        upper: np.ndarray
        size: np.ndarray
        count: np.ndarray

    # Internal attribute for index multipliers (not part of the dataclass fields)
    _index_multipliers: Optional[np.ndarray] = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self):
        """Validate the consistency of the meta parameters."""
        # Call the parent post_init to perform any additional validation
        super().__post_init__()

        # If nothing is initialized, skip validation (allows for default construction)
        if (
            np.isnan(self.lower).all()
            and np.isnan(self.upper).all()
            and np.isnan(self.size).all()
            and (self.count == -1).all()
        ):
            return

        # If anything is initialized, all must be initialized
        if (
            np.isnan(self.lower).any()
            or np.isnan(self.upper).any()
            or np.isnan(self.size).any()
            or (self.count == -1).any()
        ):
            raise ValueError(
                "If any of lower, upper, size, or count are initialized, "
                "all must be initialized with valid values"
            )

        # Validate that the count and size yield the correct upper bound
        expected_upper = self.lower + self.size * self.count
        if not np.allclose(self.upper, expected_upper, equal_nan=True):
            raise ValueError(
                "Upper must be equal to lower + size * count (within numerical precision)"
            )

    @property
    def index_multipliers(self) -> np.ndarray:
        """Get the index multipliers for converting between pixel coordinates
        and unique pixel indices.

        Returns
        -------
        np.ndarray
            Array of index multipliers for each dimension
        """
        if self._index_multipliers is None:
            if self.count is None or (self.count == -1).any():
                raise ValueError(
                    "Cannot compute index multipliers without valid count information"
                )

            self._index_multipliers = np.empty(self.dimension, dtype=np.int64)
            for i in range(self.dimension):
                self._index_multipliers[i] = np.prod(self.count[i + 1 :])

        return self._index_multipliers

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
        return np.dot(coords, self.index_multipliers).astype(np.int64)

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


@dataclass(eq=False)
class ImageMeta2D(ImageMetaBase):
    """2D rasterized image metadata.

    Represents metadata for 2D images (e.g., wire-plane detector readouts).
    Corresponds to LArCV's ImageMeta class.

    Attributes
    ----------
    lower : np.ndarray
        (2) Array of image lower bounds in detector coordinates (cm)
    upper : np.ndarray
        (2) Array of image upper bounds in detector coordinates (cm)
    size : np.ndarray
        (2) Array of pixel size in each dimension (cm)
    count : np.ndarray
        (2) Array of pixel count in each dimension
    """

    # Vector attributes
    lower: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=2, dtype=np.float32, vector=True, units="cm"),
    )
    upper: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=2, dtype=np.float32, vector=True, units="cm"),
    )
    size: np.ndarray = field(
        default_factory=lambda: np.full(2, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=2, dtype=np.float32, vector=True, units="cm"),
    )
    count: np.ndarray = field(
        default_factory=lambda: np.full(2, -1, dtype=np.int64),
        metadata=FieldMetadata(length=2, dtype=np.int64, vector=True),
    )

    @classmethod
    def from_larcv(cls, meta) -> Self:
        """Builds and returns an ImageMeta2D object from a LArCV 2D metadata object.

        Parameters
        ----------
        meta : larcv.ImageMeta
            LArCV-format 2D metadata

        Returns
        -------
        ImageMeta2D
            2D metadata object
        """
        lower = np.array([meta.min_x(), meta.min_y()], dtype=np.float32)
        upper = np.array([meta.max_x(), meta.max_y()], dtype=np.float32)
        size = np.array([meta.pixel_height(), meta.pixel_width()], dtype=np.float32)
        count = np.array([meta.rows(), meta.cols()], dtype=np.int64)

        return cls(lower=lower, upper=upper, size=size, count=count)


@dataclass(eq=False)
class ImageMeta3D(ImageMetaBase):
    """3D rasterized image metadata.

    Represents metadata for 3D voxelized images (e.g., LArTPC pixel readouts).
    Corresponds to LArCV's Voxel3DMeta class.

    Attributes
    ----------
    lower : np.ndarray
        (3) Array of image lower bounds in detector coordinates (cm)
    upper : np.ndarray
        (3) Array of image upper bounds in detector coordinates (cm)
    size : np.ndarray
        (3) Array of pixel size in each dimension (cm)
    count : np.ndarray
        (3) Array of pixel count in each dimension
    """

    # Vector attributes
    lower: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True, units="cm"),
    )
    upper: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True, units="cm"),
    )
    size: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True, units="cm"),
    )
    count: np.ndarray = field(
        default_factory=lambda: np.full(3, -1, dtype=np.int64),
        metadata=FieldMetadata(length=3, dtype=np.int64, vector=True),
    )

    @classmethod
    def from_larcv(cls, meta) -> Self:
        """Builds and returns an ImageMeta3D object from a LArCV 3D metadata object.

        Parameters
        ----------
        meta : larcv.Voxel3DMeta
            LArCV-format 3D metadata

        Returns
        -------
        ImageMeta3D
            3D metadata object
        """
        lower = np.array([meta.min_x(), meta.min_y(), meta.min_z()], dtype=np.float32)
        upper = np.array([meta.max_x(), meta.max_y(), meta.max_z()], dtype=np.float32)
        size = np.array(
            [meta.size_voxel_x(), meta.size_voxel_y(), meta.size_voxel_z()],
            dtype=np.float32,
        )
        count = np.array(
            [meta.num_voxel_x(), meta.num_voxel_y(), meta.num_voxel_z()],
            dtype=np.int64,
        )

        return cls(lower=lower, upper=upper, size=size, count=count)


# Store the original from_larcv methods before we create aliases
_ImageMeta2D_from_larcv = ImageMeta2D.from_larcv
_ImageMeta3D_from_larcv = ImageMeta3D.from_larcv


def _from_larcv_factory(_cls, meta):
    """Factory function for creating ImageMeta from LArCV metadata.

    Automatically selects ImageMeta2D or ImageMeta3D based on the metadata type.

    Parameters
    ----------
    _cls : type
        The class calling this method (not used; for classmethod compatibility)
    meta : larcv.ImageMeta or larcv.Voxel3DMeta
        LArCV-format metadata object

    Returns
    -------
    Union[ImageMeta2D, ImageMeta3D]
        Appropriate metadata object type based on input dimensionality
    """
    # Check if meta is 3D by checking for 3D-specific methods
    if hasattr(meta, "pos_z"):
        # 3D metadata (Voxel3DMeta) - call original method
        return _ImageMeta3D_from_larcv.__func__(ImageMeta3D, meta)
    else:
        # 2D metadata (ImageMeta) - call original method
        return _ImageMeta2D_from_larcv.__func__(ImageMeta2D, meta)


# Backward compatibility alias - Meta is ImageMeta3D
# For new code, use ImageMeta2D or ImageMeta3D explicitly
Meta = ImageMeta3D

# Add factory from_larcv to Meta for backward compatibility with auto-dispatch
# This allows Meta.from_larcv() to work for both 2D and 3D inputs
Meta.from_larcv = classmethod(_from_larcv_factory)  # type: ignore
