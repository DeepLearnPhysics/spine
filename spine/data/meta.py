"""Module with a data class object which represents rasterized images metadata.

This copies the internal structure of either :class:`larcv.ImageMeta` for 2D
images or :class:`larcv.Voxel3DMeta` for 3D images.
"""

from dataclasses import dataclass

import numpy as np

from .base import DataBase

__all__ = ['Meta']


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
    lower: np.ndarray = None
    upper: np.ndarray = None
    size: np.ndarray = None
    count: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = (
            ('lower', 3), ('upper', 3), ('size', 3), ('count', (3, np.int64))
    )

    # Attributes specifying vector components
    _vec_attrs = ('lower', 'upper', 'size', 'count')

    @property
    def dimension(self):
        """Number of dimensions in the image.

        Returns
        -------
        int
            Number of dimensions in the image
        """
        return len(self.lower)

    @property
    def num_elements(self):
        """Total number of pixel in the image.

        Returns
        -------
        int
            Total number of pixel in the image.
        """
        return int(np.prod(self.count))

    def index(self, coords):
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
            mult[i] = np.prod(self.count[i+1:])

        return np.dot(coords, mult).astype(np.int64)

    def to_cm(self, coords, center=False):
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
        return self.lower + (coords + .5*center)*self.size

    def to_px(self, coords, floor=False):
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
            return np.floor((coords - self.lower)/self.size)

        return (coords - self.lower)/self.size

    @classmethod
    def from_larcv(cls, meta):
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
        if hasattr(meta, 'pos_z'):
            lower = np.array([meta.min_x(), meta.min_y(),
                              meta.min_z()], dtype=np.float32)
            upper = np.array([meta.max_x(), meta.max_y(),
                              meta.max_z()], dtype=np.float32)
            size  = np.array([meta.size_voxel_x(),
                              meta.size_voxel_y(),
                              meta.size_voxel_z()], dtype=np.float32)
            count = np.array([meta.num_voxel_x(),
                              meta.num_voxel_y(),
                              meta.num_voxel_z()], dtype=np.int64)

        else:
            lower = np.array([meta.min_x(), meta.min_y()], dtype=np.float32)
            upper = np.array([meta.max_x(), meta.max_y()], dtype=np.float32)
            size  = np.array([meta.pixel_height(),
                              meta.pixel_width()], dtype=np.float32)
            count = np.array([meta.rows(), meta.cols()], dtype=np.int64)

        return cls(lower=lower, upper=upper, size=size, count=count)
