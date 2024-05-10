"""Module with a data class object which represents rasterized images metadata.

This copies the internal structure of either :class:`larcv.ImageMeta` for 2D
images or :class:`larcv.Voxel3DMeta` for 3D images.
"""

from dataclasses import dataclass

import numpy as np

from .base import DataStructBase

__all__ = ['Meta']


@dataclass
class Meta(DataStructBase):
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
    _fixed_length_attrs = {'lower': 3, 'upper': 3, 'size': 3,
                           'count': (3, np.int64)}

    # Attributes specifying vector components
    _vec_attrs = ['lower', 'upper', 'size', 'count']

    @property
    def dimension(self):
        """Number of dimensions in the image.

        Returns
        -------
        int
            Number of dimensions in the image
        """
        return len(lower)

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
            mult = np.prod(self.count[i+1:])

        return np.dot(coords, mult)

    def to_cm(self, coords, translate=True):
        """Converts pixel indexes in a tensor to detector coordinates in cm.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input pixel indices
        translate : bool, default True
            If set to `False`, this function returns the input unchanged
        """
        if not translate or len(coords) == 0:
            return coords

        out = self.lower + (coords + .5) * self.size
        return out.astype(np.float32)

    def to_px(self, coords, translate=True):
        """Converts detector coordinates in cm in a tensor to pixel indexes.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input detector coordinates
        translate : bool, default True
            If set to `False`, this function returns the input unchanged
        """
        if not translate or len(coords) == 0:
            return coords

        return (coords - self.lower) / self.size - .5

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
            size  = np.array([meta.rows(), meta.cols()], dtype=np.int64)

        return cls(lower=lower, upper=upper, size=size, count=count)
