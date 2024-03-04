"""Module with a data class object which represents rasterized images metadata.

This copies the internal structure of either :class:`larcv.ImageMeta` for 2D
images or :class:`larcv.Voxel3DMeta` for 3D images.
"""

import numpy as np
from typing import Union
from dataclasses import dataclass
from larcv import larcv


@dataclass
class Meta:
    """Meta information about a rasterized image.

    Attributes
    ----------
    lower : np.ndarray
        (2/3) Array of image lower bounds in detector coordinates (cm)
    upper : np.ndarray
        (2/3) Array of image upper bounds in detector coordinates (cm)
    size : np.ndarray
        (2/3) Array of pixel/voxel size in each dimension (cm)
    """
    lower: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    upper: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    size: np.ndarray  = np.full(3, -np.inf, dtype=np.float32)

    # Fixed-length attributes
    _fixed_length_attrs = ['lower', 'upper', 'size']

    def to_cm(self, coords, translate=True):
        """Converts pixel indexes in a tensor to detector coordinates in cm.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input pixel indices
        translate : bool, default True
            If set to `False`, this function returns the input unchanged
        """
        if not translate or not len(coords):
            return coords

        out = self.lower + (coords + .5) * self.size
        return out.astype(np.float32)

    def to_pixel(self, coords, translate=True):
        """Converts detector coordinates in cm in a tensor to pixel indexes.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2/3) Input detector coordinates
        translate : bool, default True
            If set to `False`, this function returns the input unchanged
        """
        if not translate or not len(coords):
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
        if isinstance(meta, larcv.ImageMeta):
            lower = np.array([meta.min_x(), meta.min_y()])
            upper = np.array([meta.max_x(), meta.max_y()])
            size  = np.array([meta.pixel_width(), meta.pixel_height()])
        elif isinstance(meta, larcv.Voxel3DMeta):
            lower = np.array([meta.min_x(), meta.min_y(), meta.min_z()])
            upper = np.array([meta.max_x(), meta.max_y(), meta.max_z()])
            size  = np.array([meta.size_voxel_x(),
                              meta.size_voxel_y(),
                              meta.size_voxel_z()])
        else:
            raise ValueError('Did not recognize metadata:', meta)

        return cls(lower=lower, upper=upper, size=size)
