import numpy as np

from .data_structures import Meta


def pixel_to_cm(coords, meta, translate=True):
    '''
    Converts the pixel indices in a tensor to detector coordinates
    using the metadata information.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2/3) Input pixel indices
    meta : Meta
        Metadata information
    translate : bool, default True
        If set to `False`, this function returns the input unchanged
    '''
    if not translate or not len(coords):
        return coords

    out = meta.lower + (coords + .5) * meta.size
    return out.astype(np.float32)


def cm_to_pixel(coords, meta, translate=True):
    '''
    Converts the detector coordinates in a tensor to pixel indices
    using the metadata information.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2/3) Input detector coordinates
    meta : Meta
        Metadata information
    translate : bool, default True
        If set to `False`, this function returns the input unchanged
    '''
    if not translate or not len(coords):
        return coords

    return (coords - meta.lower) / meta.size - .5
