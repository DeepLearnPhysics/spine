import numpy as np
from dataclasses import dataclass
from larcv import larcv


@dataclass
class Meta:
    '''
    Meta information about a rasterized image

    Attributes
    ----------
    lower : np.ndarray
        (2/3) Array of image lower bounds in detector coordinates (cm)
    upper : np.ndarray
        (2/3) Array of image upper bounds in detector coordinates (cm)
    size : np.ndarray
        (2/3) Array of pixel/voxel size in each dimension (cm)
    '''
    lower : np.ndarray
    upper : np.ndarray
    size : np.ndarray

    @staticmethod
    def from_larcv(meta):
        '''
        Builds and returns a Meta object from a LArCV 2D metadata object

        Parameters
        ----------
        meta : Union[larcv.ImageMeta, larcv.Voxel3DMeta]
            LArCV-format 2D metadata

        Returns
        -------
        Meta
            Metadata object
        '''
        if isinstance(meta, larcv.ImageMeta):
            lower = np.array([meta.min_x(), meta.min_y()])
            upper = np.array([meta.max_x(), meta.max_y()])
            size  = np.array([meta.pixel_width(), meta.pixel_height()])
        elif isinstance(meta, larcv.Voxel3DMeta):
            lower = np.array([meta.min_x(), meta.min_y(), meta.min_z()])
            upper = np.array([meta.max_x(), meta.max_y(), meta.max_z()])
            size  = np.array([meta.size_voxel_x(),
                meta.size_voxel_y(), meta.size_voxel_z()])
        else:
            raise ValueError('Did not recognize metadata:', meta)

        return Meta(lower = lower, upper = upper, size = size)


@dataclass
class RunInfo:
    '''
    Run information related to a specific event

    Attributes
    ----------
    run : int
        Run ID
    subrun : int
        Sub-run ID
    event : int
        Event ID
    '''
    run : int
    subrun : int
    event : int

    @staticmethod
    def from_larcv(tensor):
        '''
        Builds and returns a Meta object from a LArCV 2D metadata object

        Parameters
        ----------
        larcv_class : object
             LArCV tensor which contains the run information as attributes

        Returns
        -------
        Meta
            Metadata object
        '''
        return RunInfo(run = tensor.run(),
                subrun = tensor.subrun(),
                event = tensor.event())
