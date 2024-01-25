import numpy as np
from larcv import larcv

from mlreco.utils.data_structures import Meta, RunInfo


def parse_meta2d(sparse_event, projection_id = 0):
    '''
    Get the meta information to translate into real world coordinates (2D).

    Each entry in a dataset is a cube, where pixel coordinates typically go
    from 0 to some integer N in each dimension. If you wish to translate
    these voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block:: yaml

        schema:
          meta:
            parser: parse_meta2d
            args:
              sparse_event: sparse2d_pcluster
              projection_id: 0

    Parameters
    ----------
    sparse2d_event : Union[larcv.EventSparseTensor2D, larcv.EventClusterVoxel2D]
        Tensor which contains the metadata information as an attribute
    projection_id : int
        Projection ID to get the 2D image from

    Returns
    -------
    Meta
        Metadata information object
    '''
    tensor2d = sparse_event.sparse_tensor_2d(projection_id)

    return Meta.from_larcv(tensor2d.meta())


def parse_meta3d(sparse_event):
    '''
    Get the meta information to translate into real world coordinates (3D).

    Each entry in a dataset is a cube, where pixel coordinates typically go
    from 0 to some integer N in each dimension. If you wish to translate
    these voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block:: yaml

        schema:
          meta:
            parser: parse_meta3d
            args:
              sparse_event: sparse3d_pcluster

    Parameters
    ----------
    sparse_event : Union[larcv.EventSparseTensor3D or larcv.EventClusterVoxel3D]
        Tensor which contains the metadata information as an attribute

    Returns
    -------
    Meta
        Metadata information object
    '''
    return Meta.from_larcv(sparse_event.meta())


def parse_run_info(sparse_event):
    '''
    Parse run info (run, subrun, event number)

    .. code-block:: yaml

        schema:
          run_info:
            parser: parse_run_info
            args:
              sparse_event: sparse3d_pcluster

    Parameters
    ----------
    sparse_event : Union[larcv::EventSparseTensor3D, larcv::EventClusterVoxel3D]
        Tensor which contains the run information as attributes

    Returns
    -------
    RunInfo
        Run information object
    '''
    return RunInfo.from_larcv(sparse_event)


def parse_opflash(opflash_event):
    '''
    Copy construct OpFlash and return an array of larcv::Flash.

    .. code-block:: yaml
        schema:
          opflash_cryoE:
            parser:parse_opflash
            opflash_event: opflash_cryoE

    Configuration
    -------------
    opflash_event: larcv::EventFlash or list of larcv::EventFlash

    Returns
    -------
    list
    '''
    if not isinstance(opflash_event, list):
        opflash_event = [opflash_event]

    opflash_list = []
    for x in opflash_event:
        opflash_list.extend(x.as_vector())

    opflashes = [larcv.Flash(f) for f in opflash_list]
    return opflashes


def parse_crthits(crthit_event):
    '''
    Copy construct CRTHit and return an array of larcv::CRTHit.

    .. code-block:: yaml
        schema:
          crthits:
            parser: parse_crthits
            crthit_event: crthit_crthit

    Configuration
    -------------
    crthit_event: larcv::CRTHit

    Returns
    -------
    list
    '''
    crthits = [larcv.CRTHit(c) for c in crthit_event.as_vector()]
    return crthits


def parse_trigger(trigger_event):
    '''
    Copy construct Trigger and return an array of larcv::Trigger.

    .. code-block:: yaml
        schema:
          trigger:
            parser: parse_trigger
            trigger_event: trigger_base

    Configuration
    -------------
    trigger_event: larcv::TriggerEvent

    Returns
    -------
    list
    '''
    trigger = larcv.Trigger(trigger_event)
    return trigger
