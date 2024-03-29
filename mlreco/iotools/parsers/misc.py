"""Module that contains parsers that do not fit in other categories.

Contains the following parsers:
- :class:`Meta2DParser`
- :class:`Meta3DParser`
- :class:`RunInfoParser`
- :class:`OpFlashParser`
- :class:`CRTHitParser`
- :class:`TriggerParser`
"""

import numpy as np
from larcv import larcv

from mlreco.utils.data_structures import (
        Meta, RunInfo, Flash, CRTHit, Trigger, ObjectList)

from .parser import Parser

__all__ = ['MetaParser', 'RunInfoParser', 'OpFlashParser',
           'CRTHitParser', 'TriggerParser']


class MetaParser(Parser):
    """Get the metadata information to translate into real world coordinates.

    Each entry in a dataset is a cube, where pixel/voxel coordinates typically
    go from 0 to some integer N in each dimension. If you wish to translate
    these pixel/voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block. yaml

        schema:
          meta:
            parser: parse_meta
            sparse_event: sparse3d_pcluster
    """
    name = 'parse_meta'
    aliases = ['parse_meta2d', 'parse_meta3d']

    def __init__(self, projection_id=None, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        projection_id : int, optional
            Projection ID to get the 2D image from (if fetching from 2D)
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.projection_id = projection_id

    def process(self, sparse_event=None, cluster_event=None):
        """Fetches the metadata from one object that has it.

        Parameters
        ----------
        sparse_event : Union[larcv.EventSparseTensor2D
                             larcv.EventSparseTensor3D], optional
            Tensor which contains the metadata information as an attribute
        cluster_event : Union[larcv.EventClusterPixel2D,
                              larcv.EventClusterVoxel3D], optional
            Cluster which contains the metadata information as an attribute

        Returns
        -------
        Meta
            Metadata information for one image
        """
        # Check on the input, pick a source for the metadata
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must specify either `sparse_event` or `cluster_event`")
        ref_event = sparse_event if sparse_event is not None else cluster_event

        # Fetch a specific projection, if needed
        if isinstance(ref_event,
                      (larcv.EventSparseTensor2D, larcv.EventClusterPixel2D)):
            ref_event = ref_event.sparse_tensor_2d(self.projection_id)

        return Meta.from_larcv(ref_event.meta())


class RunInfoParser(Parser):
    """Parse run information (run, subrun, event number).

    .. code-block. yaml

        schema:
          run_info:
            parser: parse_run_info
            sparse_event: sparse3d_pcluster
    """
    name = 'parse_run_info'

    def process(self, sparse_event=None, cluster_event=None):
        """Fetches the run information from one object that has it.

        Parameters
        ----------
        sparse_event : Union[larcv.EventSparseTensor2D
                             larcv.EventSparseTensor3D], optional
            Tensor which contains the run information as an attribute
        cluster_event : Union[larcv.EventClusterPixel2D,
                              larcv.EventClusterVoxel3D], optional
            Cluster which contains the run information as an attribute

        Returns
        -------
        RunInfo
            Run information object
        """
        # Check on the input, pick a source for the run information
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must specify either `sparse_event` or `cluster_event`")
        ref_event = sparse_event if sparse_event is not None else cluster_event

        return RunInfo.from_larcv(ref_event)


class OpFlashParser(Parser):
    """Copy construct OpFlash and return an array of `Flash`.

    .. code-block. yaml
        schema:
          opflash_cryoE:
            parser:parse_opflash
            opflash_event: opflash_cryoE

    """
    name = 'parse_opflashes'
    aliases = ['parse_opflash']

    def process(self, opflash_event=None, opflash_event_list=None):
        """Fetches the list of optical flashes.

        Parameters
        -------------
        opflash_event : larcv.EventFlash, optional
            Optical flash event which contains a list of flash objects
        opflash_event_list : larcv.EventFlash, optional
            List of optical flash events, each a list of flash objects

        Returns
        -------
        List[Flash]
            List of optical flash objects
        """
        # Check on the input, aggregate the sources for the optical flashes
        assert ((opflash_event is not None) ^
                (opflash_event_list is not None)), (
                "Must specify either `opflash_event` or `opflash_event_list`")
        if opflash_event is not None:
            opflash_list = opflash_event.as_vector()
        else:
            opflash_list = []
            for opflash_event in opflash_event_list:
                opflash_list.extend(opflash_event.as_vector())

        # Output as a list of LArCV optical flash objects
        opflashes = [Flash.from_larcv(larcv.Flash(f)) for f in opflash_list]

        return ObjectList(opflashes, Flash())


class CRTHitParser(Parser):
    """Copy construct CRTHit and return an array of `CRTHit`.

    .. code-block. yaml
        schema:
          crthits:
            parser: parse_crthits
            crthit_event: crthit_crthit
    """
    name = 'parse_crthits'
    aliases = ['parse_crthit']

    def process(self, crthit_event):
        """Fetches the list of CRT hits.

        Parameters
        ----------
        crthit_event : larcv.CRTHitEvent

        Returns
        -------
        List[CRTHit]
            List of CRT hit objects
        """
        # Output as a list of LArCV CRT hit objects
        crthit_list = crthit_event.as_vector()
        crthits = [CRTHit.from_larcv(larcv.CRTHit(c)) for c in crthit_list]

        return ObjectList(crthits, CRTHit())


class TriggerParser(Parser):
    """Copy construct Trigger and return a `Trigger`.

    .. code-block. yaml
        schema:
          trigger:
            parser: parse_trigger
            trigger_event: trigger_base
    """
    name = 'parse_trigger'

    def process(self, trigger_event):
        """Fetches the trigger information.

        Parameters
        ----------
        trigger_event : larcv.TriggerEvent

        Returns
        -------
        Trigger
            Trigger object
        """
        # Output as a trigger objects
        trigger = Trigger.from_larcv(larcv.Trigger(trigger_event))

        return trigger
