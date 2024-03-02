"""Module with data classes of objects used by the reconstruction chain.

Here is the list of available data classes:
  - `TensorBatch`: Batched tensor
  - `MCParticle`: Particle truth information
  - `Meta`: Metadata information about an image
  - `RunInfo`: Information about the run
  - `Flash`: Optical flash
  - `CRTHit`: Cosmic ray tagger hit
  - `Trigger`: Trigger information
"""

import numpy as np
import torch
from typing import Union, List
from dataclasses import dataclass
from larcv import larcv

from .globals import BATCH_COL
from .utils import unique_index


@dataclass
class TensorBatch:
    """Batched tensor with the necessary methods to slice it.

    Attributes
    ----------
    tensor : Union[np.ndarray, torch.Tensor, ME.SparseTensor]
        (N, C) Batched tensor where the batch column is `BATCH_COL`
    splits : Union[List, np.ndarray, torch.Tensor]
        (B) Indexes where to split the batch to get its constituents
    batch_size : int
        Number of entries that make up the batched tensor
    """
    tensor: np.ndarray
    splits: np.ndarray
    batch_size: int

    def __init__(self, tensor, splits=None, batch_size=None, sparse=False):
        """Initialize the attributes of the class.

        Parameters
        ----------
        tensor : Union[np.ndarray, torch.Tensor, ME.SparseTensor]
            (N, C) Batched tensor where the batch column is `BATCH_COL`
        splits : Union[List, np.ndarray, torch.Tensor], optional
            (B) Indexes where to split the batch to get its constituents
        batch_size : int, optional
            Number of entries that make up the batched tensor
        sparse : bool, False
            If initializing from an ME sparse tensor, flip to True
        """
        # Should provide either the split boundaries, or the batch size
        assert (splits is not None) ^ (batch_size is not None), (
                "Provide either `splits` or `batch_size`, not both")

        # Check the typing of the input, store the split function
        self._sparse   = sparse
        self._is_numpy = not sparse and not isinstance(tensor, torch.Tensor)
        self._split_fn = np.split if self._is_numpy else torch.tensor_split

        # If the number of batches is not provided, measure it
        if batch_size is None:
            batch_size = len(splits)

        # If the split boundaries are not provided, must build them once
        if splits is None:
            # Define the array functions depending on the input type
            if self._is_numpy:
                zeros = lambda x: np.zeros(x, dtype=np.int64)
                ones = lambda x: np.ones(x, dtype=np.int64)
                unique = lambda x: np.unique(x, return_index=True)
            else:
                zeros = lambda x: torch.zeros(
                        x, dtype=torch.long, device=tensor.device)
                ones = lambda x: torch.ones(
                        x, dtype=torch.long, device=tensor.device)
                unique = unique_index

            # Get the split list
            if not len(tensor):
                # If the tensor is empty, nothing to divide
                splits = zeros(batch_size)
            else:
                # Find the first index of each batch ID in the input tensor
                ref = tensor if not sparse else tensor.C
                uni, index = unique(ref[:, BATCH_COL])
                splits = -1 * ones(batch_size)
                splits[uni[:-1]] = index[1:]
                splits[uni[-1]] = len(tensor)
                for i, s in enumerate(splits):
                    if s < 0:
                        splits[i] = splits[-1] if i > 0 else 0

        # Store the attributes
        self.tensor = tensor
        self.splits = splits
        self.batch_size = batch_size

    def __len__(self):
        """Returns the number of entries that make up the batch."""
        return self.batch_size

    def __getitem__(self, idx):
        """Returns a subset of the tensor corresponding to one entry.

        Parameters
        ----------
        idx : int
            Entry index
        """
        # Make sure the idx is sensible
        if idx >= self.batch_size:
            raise IndexError(f"Index {idx} out of bound for a batch size of "
                             f"({self.batch_size})")

        # Return
        lower = self.splits[idx-1] if idx > 0 else 0
        upper = self.splits[idx]
        if not self._sparse:
            return self.tensor[lower:upper]
        else:
            from MinkowskiEngine import SparseTensor
            return SparseTensor(
                    self.tensor.F[lower:upper],
                    coordinates=self.tensor.C[lower:upper])

    def split(self):
        """Breaks up the batch into its constituents.
        
        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one tensor per entry in the batch
        """
        if not self._sparse:
            return self._split_fn(self.tensor, self.splits[:-1])
        else:
            from MinkowskiEngine import SparseTensor
            coords = self._split_fn(self.tensor.C, self.splits[:-1])
            feat = self._split_fn(self.tensor.F, self.splits[:-1])
            return [SparseTensor(
                feat[i], coordinates=coords[i]) for i in self.batch_size]

    def to_numpy(self):
        """Cast underlying tensor to a `np.ndarray` and return a new instance.

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        assert not self._is_numpy, (
                "Must be a `torch.Tensor` to be cast to `np.ndarray`")

        tensor = self.tensor
        if self._sparse:
            tensor = torch.cat([self.tensor.C.float(), self.tensor.F], dim=1) 

        tensor = tensor.cpu().detach().numpy()
        splits = self.splits.cpu().detach().numpy()
            
        return TensorBatch(tensor, splits)

    def to_tensor(self, dtype=None, device=None):
        """Cast underlying tensor to a `torch.tensor` and return a new instance.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Data type of the tensor to create
        device : torch.device, optional
            Device on which to put the tensor

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        assert self._is_numpy, (
                "Must be a `np.ndarray` to be cast to `torch.Tensor`")

        tensor = torch.as_tensor(self.tensor, dtype=dtype, device=device)
        splits = torch.as_tensor(self.splits, dtype=torch.int64, device=device)
        return TensorBatch(tensor, splits)

    @classmethod
    def from_list(cls, tensor_list):
        """Builds a batch from a list of tensors.

        Parameters
        ----------
        tensor_list : List[Union[np.ndarray, torch.Tensor]]
            List of tensors, exactly one per batch
        """
        # Check that we are not fed an empty list of tensors
        assert len(tensor_list), (
                "Must provide at least one tensor to build a tensor batch")
        is_numpy = not isinstance(tensor_list[0], torch.Tensor)

        # Compute the splits from the input list
        counts = [len(t) for t in tensor_list]
        splits = np.cumsum(counts)
        if not is_numpy:
            device = tensor_list[0].device
            splits = torch.as_tensor(splits, dtype=torch.int64, device=device)

        # Concatenate input
        if is_numpy:
            return cls(np.concatenate(tensor_list, axis=0), splits)
        else:
            return cls(torch.cat(tensor_list, axis=0), splits)


@dataclass
class MCParticle:
    """Particle truth information.

    Attributes
    ----------
    id : int
        Index of the particle in the list
    TODO
    """
    id: int = -1


@dataclass
class Neutrino:
    """Neutrino truth information.

    Attributes
    ----------
    id : int
        Index of the neutrino in the list
    TODO
    """
    id: int = -1


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


@dataclass
class RunInfo:
    """Run information related to a specific event.

    Attributes
    ----------
    run : int
        Run ID
    subrun : int
        Sub-run ID
    event : int
        Event ID
    """
    run: int    = -1
    subrun: int = -1
    event: int  = -1

    @classmethod
    def from_larcv(cls, tensor):
        """
        Builds and returns a Meta object from a LArCV 2D metadata object

        Parameters
        ----------
        larcv_class : object
             LArCV tensor which contains the run information as attributes

        Returns
        -------
        Meta
            Metadata object
        """
        return cls(run=tensor.run(),
                   subrun=tensor.subrun(),
                   event=tensor.event())


@dataclass
class Flash:
    """Optical flash information.

    Attributes
    ----------
    id : int
        Index of the flash in the list
    time : float
        Time with respect to the trigger in microseconds
    time_width : float
        Width of the flash in microseconds
    time_abs : float
        Time in units of PMT readout clock
    frame : int
        Frame number
    in_beam_frame : bool
        Whether the flash is in the beam frame
    on_beam_time : bool
        Whether the flash time is consistent with the beam window
    total_pe : float
        Total number of PE in the flash
    fast_to_total : float
        Fraction of the total PE contributed by the fast component
    pe_per_optdet : np.ndarray
        (N) Fixed-length array of the number of PE per optical detector
    center : np.ndarray
        Barycenter of the flash in detector coordinates
    width : np.ndarray
        Spatial width of the flash in detector coordinates
    """
    id: int               = -1
    frame: int            = -1
    in_beam_frame: bool   = False
    on_beam_time: bool    = False
    time: float           = -1.0
    time_width: float     = -1.0
    time_abs: float       = -1.0
    total_pe: float       = -1.0
    fast_to_total: float  = -1.0
    pe_per_ch: np.ndarray = np.empty(0, dtype=np.float32)
    center: np.ndarray    = np.full(3, -np.inf, dtype=np.float32)
    width: np.ndarray     = np.full(3, -np.inf, dtype=np.float32)

    @classmethod
    def from_larcv(cls, flash):
        """Builds and returns a Flash object from a LArCV Flash object.

        Parameters
        ----------
        flash : larcv.Flash
            LArCV-format optical flash

        Returns
        -------
        Flash
            Flash object
        """
        # Get the physical center and width of the flash
        axes = ['x', 'y', 'z']
        center = np.array([getattr(flash, f'{a}Center') for a in axes])
        width = np.array([getattr(flash, f'{a}Width') for a in axes])

        # Get the number of PEs per optical channel
        pe_per_ch = np.array(list(flash.PEPerOpDet()), dtype=np.float32)

        return cls(id=flash.id(), frame=flash.frame(),
                   in_beam_frame=flash.inBeamFrame(),
                   on_beam_time=flash.onBeamTime(), time=flash.time(),
                   time_abs=flash.absTime(), time_width=flash.timeWidth(),
                   total_pe=flash.TotalPE(), pe_per_ch=pe_per_ch,
                   center=center, width=width)


@dataclass
class CRTHit:
    """CRT hit information.

    Attributes
    ----------
    id : int
        Index of the CRT hit in the list
    plane : int
        Index of the CRT tagger that registered the hit
    tagger : str
        Name of the CRT tagger that registered the hit
    feb_id : np.ndarray
        Address of the FEB board stored as a list of bytes (uint8)
    ts0_s : int
        Absolute time from White Rabbit (seconds component)
    ts0_ns : float
        Absolute time from White Rabbit (nanoseconds component)
    ts0_s_corr : float
        Unclear in the documentation, placeholder at this point
    ts0_ns_corr : float
        Unclear in the documentation, placeholder at this point
    ts1_ns : float
        Time relative to the trigger (nanoseconds component)
    total_pe : float
        Total number of PE in the CRT hit
    pe_per_ch : np.ndarray
        Number of PEs per FEB channel
    center : np.ndarray
        Barycenter of the CRT hit in detector coordinates
    width : np.ndarray
        Uncertainty on the barycenter of the CRT hit in detector coordinates
    """
    id: int               = -1
    plane: int            = -1
    tagger: str           = ''
    feb_id: np.ndarray    = np.empty(0, dtype=np.ubyte)
    ts0_s: int            = -1
    ts0_ns: float         = -1.0
    ts0_s_corr: float     = -1.0
    ts0_ns_corr: float    = -1.0
    ts1_ns: float         = -1.0
    total_pe: float       = -1.0
    #pe_per_ch: np.ndarray = np.empty(0, dtype=np.float32)
    center: np.ndarray    = np.full(3, -np.inf, dtype=np.float32)
    width: np.ndarray     = np.full(3, -np.inf, dtype=np.float32)

    @classmethod
    def from_larcv(cls, crthit):
        """Builds and returns a CRTHit object from a LArCV CRTHit object.

        Parameters
        ----------
        crthit : larcv.CRTHit
            LArCV-format CRT hit

        Returns
        -------
        CRTHit
            CRT hit object
        """
        # Get the physical center and width of the CRT hit
        axes = ['x', 'y', 'z']
        center = np.array([getattr(crthit, f'{a}_pos') for a in axes])
        width = np.array([getattr(crthit, f'{a}_err') for a in axes])

        # Convert the FEB address to a list of bytes
        feb_id = np.array([ord(c) for c in crthit.feb_id()], dtype=np.ubyte)

        # Get the number of PEs per FEB channel
        # TODO: This is a dictionary of dictionaries, hard to store

        return cls(id=crthit.id(), plane=crthit.plane(),
                   tagger=crthit.tagger(), feb_id=feb_id, ts0_s=crthit.ts0_s(),
                   ts0_ns=crthit.ts0_ns(), ts0_s_corr=crthit.ts0_s_corr(),
                   ts0_ns_corr=crthit.ts0_ns_corr(), ts1_ns=crthit.ts1_ns(),
                   total_pe=crthit.peshit(), center=center, width=width)


@dataclass
class Trigger:
    """Trigger information.

    Attributes
    ----------
    id : int
        Trigger ID
    time_s : int
        Integer seconds component of the UNIX trigger time
    time_ns : int
        Fractional nanoseconds component of the UNIX trigger time
    type : int
        DAQ-specific trigger type
    """
    id: int       = -1
    time_s: int   = -1
    time_ns: int  = -1
    type: int     = -1

    @classmethod
    def from_larcv(cls, trigger):
        """Builds and returns a Trigger object from a LArCV Trigger object.

        Parameters
        ----------
        trigger : larcv.Trigger
            LArCV-format trigger information

        Returns
        -------
        Trigger
            Trigger object
        """
        return cls(id=trigger.id(), time_s=trigger.time_s(),
                   time_ns=trigger.time_ns(), type=trigger.type())
