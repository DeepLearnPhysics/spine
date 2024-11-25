"""Module with a general-purpose geometry class.

This class supports the storage of:
- TPC boundaries
- Optical detector shape/locations
- CRT detector shape/locations

It also provides a plethora of useful functions to query the geometry.
"""

import os
import pathlib
from dataclasses import dataclass

import yaml
import numpy as np
from scipy.spatial.distance import cdist

from .detector import *


@dataclass
class Geometry:
    """Handles all geometry functions for a collection of box-shaped TPCs with
    a arbitrary set of optical detectors organized in optical volumes and CRT
    planes.

    Attributes
    ----------
    tpc : TPCDetector
        TPC detector properties
    optical : OptDetector, optional
        Optical detector properties
    crt : CRTDetector, optional
        CRT detector properties
    """
    tpc: TPCDetector
    optical: OptDetector = None
    crt: CRTDetector = None

    def __init__(self, detector=None, file_path=None):
        """Initializes a detector geometry object.

        The geometry configuration file is a YAML file which contains all the
        necessary information to construct the physical boundaries of the
        a detector (TPC size, positions, etc.) and (optionally) the set
        of optical detectors and CRTs.

        If the detector is already supported, the name is sufficient.
        Supported: 'icarus', 'sbnd', '2x2', '2x2_single', 'ndlar'

        Parameters
        ----------
        detector : str, optional
            Name of a recognized detector to the geometry from
        file_path : str, optional
            Path to a `.yaml` geometry configuration
        """
        # Check that we are provided with either a detector name or a file
        assert (detector is not None) ^ (file_path is not None), (
                "Must provide either a `detector` name or a geometry "
                "`file_path`, not neither, not both.")

        # If a detector name is provided, find the corresponding geometry file
        if detector is not None:
            path = pathlib.Path(__file__).parent
            file_path = os.path.join(
                    path, 'source', f'{detector.lower()}_geometry.yaml')

        # Check that the geometry configuration file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                    f"Could not find the geometry file: {file_path}")

        # Load the geometry file, parse it
        with open(file_path, 'r', encoding='utf-8') as cfg_yaml:
            cfg = yaml.safe_load(cfg_yaml)

        self.parse_configuration(**cfg)

        # Initialize place-holders for the containment conditions to be defined
        # by the end-user using :func:`define_containment_volumes`
        self._cont_volumes = None
        self._cont_use_source = False

    def parse_configuration(self, tpc, optical=None, crt=None):
        """Parse the geometry configuration.

        Parameters
        ----------
        tpc : dict
            Detector boundary configuration
        optical : dict, optional
            Optical detector configuration
        crt : dict, optional
            CRT detector configuration
        """
        # Load the charge detector boundaries
        self.tpc = TPCDetector(**tpc)

        # Load the optical detectors
        if optical is not None:
            self.parse_optical(**optical)

        # Load the CRT detectors
        if crt is not None:
            self.crt = CRTDetector(**crt)

    def parse_optical(self, volume, **optical):
        """Parse the optical detector configuration.

        Parameters
        ----------
        volume : str
            Optical volume boundaries (one of 'tpc' or 'module')
        **optical : dict
            Reset of the optical detector configuration
        """
        # Get the number of optical volumes based on the the volume type
        assert volume in ['module', 'tpc'], (
                "Optical detector positions must be provided by TPC or module.")

        if volume == 'module':
            offsets = [module.center for module in self.tpc.modules]
        else:
            offsets = [chamber.center for chamber in self.tpc.chambers]

        # Initialize the optical detector object
        self.optical = OptDetector(volume, offsets, **optical)

    def get_sources(self, sources):
        """Converts logical TPC indexes to physical TPC indexes.

        Parameters
        ----------
        sources : np.ndarray
            (N, 2) Array of logical [module ID, tpc ID] pairs, one per point

        Returns
        ----------
        np.ndarray
            (N, 2) Array of physical [module ID, tpc ID] pairs, one per point
        """
        # If logical and physical TPCs are aligned, nothing to do
        if self.tpc.det_ids is None:
            return sources.astype(int)

        # Otherwise, map logical to physical
        sources = np.copy(sources)
        sources[:, 1] = self.tpc.det_ids[sources[:, 1].astype(int)]

        return sources

    def get_contributors(self, sources):
        """Gets the list of [module ID, tpc ID] pairs that contributed to a
        particle or interaction object, as defined in this geometry.

        Parameters
        ----------
        sources : np.ndarray
            (N, 2) Array of [module ID, tpc ID] pairs, one per point

        Returns
        -------
        List[np.ndarray]
            (2, N_t) Pair of arrays: the first contains the list of
            contributing modules, the second of contributing tpcs.
        """
        # Fetch the list of unique logical [module ID, tpc ID] pairs
        sources = np.unique(sources, axis=0)

        # If the logical TPCs differ from the physical TPCs, convert
        if self.tpc.det_ids is not None:
            sources = self.get_sources(sources)
            sources = np.unique(sources, axis=0)

        # Return as a list of physical [module ID, tpc ID] pairs
        return list(sources.T)

    def get_volume_index(self, sources, module_id, tpc_id=None):
        """Gets the list of indices of points that belong to a certain
        detector volume (module or individual TPC).

        Parameters
        ----------
        sources : np.ndarray
            (N, 2) Array of [module ID, tpc ID] pairs, one per point
        module_id : int
            ID of the module
        tpc_id : int, optional
            ID of the TPC within the module. If not specified, the volume
            offsets are estimated w.r.t. the module itself

        Returns
        -------
        np.ndarray
            (N) Index of points that belong to the requested detector volume
        """
        # If the logical TPCs differ from the physical TPCs, convert
        sources = self.get_sources(sources)

        # Compute and return the index
        if tpc_id is None:
            return np.where(sources[:, 0] == module_id)[0]
        else:
            return np.where((sources == [module_id, tpc_id]).all(axis=-1))[0]

    def get_closest_tpc(self, points):
        """For each point, find the ID of the closest TPC.

        There is a natural assumption that all TPCs are boxes of identical
        sizes, so that the relative proximitity of a point to a TPC is
        equivalent to its proximity to the TPC center.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates

        Returns
        -------
        np.ndarray
            (N) List of TPC indexes, one per input point
        """
        # Get the TPC centers
        centers = np.asarray([chamber.center for chamber in self.tpc.chambers])

        # Compute the pair-wise distances between points and TPC centers
        dists = cdist(points, centers)

        # Return the closest center index as the closest centers
        return np.argmin(dists, axis=1)

    def get_closest_module(self, points):
        """For each point, find the ID of the closest module.

        There is a natural assumption that all modules are boxes of identical
        sizes, so that the relative proximitity of a point to a module is
        equivalent to its proximity to the module center.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates

        Returns
        -------
        np.ndarray
            (N) List of module indexes, one per input point
        """
        # Get the module centers
        centers = np.asarray([module.center for module in self.tpc.modules])

        # Compute the pair-wise distances between points and module centers
        dists = cdist(points, centers)

        # Return the closest center index as the closest centers
        return np.argmin(dists, axis=1)

    def get_closest_tpc_indexes(self, points):
        """For each TPC, get the list of points that live closer to it than any
        other TPC in the detector.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates

        Returns
        -------
        List[np.ndarray]
            List of index of points that belong to each TPC
        """
        # Start by finding the closest TPC to each of the points
        closest_ids = self.get_closest_tpc(points)

        # For each TPC, append the list of point indices associated with it
        tpc_indexes = []
        for t in range(self.tpc.num_chambers):
            tpc_indexes.append(np.where(closest_ids == t)[0])

        return tpc_indexes

    def get_closest_module_indexes(self, points):
        """For each module, get the list of points that live closer to it
        than any other module in the detector.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates

        Returns
        -------
        List[np.ndarray]
            List of index of points that belong to each module
        """
        # Start by finding the closest TPC to each of the points
        closest_ids = self.get_closest_module(points)

        # For each module, append the list of point indices associated with it
        module_indexes = []
        for m in range(self.tpc.num_modules):
            module_indexes.append(np.where(closest_ids == m)[0])

        return module_indexes

    def get_volume_offsets(self, points, module_id, tpc_id=None):
        """Compute how far each point is from a certain detector volume.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) : Point coordinates
        module_id : int
            ID of the module
        tpc_id : int, optional
            ID of the TPC within the module. If not specified, the volume
            offsets are estimated w.r.t. the module itself

        Returns
        -------
        np.ndarray
            (N, 3) Offsets w.r.t. to the volume boundaries
        """
        # Compute the axis-wise distances of each point to each boundary
        idx = module_id if tpc_id is None else (module_id, tpc_id)
        boundaries = self.tpc[idx].boundaries
        dists = points[..., None] - boundaries

        # If a point is between two boundaries, the distance is 0. If it is
        # outside, the distance is that of the closest boundary
        signs = (np.sign(dists[..., 0]) + np.sign(dists[..., 1]))/2
        offsets = signs * np.min(np.abs(dists), axis=-1)

        return offsets

    def get_min_volume_offset(self, points, module_id, tpc_id=None):
        """Get the minimum offset to apply to a point cloud to bring it
        within the boundaries of a volume.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) : Point coordinates
        module_id : int
            ID of the module
        tpc_id : int, optional
            ID of the TPC within the module. If not specified, the volume
            offsets are estimated w.r.t. the module itself

        Returns
        -------
        np.ndarray
            (3) Offsets w.r.t. to the volume location
        """
        # Compute the distance for each point, get the maximum necessary offset
        offsets = self.get_volume_offsets(points, module_id, tpc_id)
        offsets = offsets[np.argmax(np.abs(offsets), axis=0), np.arange(3)]

        return offsets

    def translate(self, points, source_id, target_id, factor=None):
        """Moves a point cloud from one module to another one

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates
        source_id: int
            Module ID from which to move the point cloud
        target_id : int
            Module ID to which to move the point cloud
        factor : Union[float, np.ndarray], optional
            Multiplicative factor to apply to the offset. This is necessary if
            the points are not expressed in detector coordinates

        Returns
        -------
        np.ndarray
            (N, 3) Set of translated point coordinates
        """
        # If the source and target are the same, nothing to do here
        if target_id == source_id:
            return np.copy(points)

        # Fetch the inter-module shift
        offset = self.tpc[target_id].center - self.tpc[source_id].center
        if factor is not None:
            offset *= factor

        # Translate
        return points + offset

    def split(self, points, target_id, sources=None, meta=None):
        """Migrate all points to a target module, organize them by module ID.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates
        target_id : int
            Module ID to which to move the point cloud
        sources : np.ndarray, optional
            (N, 2) Array of [module ID, tpc ID] pairs, one per voxel
        meta : Meta, optional
            Meta information about the voxelized image. If provided, the
            points are assumed to be provided in voxel coordinates.

        Returns
        -------
        np.ndarray
            (N, 3) Shifted set of points
        List[np.ndarray]
            List of index of points that belong to each module
        """
        # Check that the target ID exists
        assert target_id > -1 and target_id < self.tpc.num_modules, (
                "Target ID should be in [0, N_modules[.")

        # Get the module ID of each of the input points
        convert = False
        if sources is not None:
            # If sources are provided, simply use that
            module_indexes = []
            for m in range(self.tpc.num_modules):
                module_indexes.append(np.where(sources[:, 0] == m)[0])

        else:
            # If the points are expressed in pixel coordinates, translate
            convert = meta is not None
            if convert:
                points = meta.to_cm(points, center=True)

            # If not provided, find module each point belongs to by proximity
            module_indexes = self.get_closest_module_indexes(points)

        # Now shifts all points that are not in the target
        for module_id, module_index in enumerate(module_indexes):
            # If this is the target module, nothing to do here
            if module_id == target_id:
                continue

            # Shift the coordinates
            points[module_index] = self.translate(
                    points[module_index], module_id, target_id)

        # Bring the coordinates back to pixels, if they were shifted
        if convert:
            points = meta.to_px(points, floor=True)

        return points, module_indexes

    def check_containment(self, points, sources=None,
                          allow_multi_module=False, summarize=True):
        """Check whether a point cloud comes within some distance of the
        boundaries of a certain subset of detector volumes, depending on the
        mode.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of point coordinates
        sources : np.ndarray, optional
            (S, 2) : List of [module ID, tpc ID] pairs that created the
            point cloud
        allow_multi_module : bool, default `False`
            Whether to allow particles/interactions to span multiple modules
        summarize : bool, default `True`
            If `True`, only returns a single flag for the whole cloud.
            Otherwise, returns a boolean array corresponding to each point.

        Returns
        -------
        Union[bool, np.ndarray]
            `True` if the particle is contained, `False` if not
        """
        # If the containment volumes are not defined, throw
        if self._cont_volumes is None:
            raise ValueError("Must call `define_containment_volumes` first.")

        # If sources are provided, only consider source volumes
        if self._cont_use_source:
            # Get the contributing TPCs
            assert len(points) == len(sources), (
                    "Need to provide sources to make a source-based check.")
            contributors = self.get_contributors(sources)
            if not allow_multi_module and len(np.unique(contributors[0])) > 1:
                return False

            # Define the smallest box containing all contributing TPCs
            # TODO: this is not ideal
            index = contributors[0] * self.tpc.num_chambers_per_module + contributors[1]
            volume = self.merge_volumes(self._cont_volumes[index])
            volumes = [volume]

        else:
            volumes = self._cont_volumes

        # Loop over volumes, make sure the cloud is contained in at least one
        if summarize:
            contained = False
            for v in volumes:
                if (points > v[:, 0]).all() and (points < v[:, 1]).all():
                    contained = True
                    break
        else:
            contained = np.zeros(len(points), dtype=bool)
            for v in volumes:
                contained |= ((points > v[:, 0]).all(axis=1) &
                              (points < v[:, 1]).all(axis=1))

        return contained

    def define_containment_volumes(self, margin, cathode_margin=None,
                                   mode ='module'):
        """This function defines a list of volumes to check containment against.

        If the containment is checked against a constant volume, it is more
        efficient to call this function once and call `check_containment`
        reapitedly after.

        Parameters
        ----------
        margin : Union[float, List[float], np.array]
            Minimum distance from a detector wall to be considered contained:
            - If float: distance buffer is shared between all 6 walls
            - If [x,y,z]: distance is shared between pairs of walls facing
              each other and perpendicular to a shared axis
            - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is
              specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'tpc', makes sure it is contained within a single TPC
            - If 'module', makes sure it is contained within a single module
            - If 'detector', makes sure it is contained within in the detector
            - If 'source', use the origin of voxels to determine which TPC(s)
              contributed to them, and define volumes accordingly
        """
        # Translate the margin parameter to a (3,2) matrix
        if np.isscalar(margin):
            margin = np.full((3, 2), margin)
        elif len(np.array(margin).shape) == 1:
            assert len(margin) == 3, (
                    "Must provide one value per axis.")
            margin = np.repeat([margin], 2, axis=0).T
        else:
            assert np.array(margin).shape == (3, 2), (
                    "Must provide two values per axis.")
            margin = np.copy(margin)

        # Establish the volumes to check against
        self._cont_volumes = []
        if mode in ['tpc', 'source']:
            for m, module in enumerate(self.tpc):
                for t, tpc in enumerate(module):
                    vol = self.adapt_volume(
                            tpc.boundaries, margin, cathode_margin, m, t)
                    self._cont_volumes.append(vol)
            self._cont_use_source = mode == 'source'

        elif mode == 'module':
            for module in self.tpc:
                vol = self.adapt_volume(module.boundaries, margin)
                self._cont_volumes.append(vol)
            self._cont_use_source = False

        elif mode == 'detector':
            vol = self.adapt_volume(self.tpc.boundaries, margin)
            self._cont_volumes.append(vol)
            self._cont_use_source = False

        else:
            raise ValueError(f"Containement check mode not recognized: {mode}.")

        self._cont_volumes = np.array(self._cont_volumes)

    def adapt_volume(self, ref_volume, margin, cathode_margin=None,
                     module_id=None, tpc_id=None):
        """Apply margins from a given volume. Takes care of subtleties
        associated with the cathode, if needed.

        Parameters
        ----------
        ref_volume : np.ndarray
            (3, 2) Array of volume boundaries
        margin : np.ndarray
            Minimum distance from a detector wall to be considered contained as
            [[x_low,x_up], [y_low,y_up], [z_low,z_up]], i.e. distance is
            specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        module_id : int, optional
            ID of the module
        tpc_id : int, optional
            ID of the TPC within the module

        Returns
        -------
        np.ndarray
            (3, 2) Updated array of volume boundaries
        """
        # Reduce the volume according to the margin
        volume = np.copy(ref_volume)
        volume[:,0] += margin[:,0]
        volume[:,1] -= margin[:,1]

        # If a cathode margin is provided, adapt the cathode wall differently
        if cathode_margin is not None:
            axis = self.tpc[module_id, tpc_id].drift_axis
            side = self.tpc[module_id, tpc_id].cathode_side

            flip = (-1) ** side
            volume[axis, side] += flip * (cathode_margin - margin[axis, side])

        return volume

    @staticmethod
    def merge_volumes(volumes):
        """Given a list of volumes and their boundaries, find the smallest box
        that encompass all volumes combined.

        Parameters
        ----------
        volumes : np.ndarray
            (N, 3, 2) List of volume boundaries

        Returns
        -------
        np.ndarray
            (3, 2) Boundaries of the combined volume
        """
        volume = np.empty((3, 2))
        volume[:, 0] = np.min(volumes, axis=0)[:, 0]
        volume[:, 1] = np.max(volumes, axis=0)[:, 1]

        return volume
