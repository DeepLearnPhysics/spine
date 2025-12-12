"""Module with a general-purpose geometry class.

This class supports the storage of:
- TPC boundaries
- Optical detector shape/locations
- CRT detector shape/locations

It also provides a plethora of useful functions to query the geometry.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

from .detector import Box, CRTDetector, OptDetector, TPCDetector


@dataclass
class Geometry(Box):
    """Handles all geometry functions for a collection of box-shaped TPCs with
    a arbitrary set of optical detectors organized in optical volumes and
    auxiliary CRT planes.

    Attributes
    ----------
    name : str
        Name of the detector
    tag : str
        Tag or label for the geometry instance
    version : int
        Version number of the geometry
    tpc : TPCDetector
        TPC detector properties
    optical : OptDetector, optional
        Optical detector properties
    crt : CRTDetector, optional
        CRT detector properties
    gdml : str, optional
        GDML file name associated with the geometry
    crs_files : List[str], optional
        CRS (Charge Readout System) geometry file(s) reference (flow only)
    lrs_file : str, optional
        LRS (Light Readout System) geometry file reference (flow only)
    """

    name: str
    tag: str
    version: str
    tpc: TPCDetector
    optical: Optional[OptDetector] = None
    crt: Optional[CRTDetector] = None
    gdml: Optional[str] = None
    crs_files: Optional[List[str]] = None
    lrs_file: Optional[str] = None

    def __init__(
        self,
        name: str,
        tag: str,
        version: str,
        tpc: Dict[str, Any],
        optical: Optional[Dict[str, Any]] = None,
        crt: Optional[Dict[str, Any]] = None,
        gdml: Optional[str] = None,
        crs_files: Optional[List[str]] = None,
        lrs_file: Optional[str] = None,
    ):
        """Initialize the detector geometry.

        Parameters
        ----------
        name : str
            Name of the detector
        tag : str
            Tag or label for the geometry instance
        version : int
            Version number of the geometry
        tpc : dict
            Detector boundary configuration
        optical : dict, optional
            Optical detector configuration
        crt : dict, optional
            CRT detector configuration
        gdml : str, optional
            GDML file name associated with the geometry
        crs_files : str or list of str, optional
            CRS (Cosmic Ray System) geometry file(s)
        lrs_file : str, optional
            LRS (Light Readout System) geometry file

        Returns
        -------
        np.ndarray
            Lower boundaries of the overall geometry
        np.ndarray
            Upper boundaries of the overall geometry
        """
        # Store basic geometry information
        self.name = name
        self.tag = tag
        self.version = version
        self.gdml = gdml
        self.crs_files = crs_files
        self.lrs_file = lrs_file

        # Load the charge detector boundaries
        self.tpc = TPCDetector(**tpc)
        lower = self.tpc.lower
        upper = self.tpc.upper

        # Load the optical detectors
        if optical is not None:
            self.optical = self.parse_optical(**optical)
            lower = np.minimum(lower, self.optical.lower)
            upper = np.maximum(upper, self.optical.upper)

        # Load the CRT detectors
        if crt is not None:
            self.crt = CRTDetector(**crt)
            lower = np.minimum(lower, self.crt.lower)
            upper = np.maximum(upper, self.crt.upper)

        # Initialize the parent Box
        super().__init__(lower, upper)

    def parse_optical(self, volume: str, **optical: Any) -> OptDetector:
        """Parse the optical detector configuration.

        Parameters
        ----------
        volume : str
            Optical volume boundaries (one of 'tpc' or 'module')
        **optical : dict
            Reset of the optical detector configuration

        Returns
        -------
        OptDetector
            Optical detector object
        """
        # Get the number of optical volumes based on the the volume type
        assert volume in [
            "module",
            "tpc",
        ], "Optical detector positions must be provided by TPC or module."

        if volume == "module":
            offsets = [module.center for module in self.tpc.modules]
        else:
            offsets = [chamber.center for chamber in self.tpc.chambers]

        # Initialize the optical detector object
        return OptDetector(volume, offsets, **optical)

    @dataclass
    class ContDefinition:
        """Helper class to store containment volume definitions.

        Attributes
        ----------
        volumes : List[np.ndarray]
            (N, 3, 2) set of volume boundaries
        use_source : bool
            Whether to use the source of the points to determine the volume
        limit_normals : List[np.ndarray]
            List of limit plane normals
        limit_thresholds : List[float]
            List of limit thresholds along the plane normal
        """

        volumes: np.ndarray
        use_source: bool
        limit_normals: List[np.ndarray]
        limit_thresholds: List[float]

    def get_boundaries(
        self, with_optical: bool = True, with_crt: bool = True
    ) -> np.ndarray:
        """Fetch the overall geometry boundaries, optionally including
        optical and CRT detectors.

        Parameters
        ----------
        with_optical : bool, default True
            Whether to include optical detector boundaries
        with_crt : bool, default True
            Whether to include CRT detector boundaries

        Returns
        -------
        np.ndarray
            Lower and upper boundaries of the overall geometry
        """
        # Get the TPC boundaries
        lower = self.tpc.lower
        upper = self.tpc.upper

        # Expand to include optical and CRT detectors, if requested
        if with_optical:
            assert (
                self.optical is not None
            ), "This geometry does not have optical detectors to include."
            lower = np.minimum(lower, self.optical.lower)
            upper = np.maximum(upper, self.optical.upper)

        if with_crt:
            assert (
                self.crt is not None
            ), "This geometry does not have CRT detectors to include."
            lower = np.minimum(lower, self.crt.lower)
            upper = np.maximum(upper, self.crt.upper)

        return np.vstack((lower, upper)).T

    def get_sources(self, sources: np.ndarray) -> np.ndarray:
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

    def get_chambers(self, sources: np.ndarray) -> np.ndarray:
        """Converts logical TPC indexes to unique chamber indexes.

        Parameters
        ----------
        sources : np.ndarray
            (N, 2) Array of logical [module ID, tpc ID] pairs, one per point

        Returns
        ----------
        np.ndarray
            (N) Array of physical chamber indexes, one per point
        """
        # Convert the physical indexes to unique chamber indexes
        sources = self.get_sources(sources)
        return sources[:, 0] * self.tpc.num_chambers_per_module + sources[:, 1]

    def get_contributors(self, sources: np.ndarray) -> List[np.ndarray]:
        """Gets the list of [module ID, tpc ID] pairs that contributed to an
        object, as defined in this geometry.

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

    def get_volume_index(
        self, sources: np.ndarray, module_id: int, tpc_id: Optional[int] = None
    ) -> np.ndarray:
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

    def get_closest_tpc(self, points: np.ndarray) -> np.ndarray:
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

    def get_closest_module(self, points: np.ndarray) -> np.ndarray:
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

    def get_closest_tpc_indexes(self, points: np.ndarray) -> List[np.ndarray]:
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

    def get_closest_module_indexes(self, points: np.ndarray) -> List[np.ndarray]:
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

    def get_volume_offsets(
        self, points: np.ndarray, module_id: int, tpc_id: Optional[int] = None
    ) -> np.ndarray:
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
        module = self.tpc[module_id]
        if tpc_id is None:
            boundaries = module.boundaries
        else:
            boundaries = module[tpc_id].boundaries
        dists = points[..., None] - boundaries

        # If a point is between two boundaries, the distance is 0. If it is
        # outside, the distance is that of the closest boundary
        signs = (np.sign(dists[..., 0]) + np.sign(dists[..., 1])) / 2
        offsets = signs * np.min(np.abs(dists), axis=-1)

        return offsets

    def get_min_volume_offset(
        self, points: np.ndarray, module_id: int, tpc_id: Optional[int] = None
    ) -> np.ndarray:
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

    def translate(
        self,
        points: np.ndarray,
        source_id: int,
        target_id: int,
        factor: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
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

    def split(
        self,
        points: np.ndarray,
        target_id: int,
        sources: Optional[np.ndarray] = None,
        meta: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
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
        assert (
            target_id > -1 and target_id < self.tpc.num_modules
        ), "Target ID should be in [0, N_modules[."

        # Get the module ID of each of the input points
        convert = False
        if sources is not None:
            # If sources are provided, simply use that
            module_indexes = []
            for m in range(self.tpc.num_modules):
                module_indexes.append(np.where(sources[:, 0] == m)[0])

        else:
            # If the points are expressed in pixel coordinates, translate
            if meta is not None:
                convert = True
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
                points[module_index], module_id, target_id
            )

        # Bring the coordinates back to pixels, if they were shifted
        if meta is not None and convert:
            points = meta.to_px(points, floor=True)

        return points, module_indexes

    def check_containment(
        self,
        definition: ContDefinition,
        points: np.ndarray,
        sources: Optional[np.ndarray] = None,
        allow_multi_module: bool = False,
        summarize: bool = True,
    ):
        """Check whether a point cloud comes within some distance of the boundaries
        of a certain subset of detector volumes, depending on the mode.

        Must define containment volumes first using `define_containment_volumes`.

        Parameters
        ----------
        definition : ContDefinition
            Containment volume definition to check against
        points : np.ndarray
            (N, 3) Set of point coordinates
        sources : np.ndarray, optional
            (S, 2) : List of [module ID, tpc ID] pairs that created the
            point cloud
        allow_multi_module : bool, default `False`
            Whether to allow points to span multiple modules
        summarize : bool, default `True`
            If `True`, only returns a single flag for the whole cloud.
            Otherwise, returns a boolean array corresponding to each point.

        Returns
        -------
        Union[bool, np.ndarray]
            `True` if the points are contained, `False` if not
        """
        # If sources are provided, only consider source volumes
        if definition.use_source:
            # Get the contributing TPCs
            assert sources is not None and len(points) == len(
                sources
            ), "Need to provide sources to make a source-based check."
            contributors = self.get_contributors(sources)
            if not allow_multi_module and len(np.unique(contributors[0])) > 1:
                return False

            # Define the smallest box containing all contributing TPCs
            # TODO: this is not ideal, should check each source separately?
            index = contributors[0] * self.tpc.num_chambers_per_module + contributors[1]
            volume = self.merge_volumes(definition.volumes[index])
            volumes = [volume]

        else:
            volumes = definition.volumes

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
                contained |= (points > v[:, 0]).all(axis=1) & (points < v[:, 1]).all(
                    axis=1
                )

        # If requested, check points against the active volume limits
        if not summarize or contained:
            for normal, threshold in zip(
                definition.limit_normals, definition.limit_thresholds
            ):
                active = np.dot(points, normal) < threshold
                if summarize:
                    active = active.all()
                contained &= active

        return contained

    def define_containment_volumes(
        self,
        margin: Union[float, List[float], np.ndarray],
        cathode_margin: Optional[float] = None,
        mode: str = "module",
        include_limits: bool = True,
    ) -> ContDefinition:
        """This function defines a list of volumes to check containment against.

        If the containment is checked against a constant volume, it is a lot
        more efficient to call this function once and call `check_containment`
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
        include_limits : bool, default True
            If `True`, the TPC active region limits are checked against

        Returns
        -------
        ContDefinition
            Object containing the list of containment volumes, and other
            information to check containment against
        """
        # Translate the margin parameter to a (3,2) matrix
        margin = np.array(margin)
        if len(margin.shape) == 0:
            margin = np.full((3, 2), margin)
        elif len(margin.shape) == 1:
            assert len(margin) == 3, "Must provide one value per axis."
            margin = np.tile(margin, (2, 1)).T
        else:
            assert np.array(margin).shape == (3, 2), "Must provide two values per axis."

        # Establish the volumes to check against
        cont_volumes = []
        cont_use_source = False
        if mode in ["tpc", "source"]:
            for m, module in enumerate(self.tpc):
                for t, tpc in enumerate(module):
                    vol = self.adapt_volume(
                        tpc.boundaries, margin, cathode_margin, m, t
                    )
                    cont_volumes.append(vol)
            cont_use_source = mode == "source"

        elif mode == "module":
            for module in self.tpc:
                vol = self.adapt_volume(module.boundaries, margin)
                cont_volumes.append(vol)

        elif mode == "detector":
            vol = self.adapt_volume(self.tpc.boundaries, margin)
            cont_volumes.append(vol)

        else:
            raise ValueError(f"Containement check mode not recognized: {mode}.")

        cont_volumes = np.array(cont_volumes)

        # Establish active region limits to check against, if requested
        limit_normals, limit_thresholds = [], []
        if include_limits and self.tpc.limits is not None:
            assert np.all(margin == margin[0, 0]), (
                "No clear way to include active region limit checks when "
                "margins are different in different axes. Abort."
            )
            limit_margin = margin[0, 0]
            for limit in self.tpc.limits:
                limit_normals.append(limit.norm)
                limit_thresholds.append(limit.boundary - limit_margin)

        # Return containment conditions as a ContDefinition object
        return self.ContDefinition(
            volumes=cont_volumes,
            use_source=cont_use_source,
            limit_normals=limit_normals,
            limit_thresholds=limit_thresholds,
        )

    def adapt_volume(
        self,
        ref_volume: np.ndarray,
        margin: np.ndarray,
        cathode_margin: Optional[float] = None,
        module_id: Optional[int] = None,
        tpc_id: Optional[int] = None,
    ):
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
        volume[:, 0] += margin[:, 0]
        volume[:, 1] -= margin[:, 1]

        # If a cathode margin is provided, adapt the cathode wall differently
        if cathode_margin is not None:
            assert (
                module_id is not None and tpc_id is not None
            ), "Module and TPC ID must be provided when using a cathode margin."
            axis = self.tpc[module_id][tpc_id].drift_axis
            side = self.tpc[module_id][tpc_id].cathode_side

            flip = (-1) ** side
            volume[axis, side] += flip * (cathode_margin - margin[axis, side])

        return volume

    @staticmethod
    def merge_volumes(volumes: np.ndarray) -> np.ndarray:
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
