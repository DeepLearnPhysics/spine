"""Applies field non-uniformity corrections."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from spine.geo import GeoManager, Geometry
from spine.utils.conditional import ROOT, ROOT_AVAILABLE

if TYPE_CHECKING:  # pragma: no cover
    from spine.geo.detector import Box

__all__ = ["FieldCalibrator", "FieldMap"]

FloatArray: TypeAlias = NDArray[np.floating]


class FieldMap:
    """Dense 3D vector look-up table for position displacements."""

    _bounds_modes = ("clip", "zero", "raise")

    def __init__(
        self,
        values: FloatArray,
        ranges: list[list[float]],
        bounds: str = "zero",
    ) -> None:
        """Initialize the field map.

        Parameters
        ----------
        values : np.ndarray
            Dense ``(N_x, N_y, N_z, 3)`` array of displacement vectors.
        ranges : List[List[float]]
            Axis ranges for x, y and z.
        bounds : str, default 'zero'
            Out-of-bounds behavior. ``'clip'`` uses the closest voxel,
            ``'zero'`` returns a null displacement and ``'raise'`` fails.
        """
        if bounds not in self._bounds_modes:
            raise ValueError(
                f"Out-of-bounds mode not recognized: {bounds}. "
                f"Must be one of {self._bounds_modes}."
            )

        values = np.asarray(values, dtype=float)
        ranges_array = np.asarray(ranges, dtype=float)
        if values.ndim != 4 or values.shape[-1] != 3:
            raise ValueError("Must provide a dense (N_x, N_y, N_z, 3) map.")
        if ranges_array.shape != (3, 2):
            raise ValueError("Must provide one [min, max] range per dimension.")
        if np.any(ranges_array[:, 1] <= ranges_array[:, 0]):
            raise ValueError("Each axis range must have a positive width.")

        self.values = values
        self.range = ranges_array
        self.bins = np.asarray(values.shape[:3], dtype=int)
        self.bin_sizes = (self.range[:, 1] - self.range[:, 0]) / self.bins
        self.bounds = bounds

    @classmethod
    def from_root(
        cls,
        map_file: str | Path,
        map_prefix: str = "TrueFwd_Displacement",
        bounds: str = "zero",
    ) -> "FieldMap":
        """Load a vector displacement map from TH3 objects in a ROOT file.

        Parameters
        ----------
        map_file : Union[str, Path]
            ROOT file with one TH3 object per displacement component.
        map_prefix : str, default 'TrueFwd_Displacement'
            Prefix used to build the histogram names
            ``{map_prefix}_X``, ``{map_prefix}_Y`` and ``{map_prefix}_Z``.
        bounds : str, default 'zero'
            Out-of-bounds behavior passed to :class:`FieldMap`.

        Returns
        -------
        FieldMap
            Dense vector look-up table.
        """
        if not ROOT_AVAILABLE:
            raise ImportError("ROOT is required to load field maps from ROOT files.")

        root_file = ROOT.TFile.Open(str(map_file), "r")  # pylint: disable=E1101
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not open field map ROOT file: {map_file}")

        try:
            hists = []
            for axis in ("X", "Y", "Z"):
                hist_name = f"{map_prefix}_{axis}"
                hist = root_file.Get(hist_name)
                if not hist:
                    raise KeyError(
                        f"Could not find histogram '{hist_name}' in {map_file}."
                    )
                hists.append(hist)

            bins, ranges = cls._histogram_geometry(hists[0])
            for hist in hists[1:]:
                other_bins, other_ranges = cls._histogram_geometry(hist)
                if other_bins != bins or other_ranges != ranges:
                    raise ValueError(
                        "All displacement component histograms must share the "
                        "same binning and axis ranges."
                    )

            values = np.empty((*bins, 3), dtype=float)
            for component, hist in enumerate(hists):
                values[..., component] = cls._histogram_values(hist, bins)

        finally:
            root_file.Close()

        return cls(values, ranges, bounds=bounds)

    @property
    def edges(self) -> list[FloatArray]:
        """Returns the bin edges in each axis."""
        return [
            np.linspace(self.range[i, 0], self.range[i, 1], self.bins[i] + 1)
            for i in range(3)
        ]

    def query(self, points: FloatArray) -> FloatArray:
        """Query displacement vectors for a set of 3D points.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 3)`` array of point coordinates.

        Returns
        -------
        np.ndarray
            ``(N, 3)`` array of displacement vectors.
        """
        displacements, valid = self._query(points)

        if self.bounds == "raise" and not np.all(valid):
            raise IndexError("At least one point is outside of the field map range.")

        if self.bounds == "zero":
            displacements = np.asarray(displacements).copy()
            displacements[~valid] = 0.0

        return displacements

    def contains(self, points: FloatArray) -> NDArray[np.bool_]:
        """Check which points fall within the map boundaries."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Must provide an (N, 3) point array.")

        _, valid = self._bin_ids(points)
        return valid

    def _query(self, points: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Query displacement vectors and return the in-map mask."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Must provide an (N, 3) point array.")

        bin_ids, valid = self._bin_ids(points)
        clipped = np.clip(bin_ids, 0, self.bins - 1)
        displacements = self.values[tuple(clipped.T)]
        return displacements, valid

    def _bin_ids(
        self, points: FloatArray
    ) -> tuple[NDArray[np.integer], NDArray[np.bool_]]:
        """Return voxel indexes and in-map mask for a set of points."""
        offsets = points - self.range[:, 0]
        bin_ids = np.floor(offsets / self.bin_sizes).astype(int)
        valid = np.all((bin_ids >= 0) & (bin_ids < self.bins), axis=1)
        return bin_ids, valid

    @staticmethod
    def _histogram_geometry(
        hist: object,
    ) -> tuple[tuple[int, int, int], list[list[float]]]:
        """Extract TH3 bin counts and ranges."""
        axes = [hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()]
        bins = (hist.GetNbinsX(), hist.GetNbinsY(), hist.GetNbinsZ())
        ranges = [[axis.GetXmin(), axis.GetXmax()] for axis in axes]
        return bins, ranges

    @staticmethod
    def _histogram_values(hist: object, bins: tuple[int, int, int]) -> FloatArray:
        """Copy TH3 bin contents to a dense numpy array."""
        values = np.empty(bins, dtype=float)
        for ix in range(bins[0]):
            for iy in range(bins[1]):
                for iz in range(bins[2]):
                    values[ix, iy, iz] = hist.GetBinContent(ix + 1, iy + 1, iz + 1)
        return values


class FieldCalibrator:
    """Applies position corrections to account for field non-uniformities
    (space charge, cathode distortions, etc.)
    """

    name = "field"

    def __init__(
        self,
        map_file: str | Path | None = None,
        map_prefix: str = "TrueFwd_Displacement",
        field_map: FieldMap | None = None,
        scale: float = 1.0,
        bounds: str = "zero",
        num_tpcs: int | None = None,
    ) -> None:
        """Initialize the field calibrator.

        Parameters
        ----------
        map_file : Union[str, Path], optional
            ROOT file containing TH3 displacement maps.
        map_prefix : str, default 'TrueFwd_Displacement'
            Prefix used to load displacement components from ROOT.
        field_map : FieldMap, optional
            Pre-loaded displacement map. Mutually exclusive with ``map_file``.
        scale : float, default 1.0
            Multiplicative factor applied to the displacement before adding it
            to the input points. Use ``-1`` to subtract the stored offsets.
        bounds : str, default 'zero'
            Out-of-bounds behavior passed to the field map when loading from
            ROOT.
        num_tpcs : int, optional
            Accepted for compatibility with :class:`CalibrationManager`.
        """
        if (map_file is None) == (field_map is None):
            raise ValueError("Must provide exactly one of map_file or field_map.")

        if field_map is None:
            assert map_file is not None  # for type checker
            field_map = FieldMap.from_root(map_file, map_prefix, bounds=bounds)

        self.field_map = field_map
        self.scale = scale
        self.geo = GeoManager.get_instance()
        self.module_maps = self._build_module_maps()

    def process(
        self,
        points: FloatArray,
        tpc_id: int,
    ) -> FloatArray:
        """Apply the displacement map to a set of points.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 3)`` array of point coordinates.
        tpc_id : int
            TPC ID currently being processed.

        Returns
        -------
        np.ndarray
            ``(N, 3)`` array of displaced point coordinates.
        """
        module_id = tpc_id // self.geo.tpc.num_chambers_per_module
        displacements = self.module_maps[module_id].query(points)
        return points + self.scale * displacements

    def _build_module_maps(self) -> list[FieldMap]:
        """Build one field map in detector coordinates for each module."""
        maps: list[FieldMap | None] = [None] * self.geo.tpc.num_modules
        covered_ids = []
        for module_id, module in enumerate(self.geo.tpc.modules):
            if self._volume_overlaps(module.boundaries, self.field_map.range):
                maps[module_id] = self.field_map
                covered_ids.append(module_id)

        if not covered_ids:
            raise ValueError("The field map does not overlap any detector module.")

        for module_id, module in enumerate(self.geo.tpc.modules):
            if maps[module_id] is not None:
                continue

            source_id = self._get_source_module_id(module_id, covered_ids)
            source = self.geo.tpc.modules[source_id]
            maps[module_id] = self._transform_field_map(
                self.field_map, source, module, self.geo
            )

        return cast(list[FieldMap], maps)

    def _get_source_module_id(self, module_id: int, candidates: list[int]) -> int:
        """Pick the covered module that should seed a target module replica."""
        module = self.geo.tpc.modules[module_id]
        mirror_center = 2.0 * self.geo.tpc.center - module.center
        choices = []
        for source_id in candidates:
            source = self.geo.tpc.modules[source_id]
            if not np.allclose(source.dimensions, module.dimensions):
                continue

            _, signs = self._transform_between_volumes(source, module, self.geo)
            num_flips = int(np.count_nonzero(signs < 0.0))
            mirror_distance = float(np.linalg.norm(source.center - mirror_center))
            choices.append((-num_flips, mirror_distance, source_id))

        if not choices:
            raise ValueError(
                f"No equivalent covered module found for module {module_id}."
            )

        choices.sort()
        return int(choices[0][-1])

    @staticmethod
    def _transform_field_map(
        field_map: FieldMap,
        source: Box,
        target: Box,
        geo: Geometry,
    ) -> FieldMap:
        """Create a field-map replica in the target module frame."""
        offset, signs = FieldCalibrator._transform_between_volumes(source, target, geo)
        values = np.asarray(field_map.values, dtype=float).copy()
        ranges = np.empty_like(field_map.range)

        for axis, sign in enumerate(signs):
            low, high = field_map.range[axis]
            transformed = np.asarray([low, high]) * sign + offset[axis]
            ranges[axis] = [np.min(transformed), np.max(transformed)]
            if sign < 0.0:
                values = np.flip(values, axis=axis)
                values[..., axis] *= -1.0

        return FieldMap(values, ranges.tolist(), bounds=field_map.bounds)

    @staticmethod
    def _transform_between_volumes(
        source: Box,
        target: Box,
        geo: Geometry,
    ) -> tuple[FloatArray, FloatArray]:
        """Return a coordinate transform to an equivalent target volume frame."""
        signs = np.ones(3, dtype=float)

        for axis in range(3):
            source_center = source.center[axis]
            target_center = target.center[axis]
            detector_center = geo.tpc.center[axis]
            mirrored = np.isclose(source_center + target_center, 2.0 * detector_center)

            if mirrored and not np.isclose(source_center, target_center):
                signs[axis] = -1.0

        offset = target.center - signs * source.center
        return offset, signs

    @staticmethod
    def _volume_overlaps(first: FloatArray, second: FloatArray) -> bool:
        """Check whether two box volumes overlap."""
        lower = np.maximum(first[:, 0], second[:, 0])
        upper = np.minimum(first[:, 1], second[:, 1])
        return bool(np.all(upper > lower))
