"""Aligned voxel, fragment, particle and point-proposal supervision."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from spine.constants import PDG_TO_PID
from spine.data import Meta

from ..base import ParserBase


def _shape(code: int | np.ndarray) -> np.ndarray:
    """Translate one-based pysupera classes to SPINE semantic labels.

    Parameters
    ----------
    code : int or np.ndarray
        Source semantic code or array of codes.

    Returns
    -------
    np.ndarray
        SPINE labels of the same shape; unknown codes map to ``-1``.
    """
    code = np.asarray(code)
    return np.array([-1, 0, 1, 3, 2, 4, -1])[np.clip(code, 0, 6)]


class SuperaParserBase(ParserBase):
    """Voxelize truth deposits once, sharing winner selection across products.

    Coordinates and optional grid arguments are in mm; output metadata uses
    cm. The default size comes from the file. The origin defaults to (0,0,0);
    an event-local integer translation makes all voxel coordinates nonnegative.
    Overlaps sum dE and choose labels by track, shower, delta, Michel, LE,
    then earliest time and source row order. Unknown point classes are rejected.
    """

    def __init__(
        self,
        dtype: str,
        voxel_size_mm: float | tuple[float, ...] | None = None,
        origin_mm: tuple[float, ...] = (0.0, 0.0, 0.0),
    ) -> None:
        """Configure a common grid for all parsers in the dataset.

        Parameters
        ----------
        dtype : str
            Floating-point type used by parser products.
        voxel_size_mm : float or sequence[float], optional
            Rasterization size; defaults to the file's voxel_size attribute.
        origin_mm : sequence[float], default (0, 0, 0)
            Common grid origin in detector coordinates (mm).
        """
        super().__init__(
            dtype,
            points_event="points",
            particle_event="particles",
            metadata_event="metadata",
            interaction_event="inter",
        )

        # Normalize grid settings so all parsers can share the event cache.
        self.size = (
            None
            if voxel_size_mm is None
            else tuple(np.broadcast_to(np.asarray(voxel_size_mm, dtype=float), (3,)))
        )
        self.origin = tuple(np.asarray(origin_mm, dtype=float))
        if len(self.origin) != 3 or not np.isfinite(self.origin).all():
            raise ValueError("origin_mm must contain three finite values.")
        if self.size is not None and (
            not np.isfinite(self.size).all() or min(self.size) <= 0
        ):
            raise ValueError("voxel_size_mm must be finite and positive.")
        self.grid = (self.size, self.origin)

    def prepare(self, trees: dict[str, Any]) -> dict[str, Any]:
        """Build and cache the common voxel representation for all products.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products. A private cache is added to this dictionary.

        Returns
        -------
        dict[str, Any]
            Unique voxel coordinates, summed energy, winning labels, spatial
            metadata and the resolved particle table.

        Raises
        ------
        ValueError
            If the grid, particle associations or point values are invalid.
        """
        cache = trees.setdefault("_supera_prepared", {})
        if self.grid in cache:
            return cache[self.grid]

        # Prefer an explicit grid size; otherwise decode the file attribute.
        points, table, attrs = trees["points"], trees["particles"], trees["metadata"]
        size = self.size
        if size is None:
            raw = attrs.get("voxel_size")
            if raw is None:
                raise ValueError(
                    "Missing voxel_size: configure voxel_size_mm explicitly."
                )
            size = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        size = np.broadcast_to(np.asarray(size, dtype=float), (3,))
        if not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Invalid voxel size.")
        # Resolve sparse source IDs before expanding fragment and instance ranges.
        source_ids = table["id"].copy()
        table = self._resolve_particles(table)
        fragment, instance, semantic = self._point_labels(table, len(points))

        # Select labels using the same overlap winners as the input tensor.
        event = self._voxelize(points, size, attrs, semantic)
        winner = event.pop("winner")
        event.update(
            shape=semantic[winner],
            fragment=fragment[winner],
            instance=instance[winner],
            columns=table,
            source_ids=source_ids,
        )
        event["labels"] = self._particle_labels(event, trees["inter"])
        cache[self.grid] = event
        return event

    @staticmethod
    def _particle_labels(
        event: dict[str, Any], interactions: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Build shared particle truth using local IDs and interaction vertices.

        Parameters
        ----------
        event : dict[str, Any]
            Prepared voxel image and resolved particle columns.
        interactions : dict[str, np.ndarray]
            Event-local interaction table, with detector vertices in mm.

        Returns
        -------
        dict[str, np.ndarray]
            Particle labels shared by compact supervision and truth objects.
            Missing ancestry yields NaN vertices unless an interaction is known.
        """
        table = event["columns"]
        host = table["frag_inst_id"]
        valid = (host >= 0) & (host < len(table["id"]))
        interaction_ids = table["interaction_id"].copy()
        interaction_ids[valid] = table["interaction_id"][host[valid]]
        vertices = np.full((len(table["id"]), 3), np.nan)
        ancestors = table["ancestor_id"]
        valid_ancestors = ancestors >= 0
        vertices[valid_ancestors] = table["start_pos"][ancestors[valid_ancestors], :3]

        # A generator interaction may contain several distinct stored roots.
        # Use its common vertex instead of arbitrarily selecting one root.
        if all(key in interactions for key in ("x", "y", "z")):
            positions = np.column_stack([interactions[key] for key in ("x", "y", "z")])
            for row, interaction in enumerate(interactions["interaction_id"]):
                if np.isfinite(positions[row]).all():
                    vertices[interaction_ids == interaction] = positions[row]

        shapes = _shape(table["sem_type"])
        shapes[valid] = _shape(table["inst_sem_type"][host[valid]])
        shapes[table["frag_sem_type"] == 5] = 4

        # PID and interaction-primary supervision describe the host instance,
        # including its secondary fragments, just as in the LArCV adapter.
        pids = np.full(len(table["id"]), -1, dtype=np.int64)
        pids[valid] = [
            PDG_TO_PID.get(int(pdg), -1) for pdg in table["pdg"][host[valid]]
        ]
        primaries = np.full(len(table["id"]), -1, dtype=np.int64)
        primaries[valid] = np.where(
            table["parent_id"][host[valid]] >= 0,
            table["parent_id"][host[valid]] == host[valid],
            -1,
        )

        # All relationship fields use the same event-local particle namespace.
        return {
            "particle": table["id"].copy(),
            "group": host.copy(),
            "ancestor": table["ancestor_id"].copy(),
            "interaction": interaction_ids,
            "nu": np.full(len(table["id"]), -1),
            "pid": pids,
            "group_primary": np.where(valid, table["id"] == host, -1),
            "interaction_primary": primaries,
            "vertex": (vertices / 10.0 - event["meta"].lower) / event["meta"].size,
            "momentum": np.linalg.norm(table["start_mom"][:, :3], axis=1),
            "energy_init": table["start_mom"][:, 3].copy(),
            "shape": shapes,
        }

    def endpoints(self, trees: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Resolve collected-step endpoints without substituting track creation.

        Parameters
        ----------
        trees : dict[str, Any]
            Projected native event products.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Start and end positions in mm, with time in ns in the last column.
            Missing endpoints use fragment deposits where available, otherwise
            remain NaN. Non-track ends equal their starts.
        """
        event = self.prepare(trees)
        if "endpoints" in event:
            return event["endpoints"]
        table = event["columns"]
        starts = table["first_pos"].copy()
        ends = table["last_pos"].copy()
        time_col = next(
            i
            for i, key in enumerate(trees["metadata"]["columns"])
            if key in ("t", "time")
        )

        # Non-instance rows do not have upstream collected-step kinematics.
        for row in range(len(table["id"])):
            prefix = "inst" if table["inst_id"][row] == row else "frag"
            owner = row if prefix == "inst" else int(table["frag_id"][row])
            if owner >= 0:
                lower, upper = (
                    table[prefix + "_pc_start"][owner],
                    table[prefix + "_pc_end"][owner],
                )
                # Match upstream's preference for non-LE trajectory deposits.
                if prefix == "inst" and table["inst_pc_le_start"][row] > lower:
                    upper = table["inst_pc_le_start"][row]
                points = trees["points"][max(0, lower) : max(0, upper)]
                if len(points):
                    points = points[np.argsort(points[:, time_col], kind="stable")]
                    for endpoint, point in zip((starts, ends), points[[0, -1]]):
                        if not np.isfinite(endpoint[row]).all():
                            endpoint[row, :3] = point[:3]
                            endpoint[row, 3] = point[time_col] * 1000.0

        # A shower has one meaningful collected-step endpoint.
        ends[event["labels"]["shape"] != 1] = starts[event["labels"]["shape"] != 1]
        starts[~np.isfinite(starts)] = np.nan
        ends[~np.isfinite(ends)] = np.nan
        event["endpoints"] = (starts, ends)
        return starts, ends

    def _voxelize(
        self,
        points: np.ndarray,
        size: np.ndarray,
        attrs: dict[str, Any],
        semantic: np.ndarray,
    ) -> dict[str, Any]:
        """Rasterize deposits and retain their overlap winners and transform.

        Parameters
        ----------
        points : np.ndarray
            Event point rows in detector coordinates.
        size : np.ndarray
            Three voxel dimensions in mm.
        attrs : dict[str, Any]
            File attributes, including point column names.
        semantic : np.ndarray
            SPINE semantic label of each point.

        Returns
        -------
        dict[str, Any]
            Unique coordinates, summed energy, winning source row and Meta.
        """
        num_points = len(points)
        # Rasterize on a common grid and translate to nonnegative coordinates.
        if not np.isfinite(points[:, :3]).all():
            raise ValueError("Nonfinite point coordinates.")
        cells = np.floor((points[:, :3] - self.origin) / size).astype(np.int64)
        lower_cell = cells.min(axis=0) if num_points else np.zeros(3, dtype=np.int64)
        coords, inverse = np.unique(cells - lower_cell, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

        energy, winner = self._merge_voxels(points, attrs["columns"], semantic, inverse)

        # Metadata preserves the detector transform in SPINE centimetres.
        lower = (np.asarray(self.origin) + lower_cell * size) / 10.0
        count = coords.max(axis=0) + 1 if num_points else np.ones(3, dtype=np.int64)
        meta = Meta(
            lower=lower,
            upper=lower + count * size / 10.0,
            size=size / 10.0,
            count=count,
        )
        return {"coords": coords, "energy": energy, "winner": winner, "meta": meta}

    @staticmethod
    def _merge_voxels(
        points: np.ndarray,
        columns: tuple[str, ...],
        semantic: np.ndarray,
        inverse: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum energy and resolve overlap labels by class, time and row order.

        Parameters
        ----------
        points : np.ndarray
            Event point rows before duplicate removal.
        columns : tuple[str, ...]
            Physical column names in the point array.
        semantic : np.ndarray
            SPINE semantic label for each contribution.
        inverse : np.ndarray
            Unique voxel index of each contribution.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Summed voxel energies and winning source point indexes.
        """
        num_points = len(points)
        values = points[:, columns.index("dE")]
        times = points[:, columns.index("t" if "t" in columns else "time")]
        if not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError("Nonfinite deposit energy or time.")
        ranks = np.full(num_points, 5)
        for rank, shape in enumerate((1, 0, 3, 2, 4)):
            ranks[semantic == shape] = rank
        order = np.lexsort((np.arange(num_points), times, ranks, inverse))
        winner = (
            order[np.r_[True, np.diff(inverse[order]) != 0]] if num_points else order
        )
        energy = np.bincount(inverse, weights=values, minlength=0)

        return energy, winner

    @staticmethod
    def _resolve_particles(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Resolve sparse particle relationships to event-local table indexes.

        Parameters
        ----------
        table : dict[str, np.ndarray]
            Raw particle columns with source IDs and rebased point ranges.

        Returns
        -------
        dict[str, np.ndarray]
            Shallow copy with relationship columns replaced by row indexes.

        Raises
        ------
        ValueError
            If particle IDs are invalid or a relationship cannot be resolved.
        """
        source_ids = table["id"]
        num_particles = len(source_ids)
        if len(np.unique(source_ids)) != num_particles or np.any(source_ids < 0):
            raise ValueError("Supera particle ids must be unique and nonnegative.")
        # Contrary to io_v3's index-convention prose, layout.py retains sparse
        # source IDs. Resolve relationships explicitly, never index by an ID.
        row_of = {int(pid): row for row, pid in enumerate(source_ids)}
        table = dict(table)
        for key in (
            "id",
            "parent_id",
            "ancestor_id",
            "frag_id",
            "inst_id",
            "frag_inst_id",
        ):
            try:
                table[key] = np.array(
                    [row_of[int(v)] if v >= 0 else -1 for v in table[key]],
                    dtype=np.int64,
                )
            except KeyError as err:
                raise ValueError(
                    f"Dangling Supera particle relationship in {key}."
                ) from err
        return table

    @staticmethod
    def _point_labels(
        table: dict[str, np.ndarray], num_points: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand and validate fragment and instance ownership of every point.

        Parameters
        ----------
        table : dict[str, np.ndarray]
            Particle columns with event-local relationships and point ranges.
        num_points : int
            Number of point rows in this event.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Fragment row, instance row and SPINE semantic label for each point.

        Raises
        ------
        ValueError
            If ranges overlap, leave gaps or disagree with stored relationships
            or semantic classes.
        """
        # Each representative owns one half-open range; no point is shared.
        fragment = np.full(num_points, -1, dtype=np.int64)
        instance = np.full(num_points, -1, dtype=np.int64)
        semantic = np.full(num_points, -1, dtype=np.int64)
        for level, dest in (("frag", fragment), ("inst", instance)):
            for row in np.flatnonzero(table[level + "_id"] == table["id"]):
                start, stop = int(table[level + "_pc_start"][row]), int(
                    table[level + "_pc_end"][row]
                )
                if start == stop == -1:
                    continue
                if not 0 <= start <= stop <= num_points or np.any(
                    dest[start:stop] >= 0
                ):
                    raise ValueError(f"Invalid or overlapping {level} point ranges.")
                dest[start:stop] = row
                if level == "inst":
                    split = int(table["inst_pc_le_start"][row])
                    if not start <= split <= stop:
                        raise ValueError("Invalid instance LE range.")
                    semantic[start:split] = _shape(table["inst_sem_type"][row])
                    semantic[split:stop] = 4

        # Cross-check both partitions and the explicit fragment-to-instance join.
        if np.any(fragment < 0) or np.any(instance < 0):
            raise ValueError("Fragment and instance ranges must cover every point.")
        if np.any(semantic < 0):
            raise ValueError(
                "Unknown Supera semantic class cannot supervise a five-class model."
            )
        if num_points and not np.array_equal(table["frag_inst_id"][fragment], instance):
            raise ValueError("Fragment ownership disagrees with instance ranges.")
        if num_points and not np.array_equal(
            table["frag_sem_type"][fragment] == 5, semantic == 4
        ):
            raise ValueError("Fragment semantics disagree with instance LE ranges.")
        return fragment, instance, semantic
