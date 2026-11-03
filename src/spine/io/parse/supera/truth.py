"""Aligned voxel, fragment, particle and point-proposal supervision."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from spine.constants import PDG_TO_PID
from spine.data import ClusterLabelData, Meta, ObjectListData, Particle, TensorData

from ..base import ParserBase

__all__ = [
    "SuperaClusterLabelParser",
    "SuperaParticleParser",
    "SuperaParticlePointsParser",
    "SuperaSegmentationParser",
    "SuperaTensorParser",
]


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


class SuperaTensorParser(ParserBase):
    """Voxelize truth deposits once, sharing winner selection across products.

    Coordinates and optional grid arguments are in mm; output metadata uses
    cm. The default size comes from the file. The origin defaults to (0,0,0);
    an event-local integer translation makes all voxel coordinates nonnegative.
    Overlaps sum dE and choose labels by track, shower, delta, Michel, LE,
    then earliest time and source row order. Unknown point classes are rejected.
    """

    name = "tensor"

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
        cache[self.grid] = event
        return event

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

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return summed truth energy in MeV on the shared sparse grid.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

        Returns
        -------
        TensorData
            Sparse coordinates and summed deposit energy in MeV.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        return TensorData(
            coords=event["coords"].astype(self.itype),
            features=event["energy"].astype(self.ftype)[:, None],
            meta=event["meta"],
        )


class SuperaSegmentationParser(SuperaTensorParser):
    """Return semantic labels in exactly the input voxel order."""

    name = "segmentation"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return five-class labels; unknown point semantics are rejected.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

        Returns
        -------
        TensorData
            Semantic labels sharing the input tensor coordinates and row order.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        return TensorData(
            coords=event["coords"].astype(self.itype),
            features=event["shape"].astype(self.ftype)[:, None],
            meta=event["meta"],
        )


class SuperaClusterLabelParser(SuperaTensorParser):
    """Return fragment IDs and the winning fragment's particle table row."""

    name = "cluster_label"

    def __call__(self, trees: dict[str, Any]) -> ClusterLabelData:
        """Build compact fragment supervision with instance and ancestor links.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

        Returns
        -------
        ClusterLabelData
            Compact fragment labels and a named particle-truth sidecar.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        table = event["columns"]
        num_particles = len(table["id"])
        host = table["frag_inst_id"]
        valid = (host >= 0) & (host < num_particles)
        shapes = _shape(table["sem_type"])
        shapes[valid] = _shape(table["inst_sem_type"][host[valid]])
        shapes[table["frag_sem_type"] == 5] = 4

        # Keep source particle IDs while exposing event-local relationship fields.
        particles = {
            "particle": event["source_ids"].copy(),
            "group": host.copy(),
            "ancestor": table["ancestor_id"].copy(),
            "interaction": table["interaction_id"].copy(),
            "nu": np.full(num_particles, -1),
            "pid": np.array(
                [PDG_TO_PID.get(int(particle), -1) for particle in table["pdg"]]
            ),
            "group_primary": np.where(valid, table["id"] == host, -1),
            "interaction_primary": (table["parent_id"] == table["id"]).astype(np.int64),
            "vertex": (
                table["start_pos"][table["ancestor_id"], :3] / 10.0
                - event["meta"].lower
            )
            / event["meta"].size,
            "momentum": np.linalg.norm(table["start_mom"][:, :3], axis=1),
            "energy_init": table["start_mom"][:, 3].copy(),
            "shape": shapes,
        }

        # The winning fragment row is also the compact particle association.
        return ClusterLabelData(
            coords=event["coords"].astype(self.itype),
            features=np.column_stack(
                (event["energy"], event["fragment"], event["fragment"])
            ).astype(self.ftype),
            particles=particles,
            meta=event["meta"],
            remove_duplicates=False,
        )


class SuperaParticleParser(SuperaTensorParser):
    """Translate the full stored particle table, preserving event-local IDs."""

    name = "particle"

    def __call__(self, trees: dict[str, Any]) -> ObjectListData:
        """Return SPINE particle truth in voxel units with MeV kinematics.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

        Returns
        -------
        ObjectListData
            Complete stored particle table in voxel units with MeV kinematics.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        table, meta = event["columns"], event["meta"]
        particles = []

        # Deposited truth energy includes source contributions, not winner totals.
        event_counts = np.bincount(event["instance"], minlength=len(table["id"]))
        energy_prefix = np.r_[
            0.0,
            np.cumsum(
                trees["points"][:, trees["metadata"]["columns"].index("dE")],
                dtype=np.float64,
            ),
        ]

        # Preserve the full table so PPN and cluster references remain valid.
        for row in range(len(table["id"])):
            is_inst = table["inst_id"][row] == row
            particle = Particle(
                id=row,
                parent_id=int(table["parent_id"][row]),
                group_id=int(table["frag_inst_id"][row]),
                interaction_id=int(table["interaction_id"][row]),
                track_id=int(table["geant4_trackid"][row]),
                pdg_code=int(table["pdg"][row]),
                parent_pdg_code=int(table["parent_pdg"][row]),
                shape=int(
                    _shape(table["inst_sem_type" if is_inst else "sem_type"][row])
                ),
                energy_init=float(table["start_mom"][row, 3]),
                momentum=table["start_mom"][row, :3].copy(),
                units="px",
            )

            # Positions use voxel units; kinematic times remain in nanoseconds.
            for source, target in (
                ("start", "position"),
                ("end", "end_position"),
                ("first", "first_step"),
                ("last", "last_step"),
            ):
                setattr(
                    particle,
                    target,
                    (
                        (table[source + "_pos"][row, :3] / 10.0 - meta.lower)
                        / meta.size
                    ).astype(self.ftype),
                )
            particle.t, particle.end_t = float(table["start_pos"][row, 3]), float(
                table["end_pos"][row, 3]
            )
            particle.first_step_t, particle.last_step_t = float(
                table["first_pos"][row, 3]
            ), float(table["last_pos"][row, 3])
            particle.num_voxels = int(event_counts[row])
            if is_inst:
                start, stop = table["inst_pc_start"][row], table["inst_pc_end"][row]
                particle.energy_deposit = (
                    float(energy_prefix[stop] - energy_prefix[start])
                    if start >= 0
                    else 0.0
                )
            particles.append(particle)
        return ObjectListData(particles, Particle())


class SuperaParticlePointsParser(SuperaTensorParser):
    """PPN instance starts and track ends from collected-step kinematics."""

    name = "particle_points"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Omit unknown, LE, nonfinite and out-of-image endpoints.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

        Returns
        -------
        TensorData
            Instance starts and track ends with shape, particle and endpoint labels.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        table, meta = event["columns"], event["meta"]
        rows = []

        # Only instances represented in the voxel image can provide targets.
        for row in np.unique(event["instance"]):
            shape = int(_shape(table["inst_sem_type"][row]))
            if shape not in (0, 1, 2, 3) or abs(table["pdg"][row]) > 1000000000:
                continue
            for endpoint, key in enumerate(
                ("first_pos", "last_pos") if shape == 1 else ("first_pos",)
            ):
                point = (table[key][row, :3] / 10.0 - meta.lower) / meta.size
                if (
                    np.isfinite(point).all()
                    and np.all(point >= 0)
                    and np.all(point < meta.count)
                ):
                    rows.append([*point, shape, row, endpoint])

        # Preserve a stable six-column layout even when no endpoints survive.
        data = np.asarray(rows, dtype=self.ftype).reshape(-1, 6)
        return TensorData(
            coords=data[:, :3],
            features=data[:, 3:],
            meta=meta,
            coordinate_groups={"point": (0, 1, 2)},
            feature_fields={"shape": (0,), "particle": (1,), "endpoint": (2,)},
            index_cols=np.array([1]),
            index_shifts=np.array([len(table["id"])]),
        )
