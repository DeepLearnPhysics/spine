"""Particle truth, coordinate labels, genealogy and PPN targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import EdgeIndexData, Meta, ObjectListData, Particle, TensorData

from .base import SuperaParserBase, _shape

__all__ = [
    "SuperaParticleParser",
    "SuperaParticlePointsParser",
    "SuperaParticleCoordsParser",
    "SuperaParticleGraphParser",
]


class SuperaParticleParser(SuperaParserBase):
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
        table = event["columns"]
        particles = []

        # Charge is unavailable here. Integrate truth dE before overlap merging.
        event_counts = np.bincount(event["fragment"], minlength=len(table["id"]))
        energy_prefix = np.r_[
            0.0,
            np.cumsum(
                trees["points"][:, trees["metadata"]["columns"].index("dE")],
                dtype=np.float64,
            ),
        ]

        starts, ends = self.endpoints(trees)

        # Preserve the full table so PPN and cluster references remain valid.
        for row in range(len(table["id"])):
            particle = Particle(
                id=row,
                parent_id=int(table["parent_id"][row]),
                group_id=int(table["frag_inst_id"][row]),
                interaction_id=int(event["labels"]["interaction"][row]),
                track_id=int(table["geant4_trackid"][row]),
                pdg_code=int(table["pdg"][row]),
                parent_pdg_code=int(table["parent_pdg"][row]),
                shape=int(event["labels"]["shape"][row]),
                energy_init=float(table["start_mom"][row, 3]),
                momentum=table["start_mom"][row, :3].copy(),
                pid=int(event["labels"]["pid"][row]),
                group_primary=int(event["labels"]["group_primary"][row]),
                interaction_primary=int(event["labels"]["interaction_primary"][row]),
                ancestor_position=event["labels"]["vertex"][row].astype(self.ftype),
                children_id=np.flatnonzero(
                    (table["parent_id"] == row) & (table["id"] != row)
                ).astype(np.int32),
                end_momentum=table["end_mom"][row, :3].copy(),
                units="px",
            )

            # Positions use voxel units; kinematic times remain in nanoseconds.
            for positions, target in (
                (table["start_pos"], "position"),
                (table["end_pos"], "end_position"),
                (starts, "first_step"),
                (ends, "last_step"),
            ):
                setattr(
                    particle,
                    target,
                    (
                        (positions[row, :3] / 10.0 - event["meta"].lower)
                        / event["meta"].size
                    ).astype(self.ftype),
                )
            particle.t, particle.end_t = float(table["start_pos"][row, 3]), float(
                table["end_pos"][row, 3]
            )
            particle.first_step_t, particle.last_step_t = float(starts[row, 3]), float(
                ends[row, 3]
            )
            self._fill_ancestry(particle, row, table, event["meta"])
            particle.num_voxels = int(event_counts[row])
            # Own-particle ranges need not cover deposits absorbed from omitted
            # source particles. Fragment ranges do cover all deposits; assign
            # each fragment's energy once, to its stored representative.
            start, stop = table["frag_pc_start"][row], table["frag_pc_end"][row]
            particle.energy_deposit = 0.0
            if table["frag_id"][row] == row and start >= 0:
                particle.energy_deposit = float(
                    energy_prefix[stop] - energy_prefix[start]
                )
            particles.append(particle)
        return ObjectListData(particles, Particle())

    def _fill_ancestry(
        self, particle: Particle, row: int, table: dict[str, np.ndarray], meta: Meta
    ) -> None:
        """Populate available genealogy without dereferencing missing links.

        Parameters
        ----------
        particle : Particle
            Output truth particle, modified in place.
        row : int
            Event-local table row.
        table : dict[str, np.ndarray]
            Particle columns with normalized relationships.
        meta : Meta
            Shared detector transform in centimetres.
        """
        # Missing links must never index the last particle through -1.
        for relation in ("parent", "ancestor"):
            relative = int(table[relation + "_id"][row])
            if relative >= 0:
                setattr(
                    particle,
                    relation + "_track_id",
                    int(table["geant4_trackid"][relative]),
                )
                setattr(particle, relation + "_pdg_code", int(table["pdg"][relative]))
                setattr(
                    particle,
                    relation + "_t",
                    float(table["start_pos"][relative, 3]),
                )
        parent = int(table["parent_id"][row])
        if parent >= 0:
            particle.parent_position = (
                (table["start_pos"][parent, :3] / 10.0 - meta.lower) / meta.size
            ).astype(self.ftype)
        if "geant4_parent_trackid" in table:
            particle.parent_track_id = int(table["geant4_parent_trackid"][row])
            particle.parent_pdg_code = int(table["geant4_parent_pdg"][row])


class SuperaParticlePointsParser(SuperaParserBase):
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


class SuperaParticleCoordsParser(SuperaParserBase):
    """One pair of collected-step endpoints per stored particle-table row.

    Instance representatives use upstream collected-step kinematics. Other
    rows use their fragment's earliest/latest deposits. Missing instance
    endpoints use its non-LE deposits (all deposits if none are non-LE).
    These fallbacks are voxelized deposit positions, not exact step boundaries.
    Non-track ends equal their starts. Unavailable coordinates and
    times remain NaN; no creation/destruction positions are substituted.
    """

    name = "particle_coords"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return aligned start/end coordinates and time/shape features.

        Parameters
        ----------
        trees : dict[str, Any]
            Projected native event products.

        Returns
        -------
        TensorData
            Six coordinate columns and two features, with one row per particle.
            Point times are converted from microseconds to nanoseconds.
        """
        event = self.prepare(trees)
        meta = event["meta"]
        starts, ends = self.endpoints(trees)
        shapes = event["labels"]["shape"]
        coords = np.column_stack(
            (
                (starts[:, :3] / 10.0 - meta.lower) / meta.size,
                (ends[:, :3] / 10.0 - meta.lower) / meta.size,
            )
        )
        coords[~np.isfinite(coords)] = np.nan
        return TensorData(
            coords=coords.astype(self.ftype),
            features=np.column_stack((starts[:, 3], shapes)).astype(self.ftype),
            meta=meta,
            coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
            feature_fields={"time": (0,), "shape": (1,)},
        )


class SuperaParticleGraphParser(SuperaParserBase):
    """Directed stored-parent links in the event-local particle namespace.

    Truth construction contracts invisible rows and groups using this graph.
    Retain every stored link so contraction can recover visible descendants.
    """

    name = "particle_graph"

    def __call__(self, trees: dict[str, Any]) -> EdgeIndexData:
        """Return parent-to-child edges, excluding roots and missing parents.

        Parameters
        ----------
        trees : dict[str, Any]
            Projected native event products.

        Returns
        -------
        EdgeIndexData
            Directed edges with a span equal to the full particle-table length.
        """
        table = self.prepare(trees)["columns"]
        parents, children = table["parent_id"], table["id"]
        valid = (parents >= 0) & (parents != children)
        return EdgeIndexData(
            features=np.stack((parents[valid], children[valid])), span=len(children)
        )
