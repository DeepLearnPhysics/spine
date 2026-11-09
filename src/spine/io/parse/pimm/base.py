"""Shared PIMM hit voxelization and particle supervision."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.constants import PDG_TO_PID
from spine.data import Meta

from ..base import ParserBase


def _shape(code: int | np.ndarray) -> np.ndarray:
    """Map upstream one-based semantic codes to SPINE's five classes.

    Parameters
    ----------
    code : int or np.ndarray
        Upstream semantic codes; unsupported codes become -1.

    Returns
    -------
    np.ndarray
        SPINE class indexes with the same shape as the input.
    """
    return np.array([-1, 0, 1, 3, 2, 4, -1])[np.clip(code, 0, 6)]


class PIMMParserBase(ParserBase):
    """Share one hit grid and overlap selection across all training products.

    Pixel indexes are converted to nominal detector coordinates in mm, then
    voxelized. Metadata uses cm. Charge is summed without energy conversion.
    Label ties use track, shower, delta, Michel, LE, then source row order.
    Voxels without labelled contributions are excluded by default. Charge
    from every contribution is retained in a labelled voxel. Wire hits cannot
    supply a unique 3D position.
    """

    def __init__(
        self,
        dtype: str,
        voxel_size_mm: float | tuple[float, ...] = 3.0,
        origin_mm: tuple[float, ...] = (0.0, 0.0, 0.0),
        include_unlabeled: bool = False,
    ) -> None:
        """Configure the common detector grid.

        Parameters
        ----------
        dtype : str
            Output floating-point dtype.
        voxel_size_mm : float or sequence[float], default 3
            Three physical voxel dimensions in mm.
        origin_mm : sequence[float], default (0, 0, 0)
            Detector grid origin in mm.
        include_unlabeled : bool, default False
            Retain voxels with target -1 for inspection. Generic SPINE
            segmentation training requires these voxels to be excluded.
        """
        super().__init__(dtype, pimm_event="pimm")
        self.size = tuple(np.broadcast_to(np.asarray(voxel_size_mm, float), (3,)))
        self.origin = tuple(np.asarray(origin_mm, float))
        if len(self.origin) != 3 or not np.isfinite(self.origin).all():
            raise ValueError("origin_mm must contain three finite values.")
        if not np.isfinite(self.size).all() or min(self.size) <= 0:
            raise ValueError("voxel_size_mm must be finite and positive.")
        self.include_unlabeled = include_unlabeled
        self.grid = (self.size, self.origin, include_unlabeled)

    def prepare(self, trees: dict[str, Any]) -> dict[str, Any]:
        """Cache an aligned charge image, labels and particle-table references.

        Parameters
        ----------
        trees : dict
            Reader products containing the nested PIMM event.

        Returns
        -------
        dict
            Voxel coordinates, charge, truth references and shared metadata.
        """
        cache = trees.setdefault("_pimm_prepared", {})
        if self.grid in cache:
            return cache[self.grid]
        raw = trees["pimm"]
        hits, table = raw["hits"], raw["particle"]
        points = self._coordinates(hits, raw["metadata"])
        values = np.asarray(hits["energy"]).reshape(-1)
        semantic = np.asarray(hits["segment_motif"]).reshape(-1)
        fragment = np.asarray(hits["instance_fragment"]).reshape(-1)
        instance = np.asarray(hits["instance_particle"]).reshape(-1)
        if any(
            len(col) != len(points) for col in (values, semantic, fragment, instance)
        ):
            raise ValueError("PIMM hit and label row counts differ.")
        if not np.isfinite(values).all() or not np.isfinite(points).all():
            raise ValueError("Nonfinite PIMM hit charge or coordinates.")
        if np.any((semantic < -1) | (semantic > 4)):
            raise ValueError("Invalid PIMM semantic labels.")
        for refs in (fragment, instance):
            if np.any((refs < -1) | (refs >= len(table["parent"]))):
                raise ValueError("Invalid PIMM particle-table reference.")

        event = self._voxelize(points, values, semantic)
        winner = event.pop("winner")
        if not self.include_unlabeled:
            keep = semantic[winner] >= 0
            winner = winner[keep]
            event["coords"] = event["coords"][keep]
            event["energy"] = event["energy"][keep]
        event.update(
            shape=semantic[winner],
            fragment=fragment[winner],
            instance=instance[winner],
            columns=table,
        )
        event["labels"] = self._particle_labels(event, raw["interaction"])
        cache[self.grid] = event
        return event

    def _voxelize(
        self, points: np.ndarray, values: np.ndarray, semantic: np.ndarray
    ) -> dict[str, Any]:
        """Sum coincident charge and select one shared label owner per voxel.

        Parameters
        ----------
        points : np.ndarray
            Physical hit coordinates in mm.
        values : np.ndarray
            Signed hit charge in the source units.
        semantic : np.ndarray
            Class labels used to resolve overlaps, including unknown (-1).

        Returns
        -------
        dict
            Sparse coordinates, summed charge, winning rows and cm metadata.
        """
        size = np.asarray(self.size)
        cells = np.floor((points - self.origin) / size).astype(np.int64)
        lower_cell = cells.min(axis=0) if len(points) else np.zeros(3, dtype=np.int64)
        coords, inverse = np.unique(cells - lower_cell, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        ranks = np.full(len(points), 5)
        for rank, shape in enumerate((1, 0, 3, 2, 4)):
            ranks[semantic == shape] = rank
        order = np.lexsort((np.arange(len(points)), ranks, inverse))
        lower = (np.asarray(self.origin) + lower_cell * size) / 10.0
        count = coords.max(axis=0) + 1 if len(points) else np.ones(3, dtype=np.int64)
        return {
            "coords": coords,
            "energy": np.bincount(inverse, weights=values),
            "winner": (
                order[np.r_[True, np.diff(inverse[order]) != 0]]
                if len(points)
                else order
            ),
            "meta": Meta(
                lower=lower,
                upper=lower + count * size / 10.0,
                size=size / 10.0,
                count=count,
            ),
        }

    @staticmethod
    def _coordinates(hits: dict, metadata: dict) -> np.ndarray:
        """Convert pixel channel/tick indexes to nominal positions in mm.

        Parameters
        ----------
        hits : dict
            Upstream hit cloud in ``(py, pz, time)`` order.
        metadata : dict
            Per-volume pixel geometry copied by the PIMM label converter.

        Returns
        -------
        np.ndarray
            Detector ``(x, y, z)`` coordinates, one row per hit.
        """
        if hits["readout_type"] != "pixel":
            raise ValueError("PIMM 3D parsers require pixel hits; wire hits are 2D.")
        geometry = metadata.get("pixel_geometry")
        if geometry is None or metadata.get("hit_x_from") != "nominal":
            raise ValueError(
                "PIMM pixel input requires nominal pixel_geometry metadata."
            )
        if metadata.get("hit_energy") != "charge":
            raise ValueError("PIMM tensor input requires hit_energy=charge.")
        coordinates = np.asarray(hits["coord"], dtype=float)
        positions = np.empty_like(coordinates)
        for plane_id, plane in enumerate(hits["planes"]):
            volume = int(plane.split("_")[1])
            g = geometry[volume]
            mask = np.asarray(hits["plane_id"]).reshape(-1) == plane_id
            py, pz, ticks = coordinates[mask].T
            positions[mask, 0] = (
                g["x_anode_mm"]
                - g["drift_direction"]
                * (ticks - g["reference_tick"])
                * g["mm_per_tick"]
            )
            positions[mask, 1] = g["y_min_mm"] + (py + 0.5) * g["pitch_mm"]
            positions[mask, 2] = g["z_min_mm"] + (pz + 0.5) * g["pitch_mm"]
        return positions

    @staticmethod
    def _particle_labels(event: dict, interactions: dict) -> dict[str, np.ndarray]:
        """Build compact truth labels using upstream particle-table rows.

        Parameters
        ----------
        event : dict
            Prepared image and upstream particle table.
        interactions : dict
            Upstream interaction truth, including vertices in mm.

        Returns
        -------
        dict
            Particle sidecar used by fragment and graph supervision.
        """
        table, meta = event["columns"], event["meta"]
        rows = np.arange(len(table["parent"]))
        host = table["fragment_instance"]
        valid = (host >= 0) & (host < len(rows))
        interaction = np.full(len(rows), -1, dtype=np.int64)
        pid = np.full(len(rows), -1, dtype=np.int64)
        primary = np.full(len(rows), -1, dtype=np.int64)
        interaction[valid] = table["interaction"][host[valid]]
        pid[valid] = [
            PDG_TO_PID.get(abs(int(p)), -1) for p in table["pdg"][host[valid]]
        ]
        primary[valid] = table["is_primary"][host[valid]]
        vertices = np.full((len(rows), 3), np.nan)
        if "vertex" in interactions:
            known = (interaction >= 0) & (interaction < len(interactions["vertex"]))
            vertices[known] = interactions["vertex"][interaction[known]]
        return {
            "particle": rows,
            "group": host.copy(),
            "ancestor": table["ancestor"].copy(),
            "interaction": interaction,
            "nu": np.full(len(rows), -1),
            "pid": pid,
            "group_primary": np.where(valid, rows == host, -1),
            "interaction_primary": primary,
            "vertex": (vertices / 10.0 - meta.lower) / meta.size,
            "momentum": np.linalg.norm(table["start_mom"][:, :3], axis=1),
            "energy_init": table["start_mom"][:, 3].copy(),
            "shape": table["motif"].copy(),
        }

    def endpoints(self, trees: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return stored collected-step endpoints; missing values remain NaN.

        Parameters
        ----------
        trees : dict
            Nested PIMM event products.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Start/end positions in mm and times in ns. Non-track ends repeat
            their starts. No truth-deposit or creation-point fallback is used.
        """
        event = self.prepare(trees)
        starts, ends = (
            event["columns"][key].copy() for key in ("first_pos", "last_pos")
        )
        ends[event["labels"]["shape"] != 1] = starts[event["labels"]["shape"] != 1]
        starts[~np.isfinite(starts)] = np.nan
        ends[~np.isfinite(ends)] = np.nan
        return starts, ends
