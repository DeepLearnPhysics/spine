"""Artificial provenance-boundary track breaking for systematic studies.

The transformation emulates failures which incorrectly separate one
reconstructed track into multiple objects. Boundary eligibility and output
membership are determined exclusively from voxel-aligned logical TPC source
identifiers. Coordinates are consulted only when a boundary defines an
optional angular response.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import numpy as np

from spine.calib.function import CalibrationFunction
from spine.constants import TRACK_SHP
from spine.data import IndexBatch, TensorBatch

from ...registry import ProviderSpec, register_provider
from ...stage import ChainStage
from ...state import ChainState, StageResult

__all__ = [
    "LogicalTPCBoundary",
    "TrackBreakingStage",
    "build_track_breaking_stage",
]


@dataclass(frozen=True)
class LogicalTPCBoundary:
    """Configuration for one boundary between two logical-TPC groups.

    Parameters
    ----------
    name : str
        Unique boundary label used in deterministic random sampling and
        diagnostic outputs.
    module_id : int, optional
        Module to which the boundary applies. If omitted, the boundary applies
        to a qualifying track in any single module.
    tpc_groups : tuple of tuple of int
        Two disjoint groups of logical TPC identifiers which define opposite
        sides of the boundary.
    frequency : float
        Base probability of breaking an eligible track, in ``[0, 1]``.
    normal : numpy.ndarray, optional
        Unit normal used only to evaluate the angular response.
    response : CalibrationFunction, optional
        Multiplicative response evaluated at the absolute cosine between the
        local track direction and ``normal``.
    """

    name: str
    module_id: int | None
    tpc_groups: tuple[tuple[int, ...], tuple[int, ...]]
    frequency: float
    normal: np.ndarray | None
    response: CalibrationFunction | None

    @classmethod
    def from_config(cls, index: int, config: Any) -> "LogicalTPCBoundary":
        """Validate and normalize one configured logical-TPC boundary.

        Parameters
        ----------
        index : int
            Position of the boundary in the configured location list, used to
            generate a default name.
        config : dict
            Boundary configuration. Exactly one of ``tpc_pair`` or
            ``tpc_groups`` must be supplied. A pair is shorthand for two
            singleton groups.

        Returns
        -------
        LogicalTPCBoundary
            Immutable normalized boundary configuration.

        Raises
        ------
        TypeError
            If the configuration, module identifier, or response expression
            has an invalid type.
        ValueError
            If selectors overlap, are empty or ambiguous, or if a numeric
            option lies outside its allowed range.
        """
        if not isinstance(config, dict):
            raise TypeError("Each track-breaking location must be a mapping.")

        # Separate supported options before validating their relationships.
        cfg = dict(config)
        name = cfg.pop("name", f"boundary_{index}")
        module_id = cfg.pop("module_id", None)
        tpc_pair = cfg.pop("tpc_pair", None)
        tpc_groups = cfg.pop("tpc_groups", None)
        frequency = cfg.pop("frequency", None)
        normal = cfg.pop("normal", None)
        expression = cfg.pop("angular_response", None)
        if cfg:
            keys = ", ".join(sorted(cfg))
            raise ValueError(f"Unknown track-breaking location options: {keys}.")

        if not isinstance(name, str) or not name:
            raise ValueError("Track-breaking location names must be nonempty strings.")
        if module_id is not None and not isinstance(module_id, int):
            raise TypeError("Track-breaking `module_id` must be an integer or null.")
        if (tpc_pair is None) == (tpc_groups is None):
            raise ValueError(
                "Specify exactly one of track-breaking `tpc_pair` or `tpc_groups`."
            )
        groups: tuple[tuple[int, ...], tuple[int, ...]]
        if tpc_pair is not None:
            # A pair retains the concise configuration for one exact boundary.
            if (
                not isinstance(tpc_pair, (list, tuple))
                or len(tpc_pair) != 2
                or not all(isinstance(tpc, int) for tpc in tpc_pair)
                or tpc_pair[0] == tpc_pair[1]
            ):
                raise ValueError(
                    "Track-breaking `tpc_pair` must contain two distinct logical "
                    "TPC IDs."
                )
            groups = ((int(tpc_pair[0]),), (int(tpc_pair[1]),))
        else:
            if (
                not isinstance(tpc_groups, (list, tuple))
                or len(tpc_groups) != 2
                or not all(isinstance(group, (list, tuple)) for group in tpc_groups)
                or not all(group for group in tpc_groups)
                or not all(
                    isinstance(tpc, int) for group in tpc_groups for tpc in group
                )
            ):
                raise ValueError(
                    "Track-breaking `tpc_groups` must contain two nonempty lists "
                    "of logical TPC IDs."
                )
            # Construct each group explicitly so static type checkers retain
            # the fixed two-element outer tuple required by the dataclass.
            first_group = tuple(int(tpc) for tpc in tpc_groups[0])
            second_group = tuple(int(tpc) for tpc in tpc_groups[1])
            groups = (first_group, second_group)
            if len(set(groups[0])) != len(groups[0]) or len(set(groups[1])) != len(
                groups[1]
            ):
                raise ValueError("Logical TPC IDs within each group must be unique.")
            if set(groups[0]).intersection(groups[1]):
                raise ValueError("Track-breaking logical TPC groups must be disjoint.")

        if not isinstance(frequency, (int, float)) or not 0.0 <= frequency <= 1.0:
            raise ValueError("Track-breaking `frequency` must lie in [0, 1].")

        # A boundary normal has no role unless an angular model is requested.
        normal_array = None
        if normal is not None:
            normal_array = np.asarray(normal, dtype=np.float64)
            if normal_array.shape != (3,) or not np.all(np.isfinite(normal_array)):
                raise ValueError("Track-breaking `normal` must be a finite 3-vector.")
            magnitude = np.linalg.norm(normal_array)
            if magnitude == 0.0:
                raise ValueError("Track-breaking `normal` must be nonzero.")
            normal_array = normal_array / magnitude

        response = None
        if expression is not None:
            if not isinstance(expression, str):
                raise TypeError("Track-breaking `angular_response` must be a string.")
            if normal_array is None:
                raise ValueError(
                    "Track-breaking locations with an angular response require `normal`."
                )
            response = CalibrationFunction(expression)

        return cls(
            name,
            module_id,
            groups,
            float(frequency),
            normal_array,
            response,
        )


class TrackBreakingStage(ChainStage):
    """Split track clusters at configured logical-TPC provenance boundaries.

    The stage may operate on fragments before particle aggregation or on
    particles immediately afterward. In particle mode, cluster membership,
    semantic shape and primary indexes are replaced together so all downstream
    particle-aligned providers see one consistent object domain.
    """

    def __init__(
        self,
        name: str,
        target: str,
        locations: list[LogicalTPCBoundary],
        seed: int = 0,
        min_voxels_per_side: int | None = None,
    ) -> None:
        """Initialize a fragment- or particle-level breaking transformation.

        Parameters
        ----------
        name : str
            Unique stage name used to namespace diagnostic outputs.
        target : {"fragment", "particle"}
            Canonical object family to transform.
        locations : list of LogicalTPCBoundary
            Provenance boundaries evaluated in the configured order.
        seed : int, default 0
            Base seed used to produce deterministic event/object draws.
        min_voxels_per_side : int, optional
            Minimum number of voxels required in both children. If either side
            is smaller, the candidate is left intact.

        Raises
        ------
        TypeError
            If ``seed`` is not an integer.
        ValueError
            If the target, location list, names, or size threshold is invalid.
        """
        super().__init__(name)
        if target not in {"fragment", "particle"}:
            raise ValueError(
                "Track-breaking `target` must be `fragment` or `particle`."
            )
        if not locations:
            raise ValueError("Track breaking requires at least one location.")
        if not isinstance(seed, int):
            raise TypeError("Track-breaking `seed` must be an integer.")
        if min_voxels_per_side is not None and min_voxels_per_side < 1:
            raise ValueError("`min_voxels_per_side` must be positive when provided.")

        location_names = [location.name for location in locations]
        if len(set(location_names)) != len(location_names):
            raise ValueError("Track-breaking location names must be unique.")

        self.target = target
        self.locations = locations
        self.seed = seed
        self.min_voxels_per_side = min_voxels_per_side

        # Configure the stage contract for the selected canonical object family.
        prefix = target
        required = {"point_data", "sources", f"{prefix}_clusts", f"{prefix}_shapes"}
        replaced = {f"{prefix}_clusts", f"{prefix}_shapes"}
        if target == "particle":
            required.add("particle_primaries")
            replaced.add("particle_primaries")
        self.requires = frozenset(required)
        self.provides = frozenset(replaced)
        self.replaces = frozenset(replaced)

    @staticmethod
    def _direction(points: np.ndarray) -> np.ndarray | None:
        """Estimate an unoriented principal direction for a point subset.

        Parameters
        ----------
        points : numpy.ndarray
            ``(N, 3)`` coordinates from one side of a boundary.

        Returns
        -------
        numpy.ndarray, optional
            Principal direction, or ``None`` when fewer than two distinct
            points are available.
        """
        if len(points) < 2:
            return None
        centered = points - np.mean(points, axis=0)
        if not np.any(centered):
            return None
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        return vectors[0]

    def _probability(
        self,
        location: LogicalTPCBoundary,
        points: np.ndarray,
        side_masks: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Evaluate the location-specific angular breaking probability.

        Parameters
        ----------
        location : LogicalTPCBoundary
            Boundary containing the base frequency and optional response.
        points : numpy.ndarray
            ``(N, 3)`` candidate coordinates.
        side_masks : tuple of numpy.ndarray
            Boolean membership masks for the two boundary sides.

        Returns
        -------
        float
            Effective probability clipped to ``[0, 1]``. If neither side has
            a measurable direction, the angular probability is zero.

        Raises
        ------
        ValueError
            If the configured angular response returns a non-finite value.
        """
        if location.response is None:
            return location.frequency

        assert location.normal is not None
        # Estimate each side independently so an uncertain absolute drift
        # offset cannot bias a direction fitted across the boundary.
        cosines = []
        for mask in side_masks:
            direction = self._direction(points[mask])
            if direction is not None:
                cosines.append(abs(float(np.dot(direction, location.normal))))
        if not cosines:
            return 0.0

        x = np.asarray([np.mean(cosines)], dtype=np.float64)
        factor = float(location.response(x)[0])
        if not np.isfinite(factor):
            raise ValueError(
                f"Angular response for track-breaking location `{location.name}` "
                "returned a non-finite value."
            )
        return float(np.clip(location.frequency * factor, 0.0, 1.0))

    def _draw(
        self,
        location: LogicalTPCBoundary,
        batch_id: int,
        local_index: np.ndarray,
        run_info: Any | None,
    ) -> float:
        """Produce a stable draw from event and cluster provenance.

        Parameters
        ----------
        location : LogicalTPCBoundary
            Boundary being evaluated.
        batch_id : int
            Event position, used only when stable run information is absent.
        local_index : numpy.ndarray
            Event-local voxel membership of the candidate object.
        run_info : object, optional
            Object exposing ``run``, ``subrun`` and ``event`` identifiers.

        Returns
        -------
        float
            Deterministic uniform draw in ``[0, 1)``.
        """
        identity: tuple[int, ...]
        if run_info is not None:
            event = tuple(
                int(getattr(run_info, field, -1))
                for field in ("run", "subrun", "event")
            )
            identity = (self.seed, *event)
        else:
            # Batch position is only a fallback when no stable event identity
            # is available. With run information, draws are invariant under
            # rebatching and event reordering.
            identity = (self.seed, batch_id)
        digest = blake2b(digest_size=8)
        digest.update(np.asarray(identity, dtype=np.int64).tobytes())
        digest.update(np.asarray(local_index, dtype=np.int64).tobytes())
        digest.update(location.name.encode("utf-8"))
        return float(
            np.random.default_rng(int.from_bytes(digest.digest(), "little")).random()
        )

    def forward(self, state: ChainState) -> StageResult:
        """Split eligible tracks and replace the selected cluster family.

        Parameters
        ----------
        state : ChainState
            Chain state containing aligned point/source data and the selected
            fragment or particle products.

        Returns
        -------
        StageResult
            Replacement clusters and shapes, optional particle primaries, and
            object-aligned parent, location, probability and draw diagnostics.

        Raises
        ------
        ValueError
            If source provenance is unavailable or does not contain exactly
            ``[module, logical TPC]`` columns.
        """
        # Fetch the canonical family selected when the stage was initialized.
        point_data = state.require("point_data", self.name)
        sources_batch: TensorBatch = state.require("sources", self.name)
        clusts: IndexBatch = state.require(f"{self.target}_clusts", self.name)
        shapes: TensorBatch = state.require(f"{self.target}_shapes", self.name)
        primaries: IndexBatch | None = (
            state.require("particle_primaries", self.name)
            if self.target == "particle"
            else None
        )

        if point_data.sources is None:
            raise ValueError("Track breaking requires voxel-aligned source provenance.")

        # Structural transformations run in NumPy and rebuild portable batch
        # products before returning to downstream model providers.
        data_np = point_data.data.to_numpy()
        sources_np = sources_batch.to_numpy().tensor
        if sources_np.ndim != 2 or sources_np.shape[1] != 2:
            raise ValueError(
                "Track-breaking sources must contain [module, logical TPC] pairs."
            )
        clusts_np = clusts.to_numpy()
        shapes_np = shapes.to_numpy().tensor.astype(np.int64, copy=False)
        primaries_np = None if primaries is None else primaries.to_numpy()
        run_info = state.get("run_info")
        metadata = state.get("meta")

        # Accumulate one flattened object list plus per-event object counts.
        output_clusts: list[np.ndarray] = []
        output_shapes: list[int] = []
        output_primaries: list[np.ndarray] = []
        counts = np.zeros(clusts.batch_size, dtype=np.int64)
        parent_ids: list[int] = []
        location_ids: list[int] = []
        probabilities: list[float] = []
        draws: list[float] = []

        for batch_id in range(clusts.batch_size):
            lower, upper = clusts_np.edges[batch_id : batch_id + 2]
            offset = int(clusts_np.offsets[batch_id])
            # Resolve event context shared across module-expanded batches.
            run = None
            if run_info is not None and len(run_info) > 0:
                repeat = clusts.batch_size // len(run_info)
                if repeat * len(run_info) == clusts.batch_size:
                    run = run_info[batch_id // repeat]
            meta = None
            if metadata is not None and len(metadata) > 0:
                repeat = clusts.batch_size // len(metadata)
                if repeat * len(metadata) == clusts.batch_size:
                    meta = metadata[batch_id // repeat]

            for parent_id, object_id in enumerate(range(int(lower), int(upper))):
                index = np.asarray(clusts_np.index_list[object_id], dtype=np.int64)
                # A particle may cross multiple configured boundaries, so
                # thread the pieces produced by one location into the next.
                pieces = [(index, -1, np.nan, np.nan)]

                if int(shapes_np[object_id]) == TRACK_SHP:
                    for location_id, location in enumerate(self.locations):
                        next_pieces = []
                        for (
                            piece,
                            previous_location,
                            previous_prob,
                            previous_draw,
                        ) in pieces:
                            source = np.asarray(sources_np[piece], dtype=np.int64)
                            # A qualifying piece must live in one module, use
                            # only configured logical TPCs and touch both sides.
                            modules = np.unique(source[:, 0])
                            tpcs = set(np.unique(source[:, 1]).tolist())
                            allowed_tpcs = set(location.tpc_groups[0]).union(
                                location.tpc_groups[1]
                            )
                            module_match = len(modules) == 1 and (
                                location.module_id is None
                                or int(modules[0]) == location.module_id
                            )
                            group_match = (
                                tpcs.issubset(allowed_tpcs)
                                and bool(tpcs.intersection(location.tpc_groups[0]))
                                and bool(tpcs.intersection(location.tpc_groups[1]))
                            )
                            if not module_match or not group_match:
                                next_pieces.append(
                                    (
                                        piece,
                                        previous_location,
                                        previous_prob,
                                        previous_draw,
                                    )
                                )
                                continue

                            side_masks = (
                                np.isin(source[:, 1], location.tpc_groups[0]),
                                np.isin(source[:, 1], location.tpc_groups[1]),
                            )
                            # Apply the optional threshold before consuming a
                            # random draw so rejected candidates remain stable.
                            threshold = self.min_voxels_per_side
                            if threshold is not None and any(
                                np.count_nonzero(mask) < threshold
                                for mask in side_masks
                            ):
                                next_pieces.append(
                                    (
                                        piece,
                                        previous_location,
                                        previous_prob,
                                        previous_draw,
                                    )
                                )
                                continue

                            # Coordinates affect only the optional angular
                            # probability; provenance always owns membership.
                            points = np.asarray(
                                data_np.tensor[piece][:, data_np.coordinate_columns()]
                            )
                            if meta is not None and location.response is not None:
                                points = meta.to_cm(points, center=True)
                            probability = self._probability(
                                location, points, side_masks
                            )
                            local_index = piece - offset
                            draw = self._draw(location, batch_id, local_index, run)
                            if draw >= probability:
                                next_pieces.append((piece, -1, probability, draw))
                                continue

                            for mask in side_masks:
                                next_pieces.append(
                                    (piece[mask], location_id, probability, draw)
                                )
                        pieces = next_pieces

                for piece, location_id, probability, draw in pieces:
                    output_clusts.append(piece)
                    output_shapes.append(int(shapes_np[object_id]))
                    if primaries_np is not None:
                        # Track primaries are the full particle group. Unchanged
                        # non-track objects preserve their original primary.
                        if int(shapes_np[object_id]) == TRACK_SHP:
                            output_primaries.append(piece)
                        else:
                            output_primaries.append(
                                np.asarray(
                                    primaries_np.index_list[object_id],
                                    dtype=np.int64,
                                )
                            )
                    parent_ids.append(parent_id)
                    location_ids.append(location_id)
                    probabilities.append(probability)
                    draws.append(draw)
                    counts[batch_id] += 1

        # Reconstruct the transformed canonical batch products.
        single_counts = np.asarray(
            [len(index) for index in output_clusts], dtype=np.int64
        )
        transformed_clusts = IndexBatch(
            output_clusts,
            clusts_np.spans,
            counts,
            single_counts,
            default=np.empty(0, dtype=np.int64),
        )
        transformed_shapes = TensorBatch(np.asarray(output_shapes), counts)
        products: dict[str, Any] = {
            f"{self.target}_clusts": transformed_clusts,
            f"{self.target}_shapes": transformed_shapes,
        }
        if primaries_np is not None:
            products["particle_primaries"] = IndexBatch(
                output_primaries,
                primaries_np.spans,
                counts,
                np.asarray([len(index) for index in output_primaries], dtype=np.int64),
                default=np.empty(0, dtype=np.int64),
            )

        # Diagnostics are aligned with the transformed object list and use the
        # stage name to allow multiple independent systematic transformations.
        prefix = self.name
        outputs = {
            f"{prefix}_parent_ids": TensorBatch(
                np.asarray(parent_ids, dtype=np.int64), counts
            ),
            f"{prefix}_location_ids": TensorBatch(
                np.asarray(location_ids, dtype=np.int64), counts
            ),
            f"{prefix}_probabilities": TensorBatch(
                np.asarray(probabilities, dtype=np.float32), counts
            ),
            f"{prefix}_draws": TensorBatch(np.asarray(draws, dtype=np.float32), counts),
        }
        return StageResult(products, outputs)


def build_track_breaking_stage(
    name: str,
    config: dict[str, Any],
    _owner: Any,
) -> ChainStage:
    """Build a logical-TPC-boundary track-breaking transformation.

    Parameters
    ----------
    name : str
        Unique stage name.
    config : dict
        Stage configuration containing a ``locations`` list and optional
        ``target``, ``seed`` and ``min_voxels_per_side`` options.
    _owner : object
        Full-chain module owner, unused because this transformation has no
        trainable parameters.

    Returns
    -------
    ChainStage
        Configured track-breaking stage.

    Raises
    ------
    TypeError
        If ``locations`` or another typed option is invalid.
    ValueError
        If a location or stage value is invalid.
    """
    cfg = dict(config)
    target = cfg.pop("target", "particle")
    raw_locations = cfg.pop("locations", None)
    if not isinstance(raw_locations, list):
        raise TypeError("Track-breaking `locations` must be a list.")
    locations = [
        LogicalTPCBoundary.from_config(index, location)
        for index, location in enumerate(raw_locations)
    ]
    return TrackBreakingStage(name, target, locations, **cfg)


PROVIDER_SPEC = register_provider(
    ProviderSpec("track_breaking", build_track_breaking_stage)
)
