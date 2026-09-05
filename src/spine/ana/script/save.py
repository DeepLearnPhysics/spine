"""Analysis script used to store the reconstruction output to CSV files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spine.ana.base import AnaBase
from spine.data.out import (
    RecoFragment,
    RecoInteraction,
    RecoParticle,
    TruthFragment,
    TruthInteraction,
    TruthParticle,
)

__all__ = ["SaveAna"]


class SaveAna(AnaBase):
    """Store reconstructed or truth objects and their matches in CSV files."""

    name = "save"

    # Valid match modes
    _match_modes = (None, "reco_to_truth", "truth_to_reco", "both", "all")

    # Default object types when a match is not found
    _default_objs = (
        ("reco_fragments", RecoFragment()),
        ("truth_fragments", TruthFragment()),
        ("reco_particles", RecoParticle()),
        ("truth_particles", TruthParticle()),
        ("reco_interactions", RecoInteraction()),
        ("truth_interactions", TruthInteraction()),
    )

    def __init__(
        self,
        obj_type: str | Sequence[str],
        fragment: Sequence[str] | None = None,
        particle: Sequence[str] | None = None,
        interaction: Sequence[str] | None = None,
        lengths: Mapping[str, int] | None = None,
        run_mode: str = "both",
        match_mode: str | None = "both",
        **kwargs: Any,
    ) -> None:
        """Initialize the CSV logging class.

        If any of `fragment`, `particle` or `interaction` are specified as
        sequences of strings, only those object attributes are written.

        Parameters
        ----------
        obj_type : str or Sequence[str]
            Object types to write
        fragment : Sequence[str], optional
            List of fragment attributes to store
        particle : Sequence[str], optional
            List of particle attributes to store
        interaction : Sequence[str], optional
            List of interaction attributes to store
        lengths : Mapping[str, int], optional
            Lengths to use for variable-length object attributes
        run_mode : str, default 'both'
            Source object collection(s) that produce output rows. A
            directional matched export may select one source while loading
            the opposite collection only to populate joined columns.
        match_mode : str, default 'both'
            Best-match direction(s) to join onto the selected source rows.
            ``truth_to_reco`` with ``run_mode='truth'`` writes one row per
            truth object without producing a reco-originated output file.
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(obj_type=obj_type, run_mode=run_mode, **kwargs)

        # Store the matching mode
        self.match_mode = match_mode
        if match_mode not in self._match_modes:
            raise ValueError(
                f"Invalid matching mode: {self.match_mode}. Must be one "
                f"of {self._match_modes}."
            )
        writes_both = run_mode in ("both", "all")
        if match_mode in ("both", "all") and not writes_both:
            raise ValueError(
                f"Matching mode `{match_mode}` writes both source directions "
                "and therefore requires `run_mode: both`."
            )
        if match_mode in ("reco_to_truth", "truth_to_reco"):
            source = match_mode.split("_to_")[0]
            if not writes_both and run_mode != source:
                raise ValueError(
                    f"Matching mode `{match_mode}` requires `{source}` source "
                    f"rows, but `run_mode` is `{run_mode}`."
                )

        # Store default objects as a dictionary
        self.default_objs = dict(self._default_objs)

        # Store the list of attributes to store for each object type
        attrs: dict[str, list[str] | None] = {
            "fragments": list(fragment) if fragment is not None else None,
            "particles": list(particle) if particle is not None else None,
            "interactions": list(interaction) if interaction is not None else None,
        }
        # Matching joins need attributes from both sides even when only one
        # source direction is written. Without matching, inspect only the
        # object declinations selected by the run mode.
        attr_prefixes = (
            ("reco", "truth")
            if match_mode is not None or writes_both
            else tuple(self.prefixes)
        )
        self.attrs: dict[str, list[str] | None] = {}
        for obj_t, attrs_t in attrs.items():
            leftover = set(attrs_t) if attrs_t is not None else None
            for prefix in attr_prefixes:
                key = f"{prefix}_{obj_t}"
                if attrs_t is None:
                    self.attrs[key] = None
                    continue

                all_keys = self.default_objs[key].as_dict().keys()
                attrs_key = sorted(set(attrs_t) & set(all_keys))
                self.attrs[key] = attrs_key
                assert leftover is not None
                leftover -= set(attrs_key)

            if leftover is not None and len(leftover) > 0:
                raise ValueError(
                    "The following keys were not found in the selected reco "
                    f"or truth {obj_t}: {leftover}"
                )

        # Store the list of variable-length array lengths
        self.lengths: dict[str, int] | None = (
            dict(lengths) if lengths is not None else None
        )

        # Add the necessary keys associated with matching, if needed
        keys = {}
        if match_mode is not None:
            if self.obj_type is None:
                raise ValueError("Must provide object types when storing matches.")
            for prefix in self.prefixes:
                for obj_name in self.obj_type:
                    if self._uses_matches(prefix):
                        other = "truth" if prefix == "reco" else "reco"
                        suffix = f"{prefix[0]}2{other[0]}"
                        keys[f"{other}_{obj_name}s"] = True
                        keys[f"{obj_name}_matches_{suffix}"] = False
                        keys[f"{obj_name}_matches_{suffix}_overlap"] = False

        self.update_keys(keys)

        # Initialize one CSV writer per object type
        for key in self.obj_keys:
            self.initialize_writer(key)

        if len(self.writers) == 0:
            raise ValueError("Must request to save something.")

    def process(self, data: Mapping[str, Any]) -> None:
        """Serialize the requested object collections from one event.

        Each selected source collection is written as a batch of CSV columns.
        When matching is enabled, source and target attributes are prefixed and
        joined into the same rows. Unmatched sources are retained using the
        appropriate default target object.

        Parameters
        ----------
        data : Mapping[str, Any]
            Event data products, including the requested object collections
            and, when available, their precomputed match pairs.
        """
        # Loop over source collections selected by the run mode
        other_prefix = {"reco": "truth", "truth": "reco"}
        for key in self.obj_keys:
            prefix, obj_type = key.split("_")
            other = other_prefix[prefix]
            attrs = self.attrs[key]
            lengths = self.lengths
            if not self._uses_matches(prefix):
                objects = data[key]
                if len(objects) == 0:
                    continue

                # Serialize and submit the whole event collection together.
                row_dict = self._event_base(len(objects))
                row_dict.update(
                    self.default_objs[key].scalar_columns(
                        objects,
                        attrs,
                        lengths,
                    )
                )
                self.writers[key].append_columns(row_dict)

            else:
                # Combine each source object with its best target match. An
                # unmatched source keeps a row populated by target defaults.
                match_suffix = f"{prefix[0]}2{other[0]}"
                match_key = f"{obj_type[:-1]}_matches_{match_suffix}"
                attrs_other = self.attrs[f"{other}_{obj_type}"]
                if match_key in data:
                    pairs = data[match_key]
                    overlaps = data[f"{match_key}_overlap"]
                else:
                    targets = data[f"{other}_{obj_type}"]
                    pairs, overlaps = [], []
                    for source in data[key]:
                        match_id = source.best_match_id
                        target = (
                            targets[match_id] if 0 <= match_id < len(targets) else None
                        )
                        pairs.append((source, target))
                        overlaps.append(source.best_match_overlap)

                if len(pairs) == 0:
                    continue

                sources = [pair[0] for pair in pairs]
                default_target = self.default_objs[f"{other}_{obj_type}"]
                targets = [
                    pair[1] if pair[1] is not None else default_target for pair in pairs
                ]
                source_columns = self.default_objs[key].scalar_columns(
                    sources,
                    attrs,
                    lengths,
                )
                target_columns = default_target.scalar_columns(
                    targets,
                    attrs_other,
                    lengths,
                )

                row_dict = self._event_base(len(pairs))
                row_dict.update(
                    {
                        f"{prefix}_{name}": values
                        for name, values in source_columns.items()
                    }
                )
                row_dict.update(
                    {
                        f"{other}_{name}": values
                        for name, values in target_columns.items()
                    }
                )
                row_dict["match_overlap"] = np.asarray(overlaps)
                self.writers[key].append_columns(row_dict)

    def process_columnar(self, data: Mapping[str, Any]) -> None:
        """Serialize a chunk of projected object data without rebuilding objects.

        Object arrays are already flattened across the input events. Their
        event offsets provide the row counts needed to broadcast event metadata
        and to translate event-local match IDs into flat target-row indexes.

        Parameters
        ----------
        data : Mapping[str, Any]
            Projected columnar products for a contiguous chunk of events.
        """
        other_prefix = {"reco": "truth", "truth": "reco"}
        for key in self.obj_keys:
            prefix, obj_type = key.split("_")
            other = other_prefix[prefix]
            attrs = self.attrs[key]
            assert attrs is not None

            # Establish the source rows and attach their event metadata.
            product = data[key]
            offsets = np.asarray(product["event_offsets"], dtype=np.int64)
            counts = np.diff(offsets)
            row_dict = self._columnar_base(data, counts)
            source_columns = self._expand_columnar_attrs(
                product, attrs, self.default_objs[key]
            )

            if not self._uses_matches(prefix):
                row_dict.update(source_columns)
                self.writers[key].append_columns(row_dict)
                continue

            # Resolve event-local match IDs against the flattened target
            # collection. Invalid IDs remain populated by target defaults.
            target_key = f"{other}_{obj_type}"
            target = data[target_key]
            target_offsets = np.asarray(target["event_offsets"], dtype=np.int64)
            event_rows = self._event_rows(offsets)
            match_ids = np.asarray(product["best_match_id"], dtype=np.int64)
            target_counts = np.diff(target_offsets)
            valid = (match_ids >= 0) & (match_ids < target_counts[event_rows])
            target_rows = target_offsets[event_rows] + np.maximum(match_ids, 0)

            attrs_other = self.attrs[target_key]
            assert attrs_other is not None
            target_columns = self._expand_columnar_attrs(
                target, attrs_other, self.default_objs[target_key]
            )
            target_defaults = self.default_objs[target_key].scalar_dict(
                attrs_other, self.lengths
            )

            row_dict.update(
                {f"{prefix}_{name}": values for name, values in source_columns.items()}
            )
            for name, values in target_columns.items():
                # Allocate the complete joined column first, then replace only
                # rows with a valid target match.
                default = target_defaults[name]
                joined = np.full(len(match_ids), default, dtype=values.dtype)
                joined[valid] = values[target_rows[valid]]
                row_dict[f"{other}_{name}"] = joined
            row_dict["match_overlap"] = np.asarray(product["best_match_overlap"])
            self.writers[key].append_columns(row_dict)

    def columnar_requests(
        self,
    ) -> dict[str, tuple[tuple[str, ...] | None, bool]]:
        """Describe the input projections required by :meth:`process_columnar`.

        Source products include the configured object attributes and, when
        matching is enabled, the best-match fields. Target products are also
        projected for a match join even if they do not produce their own CSV.

        Returns
        -------
        dict[str, tuple[tuple[str, ...] | None, bool]]
            Product names mapped to their requested fields and required flag.
        """
        requests: dict[str, tuple[tuple[str, ...] | None, bool]] = {
            "run_info": (("run", "subrun", "event"), False)
        }
        for key in self.obj_keys:
            attrs = self.attrs[key]
            self._validate_columnar_attrs(key, attrs)
            assert attrs is not None

            fields = set(attrs)
            prefix, obj_type = key.split("_")
            if self._uses_matches(prefix):
                fields.update(("best_match_id", "best_match_overlap"))
            self._merge_columnar_request(requests, key, fields)

            # A directional run still projects the target columns needed for
            # the join, but does not create a target-originated writer.
            if self._uses_matches(prefix):
                other = "truth" if prefix == "reco" else "reco"
                target_key = f"{other}_{obj_type}"
                target_attrs = self.attrs[target_key]
                self._validate_columnar_attrs(target_key, target_attrs)
                assert target_attrs is not None
                self._merge_columnar_request(
                    requests,
                    target_key,
                    set(target_attrs),
                )

        return requests

    def _uses_matches(self, prefix: str) -> bool:
        """Check whether one source declination includes matched target fields.

        Parameters
        ----------
        prefix : str
            Source declination, either ``'reco'`` or ``'truth'``.

        Returns
        -------
        bool
            Whether rows originating from this declination require a join.
        """
        other = "truth" if prefix == "reco" else "reco"
        return self.match_mode in (f"{prefix}_to_{other}", "both", "all")

    def _event_base(self, count: int) -> dict[str, np.ndarray]:
        """Repeat the current event metadata for an object-row batch.

        Parameters
        ----------
        count : int
            Number of object rows produced by the current event.

        Returns
        -------
        dict[str, np.ndarray]
            Event metadata columns with ``count`` rows each.
        """
        return {
            key: np.repeat(np.asarray([value]), count, axis=0)
            for key, value in self.base_dict.items()
        }

    def _columnar_base(
        self,
        data: Mapping[str, Any],
        counts: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Broadcast chunk-level event metadata onto flattened object rows.

        Parameters
        ----------
        data : Mapping[str, Any]
            Columnar chunk containing event identifiers and optional run data.
        counts : np.ndarray
            Number of objects contributed by each event in the chunk.

        Returns
        -------
        dict[str, np.ndarray]
            Event metadata aligned with the flattened object collection.
        """
        result = {
            "index": self._repeat_event_column(data["index"], counts),
            "file_index": self._repeat_event_column(data["file_index"], counts),
        }
        if "file_entry_index" in data:
            result["file_entry_index"] = self._repeat_event_column(
                data["file_entry_index"], counts
            )
        if "run_info" in data:
            run_info = data["run_info"]
            # Match RunInfo.scalar_dict ordering rather than the compound
            # dataset's field ordering, which HDF5 does not preserve.
            for name in ("run", "subrun", "event"):
                if name in run_info:
                    result[name] = self._repeat_event_column(run_info[name], counts)
        return result

    @staticmethod
    def _repeat_event_column(values: Any, counts: np.ndarray) -> np.ndarray:
        """Broadcast one event-level column onto every corresponding object row.

        Parameters
        ----------
        values : Any
            One scalar or fixed-width value per event.
        counts : np.ndarray
            Number of object rows associated with each event.

        Returns
        -------
        np.ndarray
            Values repeated according to the per-event object counts.
        """
        return np.repeat(np.asarray(values), counts, axis=0)

    @staticmethod
    def _expand_columnar_attrs(
        product: Mapping[str, Any],
        attrs: Sequence[str],
        default_obj: Any,
    ) -> dict[str, np.ndarray]:
        """Expand projected attributes into scalar CSV columns.

        Parameters
        ----------
        product : Mapping[str, Any]
            Projected object product containing flattened attribute arrays.
        attrs : Sequence[str]
            Attributes to include in the output.
        default_obj : DataBase
            Schema-bearing object used to identify vector attributes and axes.

        Returns
        -------
        dict[str, np.ndarray]
            Scalar columns named consistently with ``DataBase.scalar_dict``.

        Raises
        ------
        ValueError
            If an attribute is neither scalar nor fixed-width.
        """
        result = {}
        for attr in attrs:
            values = np.asarray(product[attr])
            if values.ndim == 1:
                result[attr] = values
                continue

            if values.ndim != 2:
                raise ValueError(
                    f"Columnar attribute `{attr}` must be scalar or fixed-width, "
                    f"got shape {values.shape}."
                )
            # Spatial vectors use named axes; other fixed arrays use their
            # integer component indexes.
            labels = (
                default_obj.axes
                if attr in default_obj.pos_attrs + default_obj.vec_attrs
                else tuple(str(i) for i in range(values.shape[1]))
            )
            for i, label in enumerate(labels):
                column = values[:, i]
                if np.issubdtype(column.dtype, np.floating):
                    # scalar_dict converts array elements to Python scalars;
                    # promote floats to retain its established CSV spelling.
                    column = column.astype(np.float64, copy=False)
                result[f"{attr}_{label}"] = column

        return result

    @staticmethod
    def _event_rows(offsets: np.ndarray) -> np.ndarray:
        """Map every flattened object row to its chunk-local event index.

        Parameters
        ----------
        offsets : np.ndarray
            Cumulative object offsets, with one boundary beyond the last event.

        Returns
        -------
        np.ndarray
            Event index associated with each flattened object row.
        """
        return np.repeat(
            np.arange(len(offsets) - 1, dtype=np.int64),
            np.diff(offsets),
        )

    def _validate_columnar_attrs(
        self,
        key: str,
        attrs: Sequence[str] | None,
    ) -> None:
        """Validate that an object projection has fixed, explicit columns.

        Parameters
        ----------
        key : str
            Object product name whose projection is being validated.
        attrs : Sequence[str], optional
            Explicit list of attributes requested for the product.

        Raises
        ------
        ValueError
            If no projection is specified or it contains variable-length data.
        """
        if attrs is None:
            raise ValueError(
                f"Columnar save requires an explicit attribute list for `{key}`."
            )
        default_obj = self.default_objs[key]
        variable = set(attrs).intersection(default_obj.var_length_attrs)
        if variable:
            raise ValueError(
                "Columnar save currently supports fixed attributes only; "
                f"`{key}` requested variable fields {sorted(variable)}."
            )

    @staticmethod
    def _merge_columnar_request(
        requests: dict[str, tuple[tuple[str, ...] | None, bool]],
        key: str,
        fields: set[str],
    ) -> None:
        """Merge fields into a required columnar product projection.

        Parameters
        ----------
        requests : dict
            Projection requests accumulated for the analysis script.
        key : str
            Product name to add or update.
        fields : set[str]
            Fields required from the product. Existing fields are preserved.
        """
        if key in requests:
            current, _ = requests[key]
            assert current is not None
            fields.update(current)
        requests[key] = (tuple(sorted(fields)), True)
