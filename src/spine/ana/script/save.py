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
    """Class which simply saves reconstructed objects (and their matches)."""

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
            Whether to write reconstructed, truth, or both object collections.
        match_mode : str, default 'both'
            If reconstructed and truth are available, specifies which matching
            direction(s) should be saved to the log file.
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
        if match_mode is not None and run_mode != "both":
            raise ValueError(
                "When storing matches, you must load both reco and truth "
                f"objects, i.e. set `run_mode` to `both`. Got {run_mode}."
            )

        # Store default objects as a dictionary
        self.default_objs = dict(self._default_objs)

        # Store the list of attributes to store for each object type
        attrs: dict[str, list[str] | None] = {
            "fragments": list(fragment) if fragment is not None else None,
            "particles": list(particle) if particle is not None else None,
            "interactions": list(interaction) if interaction is not None else None,
        }
        self.attrs: dict[str, list[str] | None]
        if run_mode != "both":
            # If there is only one object type, the keys specified are unique
            self.attrs = {f"{run_mode}_{k}": v for k, v in attrs.items()}

        else:
            # If there are multiple object types, down select to attributes
            # each declination of the object knows, as long as either one does
            self.attrs = {}
            for obj_t, attrs_t in attrs.items():
                # Create a list specific to each object declination
                leftover = set(attrs_t) if attrs_t is not None else None
                for prefix in ["reco", "truth"]:
                    key = f"{prefix}_{obj_t}"
                    if attrs_t is not None:
                        all_keys = self.default_objs[key].as_dict().keys()
                        attrs_key = sorted(set(attrs_t) & set(all_keys))
                        self.attrs[key] = attrs_key
                        if leftover is not None:
                            leftover -= set(attrs_key)

                    else:
                        self.attrs[key] = attrs_t

                # Check that there are no leftover keys
                if leftover is not None and len(leftover) > 0:
                    raise ValueError(
                        "The following keys were not found in either the reco "
                        f"or the truth {obj_t} : {leftover}"
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
                    if prefix == "reco" and match_mode != "truth_to_reco":
                        keys[f"{obj_name}_matches_r2t"] = False
                        keys[f"{obj_name}_matches_r2t_overlap"] = False
                    if prefix == "truth" and match_mode != "reco_to_truth":
                        keys[f"{obj_name}_matches_t2r"] = False
                        keys[f"{obj_name}_matches_t2r_overlap"] = False

        self.update_keys(keys)

        # Initialize one CSV writer per object type
        for key in self.obj_keys:
            self.initialize_writer(key)

        if len(self.writers) == 0:
            raise ValueError("Must request to save something.")

    def columnar_requests(
        self,
    ) -> dict[str, tuple[tuple[str, ...] | None, bool]]:
        """Request fixed object columns needed by the columnar save path."""
        requests: dict[str, tuple[tuple[str, ...] | None, bool]] = {
            "run_info": (("run", "subrun", "event"), False)
        }
        other_prefix = {"reco": "truth", "truth": "reco"}
        for key in self.obj_keys:
            attrs = self.attrs[key]
            if attrs is None:
                raise ValueError(
                    "Columnar save requires an explicit attribute list for " f"`{key}`."
                )
            default_obj = self.default_objs[key]
            variable = set(attrs).intersection(default_obj._var_length_attrs)
            if variable:
                raise ValueError(
                    "Columnar save currently supports fixed attributes only; "
                    f"`{key}` requested variable fields {sorted(variable)}."
                )

            fields = set(attrs)
            prefix, obj_type = key.split("_")
            other = other_prefix[prefix]
            if (
                self.match_mode is not None
                and self.match_mode != f"{other}_to_{prefix}"
            ):
                fields.update(("best_match_id", "best_match_overlap"))
            requests[key] = (tuple(sorted(fields)), True)

        return requests

    @staticmethod
    def _event_rows(offsets: np.ndarray) -> np.ndarray:
        """Map each flattened object row to its chunk-local event."""
        return np.repeat(
            np.arange(len(offsets) - 1, dtype=np.int64),
            np.diff(offsets),
        )

    @staticmethod
    def _expand_columnar_attrs(
        product: Mapping[str, Any],
        attrs: Sequence[str],
        default_obj: Any,
    ) -> dict[str, np.ndarray]:
        """Expand scalar and fixed-width columns using scalar_dict names."""
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
            labels = (
                default_obj._axes
                if attr in default_obj._pos_attrs + default_obj._vec_attrs
                else tuple(str(i) for i in range(values.shape[1]))
            )
            for i, label in enumerate(labels):
                result[f"{attr}_{label}"] = values[:, i]

        return result

    @staticmethod
    def _repeat_event_column(values: Any, counts: np.ndarray) -> np.ndarray:
        """Broadcast one event-level value onto every object row."""
        return np.repeat(np.asarray(values), counts, axis=0)

    def _columnar_base(
        self,
        data: Mapping[str, Any],
        counts: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build event metadata columns repeated for one object collection."""
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

    def process_columnar(self, data: Mapping[str, Any]) -> None:
        """Write projected object columns and best-match joins in bulk."""
        other_prefix = {"reco": "truth", "truth": "reco"}
        for key in self.obj_keys:
            prefix, obj_type = key.split("_")
            other = other_prefix[prefix]
            attrs = self.attrs[key]
            assert attrs is not None
            product = data[key]
            offsets = np.asarray(product["event_offsets"], dtype=np.int64)
            counts = np.diff(offsets)
            row_dict = self._columnar_base(data, counts)
            source_columns = self._expand_columnar_attrs(
                product, attrs, self.default_objs[key]
            )

            if self.match_mode is None or self.match_mode == f"{other}_to_{prefix}":
                row_dict.update(source_columns)
                self.writers[key].append_columns(row_dict)
                continue

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
                default = target_defaults[name]
                joined = np.full(len(match_ids), default, dtype=values.dtype)
                joined[valid] = values[target_rows[valid]]
                row_dict[f"{other}_{name}"] = joined
            row_dict["match_overlap"] = np.asarray(product["best_match_overlap"])
            self.writers[key].append_columns(row_dict)

    def process(self, data: Mapping[str, Any]) -> None:
        """Store the information from one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the keys to store
        other_prefix = {"reco": "truth", "truth": "reco"}
        for key in self.obj_keys:
            # Dispatch
            prefix, obj_type = key.split("_")
            other = other_prefix[prefix]
            attrs = self.attrs[key]
            lengths = self.lengths
            if self.match_mode is None or self.match_mode == f"{other}_to_{prefix}":
                # If there is no matches, save objects by themselves
                for obj in data[key]:
                    self.append(key, **obj.scalar_dict(attrs, lengths))

            else:
                # If there are matches, combine the objects with their best
                # match on a single row
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

                for idx, (obj_i, obj_j) in enumerate(pairs):
                    src_dict = obj_i.scalar_dict(attrs, lengths)
                    if obj_j is not None:
                        tgt_dict = obj_j.scalar_dict(attrs_other, lengths)
                    else:
                        default_obj = self.default_objs[f"{other}_{obj_type}"]
                        tgt_dict = default_obj.scalar_dict(attrs_other, lengths)

                    src_dict = {f"{prefix}_{k}": v for k, v in src_dict.items()}
                    tgt_dict = {f"{other}_{k}": v for k, v in tgt_dict.items()}
                    overlap = overlaps[idx]

                    row_dict = {**src_dict, **tgt_dict}
                    row_dict.update({"match_overlap": overlap})
                    self.append(key, **row_dict)
