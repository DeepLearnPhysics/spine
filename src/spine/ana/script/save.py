"""Analysis script used to store the reconstruction output to CSV files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
                        keys[f"{obj_name}_matches_r2t"] = True
                        keys[f"{obj_name}_matches_r2t_overlap"] = True
                    if prefix == "truth" and match_mode != "reco_to_truth":
                        keys[f"{obj_name}_matches_t2r"] = True
                        keys[f"{obj_name}_matches_t2r_overlap"] = True

        self.update_keys(keys)

        # Initialize one CSV writer per object type
        for key in self.obj_keys:
            self.initialize_writer(key)

        if len(self.writers) == 0:
            raise ValueError("Must request to save something.")

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
                for idx, (obj_i, obj_j) in enumerate(data[match_key]):
                    src_dict = obj_i.scalar_dict(attrs, lengths)
                    if obj_j is not None:
                        tgt_dict = obj_j.scalar_dict(attrs_other, lengths)
                    else:
                        default_obj = self.default_objs[f"{other}_{obj_type}"]
                        tgt_dict = default_obj.scalar_dict(attrs_other, lengths)

                    src_dict = {f"{prefix}_{k}": v for k, v in src_dict.items()}
                    tgt_dict = {f"{other}_{k}": v for k, v in tgt_dict.items()}
                    overlap = data[f"{match_key}_overlap"][idx]

                    row_dict = {**src_dict, **tgt_dict}
                    row_dict.update({"match_overlap": overlap})
                    self.append(key, **row_dict)
