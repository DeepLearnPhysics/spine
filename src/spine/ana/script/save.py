"""Analysis script used to store the reconstruction output to CSV files."""

from warnings import warn

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
        obj_type,
        fragment=None,
        particle=None,
        interaction=None,
        lengths=None,
        run_mode="both",
        match_mode="both",
        **kwargs,
    ):
        """Initialize the CSV logging class.

        If any of the `fragments`, `particles` or `interactions` are specified
        as lists of strings, it will be used to restrict the list of
        object attributes which get stored to CSV.

        Parameters
        ----------
        obj_type : Union[str, List[str]], default ['particle', 'interaction']
            Objects to build files from
        fragment : List[str], optional
            List of fragment attributes to store
        particle : List[str], optional
            List of particle attributes to store
        interaction : List[str], optional
            List of interaction attributes to store
        lengths : Dict[str, int], optional
            Lengths to use for variable-length object attributes
        match_mode : str, default 'both'
            If reconstructed and truth are available, specified which matching
            direction(s) should be saved to the log file.
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(obj_type=obj_type, run_mode=run_mode, **kwargs)

        # Store the matching mode
        self.match_mode = match_mode
        assert match_mode in self._match_modes, (
            f"Invalid matching mode: {self.match_mode}. Must be one "
            f"of {self._match_modes}."
        )
        assert match_mode is None or run_mode == "both", (
            "When storing matches, you must load both reco and truth "
            f"objects, i.e. set `run_mode` to `True`. Got {run_mode}."
        )

        # Store default objects as a dictionary
        self.default_objs = dict(self._default_objs)

        # Store the list of attributes to store for each object type
        attrs = {
            "fragments": fragment,
            "particles": particle,
            "interactions": interaction,
        }
        if run_mode != "both":
            # If there is only one object type, the keys specified are unique
            self.attrs = {f"{run_mode}_{k}": v for k, v in attrs.items()}

        else:
            # If there are multiple object types, down select to attributes
            # each declination of the object knows, as long as either one does
            self.attrs = {}
            for obj_t in attrs.keys():
                # Create a list speicific to each object declination
                leftover = set(attrs[obj_t]) if attrs[obj_t] is not None else None
                for run_mode in ["reco", "truth"]:
                    key = f"{run_mode}_{obj_t}"
                    if attrs[obj_t] is not None:
                        all_keys = self.default_objs[key].as_dict().keys()
                        self.attrs[key] = set(attrs[obj_t]) & set(all_keys)
                        leftover -= leftover & self.attrs[key]

                    else:
                        self.attrs[key] = attrs[obj_t]

                # Check that there are no leftover keys
                assert leftover is None or len(leftover) == 0, (
                    "The following keys were not found in either the reco "
                    f"or the truth {obj_t} : {leftover}"
                )

        # Store the list of variable-length array lengths
        self.lengths = lengths

        # Add the necessary keys associated with matching, if needed
        keys = {}
        if match_mode is not None:
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

        assert len(self.writers), "Must request to save something."

    def process(self, data):
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
                for i, obj in enumerate(data[key]):
                    self.append(key, **obj.scalar_dict(attrs, lengths))

            else:
                # If there are matches, combine the objects with their best
                # match on a single row
                match_suffix = f"{prefix[0]}2{other[0]}"
                match_key = f"{obj_type[:-1]}_matches_{match_suffix}"
                attrs_other = self.attrs[f"{other}_{obj_type}"]
                lengths_other = self.lengths  # TODO
                for idx, (obj_i, obj_j) in enumerate(data[match_key]):
                    src_dict = obj_i.scalar_dict(attrs, lengths)
                    if obj_j is not None:
                        tgt_dict = obj_j.scalar_dict(attrs_other, lengths_other)
                    else:
                        default_obj = self.default_objs[f"{other}_{obj_type}"]
                        tgt_dict = default_obj.scalar_dict(attrs_other, lengths_other)

                    src_dict = {f"{prefix}_{k}": v for k, v in src_dict.items()}
                    tgt_dict = {f"{other}_{k}": v for k, v in tgt_dict.items()}
                    overlap = data[f"{match_key}_overlap"][idx]

                    row_dict = {**src_dict, **tgt_dict}
                    row_dict.update({"match_overlap": overlap})
                    self.append(key, **row_dict)
