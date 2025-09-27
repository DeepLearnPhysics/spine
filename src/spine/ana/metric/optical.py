"""Analysis script used to evaluate the semantic segmentation accuracy."""

import numpy as np

from spine.ana.base import AnaBase
from spine.data.out import RecoInteraction, TruthInteraction

__all__ = ["FlashMatchingAna"]


class FlashMatchingAna(AnaBase):
    """Class which computes and stores the necessary data to build a
    semantic segmentation confusion matrix.
    """

    # Name of the analysis script (as specified in the configuration)
    name = "flash_match_eval"

    # Valid match modes
    _match_modes = ("reco_to_truth", "truth_to_reco", "both", "all")

    # Default object types when a match is not found
    _default_objs = (
        ("reco_interactions", RecoInteraction()),
        ("truth_interactions", TruthInteraction()),
    )

    def __init__(
        self,
        time_window=None,
        neutrino_only=True,
        max_num_flashes=1,
        match_mode="both",
        **kwargs,
    ):
        """Initialize the analysis script.

        Parameters
        ----------
        time_window : List[float], optional
            Time window (in ns) for which interactions must have matched flash
        neutrino_only : bool, default False
            If `True`, only check if neutrino in-time activity is matched for
            the efficiency measurement (as opposed to any in-time activity)
        max_num_flashes : int
            Maximum number of flash matches to store
        match_mode : str, default 'both'
            If reconstructed and truth are available, specified which matching
            direction(s) should be saved to the log file.
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__("interaction", "both", **kwargs)

        # Store basic parameters
        self.time_window = time_window
        self.neutrino_only = neutrino_only

        # Store default objects as a dictionary
        self.default_objs = dict(self._default_objs)

        # Store the matching mode
        self.match_mode = match_mode
        assert match_mode in self._match_modes, (
            f"Invalid matching mode: {self.match_mode}. Must be one "
            f"of {self._match_modes}."
        )

        # Make sure the matches are loaded, initialize the output files
        keys = {}
        for prefix in self.prefixes:
            if prefix == "reco" and match_mode != "truth_to_reco":
                keys["interaction_matches_r2t"] = True
                keys["interaction_matches_r2t_overlap"] = True
                self.initialize_writer("reco")
            if prefix == "truth" and match_mode != "reco_to_truth":
                keys["interaction_matches_t2r"] = True
                keys["interaction_matches_t2r_overlap"] = True
                self.initialize_writer("truth")

        self.update_keys(keys)

        # List the interaction attributes to be stored
        nu_attrs = ("energy_init",) if neutrino_only else ()
        flash_attrs = (
            "is_flash_matched",
            "flash_ids",
            "flash_times",
            "flash_scores",
            "flash_total_pe",
            "flash_hypo_pe",
        )
        flash_lengths = {
            k: max_num_flashes for k in ["flash_ids", "flash_times", "flash_scores"]
        }

        self.reco_attrs = ("id", "size", "is_contained", "topology", *flash_attrs)
        self.truth_attrs = (
            "id",
            "size",
            "is_contained",
            "nu_id",
            "t",
            "topology",
            *nu_attrs,
        )

        self.reco_lengths = flash_lengths
        self.truth_lengths = None

    def process(self, data):
        """Store the flash matching metrics for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the matching directions
        prefixes = {"reco": "truth", "truth": "reco"}
        for source, target in prefixes.items():
            # Loop over the match pairs
            src_attrs = getattr(self, f"{source}_attrs")
            src_lengths = getattr(self, f"{source}_lengths")
            tgt_attrs = getattr(self, f"{target}_attrs")
            tgt_lengths = getattr(self, f"{target}_lengths")
            match_suffix = f"{source[0]}2{target[0]}"
            match_key = f"interaction_matches_{match_suffix}"
            for idx, (obj_i, obj_j) in enumerate(data[match_key]):
                # Check that the source interaction is of interest
                if obj_i.is_truth:
                    # If the source object is a true interaction, check if it
                    # should be matched or not (in time or not)
                    if self.time_window is not None and (
                        obj_i.t < self.time_window[0] or obj_i.t > self.time_window[1]
                    ):
                        continue

                    # If requested, check that the in-time activity is a neutrino
                    if self.neutrino_only and obj_i.nu_id < 0:
                        continue

                else:
                    # If the source object is a reco interaction, check if it
                    # is matched to a flash or not
                    if not obj_i.is_flash_matched:
                        continue

                # Store information about the corresponding reco interaction
                # and the flash associated with it (if any)
                src_dict = obj_i.scalar_dict(src_attrs, src_lengths)
                if obj_j is not None:
                    tgt_dict = obj_j.scalar_dict(tgt_attrs, tgt_lengths)
                else:
                    default_obj = self.default_objs[f"{target}_interactions"]
                    tgt_dict = default_obj.scalar_dict(tgt_attrs, tgt_lengths)

                src_dict = {f"{source}_{k}": v for k, v in src_dict.items()}
                tgt_dict = {f"{target}_{k}": v for k, v in tgt_dict.items()}

                # Get the match quality
                overlap = data[f"{match_key}_overlap"][idx]

                # Build row dictionary and store
                row_dict = {**src_dict, **tgt_dict}
                row_dict.update({"match_overlap": overlap})

                self.append(source, **row_dict)
