"""Analysis script used to store basic event information into CSV files."""

import copy
from collections import defaultdict

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.globals import ELEC_PID, SHOWR_SHP

__all__ = ["EventAna"]


class EventAna(AnaBase):
    """Class which saves basic event information (and their matches)."""

    # Name of the analysis script (as specified in the configuration)
    name = "event"

    # Set of data keys needed for this analysis script to operate
    _keys = (("reco_particles", True), ("truth_particles", True))

    def __init__(self, **kwargs):
        """Initialize the CSV event logging class.

        Parameters
        ----------
        **kwargs : dict, optional
            Parameters to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Initialize the output log file
        self.initialize_writer("events")

    def process(self, data):
        """Store basic event information for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products containing particle representations
        """
        # Get the basic event-level information
        # TODO: this should be using run_mode
        reco_p, truth_p = data["reco_particles"], data["truth_particles"]
        cnts = defaultdict(int)
        for out_type in ["reco", "truth"]:
            particles = data[f"{out_type}_particles"]
            pre = f"num_{out_type}"
            for p in particles:
                cnts[f"{pre}_particles"] += 1
                cnts[f"{pre}_primaries"] += p.is_primary
                cnts[f"{pre}_voxels"] += p.size
                cnts[f"{pre}_showers"] += p.shape == SHOWR_SHP
                cnts[f"{pre}_primary_showers"] += p.shape == SHOWR_SHP and p.is_primary
                cnts[f"{pre}_electron_showers"] += p.pid == ELEC_PID

        row_dict.update(**self.base_dict, **cnts)
        self.writers["events"].append(row_dict)
