import os

import numpy as np
import pandas as pd

from spine.data import Trigger
from spine.post.base import PostBase

__all__ = ["TriggerProcessor"]


class TriggerProcessor(PostBase):
    """Parses trigger information from a CSV file and adds a new trigger_info
    data product to the data dictionary.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "trigger"

    # Alternative allowed names of the post-processor
    aliases = ("parse_trigger",)

    # Set of data keys needed for this post-processor to operate
    _keys = (("run_info", True),)

    def __init__(
        self, file_path, correct_flash_times=True, flash_keys=None, flash_time_corr_us=4
    ):
        """Initialize the trigger information parser.

        Parameters
        ----------
        file_path : str
            Path to the csv file which contains the trigger information
        correct_flash_times : bool, default True
            If True, corrects the flash times using w.r.t. the trigger times
        flash_keys : List[str], optional
            When correcting flash times, provide the list of flash products
            to correct times for
        flash_time_corr_us : float, default 4
            Systematic correction between the trigger time and the flash time
            in us
        """
        # Initialize the parent class
        super().__init__()

        # Load the trigger information
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Cannot find the trigger CSV file: {file_path}")

        self.trigger_dict = pd.read_csv(file_path)

        # Store the parameters
        self.correct_flash_times = correct_flash_times
        if self.correct_flash_times:
            assert flash_keys is not None and len(
                flash_keys
            ), "When correcting flash times, must provide the flash keys."
            self.flash_keys = flash_keys
            self.flash_time_corr_us = flash_time_corr_us

            # Add flash keys to the required data products
            self.update_keys({k: True for k in self.flash_keys})

    def process(self, data):
        """Parse the trigger information of one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the run info, find the corresponding trigger
        run_info = data["run_info"]
        run_id, event_id = run_info.run, run_info.event
        trigger_mask = (self.trigger_dict["run_number"] == run_id) & (
            self.trigger_dict["event_no"] == event_id
        )
        trigger_info = self.trigger_dict[trigger_mask]

        if not len(trigger_info):
            raise KeyError(
                f"Could not find run {run_id}, event {event_id} in the " "trigger file."
            )
        elif len(trigger_info) > 1:
            raise KeyError(
                f"Found more than one trigger associated with {run_id} "
                f"event {event_id} in the trigger file."
            )

        trigger_info = trigger_info.to_dict(orient="records")[0]

        # Build trigger object
        trigger = Trigger(
            time_s=trigger_info["wr_seconds"],
            time_ns=trigger_info["wr_nanoseconds"],
            beam_time_s=trigger_info["beam_seconds"],
            beam_time_ns=trigger_info["beam_nanoseconds"],
            type=trigger_info.get("trigger_type", -1),
        )

        # If requested, loop over the interaction objects, modify flash times
        if self.correct_flash_times:
            # Loop over flashes, correct the timing (flash times are in us)
            offset = (
                (trigger.time_s - trigger.beam_time_s) * 1e6
                + (trigger.time_ns - trigger.beam_time_ns) * 1e-3
                - self.flash_time_corr_us
            )

            for key in self.flash_keys:
                for flash in data[key]:
                    flash.time += offset

        return {"trigger": trigger}
