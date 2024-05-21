"""Analysis script used to store basic event information into CSV files."""

from collections import defaultdict

import numpy as np
import copy

from mlreco.ana.base import AnaBase

from mlreco.utils.globals import SHOWR_SHP

__all__ = ['EventAna']


class EventAna(AnaBase):
    """Class which saves basic event information (and their matches)."""
    name = 'event'
    req_keys = ['index', 'particles', 'truth_particles']
    opt_keys = ['run_info']

    def __init__(self, run_mode='both', append=False):
        """Initialize the CSV event logging class."""
        # Initialize the parent class
        super().__init__(run_mode=run_mode, append=append)

        # Initialize the output log file
        self.initialzie('events')

    def process(self, data):
        """Store basic event information for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products containing particle representations
        """
        # Extract basic information to store in every row
        # TODO add file index + index within the file?
        # TODO remove all assert checks below and add them to required keys!!
        row_dict = {'index': data['index']}
        if 'run_info' in data:
            row_dict.update(**data['run_info'].scalar_dict())
        else:
            warn("`run_info` is missing; will not be included in CSV file.")

        # Get the basic event-level information
        # TODO: this should be using run_mode
        reco_p, truth_p = data['reco_particles'], data['truth_particles']
        cnts = defaultdict(int)
        for out_type in ['reco', 'truth']:
            particles = data[f'{out_type}_particles']
            pre = f'num_{out_type}'
            for p in particles:
                cnts[f'{pre}_particles'] += 1
                cnts[f'{pre}_primaries'] += p.is_primary
                cnts[f'{pre}_voxels'] += p.size
                cnts[f'{pre}_showers'] += p.shape == SHOWR_SHP
                cnts[f'{pre}_primary_showers'] += (
                        p.shape == SHOWR_SHP and p.is_primary)
                cnts[f'{pre}_electron_showers'] += p.pid == ELEC_PID

        row_dict.update(**cnts)
        self.writers['events'].append(row_dict)
