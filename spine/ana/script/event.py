"""Analysis script used to store basic event information into CSV files."""

from collections import defaultdict

import numpy as np
import copy

from spine.ana.base import AnaBase

from spine.utils.globals import SHOWR_SHP, ELEC_PID

__all__ = ['EventAna']


class EventAna(AnaBase):
    """Class which saves basic event information (and their matches)."""
    name = 'event'
    keys = {'index': True, 'reco_particles': True,
            'truth_particles': True, 'run_info': False}

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
        self.initialize_writer('events')

    def process(self, data):
        """Store basic event information for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products containing particle representations
        """
        # Extract basic information to store in every row
        # TODO add file index + index within the file?
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
