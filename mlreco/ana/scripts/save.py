"""Analysis script used to store the reconstruction output to CSV files."""

from warnings import warn

from mlreco.ana.base import AnaBase

__all__ = ['SaveAna']


class SaveAna(AnaBase):
    """Class which simply saves reconstructed objects (and their matches)."""
    name = 'save'
    req_keys = ['index']
    opt_keys = ['run_info']

    # Valid match modes
    _match_modes = [None, 'reco_to_truth', 'truth_to_reco', 'both', 'all']

    def __init__(self, fragments=False, particles=False, interactions=False,
                 match_mode='both', run_mode='both', append=False):
        """Initialize the CSV logging class.

        If any of the `fragments`, `particles` or `interactions` are specified
        as lists of strings, it will be used to restrict the list of
        object attributes which get stored to CSV.

        Parameters
        ----------
        fragments : Union[bool, List[str]], default False
            Whether to save fragment objects
        particles : Union[bool, List[str]], default False
            Whether to save fragment objects
        interactions : Union[bool, List[str]], default False
            Whether to save interaction objects
        match_mode : str, default 'both'
            If reconstructed and truth are available, specified which matching
            direction(s) should be saved to the log file.
        """
        # Initialize the parent class
        super().__init__(run_mode=run_mode, append=append)

        # Store the matching mode
        assert self.match_mode in self._match_modes, (
            f"Invalid matching mode: {self.match_mode}. Must be one "
            f"of {self._match_modes}.")

        self.match_mode = match_mode

        # Initialize the CSV writers
        self.save_keys = {'fragments': fragments,
                          'particles': particles,
                          'interactions': interactions}
        for key, save in self.save_keys.items():
            if save:
                if self.match_mode is None:
                    self.req_keys.append(f'reco_{key}')
                    self.initialize(key)
                    continue
                if self.match_mode != 'truth_to_reco':
                    self.req_keys.append(f'reco_{key}')
                    self.initialize(f'{key}_r2t')
                if self.match_mode != 'reco_to_truth':
                    self.req_keys.append(f'truth_{key}')
                    self.initialize(f'{key}_t2r')

        assert len(self.writers), (
                "Must select at least one of 'save_fragments', "
                "'save_particles' or 'save_interactions'.")

    def process(self, data):
        """Store the information from one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data containing object representation
        """
        # Extract basic information to store in every row
        # TODO add file index + index within the file?
        # TODO remove all assert checks below and add them to required keys!!
        base_dict = {'index': data['index']}
        if 'run_info' in data:
            base_dict.update(**data['run_info'].scalar_dict())
        else:
            warn("`run_info` is missing; will not be included in CSV file.")

        # Loop over the keys to store
        for key, save in self.save_keys.items():
            # Skip if not required to save
            if not save:
                continue

            # If there is no matches, save reconstructed objects by themselves
            if self.match_mode is None:
                reco_key = f'reco_{key}'
                assert reco_key in data, (
                        f"Must build representations for {key} to save them.")
                for i, obj in enumerate(data[reco_key]):
                    attrs = save if isinstance(save, list) else None
                    row_dict = {**base_dict, **obj.scalar_dict(attrs)}
                    self.writers[key].append(row_dict)
                continue

            # If there are matches, store relevant information
            if self.match_mode != 'truth_to_reco':
                match_key = f'{key[:-1]}_matches_r2t'
                assert match_key in data, (
                         "Must run the reco to truth matching post-processor "
                        f"for {key} to store them.")
                for k, (obj_i, obj_j) in enumerate(data[match_key]):
                    # TODO: handle attributes not shared between reco/truth
                    # TODO: give default truth particle if there is no match
                    attrs = save if isinstance(save, list) else None
                    reco_dict = {f'reco_{k}':v for k, v in obj_i.scalar_dict(attrs).items()}
                    truth_dict = {f'truth_{k}':v for k, v in obj_j.scalar_dict(attrs).items()}
                    overlap = data[f'{match_key}_overlap'][k]

                    row_dict = {**base_dict, **reco_dict, **truth_dict}
                    row_dict.update({'match_overlap': overlap})
                    self.writers[f'{key}_r2t'].append(row_dict)

            if self.match_mode != 'reco_to_truth':
                match_key = f'{key[:-1]}_matches_t2r'
                assert match_key in data, (
                         "Must run the truth to reco matching post-processor "
                        f"for {key} to store them.")
                for k, (obj_i, obj_j) in data[match_key]:
                    # TODO: handle attributes not shared between reco/truth
                    # TODO: give default reco particle if there is no match
                    attrs = save if isinstance(save, list) else None
                    truth_dict = {f'truth_{k}':v for k, v in obj_i.scalar_dict(attrs).items()}
                    reco_dict = {f'reco_{k}':v for k, v in obj_j.scalar_dict(attrs).items()}
                    overlap = data[f'{match_key}_overlap'][k]

                    row_dict = {**base_dict, **truth_dict, **reco_dict}
                    row_dict.update({'match_overlap': overlap})
                    self.writers[f'{key}_t2r'].append(row_dict)
