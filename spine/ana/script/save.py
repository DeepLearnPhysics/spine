"""Analysis script used to store the reconstruction output to CSV files."""

from warnings import warn

from spine.data.out import (
        RecoFragment, TruthFragment, RecoParticle, TruthParticle,
        RecoInteraction, TruthInteraction)

from spine.ana.base import AnaBase

__all__ = ['SaveAna']


class SaveAna(AnaBase):
    """Class which simply saves reconstructed objects (and their matches)."""
    name = 'save'

    # Valid match modes
    _match_modes = [None, 'reco_to_truth', 'truth_to_reco', 'both', 'all']

    # Default object types when a match is not found
    _default_objs = {
            'reco_fragment': RecoFragment(), 'truth_fragment': TruthFragment(),
            'reco_particle': RecoParticle(), 'truth_particle': TruthParticle(),
            'reco_interaction': RecoInteraction(),
            'truth_interaction': TruthInteraction()
    }

    def __init__(self, obj_type, fragment=None, particle=None, interaction=None,
                 run_mode='both', match_mode='both', **kwargs):
        """Initialize the CSV logging class.

        If any of the `fragments`, `particles` or `interactions` are specified
        as lists of strings, it will be used to restrict the list of
        object attributes which get stored to CSV.

        Parameters
        ----------
        obj_type : Union[str, List[str]], default ['particle', 'interaction']
            Objects to build files from
        attrs : List[str]
            List of object attributes to store
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
            f"of {self._match_modes}.")

        # Store the list of attributes to store for each object type
        self.attrs = {'fragment': fragment, 'particle': particle,
                      'interaction': interaction}

        # Add the necessary keys associated with matching, if needed
        if match_mode is not None:
            if isinstance(obj_type, str):
                obj_type = [obj_type]
            for prefix in self.prefixes:
                for obj_name in obj_type:
                    if prefix == 'reco' and match_mode != 'truth_to_reco':
                        self.keys[f'{obj_name}_matches_r2t'] = True
                        self.keys[f'{obj_name}_matches_r2t_overlap'] = True
                    if prefix == 'truth' and match_mode != 'reco_to_truth':
                        self.keys[f'{obj_name}_matches_t2r'] = True
                        self.keys[f'{obj_name}_matches_t2r_overlap'] = True

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
        other_prefix = {'reco': 'truth', 'truth': 'reco'}
        for key in self.obj_keys:
            # Dispatch
            prefix, obj_type = key.split('_')
            obj_type = obj_type[:-1]
            other = other_prefix[prefix]
            attrs = self.attrs[obj_type]
            if (self.match_mode is None or 
                self.match_mode == f'{other}_to_{prefix}'):
                # If there is no matches, save objects by themselves
                for i, obj in enumerate(data[key]):
                    self.append(key, **obj.scalar_dict(attrs))

            else:
                # If there are matches, combine the objects with their best
                # match on a single row
                match_suffix = f'{prefix[0]}2{other[0]}'
                match_key = f'{obj_type}_matches_{match_suffix}'
                for idx, (obj_i, obj_j) in enumerate(data[match_key]):
                    src_dict = obj_i.scalar_dict(attrs)
                    if obj_j is not None:
                        tgt_dict = obj_j.scalar_dict(attrs)
                    else:
                        default_obj = self._default_objs[f'{other}_{obj_type}']
                        tgt_dict = default_obj.scalar_dict(attrs)

                    src_dict = {f'{prefix}_{k}':v for k, v in src_dict.items()}
                    tgt_dict = {f'{other}_{k}':v for k, v in tgt_dict.items()}
                    overlap = data[f'{match_key}_overlap'][idx]

                    row_dict = {**src_dict, **tgt_dict}
                    row_dict.update({'match_overlap': overlap})
                    self.append(key, **row_dict)
