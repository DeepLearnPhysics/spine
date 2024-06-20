import numpy as np
from warnings import warn

from spine.post.base import PostBase

from .barycenter import BarycenterFlashMatcher
from .likelihood import LikelihoodFlashMatcher

__all__ = ['FlashMatchProcessor']


class FlashMatchProcessor(PostBase):
    """Associates TPC interactions with optical flashes."""
    name = 'flash_match'
    aliases = ['run_flash_matching']

    def __init__(self, opflash_map, method='likelihood',
                 run_mode='reco', truth_point_mode='points', **kwargs):
        """Initialize the flash matching algorithm.

        Parameters
        ----------
        method : str, default 'likelihood'
            Flash matching method (one of 'likelihood' or 'barycenter')
        opflash_map : dict
            Maps a flash data product key in the data ditctionary to an
            optical volume in the detector
        **kwargs : dict
            Keyword arguments to pass to specific flash matching algorithms
        """
        # Initialize the parent class
        super().__init__('interaction', run_mode, truth_point_mode)

        # If there is no map from flash data product to volume ID, throw
        self.opflash_map = opflash_map
        for key in self.opflash_map:
            self.keys[key] = True

        # Initialize the flash matching algorithm
        if method == 'barycenter':
            self.matcher = BarycenterFlashMatcher(**kwargs)

        elif method == 'likelihood':
            self.matcher = LikelihoodFlashMatcher(
                    **kwargs, parent_path=self.parent_path)

        else:
            raise ValueError(f'Flash matching method not recognized: {method}')

    def process(self, data):
        """Find [interaction, flash] pairs.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Notes
        -----
        This post-processor modifies the list of `interaction` objectss
        in-place by adding the following attributes:
        - interaction.is_flash_matched: (bool)
               Indicator for whether the given interaction has a flash match
        - interaction.flash_time: float
               The flash time in microseconds
        - interaction.flash_total_pe: float
        - interaction.flash_hypo_pe: float
        """
        # Loop over the keys to match
        for k in self.inter_keys:
            # Fetch interactions, nothing to do if there are not any
            interactions = data[k]
            if not len(interactions):
                continue

            # Make sure the interaction coordinates are expressed in cm
            self.check_units(interactions[0])

            # Clear previous flash matching information
            for inter in interactions:
                if inter.is_flash_matched:
                    inter.is_flash_matched = False
                    inter.flash_id = -1
                    inter.flash_time = -np.inf
                    inter.flash_total_pe = -1.0
                    inter.flash_hypo_pe = -1.0

            # Loop over flash keys
            for key, volume_id in self.opflash_map.items():
                # Get the list of flashes associated with that key
                opflashes = data_dict[key]

                # Get the list of interactions that share the same volume
                # TODO: this only works for interactions coming from a single
                # volume. Must fix this.
                ints = [inter for inter in interactions if inter.volume_id == volume_id]

                # Run flash matching
                matches = self.matcher.get_matches(ints, opflashes)

                # Store flash information
                for inter, flash, match in matches:
                    inter.is_flash_matched = True
                    inter.flash_id = int(flash.id())
                    inter.flash_time = float(flash.time())
                    inter.flash_total_pe = float(flash.TotalPE())
                    if hasattr(match, 'hypothesis'):
                        inter.flash_hypo_pe = float(np.sum(match.hypothesis))
