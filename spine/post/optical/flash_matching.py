"""Post-processor in charge of finding matches between charge and light.
"""

import numpy as np
from warnings import warn

from spine.post.base import PostBase

from spine.data.out.base import OutBase

from spine.utils.geo import Geometry

from .barycenter import BarycenterFlashMatcher
from .likelihood import LikelihoodFlashMatcher

__all__ = ['FlashMatchProcessor']


class FlashMatchProcessor(PostBase):
    """Associates TPC interactions with optical flashes."""

    # Name of the post-processor (as specified in the configuration)
    name = 'flash_match'

    # Alternative allowed names of the post-processor
    aliases = ('run_flash_matching',)

    def __init__(self, flash_key, volume, ref_volume_id=None,
                 method='likelihood', detector=None, geometry_file=None,
                 run_mode='reco', truth_point_mode='points',
                 truth_dep_mode='depositions', parent_path=None, **kwargs):
        """Initialize the flash matching algorithm.

        Parameters
        ----------
        flash_key : str
            Flash data product name. In most cases, this is unambiguous, unless
            there are multiple types of segregated optical detectors
        volume : str
            Physical volume corresponding to each flash ('module' or 'tpc')
        ref_volume_id : str, optional
            If specified, the flash matching expects all interactions/flashes
            to live into a specific optical volume. Must shift everything.
        method : str, default 'likelihood'
            Flash matching method (one of 'likelihood' or 'barycenter')
        detector : str, optional
            Detector to get the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        parent_path : str, optional
            Path to the parent directory of the main analysis configuration.
            This allows for the use of relative paths in the post-processors.
        **kwargs : dict
            Keyword arguments to pass to specific flash matching algorithms
        """
        # Initialize the parent class
        super().__init__(
                'interaction', run_mode, truth_point_mode, truth_dep_mode,
                parent_path=parent_path)

        # Make sure the flash data product is available, store
        self.flash_key = flash_key
        self.update_keys({flash_key: True})

        # Initialize the detector geometry
        self.geo = Geometry(detector, geometry_file)

        # Get the volume within which each flash is confined
        assert volume in ['tpc', 'module'], (
                "The `volume` must be one of 'tpc' or 'module'.")
        self.volume = volume
        self.ref_volume_id = ref_volume_id

        # Initialize the flash matching algorithm
        if method == 'barycenter':
            self.matcher = BarycenterFlashMatcher(**kwargs)

        elif method == 'likelihood':
            self.matcher = LikelihoodFlashMatcher(
                    detector=detector, parent_path=self.parent_path, **kwargs)

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
        in-place by filling the following attributes
        - interaction.is_flash_matched: (bool)
               Indicator for whether the given interaction has a flash match
        - interaction.flash_ids: np.ndarray
               The flash IDs in the flash list
        - interaction.flash_volume_ids: np.ndarray
               The flash optical volume IDs in the flash list
        - interaction.flash_times: np.ndarray
               The flash time(s) in microseconds
        - interaction.flash_total_pe: float
               Total number of PEs associated with the matched flash(es)
        - interaction.flash_hypo_pe: float, optional
               Total number of PEss associated with the hypothesis flash
        """
        # Fetch the optical volume each flash belongs to
        flashes = data[self.flash_key]
        volume_ids = np.asarray([f.volume_id for f in flashes])
        
        # Loop over the optical volumes, run flash matching
        for k in self.interaction_keys:
            # Fetch interactions, nothing to do if there are not any
            interactions = data[k]
            if not len(interactions):
                continue

            # Make sure the interaction coordinates are expressed in cm
            self.check_units(interactions[0])

            # Clear previous flash matching information
            for inter in interactions:
                inter.flash_ids = []
                inter.flash_volume_ids = []
                inter.flash_times = []
                inter.flash_scores = []
                if inter.is_flash_matched:
                    inter.is_flash_matched = False
                    inter.flash_total_pe = -1.
                    inter.flash_hypo_pe = -1.

            # Loop over the optical volumes
            for volume_id in np.unique(volume_ids):
                # Get the list of flashes associated with this optical volume
                flashes_v = []
                for flash in flashes:
                    # Skip if the flash is not associated with the right volume
                    if flash.volume_id != volume_id:
                        continue

                    # Reshape the flash based on geometry
                    pe_per_ch = np.zeros(
                            self.geo.optical.num_detectors_per_volume,
                            dtype=flash.pe_per_ch.dtype)
                    if self.ref_volume_id is not None:
                        lower = flash.volume_id*len(pe_per_ch)
                        upper = (flash.volume_id + 1)*len(pe_per_ch)
                        pe_per_ch = flash.pe_per_ch[lower:upper]
                    else:
                        pe_per_ch[:len(flash.pe_per_ch)] = flash.pe_per_ch

                    flash.pe_per_ch = pe_per_ch
                    flashes_v.append(flash)

                # Crop interactions to only include depositions in the optical volume
                interactions_v = []
                for inter in interactions:
                    # Fetch the points in the current optical volume
                    sources = self.get_sources(inter)
                    if self.volume == 'module':
                        index = self.geo.get_volume_index(sources, volume_id)

                    elif self.volume == 'tpc':
                        num_cpm = self.geo.tpc.num_chambers_per_module
                        module_id, tpc_id = volume_id//num_cpm, volume_id%num_cpm
                        index = self.geo.get_volume_index(sources, module_id, tpc_id)

                    # If there are no points in this volume, proceed
                    if len(index) == 0:
                        continue

                    # Fetch points and depositions
                    points = self.get_points(inter)[index]
                    depositions = self.get_depositions(inter)[index]
                    if self.ref_volume_id is not None:
                        # If the reference volume is specified, shift positions
                        points = self.geo.translate(
                                points, volume_id, self.ref_volume_id)

                    # Create an interaction which holds positions/depositions
                    inter_v = OutBase(
                            id=inter.id, points=points, depositions=depositions)
                    interactions_v.append(inter_v)

                # Run flash matching
                matches = self.matcher.get_matches(interactions_v, flashes_v)

                # Store flash information
                for inter_v, flash, match in matches:
                    # Get the interaction that matches the cropped version
                    inter = interactions[inter_v.id]

                    # Get the flash hypothesis (if the matcher produces one)
                    hypo_pe, score = -1., -1.
                    if hasattr(match, 'hypothesis'):
                        hypo_pe = float(np.sum(list(match.hypothesis)))
                    if hasattr(match, 'score'):
                        score = float(match.score)

                    # Append
                    inter.flash_ids.append(int(flash.id))
                    inter.flash_volume_ids.append(int(flash.volume_id))
                    inter.flash_times.append(float(flash.time))
                    inter.flash_scores.append(score)
                    if inter.is_flash_matched:
                        inter.flash_total_pe += float(flash.total_pe)
                        inter.flash_hypo_pe += hypo_pe

                    else:
                        inter.is_flash_matched = True
                        inter.flash_total_pe = float(flash.total_pe)
                        inter.flash_hypo_pe = hypo_pe

            # Cast list attributes to numpy arrays
            for inter in interactions:
                inter.flash_ids = np.asarray(inter.flash_ids, dtype=np.int32)
                inter.flash_volume_ids = np.asarray(inter.flash_volume_ids, dtype=np.int32)
                inter.flash_times = np.asarray(inter.flash_times, dtype=np.float32)
                inter.flash_scores = np.asarray(inter.flash_scores, dtype=np.float32)
