"""
Post-processor in charge of filling the hypothesis into the data product.
"""

from spine.post.base import PostBase
from spine.utils.geo import Geometry
from spine.data.out.base import OutBase
import numpy as np

from .hypothesis import Hypothesis

__all__ = ['FillFlashHypothesisProcessor']

class FillFlashHypothesisProcessor(PostBase):
    """Fills the hypothesis into the data product."""

    # Name of the post-processor (as specified in the configuration)
    name = 'fill_hypothesis'

    # Alternative allowed names of the post-processor
    aliases = ('fill_hypo',)

    def __init__(self, flash_key, volume, ref_volume_id=None, detector=None, parent_path=None,
                 geometry_file=None, run_mode='reco', truth_point_mode='points',
                 truth_dep_mode='depositions', hypothesis_key='flash_hypos', **kwargs):
        """Initialize the fill hypothesis processor.

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
        detector : str, optional
            Detector to get the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        parent_path : str, optional
            Path to the parent directory of the main analysis configuration.
            This allows for the use of relative paths in the post-processors.
        hypothesis_key : str, default 'flash_hypo'
            Key to use for the hypothesis data product
        """
        # Initialize the parent class
        super().__init__(
                'interaction', run_mode, truth_point_mode, truth_dep_mode,
                parent_path=parent_path)
        
        # Initialize the hypothesis key
        self.flash_key = flash_key
        self.hypothesis_key = hypothesis_key

        # Initialize the detector geometry
        self.geo = Geometry(detector, geometry_file)

        # Get the volume within which each flash is confined
        assert volume in ('tpc', 'module'), (
                "The `volume` must be one of 'tpc' or 'module'.")
        self.volume = volume
        self.ref_volume_id = ref_volume_id

        # Initialize the hypothesis algorithm
        self.hypothesis = Hypothesis(detector=detector, parent_path=self.parent_path, **kwargs)

        #Assert that we have flashes and interactions
        self.update_keys({self.flash_key: True})

    def match_hypothesis(self, hypothesis_v, flash_info_v):
        """Match the hypothesis to the flash. The hypothesis has the interaction ID,
        whereas the interaction has the flash ID. So we will match the hypothesis interaction ID to 
        the interaction that's matched to the flash, then set the flash ID to the hypothesis ID.

        Parameters
        ----------
        hypothesis_v : list
            List of hypothesis objects
        flash_info_v : list
            List of tuples of interaction ID, flash IDs, and flash volumes

        Returns
        -------
        None
            Modifies the hypothesis objects in place
        """

        #Make a dictionary of the flash IDs and volumes
        int_id_dict = {ii[0]: (ii[1],ii[2]) for ii in flash_info_v}
        
        #Modify the hypothesis objects
        for hypo in hypothesis_v:
            #If the interaction ID is in the dictionary, and the hypothesis volume matches the flash volume, set the flash IDs
            if hypo.interaction_id in int_id_dict and int_id_dict[hypo.interaction_id][1] == hypo.volume_id:
                hypo.id = int_id_dict[hypo.interaction_id][0]
        
        
    def process(self, data):
        """Fills the hypothesis into the data product.

        Parameters
        ----------
        data : dict
            Data product to fill the hypothesis into
        """

        volume_ids = np.asarray([f.volume_id for f in data[self.flash_key]])

        #Loop over optical volumes, make the hypotheses in each
        for k in self.interaction_keys:
            # Fetch interactions, nothing to do if there are not any
            interactions = data[k]
            if not len(interactions):
                continue
            
            # Make sure the interaction coordinates are expressed in cm
            self.check_units(interactions[0])

            # Loop over the optical volumes
            id_offset = 0
            hypothesis_v = []
            for volume_id in np.unique(volume_ids):
                # Crop interactions to only include depositions in the optical volume
                interactions_v = []
                flash_info_v = []
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
                    for fid,fvol in zip(inter.flash_ids,inter.flash_volume_ids):
                        if fvol == volume_id:
                            flash_info_v.append((inter.id,fid,fvol)) #needed for matching
                # Make the hypothesis
                _hypo_v = self.hypothesis.make_hypothesis_list(interactions_v, id_offset, volume_id)
                hypothesis_v.extend(_hypo_v)
                id_offset += len(_hypo_v) #increment the offset for the next volume

                # Match the matched hypothesis to the flash if provided in interactions_v
                self.match_hypothesis(hypothesis_v, flash_info_v)
                
        # Fill the hypothesis into the data product
        return {self.hypothesis_key: hypothesis_v}