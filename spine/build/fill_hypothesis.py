"""Builder in charge of creating flash hypothesis objects."""

import numpy as np

from spine.build.base import BuilderBase
from spine.utils.geo import Geometry
from spine.post.optical.hypothesis import Hypothesis
from spine.data.optical import Flash
from spine.data.out.base import OutBase

__all__ = ['FlashHypothesisBuilder']

class FlashHypothesisBuilder(BuilderBase):
    """Builds flash hypothesis objects from interaction data."""

    # Builder name (used to create keys like 'reco_flash_hypos')
    name = 'flash_hypo'

    # Types of objects constructed by the builder
    _reco_type = Flash
    _truth_type = Flash

    # Override required keys for building
    _build_reco_keys = (
        ('reco_interactions', True),
    )
    _build_truth_keys = (
        ('truth_interactions', True),
    )

    # Necessary/optional data products to load a reconstructed object
    _load_reco_keys = (
        ('reco_flash_hypos', True),
    )

    # Necessary/optional data products to load a truth object
    _load_truth_keys = (
        ('truth_flash_hypos', True),
    )

    def __init__(self, mode='reco', units='cm', cfg=None, volume=None, ref_volume_id=None, 
                 detector=None, parent_path=None, geometry_file=None,truth_point_mode='points', truth_dep_mode='depositions', 
                 **kwargs):
        """Initialize the flash hypothesis builder.

        Parameters
        ----------
        cfg : str
            Flash matching configuration file path
        volume : str
            Physical volume corresponding to each flash ('module' or 'tpc')
        mode : str, default 'reco'
            Whether to construct reconstructed objects, true objects or both
            (one of 'reco', 'truth', 'both' or 'all')
        units : str, default 'cm'
            Units in which the position arguments of the constructed objects
            should be expressed (one of 'cm' or 'px')
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
        """
        # Initialize the parent class
        super().__init__(mode,units)

        # Initialize the flash hypothesis builder if building the representation
        if cfg is not None: #TODO: Implement a way to initialize fig only when building the representation
            self.initialize(cfg, volume, mode, units, ref_volume_id, detector, parent_path, geometry_file,truth_point_mode, truth_dep_mode, **kwargs)

    def initialize(self, cfg=None, volume=None, mode='reco', units='cm', ref_volume_id=None, 
                 detector=None, parent_path=None, geometry_file=None,truth_point_mode='points', truth_dep_mode='depositions', 
                 **kwargs):
        """Initialize the flash hypothesis builder. Use this only for building the represenation.

        Parameters
        ----------
        cfg : str
            Flash matching configuration file path
        volume : str
            Physical volume corresponding to each flash ('module' or 'tpc')
        mode : str, default 'reco'
            Whether to construct reconstructed objects, true objects or both
            (one of 'reco', 'truth', 'both' or 'all')
        units : str, default 'cm'
            Units in which the position arguments of the constructed objects
            should be expressed (one of 'cm' or 'px')
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
        truth_point_mode : str, optional
            If specified, the flash matching expects all interactions/flashes
            to live into a specific optical volume. Must shift everything.
        truth_dep_mode : str, optional
            Detector to get the geometry from
        cfg : str, optional
            Flash matching configuration file path
        """
        #Store point mode to use for truth objects
        self.truth_point_mode = truth_point_mode
        self.truth_dep_mode = truth_dep_mode
        self.truth_source_mode = truth_point_mode.replace('points', 'sources')

        # Initialize the detector geometry
        self.geo = Geometry(detector, geometry_file)

        # Get the volume within which each flash is confined
        assert volume in ('tpc', 'module'), (
                "The `volume` must be one of 'tpc' or 'module'.")
        self.volume = volume
        self.ref_volume_id = ref_volume_id

        # Initialize the hypothesis algorithm
        self.hypothesis = Hypothesis(cfg,detector=detector, parent_path=parent_path, **kwargs)
        print('Configuring FlashHypothesisBuilder')
        
    def build_reco(self, data):
        """Build flash hypothesis objects from reconstructed interactions.
        
        Parameters
        ----------
        data : dict
            Dictionary containing interactions data
            
        Returns
        -------
        list
            List of flash hypothesis objects
        """
        return self._build_hypotheses(
            data['reco_interactions']
        )
        
    def build_truth(self, data):
        """Build flash hypothesis objects from truth interactions.
        
        Parameters
        ----------
        data : dict
            Dictionary containing truth interactions data
            
        Returns
        -------
        list
            List of flash hypothesis objects
        """
        return self._build_hypotheses(
            data['truth_interactions']
        )
    
    def load_reco(self, data):
        """Load pre-computed reconstructed flash hypotheses.
        
        Parameters
        ----------
        data : dict
            Dictionary containing pre-computed reco hypotheses
            
        Returns
        -------
        list
            List of flash hypothesis objects
        """
        return data['reco_flash_hypos']
    
    def load_truth(self, data):
        """Load pre-computed truth flash hypotheses.
        
        Parameters
        ----------
        data : dict
            Dictionary containing pre-computed truth hypotheses
            
        Returns
        -------
        list
            List of flash hypothesis objects
        """
        return data['truth_flash_hypos']
    
    def get_sources(self, obj):
        """Get a certain pre-defined sources attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects sources are obtained using the
        `truth_source_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
            (N, 2) Object sources
        """
        if not obj.is_truth:
            return obj.sources
        else:
            return getattr(obj, self.truth_source_mode)
    
    def get_points(self, obj):
        """Get a certain pre-defined point attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects points are obtained using the
        `truth_point_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
            (N, 3) Point coordinates
        """
        if not obj.is_truth:
            return obj.points
        else:
            return getattr(obj, self.truth_point_mode)
    def get_depositions(self, obj):
        """Get a certain pre-defined deposition attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects points are obtained using the
        `truth_dep_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
            (N) Depositions
        """
        if not obj.is_truth:
            return obj.depositions
        else:
            return getattr(obj, self.truth_dep_mode)


    @property
    def source_modes(self):
        """Dictionary which makes the correspondance between the name of a true
        object source attribute with the underlying source tensor it points to.

        Returns
        -------
        Dict[str, str]
            Dictionary of (attribute, key) mapping for point sources
        """
        return dict(self._source_modes)
        
    def _build_hypotheses(self, interactions):
        """Build flash hypotheses from interaction data.
        
        Parameters
        ----------
        interactions : list
            List of interaction objects
            
        Returns
        -------
        list
            List of flash hypothesis objects
        """
                

        volume_ids = [0, 1]  # TODO: Assuming 2 modules
        hypotheses = []
        id_offset = 0
        
        # Loop over optical volumes, make hypotheses for each
        for volume_id in volume_ids:
            #Crop interactions to only include depositions in the optical volume
            interactions_v = []
            for inter in interactions:
                #Fetch the points in the current optical volume
                sources = self.get_sources(inter)

                # Filter points by volume
                if self.volume == 'module':
                    index = self.geo.get_volume_index(sources, volume_id)
                elif self.volume == 'tpc':
                    num_cpm = self.geo.tpc.num_chambers_per_module
                    module_id, tpc_id = volume_id//num_cpm, volume_id%num_cpm
                    index = self.geo.get_volume_index(sources, module_id, tpc_id)
            
                # If there are no points in this volume, skip
                if len(index) == 0:
                    continue
                
                # Fetch points and depositions
                points = self.get_points(inter)[index]
                depositions = self.get_depositions(inter)[index]
                if self.ref_volume_id is not None:
                    # If the reference volume is specified, shift positions
                    points = self.geo.translate(
                            points, volume_id, self.ref_volume_id)
                
                # Create an interaction object for this volume
                inter = OutBase(
                    id=inter.id, points=points, depositions=depositions)
                interactions_v.append(inter)
            
            # Make the hypothesis
            _hypo_v = self.hypothesis.make_hypothesis_list(interactions_v, id_offset)
            hypotheses.extend(_hypo_v)
            id_offset += len(_hypo_v)
             
        return hypotheses


# """
# Post-processor in charge of filling the hypothesis into the data product.
# """

# from spine.post.base import PostBase
# from spine.utils.geo import Geometry
# from spine.data.out.base import OutBase
# import numpy as np

# from .hypothesis import Hypothesis

# __all__ = ['FillFlashHypothesisProcessor']

# class FillFlashHypothesisProcessor(PostBase):
#     """Fills the hypothesis into the data product."""

#     # Name of the post-processor (as specified in the configuration)
#     name = 'fill_flash_hypothesis'

#     # Alternative allowed names of the post-processor
#     aliases = ('fill_hypothesis',)

#     def __init__(self, volume, ref_volume_id=None, detector=None, parent_path=None,
#                  geometry_file=None, run_mode='reco', truth_point_mode='points',
#                  truth_dep_mode='depositions', hypothesis_key='flash_hypo', **kwargs):
#         """Initialize the fill hypothesis processor.

#         Parameters
#         ----------
#         volume : str
#             Physical volume corresponding to each flash ('module' or 'tpc')
#         ref_volume_id : str, optional
#             If specified, the flash matching expects all interactions/flashes
#             to live into a specific optical volume. Must shift everything.
#         detector : str, optional
#             Detector to get the geometry from
#         geometry_file : str, optional
#             Path to a `.yaml` geometry file to load the geometry from
#         parent_path : str, optional
#             Path to the parent directory of the main analysis configuration.
#             This allows for the use of relative paths in the post-processors.
#         hypothesis_key : str, default 'flash_hypo'
#             Key to use for the hypothesis data product
#         """
#         # Initialize the parent class
#         super().__init__(
#                 'interaction', run_mode, truth_point_mode, truth_dep_mode,
#                 parent_path=parent_path)
        
#         # Initialize the hypothesis key
#         self.hypothesis_key = hypothesis_key

#         # Initialize the detector geometry
#         self.geo = Geometry(detector, geometry_file)

#         # Get the volume within which each flash is confined
#         assert volume in ('tpc', 'module'), (
#                 "The `volume` must be one of 'tpc' or 'module'.")
#         self.volume = volume
#         self.ref_volume_id = ref_volume_id

#         # Initialize the hypothesis algorithm
#         self.hypothesis = Hypothesis(detector=detector, parent_path=self.parent_path, **kwargs)
        
#     def process(self, data):
#         """Fills the hypothesis into the data product.

#         Parameters
#         ----------
#         data : dict
#             Data product to fill the hypothesis into
#         """

#         #Loop over optical volumes, make the hypotheses in each
#         for k in self.interaction_keys:
#             # Fetch interactions, nothing to do if there are not any
#             interactions = data[k]
#             if not len(interactions):
#                 continue
            
#             # Make sure the interaction coordinates are expressed in cm
#             self.check_units(interactions[0])

#             # Loop over the optical volumes
#             #TODO: Use the specific detector or geometry file to get the list of optical volumes
#             id_offset = 0
#             hypothesis_v = []
#             for volume_id in [0,1]:
#                 # Crop interactions to only include depositions in the optical volume
#                 interactions_v = []
#                 for inter in interactions:
#                     # Fetch the points in the current optical volume
#                     sources = self.get_sources(inter)
#                     if self.volume == 'module':
#                         index = self.geo.get_volume_index(sources, volume_id)

#                     elif self.volume == 'tpc':
#                         num_cpm = self.geo.tpc.num_chambers_per_module
#                         module_id, tpc_id = volume_id//num_cpm, volume_id%num_cpm
#                         index = self.geo.get_volume_index(sources, module_id, tpc_id)

#                     # If there are no points in this volume, proceed
#                     if len(index) == 0:
#                         continue

#                     # Fetch points and depositions
#                     points = self.get_points(inter)[index]
#                     depositions = self.get_depositions(inter)[index]
#                     if self.ref_volume_id is not None:
#                         # If the reference volume is specified, shift positions
#                         points = self.geo.translate(
#                                 points, volume_id, self.ref_volume_id)

#                     # Create an interaction which holds positions/depositions
#                     inter_v = OutBase(
#                             id=inter.id, points=points, depositions=depositions)
#                     interactions_v.append(inter_v)
                
#                 # Make the hypothesis
#                 _hypo_v = self.hypothesis.make_hypothesis_list(interactions_v, id_offset)
#                 hypothesis_v.extend(_hypo_v)
#                 id_offset += len(_hypo_v) #increment the offset for the next volume
                
#         # Fill the hypothesis into the data product
#         data[self.hypothesis_key] = hypothesis_v
#         return data