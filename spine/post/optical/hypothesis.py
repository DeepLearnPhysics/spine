"""Module for storing flash hypotheses into flash objects."""

#TODO: Make base class between likelihood and this

import os
import sys
import numpy as np
import re

# Import the base class and Flash data structure
from .opt0_interface import OpT0Interface
from spine.data.optical import Flash


class Hypothesis(OpT0Interface):
    """
    Interface class between flash hypothesis generation and OpT0Finder.

    Inherits common initialization and QCluster creation from OpT0Interface.
    Uses an OpT0Finder hypothesis algorithm (e.g., SemiAnalyticalModel,
    PhotonLibHypothesis) to generate predicted flash PEs from TPC interactions.
    """

    def __init__(self, cfg, detector, parent_path=None, scaling=1., alpha=0.21,
                 recombination_mip=0.6, legacy=False):
        """Initialize the flash hypothesis algorithm.

        Parameters
        ----------
        cfg : str
            Flash matching configuration file path
        detector : str, optional
            Detector to get the geometry from
        parent_path : str, optional
            Path to the parent configuration file (allows for relative paths)
        scaling : Union[float, str], default 1.
            Global scaling factor for the depositions (can be an expression)
        alpha : float, default 0.21
            Number of excitons (Ar*) divided by number of electron-ion pairs (e-,Ar+)
        recombination_mip : float, default 0.6
            Recombination factor for MIP-like particles in LAr
        legacy : bool, default False
            Use the legacy OpT0Finder function(s). TODO: remove when dropping legacy
        """
        # Call the parent class initializer for common setup
        super().__init__(cfg, detector, parent_path, scaling, alpha,
                         recombination_mip, legacy)

        # Initialize hypothesis-specific attributes
        self.hypothesis_v = None

    def _initialize_algorithm(self, cfg_params):
        """
        Initialize the specific Hypothesis algorithm based on configuration.

        Parameters
        ----------
        cfg_params : flashmatch::PSet
            The loaded OpT0Finder configuration parameters.
        """
        from flashmatch import flashmatch
        # Get FlashMatchManager configuration section to find the HypothesisAlgo name
        # Assuming the relevant parameters are under 'FlashMatchManager' PSet
        # Adjust 'FlashMatchManager' if your config structure is different
        manager_params = cfg_params.get['flashmatch::FMParams']('FlashMatchManager')

        # Parse the configuration dump to find the HypothesisAlgo value
        config_dump = manager_params.dump()
        match = re.search(r'HypothesisAlgo\s*:\s*"([^"]+)"', config_dump)
        if match:
            algo_name = match.group(1)
        else:
            # Fallback: Check if the hypothesis algo config exists directly under top level
            # This depends on how the .cfg file is structured
            found_algo = False
            for name in ['SemiAnalyticalModel', 'PhotonLibHypothesis']: # Add other known hypothesis algos
                 if cfg_params.contains_pset(name):
                     algo_name = name
                     found_algo = True
                     break
            if not found_algo:
                raise ValueError(f"Could not find HypothesisAlgo parameter within "
                                 f"'FlashMatchManager' PSet in configuration: {config_dump}")


        print(f'HypothesisAlgo: {algo_name}')

        # Create the hypothesis algorithm based on the extracted name
        # Ensure the factory name matches the class name used in OpT0Finder registration
        try:
            self.hypothesis = flashmatch.FlashHypothesisFactory.get().create(
                algo_name, algo_name) # Factory name and instance name often match
        except Exception as e:
             raise ValueError(f"Failed to create hypothesis algorithm '{algo_name}'. "
                              f"Is it registered correctly in OpT0Finder? Error: {e}")

        # Configure the hypothesis algorithm using its own PSet
        try:
            algo_pset = cfg_params.get['flashmatch::FMParams'](algo_name)
            self.hypothesis.Configure(algo_pset)
        except Exception as e:
             raise ValueError(f"Failed to configure hypothesis algorithm '{algo_name}' "
                              f"using PSet '{algo_name}'. Error: {e}")

    def make_hypothesis_list(self, interactions, id_offset=0, volume_id=None):
        """
        Runs the hypothesis algorithm on a list of interactions to create
        a list of spine Flash objects representing the predicted light.

        Parameters
        ----------
        interactions : List[Union[Interaction, TruthInteraction]]
            List of TPC interactions
        id_offset : int, default 0
            Offset to add to the flash ID
        volume_id : int, optional
            Volume ID to use for the hypothesis

        Returns
        -------
        List[Flash]
            List of generated spine Flash objects.
        """
        # Make the QCluster_t objects using the base class method
        # Store them in self.qcluster_v as the base class expects
        self.qcluster_v = self.make_qcluster_list(interactions)

        # Map original interaction index to qcluster object for easy lookup
        qcluster_map = {qc.idx: qc for qc in self.qcluster_v}

        # Initialize the list of generated spine Flash objects
        self.hypothesis_v = []

        # Run the hypothesis algorithm for each interaction that produced a valid qcluster
        for i, inter in enumerate(interactions):
            # Find the corresponding QCluster_t object using the original index
            qcluster = qcluster_map.get(inter.id) # Assuming inter.id is the original index used in make_qcluster_list

            # Skip if no valid qcluster was created for this interaction
            if qcluster is None:
                continue

            # Run the hypothesis algorithm
            flash_hypothesis_fm = self.hypothesis.GetEstimate(qcluster) # flashmatch::Flash_t

            # Create a new spine Flash object from the hypothesis result
            # Pass the original interaction ID (inter.id)
            flash = Flash.from_hypothesis(flash_hypothesis_fm, inter.id, i + id_offset, volume_id)

            # Append the generated spine Flash object
            self.hypothesis_v.append(flash)

        return self.hypothesis_v
    
