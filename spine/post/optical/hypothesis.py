"""Module for storing flash hypotheses into flash objects."""

#TODO: Make base class between likelihood and this

import os
import sys
import numpy as np
import re
from spine.data.optical import Flash

class Hypothesis:
    """Interface class between flash hypothesis and OpT0Finder."""

    def __init__(self, cfg, detector, parent_path=None,scaling=1., alpha=0.21,
                 recombination_mip=0.65, legacy=False):
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
        recombination_mip : float, default 0.65
            Recombination factor for MIP-like particles in LAr
        legacy : bool, default False
            Use the legacy OpT0Finder function(s). TODO: remove when dropping legacy
        """
        # Initialize the flash manager (OpT0Finder wrapper)
        self.initialize_backend(cfg, detector, parent_path)

        # Get the external parameters
        self.scaling = scaling
        if isinstance(self.scaling, str):
            self.scaling = eval(self.scaling)
        self.alpha = alpha
        if isinstance(self.alpha, str):
            self.alpha = eval(self.alpha)
        self.recombination_mip = recombination_mip
        if isinstance(self.recombination_mip, str):
            self.recombination_mip = eval(self.recombination_mip)
        self.legacy = legacy
        
        #Initialize hypotheses
        self.hypothesis_v = None

    def initialize_backend(self, cfg, detector, parent_path):
        """Initialize the flash manager (OpT0Finder wrapper).

        Expects that the environment variable `FMATCH_BASEDIR` is set.
        You can either set it by hand (to the path where one can find
        OpT0Finder) or you can source `OpT0Finder/configure.sh` if you
        are running code from a command line.

        Parameters
        ----------
        cfg : str
            Path to config for OpT0Finder
        detector : str, optional
            Detector to get the geometry from
        parent_path : str, optional
            Path to the parent configuration file (allows for relative paths)
        """
        # Add OpT0finder python interface to the python path
        basedir = os.getenv('FMATCH_BASEDIR')
        assert basedir is not None, (
                "You need to source OpT0Finder's configure.sh or set the "
                "FMATCH_BASEDIR environment variable before running flash "
                "matching.")
        sys.path.append(os.path.join(basedir, 'python'))

        # Add the OpT0Finder library to the dynamic link loader
        lib_path = os.path.join(basedir, 'build/lib')
        os.environ['LD_LIBRARY_PATH'] = '{}:{}'.format(
                lib_path, os.environ['LD_LIBRARY_PATH'])

        # Add the OpT0Finder data directory if it is not yet set
        if 'FMATCH_DATADIR' not in os.environ:
            os.environ['FMATCH_DATADIR'] = os.path.join(basedir, 'dat')

        # Load up the detector specifications
        if detector is None:
            det_cfg = os.path.join(basedir, 'dat/detector_specs.cfg')
        else:
            det_cfg = os.path.join(basedir, f'dat/detector_specs_{detector}.cfg')

        if not os.path.isfile(det_cfg):
            raise FileNotFoundError(
                    f"Cannot file detector specification file: {det_cfg}.")

        from flashmatch import flashmatch
        flashmatch.DetectorSpecs.GetME(det_cfg)

        # Fetch and initialize the OpT0Finder configuration
        if parent_path is not None and not os.path.isfile(cfg):
            cfg = os.path.join(parent_path, cfg)
        if not os.path.isfile(cfg):
            raise FileNotFoundError(
                    f"Cannot find flash-matcher config: {cfg}")

        cfg = flashmatch.CreateFMParamsFromFile(cfg)
        
        # Get FlashMatchManager configuration
        fmatch_params = cfg.get['flashmatch::FMParams']('FlashMatchManager')
        
        # Parse the configuration dump to find the HypothesisAlgo value
        config_dump = fmatch_params.dump()
        match = re.search(r'HypothesisAlgo\s*:\s*"([^"]+)"', config_dump)
        if match:
            algo = match.group(1)
        else:
            raise ValueError(f"Could not find HypothesisAlgo in configuration: {config_dump}")

        print(f'HypothesisAlgo: {algo}')
        
        # Get the light path algorithm to produce QCluster_t objects
        self.light_path = flashmatch.CustomAlgoFactory.get().create(
                'LightPath', 'ToyMCLightPath')
        self.light_path.Configure(cfg.get['flashmatch::FMParams']('LightPath'))

        # Create the hypothesis algorithm based on the extracted name
        if algo == 'SemiAnalyticalModel':
            self.hypothesis = flashmatch.FlashHypothesisFactory.get().create(
                'SemiAnalyticalModel','SemiAnalyticalModel')
        elif algo == 'PhotonLibHypothesis':
            self.hypothesis = flashmatch.FlashHypothesisFactory.get().create(
                'PhotonLibHypothesis','PhotonLibHypothesis')
        else:
            raise ValueError(f"Unknown hypothesis algorithm: {algo}")
        self.hypothesis.Configure(cfg.get['flashmatch::FMParams'](f'{algo}'))

    def make_qcluster_list(self, interactions):
        """Converts a list of SPINE interaction into a list of OpT0Finder
        flashmatch.QCluster_t objects.

        Parameters
        ----------
        interactions : List[Union[Interaction, TruthInteraction]]
            List of TPC interactions

        Returns
        -------
        List[QCluster_t]
           List of OpT0Finder flashmatch::QCluster_t objects
        """
        # Loop over the interacions
        from flashmatch import flashmatch
        qcluster_v = []
        for idx, inter in enumerate(interactions):
            # Produce a mask to remove negative value points (can happen)
            valid_mask = np.where(inter.depositions > 0.)[0]

            # Skip interactions with less than 2 points
            if len(valid_mask) < 2:
                continue

            # Initialize qcluster
            qcluster = flashmatch.QCluster_t()
            qcluster.idx = idx
            qcluster.time = 0

            # Get the point coordinates
            points = inter.points[valid_mask]

            # Get the depositions
            depositions = inter.depositions[valid_mask]

            # Fill the trajectory
            pytraj = np.hstack([points, depositions[:, None]])
            traj = flashmatch.as_geoalgo_trajectory(pytraj)
            if self.legacy:
                qcluster += self.light_path.MakeQCluster(traj, self.scaling)
            else:
                qcluster += self.light_path.MakeQCluster(
                        traj, self.scaling, self.alpha, self.recombination_mip)

            # Append
            qcluster_v.append(qcluster)

        return qcluster_v
    
    def make_hypothesis_list(self, interactions, id_offset=0, volume_id=None):
        """
        Runs the hypothesis algorithm on a list of interactions to create
        a list of flashmatch::Flash_t objects.

        Parameters
        ----------
        interactions : List[Union[Interaction, TruthInteraction]]
            List of TPC interactions
        id_offset : int, default 0
            Offset to add to the flash ID
        volume_id : int, optional
            Volume ID to use for the hypothesis
        """
        # Make the QCluster_t objects
        qcluster_v = self.make_qcluster_list(interactions)

        # Initialize the list of flashmatch::Flash_t objects
        self.hypothesis_v = []

        # Run the hypothesis algorithm
        for i,int in enumerate(interactions):
            # Make the QCluster_t object
            qcluster = qcluster_v[i]

            # Run the hypothesis algorithm
            flash = self.hypothesis.GetEstimate(qcluster)

            # Create a new Flash object
            flash = Flash.from_hypothesis(flash, int.id, i+id_offset, volume_id)

            # Append
            self.hypothesis_v.append(flash)

        return self.hypothesis_v
    
