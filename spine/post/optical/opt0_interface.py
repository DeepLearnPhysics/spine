"""Base module for interfacing with OpT0Finder algorithms."""

import os
import sys
import numpy as np
import re
from abc import ABC, abstractmethod

class OpT0Interface(ABC):
    """
    Abstract base class for OpT0Finder interfaces (Likelihood and Hypothesis).

    Handles common initialization logic, environment setup, configuration loading,
    and QCluster creation.
    """

    def __init__(self, cfg, detector, parent_path=None, scaling=1., alpha=0.21,
                 recombination_mip=0.6, legacy=False):
        """
        Initialize common attributes and the OpT0Finder backend.

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
        # Store external parameters
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

        # Initialize the flash manager (OpT0Finder wrapper)
        self.initialize_backend(cfg, detector, parent_path)

        # Initialize common attributes potentially used by subclasses
        self.qcluster_v = None

    def initialize_backend(self, cfg, detector, parent_path):
        """
        Initialize the common OpT0Finder backend components.

        Sets up environment variables, loads detector specs, loads configuration,
        and initializes the LightPath algorithm. Calls the abstract method
        `_initialize_algorithm` for subclass-specific initialization.

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
        # Avoid prepending if already present to prevent excessive path length
        if lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
            os.environ['LD_LIBRARY_PATH'] = '{}:{}'.format(
                    lib_path, os.environ.get('LD_LIBRARY_PATH', ''))

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
        if parent_path is not None and not os.path.isabs(cfg):
            cfg = os.path.join(parent_path, cfg)
        if not os.path.isfile(cfg):
            raise FileNotFoundError(
                    f"Cannot find flash-matcher config: {cfg}")

        cfg_params = flashmatch.CreateFMParamsFromFile(cfg)

        # Get the light path algorithm to produce QCluster_t objects
        self.light_path = flashmatch.CustomAlgoFactory.get().create(
                'LightPath', 'ToyMCLightPath')
        self.light_path.Configure(cfg_params.get['flashmatch::FMParams']('LightPath'))

        # Initialize the specific algorithm (FlashMatchManager or Hypothesis)
        self._initialize_algorithm(cfg_params)


    @abstractmethod
    def _initialize_algorithm(self, cfg_params):
        """
        Abstract method for initializing the specific OpT0Finder algorithm.

        Subclasses must implement this to initialize their specific backend
        (e.g., FlashMatchManager for likelihood, HypothesisAlgo for hypothesis).

        Parameters
        ----------
        cfg_params : flashmatch::PSet
            The loaded OpT0Finder configuration parameters.
        """
        pass

    def make_qcluster_list(self, interactions):
        """
        Converts a list of SPINE interaction into a list of OpT0Finder
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
            qcluster.time = 0 # Assume t=0 for hypothesis/likelihood generation

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

