"""Module which supports likelihood-based flash matchin (OpT0Finder)."""

import os, sys
import numpy as np
import time


class LikelihoodFlashMatcher:
    """Interface class between full chain outputs and OpT0Finder

    See https://github.com/drinkingkazu/OpT0Finder for more details about it.
    """

    def __init__(self, cfg, detector, parent_path=None,
                 reflash_merging_window=None, scaling=1., alpha=0.21,
                 recombination_mip=0.65, legacy=False):
        """Initialize the likelihood-based flash matching algorithm.

        Parameters
        ----------
        cfg : str
            Flash matching configuration file path
        detector : str, optional
            Detector to get the geometry from
        parent_path : str, optional
            Path to the parent configuration file (allows for relative paths)
        reflash_merging_window : float, optional
            Maximum time between successive flashes to be considered a reflash
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
        self.reflash_merging_window = reflash_merging_window
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

        # Initialize flash matching attributes
        self.matches = None
        self.qcluster_v = None
        self.flash_v = None

    def initialize_backend(self, cfg, detector, parent_path):
        """Initialize OpT0Finder (backend).

        Expects that the environment variable `FMATCH_BASEDIR` is set.
        You can either set it by hand (to the path where one can find
        OpT0Finder) or you can source `OpT0Finder/configure.sh` if you
        are running code from a command line.

        Parameters
        ----------
        cfg: str
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

        cfg = flashmatch.CreatePSetFromFile(cfg)

        # Initialize The OpT0Finder flash match manager
        self.mgr = flashmatch.FlashMatchManager()
        self.mgr.Configure(cfg)

        # Get the light path algorithm to produce QCluster_t objects
        self.light_path = flashmatch.CustomAlgoFactory.get().create(
                'LightPath', 'ToyMCLightPath')
        self.light_path.Configure(cfg.get['flashmatch::PSet']('LightPath'))

    def get_matches(self, interactions, flashes):
        """Find TPC interactions compatible with optical flashes.

        Parameters
        ----------
        interactions : List[Union[Interaction, TruthInteraction]]
            List of TPC interactions
        flashes : List[Flash]
            List of optical flashes

        Returns
        -------
        List[Tuple[Interaction, Flash, flashmatch::FlashMatch_t]]
            Set of interaction/flash matches with their matching characteristics
        """
        # If there is no interaction or no flashe, nothing to do
        if not len(interactions) or not len(flashes):
            return []

        # Build a list of QCluster_t (OpT0Finder interaction representation)
        self.qcluster_v = self.make_qcluster_list(interactions)

        # Build a list of Flash_t (OpT0Finder optical flash representation)
        self.flash_v, flashes = self.make_flash_list(flashes)

        # Running flash matching and caching the results
        self.matches = self.run_flash_matching()

        # Build result, return
        result = []
        for m in self.matches:
            result.append((interactions[m.tpc_id], flashes[m.flash_id], m))

        return result

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
        for inter in interactions:
            # Produce a mask to remove negative value points (can happen)
            valid_mask = np.where(inter.depositions > 0.)[0]

            # If the interaction has less than 2 points, skip
            if len(valid_mask) < 2:
                continue

            # Initialize qcluster
            qcluster = flashmatch.QCluster_t()
            qcluster.idx = int(inter.id)
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

    def make_flash_list(self, flashes):
        """Creates a list of flashmatch.Flash_t from the local class.

        Parameters
        ----------
        flashes : List[Flash]
            List of optical flashes

        Returns
        -------
        List[Flash_t]
            List of flashmatch::Flash_t objects
        """
        # If requested, merge flashes that are compatible in time
        if self.reflash_merging_window is not None:
            times = [f.time for f in flashes]
            perm = np.argsort(times)
            new_flashes = [flashes[perm[0]]]
            for i in range(1, len(perm)):
                prev, curr = perm[i-1], perm[i]
                if ((flashes[curr].time - flashes[prev].time)
                    < self.reflash_merging_window):
                    # If compatible, simply add up the PEs
                    pe_v = flashes[prev].pe_per_ch + flashes[curr].pe_per_ch
                    new_flashes[-1].pe_per_ch = pe_v
                else:
                    new_flashes.append(flashes[curr])

            flashes = new_flashes

        # Loop over the optical flashes
        from flashmatch import flashmatch
        flash_v = []
        for idx, f in enumerate(flashes):
            # Initialize the Flash_t object
            flash = flashmatch.Flash_t()
            flash.idx = int(f.id)  # Assign a unique index
            flash.time = f.time  # Flash timing, a candidate T0

            # Assign the flash position and error on this position
            flash.x, flash.y, flash.z = 0, 0, 0
            flash.x_err, flash.y_err, flash.z_err = 0, 0, 0

            # Assign the individual PMT optical hit PEs
            for i in range(len(f.pe_per_ch)):
                flash.pe_v.push_back(f.pe_per_ch[i])
                flash.pe_err_v.push_back(0.)

            # Append
            flash_v.append(flash)

        return flash_v, flashes

    def run_flash_matching(self):
        """Drive the OpT0Finder flash matching.

        Returns
        -------
        List[flashmatch::FlashMatch_t]
            List of matches
        """
        # Make sure the interaction and flash objects were set
        assert self.qcluster_v is not None and self.flash_v is not None, (
                "Must make_qcluster_list and make_flash_list first.")

        # Register all objects in the manager
        self.mgr.Reset()
        for x in self.qcluster_v:
            self.mgr.Add(x)
        for x in self.flash_v:
            self.mgr.Add(x)

        # Run the matching
        all_matches = self.mgr.Match()

        # Adjust the output position to account for the module shift
        for m in all_matches:
            pos = np.array([m.tpc_point.x, m.tpc_point.y, m.tpc_point.z])
            m.tpc_point.x = pos[0]
            m.tpc_point.y = pos[1]
            m.tpc_point.z = pos[2]

        return all_matches

    def get_qcluster(self, idx, array=False):
        """Fetch a given flashmatch::QCluster_t object.

        Parameters
        ----------
        idx : int
            ID of the interaction to fetch
        array : bool, default `False`
            If `True`, The QCluster is returned as an np.ndarray

        Returns
        -------
        Union[flashmatch::QCluster_t, np.ndarray]
            QCluster object
        """
        if self.qcluster_v is None:
            raise Exception('self.qcluster_v is None')

        for qcluster in self.qcluster_v:
            if qcluster.idx != idx: continue
            if array: return flashmatch.as_ndarray(qcluster)
            else: return qcluster

        raise Exception(f'TPC object {idx} does not exist in self.qcluster_v')

    def get_flash(self, idx, array=False):
        """Fetch a given flashmatch::Flash object.

        Parameters
        ----------
        idx : int
            ID of the flash to fetch
        array : bool, default `False`
            If `True`, The flash is returned as an np.ndarray

        Returns
        -------
        Union[flashmatch::Flash, np.ndarray]
            Flash object
        """
        if self.flash_v is None:
            raise Exception('self.flash_v is None')

        for flash in self.flash_v:
            if flash.idx != idx: continue
            if array: return flashmatch.as_ndarray(flash)
            else: return flash

        raise Exception('Flash {idx} does not exist in self.flash_v')


    def get_match(self, idx):
        """Fetch a match for a given TPC interaction ID.

        Parameters
        ----------
        idx : int
            Index of TPC object for which we want to retrieve a match

        Returns
        -------
        flashmatch::FlashMatch_t
            Flash match associated with interaction idx
        """
        if self.matches is None:
            raise Exception('Need to run flash matching first')

        for m in self.matches:
            if self.qcluster_v[m.tpc_id].idx != idx: continue
            return m

        return None

    def get_matched_flash(self, idx):
        """Fetch a matched flash for a given TPC interaction ID.

        Parameters
        ----------
        idx : int
            Index of TPC object for which we want to retrieve a match

        Returns
        -------
        flashmatch::Flash_t
            Optical flash that matches interaction idx
        """
        # Get a match, if any
        m = self.get_match(idx)
        if m is None: return None

        # Get the flash that corresponds to the match
        flash_id = m.flash_id
        if flash_id is None: return None
        if flash_id > len(self.flash_v):
            raise Exception('Flash {flash_id} does not exist in self.flash_v')

        return self.flash_v[flash_id]

    def get_t0(self, idx):
        """Fetch a matched flash time for a given TPC interaction ID.

        Parameters
        ----------
        idx : int
            Index of TPC object for which we want to retrieve a match

        Returns
        -------
        float
            Time in us with respect to simulation time reference
        """
        # Get the matched flash, if any
        flash = self.get_matched_flash(idx)

        return None if flash is None else flash.time
