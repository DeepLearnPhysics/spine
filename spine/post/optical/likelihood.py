import os, sys
import numpy as np
import time

from spine.utils.geo import Geometry


class LikelihoodFlashMatcher:
    """Interface class between full chain outputs and OpT0Finder

    See https://github.com/drinkingkazu/OpT0Finder for more details about it.
    """
    def __init__(self, cfg, parent_path=None, reflash_merging_window=None,
                 detector=None, geometry_file=None, scaling=1.,
                 alpha=0.21, recombination_mip=0.65,
                 truth_dep_mode='depositions'):
        """Initialize the likelihood-based flash matching algorithm.

        Parameters
        ----------
        cfg : str
            Flash matching configuration file path
        parent_path : str, optional
            Path to the parent configuration file (allows for relative paths)
        reflash_merging_window : float, optional
            Maximum time between successive flashes to be considered a reflash
        detector : str, optional
            Detector to get the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        scaling : Union[float, str], default 1.
            Global scaling factor for the depositions (can be an expression)
        alpha : float, default 0.21
            Number of excitons (Ar*) divided by number of electron-ion pairs (e-,Ar+)
        recombination_mip : float, default 0.65
            Recombination factor for MIP-like particles in LAr
        truth_dep_mode : str, default 'depositions'
            Attribute used to fetch deposition values for truth interactions
        """
        # Initialize the flash manager (OpT0Finder wrapper)
        self.detector = detector
        self.initialize_backend(cfg, parent_path)

        # Initialize the geometry
        self.geo = Geometry(detector, geometry_file)

        # Get the external parameters
        self.truth_dep_mode = truth_dep_mode
        self.reflash_merging_window = reflash_merging_window
        self.scaling = scaling
        if isinstance(self.scaling, str):
            self.scaling = eval(self.scaling)
        self.alpha = alpha
        if isinstance(self.alpha, str):
            self.alpha = eval(self.alpha)
        self.recombination_mip = recombination_mip
        if isinstance(self.recombination_mip, str):
            self.recobination_mip = eval(self.recombination_mip)

        # Initialize flash matching attributes
        self.matches = None
        self.qcluster_v = None
        self.flash_v = None

    def initialize_backend(self, cfg, parent_path):
        """Initialize OpT0Finder (backend).

        Expects that the environment variable `FMATCH_BASEDIR` is set.
        You can either set it by hand (to the path where one can find
        OpT0Finder) or you can source `OpT0Finder/configure.sh` if you
        are running code from a command line.

        Parameters
        ----------
        cfg: str
            Path to config for OpT0Finder
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
        from flashmatch import flashmatch
        #Find the detector configuration file
        if self.detector is None:
            det_cfg = os.path.join(basedir, 'dat/detector_specs.cfg')
        else:
            det_cfg = os.path.join(basedir, f'dat/detector_specs_{self.detector}.cfg')
        det_cfg = os.path.join(basedir, f'dat/detector_specs_sbnd.cfg')
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

        # Get the module ID in which all the interactions live
        module_ids = np.empty(len(interactions), dtype=np.int64)
        for i, inter in enumerate(interactions):
            ids = inter.module_ids
            assert len(ids) > 0, (
                    "The interaction object does not contain any information "
                    "about which optical module produced it; must be provided.")
            assert len(ids) == 1, (
                    "Cannot match interactions that are composed of points "
                    "originating for more than one optical module.")
            module_ids[i] = ids[0]

        # Check that all interactions live in one module, store it
        assert len(np.unique(module_ids)) == 1, (
                "Should only provide interactions from a single optical module.")
        self.module_id = module_ids[0]

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
            # FIXME: This is a temporary fix for the SBND geometry, we don't need to shift points
            points = inter.points[valid_mask]
            #points = self.geo.translate(inter.points[valid_mask],
            #        self.module_id, 0)

            # Get the depositions
            if not inter.is_truth:
                depositions = inter.depositions[valid_mask]
            else:
                depositions = getattr(inter, self.truth_dep_mode)

            # Fill the trajectory
            pytraj = np.hstack([points, depositions[:, None]])
            traj = flashmatch.as_geoalgo_trajectory(pytraj)
            qcluster += self.light_path.MakeQCluster(traj, self.scaling, self.alpha, self.recombination_mip)

            # Append
            qcluster_v.append(qcluster)

        return qcluster_v

    def make_flash_list(self, flashes,n_pds=312):
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
        #FIXME - do this upstream when parsing flashes
        #if the PEPerOPDet is not the same length as the number of PDs, pad with zeros
        for f in flashes:
            if len(f.pe_per_ch) != n_pds:
                f.pe_per_ch = np.resize(f.pe_per_ch,n_pds)

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
            offset = 0 if len(f.pe_per_ch) == n_pds else n_pds
            for i in range(n_pds):
                flash.pe_v.push_back(f.pe_per_ch[i + offset])
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
            #FIXME: This is a temporary fix for the SBND geometry, we don't need to shift points
            #pos = self.geo.translate(pos, 0, self.module_id)
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
