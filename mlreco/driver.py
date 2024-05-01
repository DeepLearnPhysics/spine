"""SPINE driver class.

Takes care of everything in one centralized place:
    - Data loading
    - ML model forward path
    - Batch unwrapping
    - Representation building
    - Post-processing
    - Analysis scripts
    - Writing
"""

import os

from .io import loader_factory, reader_factory, writer_factory

#from .trainval import TrainVal

from .utils.unwrap import Unwrapper
from .build import BuildManager
from .post import PostManager
from .ana import AnaManager


class Driver:
    """Central SPINE driver.

    Processes global configuration and runs the appropriate modules:
      1. Load data
      2. Run the model forward
      3. Unwrap batched data
      4. Build representations
      5. Run post-processing
      6. Run analysis scripts
      7. Write to file
    """

    def __init__(self, io, base, model=None, post=None, ana=None):
        """Initializes the class attributes.

        Parameters
        ----------
        io : dict
           Input/output configuration dictionary
        main_cfg : dict
           Main configuration dictionary
        model : dict
           Model configuration dictionary
        post : dict, optional
            Post-processor configuration, if there are any to be run
        ana : dict, optional
            Analysis script configuration (writes to CSV files)
        rank : int, default 0
           Rank of the GPU in the multi-GPU training process
        """
        # Initialize the timers
        self.watch = StopwatchManager()
        self.watch.initialize('iteration')

        # Initialize the main analysis configuration parameters
        self.initialize_base(**base)

        # Initialize the input/output
        self.initialize_io(**io)

        # Initialize the ML model
        self.geo = None
        self.model = None
        if model is not None:
            self.model = TrainVal(io, model)

        # Initialize the data representation builder
        self.builder = None
        if build is not None:
            self.watch.initialize('build')
            self.builder = BuildManager(**build)

        # Initialize the post-processors
        self.post = None
        if post is not None:
            self.watch.initialize('post')
            self.post = PostManager(**post, parent_path=parent_path)

        # Initialize the analysis scripts
        self.ana = None
        if ana is not None:
            self.watch.initialize('ana')
            self.ana = AnaManager(**ana)

    def initialize_base(self,
                        log_dir = './',
                        prefix_log = False,
                        parent_path = None,
                        iteration = -1,
                        chain_config = None,
                        data_builders = None,
                        event_list = None):
        '''
        Initialize the main analysis tool parameters

        Parameters
        ----------
        log_dir : str, default './'
            Path to the log directory where the logs will be written to
        prefix_log : bool, default False
            If True, use the input file name to prefix the log name
        parent_path : str, optional
            Path to the parent directory of the analysis configuration file
        iteration : int, default -1
            Number of times to iterate over the data (-1 means all entries)
        convert_to_cm : bool, default True
            If `True`, convert pixel coordinates to detector coordinates
        chain_config : str, optional
            Path to the full ML chain configuration
        data_builders : dict, optional
            Data builder function configuration
        '''
        # Store general parameters
        self.parent_path = parent_path
        self.max_iteration = iteration
        self.convert_to_cm = convert_to_cm
        self.event_list = event_list

        # Create the log directory if it does not exist and initialize log file
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.logger = CSVWriter(os.path.join(log_dir, 'ana_profile.csv'))
        self.logger_dict = {}
        self.prefix_log = prefix_log

        # Load the full chain configuration, if it is provided
        self.chain_config = chain_config
        if chain_config is not None:
            cfg = yaml.safe_load(open(chain_config, 'r').read())
            process_config(cfg, verbosity='info')
            self.chain_config = cfg

        # Initialize data product builders
        self.builders = {}
        if data_builders is not None:
            for builder_name in data_builders:
                if builder_name not in SUPPORTED_BUILDERS:
                    msg = f'{builder_name} is not a valid data product builder!'
                    raise ValueError(msg)
                builder = eval(builder_name)(convert_to_cm=convert_to_cm)
                self.builders[builder_name] = builder

    def initialize_io(self, loader=None, reader=None, writer=None,
                      unwrap=False):
        """Initializes the input/output scripts.

        Parameters
        ----------
        loader : dict, optional
            PyTorch DataLoader configuration dictionary
        reader : dict, optional
            Reader configuration dictionary
        writer : dict, optional
            Writer configuration dictionary
        unwrap : dict, default False
            Unwrap batched data
        """
        # Make sure that we have either a data loader or a reader, not both
        assert (loader is not None) ^ (reader is not None), (
                "Must provide either a loader or a reader configuration.")

        # Initialize the data loader, if provided
        self.loader = None
        if loader is not None:
            self.loader = loader_factory(
                    distributed=self.distributed,
                    world_size=self.world_size, rank=self.rank, **io_cfg)
            self.reader = self.loader.dataset.reader

        # Initialize the data reader, if provided
        self.reader = None
        if reader is not None:
            self.reader = reader_factory(reader)

        # Initialize the data writer, if provided
        self.writer = None
        if writer is not None:
            self.writer = writer_factory(writer)

        # Infer the total number of epochs from iterations or vice-versa
        iter_per_epoch = len(self.loader) / self.world_size # TODO: check on that
        if self.iterations is not None:
            self.epochs = self.iterations / iter_per_epoch
        else:
            self.iterations = self.epochs * iter_per_epoch

        # TODO
        if self.prefix_log and len(self._file_list) == 1:
            prefix = pathlib.Path(self._file_list[0]).stem
            self.logger = CSVWriter(os.path.join(
                self.log_dir, f'{prefix}_ana_profile.csv'))

    def run(self):
        """Loop over the requested number of iterations, process them."""
        # Loop and process each iteration
        for iteration in range(self.iterations):
            self.process(iteration)

    def process(self, iteration=None, run=None, event=None):
        """Process one entry or one batch of data.

        Run single step of analysis tools workflow. This includes
        data forwarding, building data structures, running post-processing,
        and appending desired information to each row of output csv files.

        Parameters
        ----------
        iteration : int, optional
            Iteration number for current step.
        run : int, optional
            Run number
        event : int, optional
            Event number
        """
        # 0. Start the timer for the iteration
        self.watch.start('iteration')

        # 1. Load data
        self.watch.start('read')
        data, result = self.load(iteration, run, event)
        self.watch.stop('read')

        # 2. Pass through the model
        if self.model is not None:
            pass # TODO

        # 3. Unwrap
        if self.unwrap:
            self.watch.start('unwrap')
            data, result = self.unwrapper(data, result)
            self.watch.start('unwrap')

        # 4. Build representations
        if self.builder is not None:
            self.watch.start('build')
            self.builder(data, result)
            self.watch.stop('build')

        # 5. Run post-processing, if requested
        if self.post is not None:
            self.watch.start('post')
            self.post_processor(data, result)
            self.watch.stop('post')

        # 6. Run scripts, if requested
        if self.analyzer is not None:
            self.watch.start('ana')
            self.analyzer(data, result)
            self.watch.stop('ana')

        # 7. Write output to file, if requested
        if self.writer is not None:
            self.watch.start('write')
            self.writer(data, result)
            self.watch.stop('write')

        # Log to file
        self.stop('iteration')
        self.log() # TODO

        # Return
        return data, result

    def load(self, iteration=None, run=None, event=None):
        '''
        Read one minibatch worth of image from dataset.

        Parameters
        ----------
        iteration : int, optional
            Iteration number, needed for reading entries from
            HDF5 files, by default None.
        run : int, optional
            Run number
        event : int, optional
            Event number

        Returns
        -------
        data: dict
            Data dictionary containing the input
        '''
        # Dispatch to the appropriate loader
        if self.loader is not None:
            # Can only load batches by index
            assert ((iteration is not None) and
                    (run is None and event is None)), (
                           "Provide the iteration number only.")

            return next(self.loader_iter)

        else:
            # Must provide either iteration or both run and event numbers
            assert ((iteration is not None) or
                    (run is not None and event is not None)), (
                           "Provide either the iteration number or both the "
                           "run number and the event number to load.")

            if iteration is not None:
                return self.reader.get(iteration)
            else:
                return self.reader.get_run_event(run, event)

    def convert_pixels_to_cm(self, data, result):
        '''Convert pixel coordinates to real world coordinates (in cm)
        for all tensors that have spatial coordinate information, using
        information in meta (operation is in-place).

        Parameters
        ----------
        data : dict
            Data and label dictionary
        result : dict
            Result dictionary
        '''

        data_has_voxels = set([
            'input_data', 'segment_label',
            'particles_label', 'cluster_label', 'kinematics_label', 'sed'
        ])
        result_has_voxels = set([
            'input_rescaled',
            'cluster_label_adapted',
            'shower_fragment_start_points',
            'shower_fragment_end_points',
            'track_fragment_start_points',
            'track_fragment_end_points',
            'particle_start_points',
            'particle_end_points',
        ])

        data_products = set([
            'particles', 'truth_particles', 'interactions', 'truth_interactions',
            'particle_fragments', 'truth_particle_fragments'
        ])

        meta = data['meta'][0]
        assert len(meta) == 9

        print('Converting units from px to cm...')

        # for key, val in data.items():
        #     if key in data_has_voxels:
        #         data[key] = [self._pixel_to_cm(arr, meta) for arr in val]
        for key, val in result.items():
        #     if key in result_has_voxels:
        #         result[key] = [self._pixel_to_cm(arr, meta) for arr in val]
            if key in data_products:
                for plist in val:
                    for p in plist:
                        p.convert_to_cm(meta)

    def _build_reco_reps(self, data, result):
        '''
        Build representations for reconstructed objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity.
        '''
        length_check = []
        if 'ParticleBuilder' in self.builders:
            result['particles']         = self.builders['ParticleBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['particles']))
        if 'InteractionBuilder' in self.builders:
            result['interactions']      = self.builders['InteractionBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['interactions']))
        if 'FragmentBuilder' in self.builders:
            result['particle_fragments'] = self.builders['FragmentBuilder'].build(data, result, mode='reco')
            length_check.append(len(result['particle_fragments']))
        return length_check

    def _build_truth_reps(self, data, result):
        '''
        Build representations for true objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity.
        '''
        length_check = []
        if 'ParticleBuilder' in self.builders:
            result['truth_particles']    = self.builders['ParticleBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_particles']))
        if 'InteractionBuilder' in self.builders:
            result['truth_interactions'] = self.builders['InteractionBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_interactions']))
        if 'FragmentBuilder' in self.builders:
            result['truth_particle_fragments'] = self.builders['FragmentBuilder'].build(data, result, mode='truth')
            length_check.append(len(result['truth_particle_fragments']))
        return length_check

    def build_representations(self, data, result, mode='all'):
        '''
        Build human readable data structures from full chain output.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary
        mode : str, optional
            Whether to build only reconstructed or true objects.
            'reco', 'truth', and 'all' are available (by default 'all').
        '''
        num_batches = len(data['index'])
        lcheck_reco, lcheck_truth = [], []

        if mode == 'reco':
            lcheck_reco = self._build_reco_reps(data, result)
        elif mode == 'truth':
            lcheck_truth = self._build_truth_reps(data, result)
        elif mode == 'all':
            lcheck_reco = self._build_reco_reps(data, result)
            lcheck_truth = self._build_truth_reps(data, result)
        else:
            raise ValueError(f'DataBuilder mode {mode} is not supported!')
        for lreco in lcheck_reco:
            assert lreco == num_batches
        for ltruth in lcheck_truth:
            assert ltruth == num_batches

    def _load_reco_reps(self, data, result):
        '''
        Load representations for reconstructed objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity.
        '''
        if 'ParticleBuilder' in self.builders:
            result['particles']         = self.builders['ParticleBuilder'].load(data, result, mode='reco')

        if 'InteractionBuilder' in self.builders:
            result['interactions']      = self.builders['InteractionBuilder'].load(data, result, mode='reco')

    def _load_truth_reps(self, data, result):
        '''
        Load representations for true objects.

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        length_check: List[int]
            List of integers representing the length of each data structure
            from DataBuilders, used for checking validity.
        '''
        if 'ParticleBuilder' in self.builders:
            result['truth_particles']    = self.builders['ParticleBuilder'].load(data, result, mode='truth')
        if 'InteractionBuilder' in self.builders:
            result['truth_interactions'] = self.builders['InteractionBuilder'].load(data, result, mode='truth')

    def load_representations(self, data, result, mode='all'):
        if mode == 'reco':
            self._load_reco_reps(data, result)
        elif mode == 'truth':
            self._load_truth_reps(data, result)
        elif mode is None or mode == 'all':
            self._load_reco_reps(data, result)
            self._load_truth_reps(data, result)
            if 'ParticleBuilder' in self.builders:
                matches = generate_match_pairs(result['truth_particles'][0],
                        result['particles'][0], 'matched_particles', only_principal=self.load_principal_matches)
                result.update({k:[v] for k, v in matches.items()})
                result['particle_match_overlap_t2r'] = result.pop('matched_particles_t2r_values')
                result['particle_match_overlap_r2t'] = result.pop('matched_particles_r2t_values')
            if 'InteractionBuilder' in self.builders:
                matches = generate_match_pairs(result['truth_interactions'][0],
                        result['interactions'][0], 'matched_interactions', only_principal=self.load_principal_matches)
                result.update({k:[v] for k, v in matches.items()})
                result['interaction_match_overlap_t2r'] = result.pop('matched_interactions_t2r_values')
                result['interaction_match_overlap_r2t'] = result.pop('matched_interactions_r2t_values')
        else:
            raise ValueError(f'DataBuilder mode {mode} is not supported!')

    def run_ana_scripts(self, data, result, iteration):
        '''Run all registered analysis scripts (under producers/scripts)

        Parameters
        ----------
        data : dict
            Data dictionary
        result : dict
            Result dictionary

        Returns
        -------
        out: dict
            Dictionary of column name : value mapping, which corresponds to
            each row in the output csv file.
        '''
        out = {}
        if self.scripts is not None and len(self.scripts):
            script_processor = ScriptProcessor(data, result)
            for processor_name, pcfg in self.scripts.items():
                priority = pcfg.pop('priority', -1)
                pcfg['iteration'] = iteration
                processor_name = processor_name.split('+')[0]
                processor = getattr(scripts,str(processor_name))
                script_processor.register_function(processor,
                                                   priority,
                                                   script_cfg=pcfg)
            fname_to_update_list = script_processor.process(iteration)
            out[processor_name] = fname_to_update_list # TODO: Questionable

        return out

    def write(self, ana_output):
        '''Method to gather logging information from each analysis script
        and save to csv files.

        Parameters
        ----------
        ana_output : dict
            Dictionary of column name : value mapping, which corresponds to
            each row in the output csv file.

        Raises
        ------
        RuntimeError
            If two filenames specified by the user point to the same path.
        '''

        if self.scripts is None:
            self.scripts = {}
        if self.csv_writers is None:
            self.csv_writers = {}

        for script_name, fname_to_update_list in ana_output.items():
            append  = self.scripts[script_name].get('append', False)
            filenames = list(fname_to_update_list.keys())
            if len(filenames) != len(set(filenames)):
                msg = f'Duplicate filenames: {str(filenames)} in {script_name} '\
                'detected. you need to change the output filename for '\
                f'script {script_name} to something else.'
                raise RuntimeError(msg)
            if len(self.csv_writers) == 0:
                for fname in filenames:
                    path = os.path.join(self.log_dir, fname+'.csv')
                    self.csv_writers[fname] = CSVWriter(path, append)
            for i, fname in enumerate(fname_to_update_list):
                for row_dict in ana_output[script_name][fname]:
                    self.csv_writers[fname].append(row_dict)

    def log(self, iteration):
        '''
        Generate analysis tools iteration log. This is a separate logging
        operation from the subroutines in analysis.producers.loggers.

        Parameters
        ----------
        iteration : int
            Current iteration number
        '''
        row_dict = {'iteration': iteration}
        row_dict.update(self.logger_dict)
        self.logger.append(row_dict)


    def _set_iteration(self, dataset):
        '''
        Sets maximum number of iteration given dataset
        and max_iteration input.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Torch dataset containing images.
        '''
        if self.max_iteration == -1:
            self.max_iteration = len(dataset)
        assert self.max_iteration <= len(dataset)

    @staticmethod
    def _pixel_to_cm(arr, meta):
        '''
        Converts tensor pixel coordinates to detector coordinates

        Parameters
        ----------
        arr : np.ndarray
            Tensor of which to convert the coordinate columns
        meta : np.ndarray
            Metadata information to operate the translation
        '''
        arr[:, COORD_COLS] = pixel_to_cm(arr[:, COORD_COLS], meta)
        return arr
