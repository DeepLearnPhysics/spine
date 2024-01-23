import os, time, glob, yaml
import subprocess as sc
import numpy as np
import torch

from warnings import filterwarnings
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

from .iotools.factories import loader_factory

from .utils import cycle

from .trainval import TrainVal

# If MinkowskiEngine is available, load it with the right number of threads
try:
    if os.environ.get('OMP_NUM_THREADS') is None:
        os.environ['OMP_NUM_THREADS'] = '16'
    import MinkowskiEngine as ME
except:
    print('MinkowskiEngine could not be found, cannot run dependant models')


@dataclass
class Handlers:
    '''
    Simple dataclass that holds all the necessary information
    to run the training.
    '''
    cfg: dict               = None
    data_io: DataLoader     = None
    data_io_iter: iter      = None
    trainval: TrainVal      = None

    def keys(self):
        return list(self.__dict__.keys())

def run(cfg):
    '''
    Execute a model in one or more processes

    Parameters
    ----------
    cfg : dict
        IO, model and training/inference configuration
    '''
    # Process the configuration
    cfg = process_config(**cfg)

    # Check if we are running in training or inference mode
    assert 'train' in cfg['trainval'], \
            'Must specify the `train` parameter in the `trainval` block'
    train = cfg['trainval']['train']

    # Launch the training/inference process
    if not cfg['trainval']['distributed']:
        run_single(0, cfg, train)
    else:
        world_size = torch.cuda.device_count() \
                if torch.cuda.is_available() else 1
        torch.multiprocessing.spawn(run_single,
                args = (cfg, train, world_size,), nprocs = world_size)


def run_single(rank, cfg, train, world_size = None):
    '''
    Execute a model on a single process

    Parameters
    ----------
    rank : int
        Process rank
    cfg : dict
        IO, model and training/inference configuration
    train : bool
        Whether to train the network or simply execute it
    world_size : int, optional
        How many processes are being run in parallel
    '''
    # Treat distributed and undistributed training/inference differently
    if world_size is None:
        train_single(cfg) if train else inference_single(cfg)
    else:
        setup_ddp(rank, world_size)
        train_single(cfg, rank) if train else inference_single(cfg, rank)
        destroy_process_group()


def train_single(cfg, rank = 0):
    '''
    Train a model in a single process

    Parameters
    ----------
    cfg : dict
        IO, model and training/inference configuration
    rank : int, default 0
        Process rank
    '''
    # Prepare the handlers
    handlers = prepare(cfg, rank)

    # Run the training process
    handlers.trainval.run()

def inference_single(cfg, rank = 0):
    '''
    Execute a model in inference mode in a single process

    Parameters
    ----------
    cfg : dict
        IO, model and training/inference configuration
    rank : int, default 0
        Process rank
    '''
    # Prepare the handlers
    handlers = prepare(cfg, rank)

    # Find the set of weights to run the inference on
    preloaded = os.path.isfile(self.model_path)
    weights = sorted(glob.glob(self.model_path))
    if not preloaded and len(weights):
        # TODO: use logger
        print('Looping over {len(weights)} set of weights:')
        for w in weights: print('  -',w)
    elif not len(weights):
        weights = [None]

    # Loop over the weights, run the inference loop
    for weight in weights:
        if weight is not None and not preloaded:
            handlers.load_weights(model_path)
            handlers.make_diretories()

        handlers.trainval.run()


def prepare(cfg, rank = 0):
    '''
    Prepares high level API handlers, namely the torch DataLoader (and an
    iterator) and a TrainVal instance.

    Parameters
    ----------
    cfg : dict
        IO, model and training/infernece configuration
    rank : int, default 0
        Process rank

    Returns
    -------
    Handlers
        Handler instances needed for training/inference
    '''
    # Initialize the handlers
    handlers = Handlers()

    # If there is no `trainval` block, treat config as data loading config.
    # Otherwise, intiliaze the train/validation class
    if 'trainval' not in cfg:
        # Instantiate the data loader
        handlers.data_io = loader_factory(**cfg['iotool'])

        # Instantiate a cyclic iterator over the dataloader
        handlers.data_io_iter = iter(cycle(handlers.data_io))

    else:
        # Instantiate the training/inference object
        handlers.trainval = TrainVal(**cfg, rank = rank)

        # Expose the dataloader
        handlers.data_io = handlers.trainval.loader

        # Instantiate a cyclic iterator over the dataloader
        handlers.data_io_iter = handlers.trainval.loader_iter

    return handlers


def process_config(iotool, model = None, trainval = None, verbose = True):
    '''
    Do all the necessary cross-checks to ensure the that the configuration
    can be used. Parse the necessary arguments to make them useable downstream.
    Modifies the input in place.

    Parameters
    ----------
    iotool : dict
        I/O configuration dictionary
    model : dict, optional
        Model configuration dictionary
    trainval : dict, optional
        Training/inference configuration dictionary
    verbose : bool, default True
        Whether or not to print configuration
        TODO: Make this a general verbosity setting for the whole process
        TODO: and make it an enumerator rather than a boolean
    '''
    # Convert the gpus parameter to a list of GPU IDs
    distributed = False
    world_size = 1
    if trainval is not None and 'gpus' in trainval:
        # Check that the list of GPUs is a comma-separated string
        for val in trainval['gpus'].split(','):
            assert val.isdigit(), 'The `gpus` parameter must ' \
                    'be specified as a comma-separated string of numbers.' \
                    f'instead, got: {trainval["gpus"]}'

        # Set GPUs visible to CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = trainval['gpus']

        # Convert the comma-separated string to a list
        trainval['gpus'] = list(range(len(trainval['gpus'].split(','))))
        world_size = len(trainval['gpus'])

        # If there is more than one GPU, must distribute
        if world_size > 1:
            trainval['distributed'] = True
        elif 'distributed' not in trainval:
            trainval['distributed'] = False
        distributed = trainval['distributed']

    # If the seed is not set for the sampler, randomize it
    if 'sampler' in iotool:
        if 'seed' not in iotool['sampler'] or iotool['sampler']['seed'] < 0:
            current = int(time.time())
            if not distributed:
                iotool['sampler']['seed'] = current
            else:
                iotool['sampler']['seed'] = \
                        [current + i for i in range(world_size)]
        else:
            if distributed:
                seed = int(iotool['sampler']['seed'])
                iotool['sampler']['seed'] = \
                        [seed + i for i in range(world_size)]

    # If the seed is not set for the training/inference process, randomize it
    if trainval is not None:
        if 'seed' not in trainval or trainval['seed'] < 0:
            trainval['seed'] = int(time.time())
        else:
            trainval['seed'] = int(trainval['seed'])

    # Make sure that warnings are reported once
    # TODO: should be part of the verbosity configuration
    # TODO: deal with vebosity properly
    filterwarnings('once', message='Deprecated', category=DeprecationWarning)

    # Report configuration
    print('\nConfig processed at:', sc.getstatusoutput('uname -a')[1])
    print('\n$CUDA_VISIBLE_DEVICES="%s"\n' \
            % os.environ.get('CUDA_VISIBLE_DEVICES',None))

    cfg = {'iotool': iotool, 'model': model, 'trainval': trainval}
    print(yaml.dump(cfg, default_flow_style = None))

    return cfg


def apply_event_filter(handlers, event_list):
    '''
    Reinstantiate data loader with a restricted set of events.

    Parameters
    ----------
    handlers : Handlers
        Handler instances needed for training/inference
    event_list : list
        List of events to load
    '''
    # Instantiate DataLoader
    handlers.cfg['iotool']['event_list'] = event_list
    handlers.data_io = loader_factory(**handlers.cfg['iotool'])

    # IO iterator
    handlers.data_io_iter = iter(cycle(handlers.data_io))


def setup_ddp(rank, world_size):
    '''
    Sets up the DistributedDataParallel environment
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    init_process_group(backend = 'nccl', rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)
