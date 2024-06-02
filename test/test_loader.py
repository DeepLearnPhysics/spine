"""Test that the loading of data using a full-fledged configuration."""

import os
import yaml
import time
import pytest

import numpy as np

from spine.io.factories import loader_factory
from spine.io.write import CSVWriter

MAX_ITER = 10


@pytest.mark.parametrize('cfg_file', ['test_loader.cfg'])
def test_loader(cfg_file, larcv_data, quiet=True, csv=False):
    """Tests the loading of data using a full IO configuration."""
    # Find the top-level directory of the package
    main_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(main_dir)

    # Fetch the configuration
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(main_dir, 'config', cfg_file)
    if not os.path.isfile(cfg_file):
        raise ValueError(f"Configuration file not found: {cfg_file}")

    # If requested, intialize a CSV output
    if csv:
        csv = CSVWriter('test.csv')

    # Initialize the loader
    with open(cfg_file, 'r', encoding='utf-8') as cfg_str:
        # Load configuration dictionary
        cfg = yaml.safe_load(cfg_str)

        # Update the path to the file
        print(cfg.keys())
        cfg['io']['loader']['dataset']['file_keys'] = larcv_data

    loader = loader_factory(**cfg['io']['loader'])

    # Loop
    tstart = time.time()
    tsum = 0.
    t0 = 0.
    for batch_id, data in enumerate(loader):
        titer = time.time() - tstart
        if not quiet:
            print('Batch', batch_id)
            for key, value in data.items():
                print('   ', key, np.shape(value))
            print('Duration', titer, '[s]')
        if batch_id < 1:
            t0 = titer
        tsum += (titer)
        if csv:
            csv.record(['iter', 't'], [batch_id, titer])
            csv.write()
        if (batch_id + 1) == MAX_ITER:
            break
        tstart = time.time()

    if not quiet:
        print('Total time:',tsum,'[s] ... Average time:',tsum/MAX_BATCH_ID,'[s]')
        if MAX_BATCH_ID>1:
            print('First iter:',t0,'[s] ... Average w/o first iter:',(tsum - t0)/(MAX_BATCH_ID-1),'[s]')
