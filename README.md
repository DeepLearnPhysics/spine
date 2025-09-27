<h1 align="center">
<img src="https://github.com/DeepLearnPhysics/spine/blob/main/docs/source/_static/img/spine-logo-dark.png" alt='SPINE', width="400">
</h1><br>

[![PyPI version](https://badge.fury.io/py/spine-ml.svg)](https://badge.fury.io/py/spine-ml)
[![Python version](https://img.shields.io/pypi/pyversions/spine-ml.svg)](https://pypi.org/project/spine-ml/)
[![Documentation Status](https://readthedocs.org/projects/spine-ml/badge/?version=latest)](https://spine-ml.readthedocs.io/en/latest/?badge=latest)

The Scalable Particle Imaging with Neural Embeddings (SPINE) package leverages state-of-the-art Machine Learning (ML) algorithms -- in particular Deep Neural Networks (DNNs) -- to reconstruct particle imaging detector data. This package was primarily developed for Liquid Argon Time-Projection Chamber (LArTPC) data and relies on Convolutional Neural Networks (CNNs) for pixel-level feature extraction and Graph Neural Networks (GNNs) for superstructure formation. The schematic below breaks down the full end-to-end reconstruction flow.

![Full chain](https://github.com/DeepLearnPhysics/spine/blob/main/docs/source/_static/img/spine-chain-alpha.png)

## Installation

SPINE is now available on PyPI with flexible installation options to suit different needs:

### Quick Start (Recommended)

For most users, install with all optional dependencies:

```bash
pip install spine-ml[all]
```

### Installation Options

**1. Core Package (minimal dependencies)**
```bash
# Essential dependencies: numpy, scipy, pandas, PyYAML, h5py, numba
pip install spine-ml
```

**2. With Deep Learning Support**
```bash
# Adds PyTorch, MinkowskiEngine, and torch-geometric for neural networks
pip install spine-ml[model]
```

**3. With Visualization Tools**
```bash
# Adds plotly, matplotlib, seaborn for data visualization
pip install spine-ml[viz]
```

**4. Development Environment**
```bash
# Adds testing, formatting, and documentation tools
pip install spine-ml[dev]
```

**5. Everything**
```bash
# All optional dependencies included
pip install spine-ml[all]
```

### Special Dependencies

**MinkowskiEngine** (for sparse CNNs):
```bash
# Install via pip (recommended)
pip install MinkowskiEngine

# For CUDA support, ensure CUDA toolkit is installed first
# See: https://nvidia.github.io/MinkowskiEngine/quick_start.html
```

**LArCV** (for LArTPC data):
```bash
# Install from conda-forge
conda install -c conda-forge larcv
```

### Development Installation

For developers who want to work with the source code:
```bash
git clone https://github.com/DeepLearnPhysics/spine.git
cd spine
pip install -e .[dev]
```

To build and test packages locally:
```bash
# Build the package
./build_packages.sh

# Install locally built package
pip install dist/spine_ml-*.whl[all]
```

### Docker/Singularity (Alternative)

We provide containers with all dependencies pre-installed:
```bash
# Pull the latest container
docker pull deeplearnphysics/spine-ml
```

The container includes:
* `MinkowskiEngine` (sparse convolutions)
* `larcv2` (LArTPC data I/O)
* `torch` (deep learning framework)
* `torch_geometric` (graph neural networks)
* `numba` (just-in-time compilation)
* All Python scientific libraries

## Usage

### Command Line Interface

After installation, use the `spine` command:

```bash
# Run training/inference/analysis
spine --config config/train_uresnet.cfg --source /path/to/data.h5
```

### Python API

Basic example:
```python
# Necessary imports
import yaml
from spine.driver import Driver

# Load configuration file  
cfg_path = 'config/train_uresnet.cfg'  # or your config file
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Initialize driver class
driver = Driver(cfg)

# Execute model following the configuration regimen
driver.run()
```

* Documentation is available at https://spine-ml.readthedocs.io/.
* Tutorials and examples can be found in the documentation.

### Example Configuration Files

Example configurations are available in the `config` folder:

| Configuration name            | Model          |
| ------------------------------|----------------|
| `train_uresnet.cfg`           | UResNet alone  |
| `train_uresnet_ppn.cfg`       | UResNet + PPN  |
| `train_graph_spice.cfg`       | GraphSpice     |
| `train_grappa_shower.cfg`     | GrapPA for shower fragments clustering |
| `train_grappa_track.cfg`      | GrapPA for track fragments clustering |
| `train_grappa_inter.cfg`      | GrapPA for interaction clustering |

To switch from training to inference mode, set `trainval.train: False` in your configuration file.

Key configuration parameters you may want to modify:
* `batch_size` - batch size for training/inference
* `weight_prefix` - directory to save model checkpoints
* `log_dir` - directory to save training logs
* `iterations` - number of training iterations
* `model_path` - path to checkpoint to load (optional)
* `train` - boolean flag for training vs inference mode
* `gpus` - GPU IDs to use (leave empty '' for CPU)


For more information on storing analysis outputs and running custom analysis scripts, see the documentation on `outputs` (formatters) and `analysis` (scripts) configurations.

### Running A Configuration File

Basic usage with the `spine` command:
```bash
# Run training/inference directly
spine --config config/train_uresnet.cfg --source /path/to/data.h5

# Or run in background with logging
nohup spine --config config/train_uresnet.cfg --source /path/to/data.h5 > log_uresnet.txt 2>&1 &
```

You can load a configuration file into a Python dictionary using:
```python
import yaml
# Load configuration file
with open('config/train_uresnet.cfg', 'r') as f:
    cfg = yaml.safe_load(f)
```

### Reading a Log

A quick example of how to read a training log, and plot something
```python
import pandas as pd
import matplotlib.pyplot as plt
fname = 'path/to/log.csv'
df = pd.read_csv(fname)

# plot moving average of accuracy over 10 iterations
df.accuracy.rolling(10, min_periods=1).mean().plot()
plt.ylabel("accuracy")
plt.xlabel("iteration")
plt.title("moving average of accuracy")
plt.show()

# list all column names
print(df.columns.values)
```

### Recording network output or running analysis
Documentation for analysis tools and output formatting is available in the main documentation at https://spine-ml.readthedocs.io/.

## Repository Structure
* `bin` contains utility scripts for data processing
* `config` has example configuration files
* `docs` contains documentation source files  
* `src/spine` contains the main package code
* `test` contains unit tests using pytest

Please consult the documentation for detailed information about each component.

## Contributing

Before you start contributing to the code, please see the [contribution guidelines](CONTRIBUTING.md).

### Adding a new model

The SPINE framework is designed to be extensible. To add a new model:

1. **Data Loading**: Parsers exist for various sparse tensor and particle outputs in `spine.io.parse`. If you need fundamentally different data formats, you may need to add new parsers or collation functions.

2. **Model Implementation**: Add your model to the `spine.model` package. Include your model in the factory dictionary in `spine.model.factories` so it can be found by the configuration system.

3. **Configuration**: Create a configuration file in the `config/` folder that specifies your model architecture and training parameters.

Once these steps are complete, you should be able to train your model using the standard SPINE workflow.
