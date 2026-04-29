# SPINE Docker Container

This directory contains the Docker container definition for SPINE (Scalable Particle Imaging with Neural Embeddings).

## Container: `ghcr.io/deeplearnphysics/spine:<release>`

The recommended way to run SPINE is to use the image published for each SPINE release. Prefer explicit release tags for reproducibility. When in doubt, use `latest`, or omit the tag entirely in Docker-style image references, which is equivalent.

**Complete ML stack with GPU support**

- **Base Image**: `ghcr.io/deeplearnphysics/larcv2:2.3.4-ubuntu22.04` (ROOT + LArCV2)
- **Size**: ~5.7 GB as a pulled Apptainer/Singularity `.sif` image
- **Includes**: 
  - PyTorch 2.5.1 (CUDA 12.1)
  - MinkowskiEngine (4D sparse convolutions)
  - torch-geometric ecosystem (torch-scatter, torch-cluster, torch-geometric)
  - ROOT 6.30+ and LArCV2 2.3.4
  - OpT0Finder v1.0.0 for likelihood-based SBND/ICARUS flash matching
  - XRootD client with token authentication (for dCache streaming)
  - JupyterLab and the classic Jupyter Notebook interface for tutorials and interactive analysis
  - Basic terminal editors (`vim`, `nano`) for quick in-container inspection/debugging
  - SPINE with all dependencies
- **GPU Support**: Built for NVIDIA datacenter and workstation GPUs:
  - **Cluster / datacenter**: **V100** (7.0), **A100** (8.0), **H100/H200** (9.0)
  - **Consumer / workstation**: **RTX 20xx** (7.5), **RTX 30xx** (8.6), **RTX 40xx** (8.9)
- **Use Cases**:
  - Full ML training and inference
  - Processing ROOT/LArCV2 detector data  
  - Streaming data from dCache via XRootD (with token authentication)
  - GPU-accelerated reconstruction
- **Requirements**: NVIDIA GPU with CUDA 12.1+ support (compute capability ≥7.0)

## Quick Start

### Pull and Run

```bash
docker pull ghcr.io/deeplearnphysics/spine:latest

# Or pin to a specific release
docker pull ghcr.io/deeplearnphysics/spine:<release>
docker run --gpus all -v $(pwd):/workspace ghcr.io/deeplearnphysics/spine:<release> --help
```

`/workspace` is just the default working directory inside the container. To work on host files, bind-mount a host path there as shown above.

### Run Training Example

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/config:/workspace/config \
  -v $(pwd)/output:/workspace/output \
  ghcr.io/deeplearnphysics/spine:<release> \
  --config /workspace/config/train_uresnet.yaml \
  --source /workspace/data/training_data.root
```

### Interactive Shell

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  ghcr.io/deeplearnphysics/spine:<release> \
  bash
```

The image includes `vim` and `nano` for lightweight in-container editing during
tutorials, debugging sessions, or quick configuration changes.

### Run JupyterLab

```bash
docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  ghcr.io/deeplearnphysics/spine:<release> \
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

This launches the JupyterLab web UI with notebook support enabled. Open the
URL printed by Jupyter in your browser on the host machine.

### Run Classic Jupyter Notebook

```bash
docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  ghcr.io/deeplearnphysics/spine:<release> \
  jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

Use this if you specifically want the classic notebook UI rather than the Lab
interface.

### Run Python Script

```bash
docker run --gpus all -v $(pwd):/workspace \
  ghcr.io/deeplearnphysics/spine:<release> \
  python /workspace/my_script.py
```

## XRootD Support (dCache Streaming with Token Authentication)

The container includes XRootD client with token authentication support, enabling direct streaming of LArCV files from dCache without local copies.

### Basic XRootD Usage

```bash
# Process file directly from XRootD endpoint
docker run --gpus all \
  ghcr.io/deeplearnphysics/spine:<release> \
  --config config.yaml \
  --source root://dcache.example.com//path/to/file.root
```

### Using Token Authentication

**Method 1: Mount token file**
```bash
# Get your bearer token and save to file
export BEARER_TOKEN="your_token_here"

# Run with mounted token
docker run --gpus all \
  -v $(pwd)/bearer_token:/tmp/bearer_token:ro \
  -e BEARER_TOKEN_FILE=/tmp/bearer_token \
  ghcr.io/deeplearnphysics/spine:<release> \
  --source root://dcache.example.com//path/to/file.root
```

**Method 2: Pass token as environment variable**
```bash
docker run --gpus all \
  -e BEARER_TOKEN="your_token_here" \
  ghcr.io/deeplearnphysics/spine:<release> \
  --source root://dcache.example.com//path/to/file.root
```

### Verify XRootD Installation

```bash
# Check XRootD client is installed
docker run --rm \
  --entrypoint xrdcp \
  ghcr.io/deeplearnphysics/spine:<release> \
  --help

# Test access to XRootD endpoint (replace with your endpoint)
docker run --rm \
  --entrypoint xrdfs \
  ghcr.io/deeplearnphysics/spine:<release> \
  root://dcache.example.com/ ls /path/to/directory
```

### XRootD with Apptainer

```bash
# Pass token file
apptainer run --nv \
  --bind /path/to/bearer_token:/tmp/bearer_token:ro \
  --env BEARER_TOKEN_FILE=/tmp/bearer_token \
  spine_latest.sif \
  --source root://dcache.example.com//path/to/file.root

# Or pass token as environment variable
export BEARER_TOKEN="your_token_here"
apptainer run --nv \
  --env BEARER_TOKEN \
  spine_latest.sif \
  --source root://dcache.example.com//path/to/file.root

# Note: singularity command still works on many systems (legacy compatibility)
```

## Apptainer Usage (HPC Clusters)

Many HPC clusters don't allow Docker but support Apptainer (formerly Singularity). In that case, pull the exact same released SPINE image from GHCR and run it through Apptainer.

### Pull and Convert

```bash
# Apptainer will automatically convert from Docker
apptainer pull spine_latest.sif docker://ghcr.io/deeplearnphysics/spine:latest

# Or pin to a specific release
apptainer pull spine_<release>.sif docker://ghcr.io/deeplearnphysics/spine:<release>

# This creates spine_<release>.sif
# Note: 'singularity' command works too (legacy compatibility)
```

### Run with GPU Support

**Key difference**: Use `--nv` flag instead of Docker's `--gpus all`

```bash
# Run SPINE CLI
apptainer run --nv spine_<release>.sif --help

# Interactive shell
apptainer shell --nv spine_<release>.sif

# Execute Python script
apptainer exec --nv spine_<release>.sif python my_script.py

# Run with bound directories (like Docker -v)
apptainer run --nv \
  --bind /path/to/data:/workspace/data \
  --bind /path/to/config:/workspace/config \
  spine_<release>.sif \
  --config /workspace/config/train_uresnet.yaml
```

When pulled with Apptainer/Singularity, the fully built image is about **5.7 GB**.

### HPC SLURM Example

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Load Apptainer module (if required by your cluster)
# Some clusters may still use 'module load singularity'
module load apptainer

# Run training
apptainer exec --nv \
  --bind $PWD:/workspace \
  /path/to/spine_<release>.sif \
  spine --config /workspace/config/train.yaml
```

### Docker vs Apptainer Quick Reference

| Feature | Docker | Apptainer |
|---------|--------|----------------------|
| Pull image | `docker pull ghcr.io/deeplearnphysics/spine:<release>` | `apptainer pull docker://ghcr.io/deeplearnphysics/spine:<release>` |
| GPU flag | `--gpus all` | `--nv` |
| Mount volumes | `-v /host:/container` | `--bind /host:/container` |
| Interactive shell | `docker run -it --rm IMAGE bash` | `apptainer shell` |
| Run command | `docker run IMAGE cmd` | `apptainer exec cmd` |
| Root required | Yes (Docker daemon) | No (user-level) |
| Legacy name | N/A | Singularity (commands still work) |

## Building Locally

### Prerequisites
- Docker installed with GPU support
- Access to build from source
- **Note**: Container requires `linux/amd64` platform (larcv2 dependency)
  - On Apple Silicon: Docker Desktop with Rosetta emulation enabled
  - On Linux: Native build

### Build Commands

```bash
# Build container
cd docker
./build.sh

# Build and push to registry (requires authentication)
./build.sh --push
```

**Apple Silicon Users**: The container builds for `linux/amd64` due to the larcv2 dependency. Docker Desktop will use Rosetta 2 emulation. For best performance, build and run on Linux x86_64 machines.

### Build Arguments

The container supports build-time arguments:

```bash
# Use specific larcv2 version
docker build --build-arg LARCV2_VERSION=2.3.3 \
  --platform linux/amd64 \
  -t spine:latest \
  -f spine/Dockerfile ..

# Use specific MinkowskiEngine commit
docker build --build-arg ME_COMMIT=02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
  --platform linux/amd64 \
  -t spine:latest \
  -f spine/Dockerfile ..

# Customize GPU architectures (for specific hardware)
# Default: "7.0 7.5 8.0 8.6 8.9 9.0" (V100, A100, H100/H200, RTX 30xx/40xx)
docker build --build-arg LARCV2_VERSION=2.3.3 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0 9.0" \
  --platform linux/amd64 \
  -t spine:latest \
  -f spine/Dockerfile ..

# Override the bundled ICARUS PhotonLibrary used by OpT0Finder likelihood flash matching
docker build \
  --build-arg ICARUS_PHOTONLIB_URL="https://example.org/PhotonLibrary.root" \
  --build-arg ICARUS_PHOTONLIB_SHA256="<sha256>" \
  --platform linux/amd64 \
  -t spine:latest \
  -f spine/Dockerfile ..
```

By default, the image bundles the ICARUS PhotonLibrary from
`https://s3df.slac.stanford.edu/data/neutrino/OpT0Finder/dat/PhotonLibrary-20201209.root`
and verifies it with SHA256
`9047b736467a04866751810a2648c5b8b73d7647b86b401010f65c4d3325c610`.

## GPU Architecture Support

The container is built with support for multiple NVIDIA GPU architectures. The `TORCH_CUDA_ARCH_LIST` controls which architectures MinkowskiEngine is compiled for:

| Architecture | Compute | GPUs |
|--------------|---------|------|
| `7.0` | sm_70 | V100 |
| `7.5` | sm_75 | T4, Quadro RTX 4000/5000/6000/8000 |
| `8.0` | sm_80 | A10, A30, A100 |
| `8.6` | sm_86 | RTX 30xx series, RTX A2000/A3000/A4000/A5000/A6000 |
| `8.9` | sm_89 | L4, L40, RTX 4000 (Ada) |
| `9.0` | sm_90 | H100, H200 |

**Default**: All architectures are included (`7.0 7.5 8.0 8.6 8.9 9.0`) for maximum compatibility.

To build for specific GPUs only (faster builds, smaller binaries):
```bash
# Example: A100 and H100 only
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0 9.0" \
  --platform linux/amd64 \
  -f spine/Dockerfile ..
```
| Platform | Multi | linux/amd64 |
| Size | ~1GB | ~5.7GB pulled `.sif` |

## Development Workflow

### Testing Changes Locally

```bash
# Make changes to SPINE code
$ vim src/spine/model.py

# Rebuild container
$ cd docker
$ ./build.sh

# Test
$ docker run --gpus all --rm spine:latest python -c "from spine.model import UResNet; print('OK')"
```

### Installing Additional Packages

```bash
# For one-off testing, mount as volume and install in container
docker run --gpus all -it -v $(pwd):/workspace -v spine-pip-cache:/root/.cache/pip \
  ghcr.io/deeplearnphysics/spine:latest bash

# Inside container:
pip install my-package
```

For permanent additions, modify the Dockerfile.

## Troubleshooting

### GPU not detected

**Docker:**
```bash
# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If that fails, install nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

**Apptainer:**
```bash
# Verify Apptainer GPU support
apptainer exec --nv docker://nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If that fails, check if NVIDIA drivers are installed and --nv is supported
# Contact your HPC admin for Apptainer/Singularity configuration
```

### Check GPU compute capability

**Docker:**
```bash
# Verify your GPU is supported (compute capability ≥7.0)
docker run --rm --gpus all ghcr.io/deeplearnphysics/spine:latest python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'Supported: {props.major >= 7}')
"
```

**Apptainer:**
```bash
# Same check with Apptainer
apptainer exec --nv spine_latest.sif python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'Supported: {props.major >= 7}')
"
```

Supported GPUs include:
- **Datacenter**: V100, T4, A10, A30, A100, L4, L40, H100, H200
- **Workstation**: RTX 20xx, RTX 30xx, RTX 40xx, RTX A2000/A3000/A4000/A5000/A6000
- **Cloud**: T4, A10G, A100, L4, H100

### Import errors

**Docker:**
```bash
# Verify all imports work
docker run --rm ghcr.io/deeplearnphysics/spine:latest python -c "
import torch
import MinkowskiEngine as ME
import ROOT
import larcv
from spine.model import UResNet
print('All imports OK')
"
```

**Apptainer:**
```bash
# Same verification with Apptainer
apptainer exec spine_latest.sif python -c "
import torch
import MinkowskiEngine as ME
import ROOT
import larcv
from spine.model import UResNet
print('All imports OK')
"
```

### Permission issues with mounted volumes

**Docker:**
```bash
# Run as your user ID
docker run --gpus all --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  ghcr.io/deeplearnphysics/spine:latest --help
```

**Apptainer:**
```bash
# Apptainer runs as your user by default (no permission issues)
# This is one of the advantages of Apptainer on HPC systems
apptainer run --nv --bind $(pwd):/workspace spine_latest.sif --help
```

## Version Tags

Containers are tagged with semantic versioning:

- `latest`: Most recent release (tracks SPINE version tags)
- `X.Y.Z`: Specific SPINE version (e.g., `0.10.5`)

## Dependencies

The `spine:latest` container depends on:
- **`ghcr.io/deeplearnphysics/larcv2:2.3.3`** - Provides ROOT + LArCV2

The larcv2 container is maintained separately in the [DeepLearnPhysics/larcv2](https://github.com/DeepLearnPhysics/larcv2) repository.

## CI/CD

Containers are automatically built and published via GitHub Actions when:
- A new version tag is pushed (e.g., `v0.10.6`)
- Manual workflow dispatch

See [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml) for details.

## Support

For issues specific to:
- **SPINE package**: [GitHub Issues](https://github.com/DeepLearnPhysics/spine/issues)
- **LArCV2 base container**: [larcv2 repository](https://github.com/DeepLearnPhysics/larcv2)
- **Container builds**: [GitHub Actions logs](https://github.com/DeepLearnPhysics/spine/actions)
