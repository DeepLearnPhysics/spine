# SPINE Docker Container

This directory contains the Docker container definition for SPINE (Scalable Particle Imaging with Neural Embeddings).

## Container: `ghcr.io/deeplearnphysics/spine:latest`

**Complete ML stack with GPU support**

- **Base Image**: `ghcr.io/deeplearnphysics/larcv2:2.3.4-ubuntu22.04` (ROOT + LArCV2)
- **Size**: ~8-10GB
- **Includes**: 
  - PyTorch 2.5.1 (CUDA 12.1)
  - MinkowskiEngine (4D sparse convolutions)
  - torch-geometric ecosystem (torch-scatter, torch-cluster, torch-geometric)
  - ROOT 6.30+ and LArCV2 2.3.4
  - XRootD client with token authentication (for dCache streaming)
  - SPINE with all dependencies
- **GPU Support**: Built for NVIDIA datacenter and workstation GPUs:
  - **V100** (compute 7.0), **A10/A100** (8.0), **H100/H200** (9.0)
  - **RTX 30xx/40xx** series (8.6), plus T4 (7.5) and A30 (8.9)
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
docker run --gpus all -v $(pwd):/workspace ghcr.io/deeplearnphysics/spine:latest --help
```

### Run Training Example

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/config:/workspace/config \
  -v $(pwd)/output:/workspace/output \
  ghcr.io/deeplearnphysics/spine:latest \
  --config /workspace/config/train_uresnet.cfg \
  --source /workspace/data/training_data.h5
```

### Interactive Shell

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  ghcr.io/deeplearnphysics/spine:latest \
  bash
```

### Run Python Script

```bash
docker run --gpus all -v $(pwd):/workspace \
  ghcr.io/deeplearnphysics/spine:latest \
  python /workspace/my_script.py
```

## XRootD Support (dCache Streaming with Token Authentication)

The container includes XRootD client with token authentication support, enabling direct streaming of LArCV files from dCache without local copies.

### Basic XRootD Usage

```bash
# Process file directly from XRootD endpoint
docker run --gpus all \
  ghcr.io/deeplearnphysics/spine:latest \
  --config config.cfg \
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
  ghcr.io/deeplearnphysics/spine:latest \
  --source root://dcache.example.com//path/to/file.root
```

**Method 2: Pass token as environment variable**
```bash
docker run --gpus all \
  -e BEARER_TOKEN="your_token_here" \
  ghcr.io/deeplearnphysics/spine:latest \
  --source root://dcache.example.com//path/to/file.root
```

### Verify XRootD Installation

```bash
# Check XRootD client is installed
docker run --rm \
  --entrypoint xrdcp \
  ghcr.io/deeplearnphysics/spine:latest \
  --help

# Test access to XRootD endpoint (replace with your endpoint)
docker run --rm \
  --entrypoint xrdfs \
  ghcr.io/deeplearnphysics/spine:latest \
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

Many HPC clusters don't allow Docker but support Apptainer (formerly Singularity). You can use the Docker image directly:

### Pull and Convert

```bash
# Apptainer will automatically convert from Docker
apptainer pull docker://ghcr.io/deeplearnphysics/spine:latest

# This creates spine_latest.sif
# Note: 'singularity' command works too (legacy compatibility)
```

### Run with GPU Support

**Key difference**: Use `--nv` flag instead of Docker's `--gpus all`

```bash
# Run SPINE CLI
apptainer run --nv spine_latest.sif --help

# Interactive shell
apptainer shell --nv spine_latest.sif

# Execute Python script
apptainer exec --nv spine_latest.sif python my_script.py

# Run with bound directories (like Docker -v)
apptainer run --nv \
  --bind /path/to/data:/workspace/data \
  --bind /path/to/config:/workspace/config \
  spine_latest.sif \
  --config /workspace/config/train_uresnet.cfg
```

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
  /path/to/spine_latest.sif \
  spine --config /workspace/config/train.cfg
```

### Docker vs Apptainer Quick Reference

| Feature | Docker | Apptainer |
|---------|--------|----------------------|
| Pull image | `docker pull ghcr.io/deeplearnphysics/spine:latest` | `apptainer pull docker://ghcr.io/deeplearnphysics/spine:latest` |
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
```

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
| Size | ~1GB | ~10GB |

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
- **Datacenter**: V100, A100, A10, A30, H100, H200
- **Workstation**: RTX 3070/3080/3090, RTX 4070/4080/4090, RTX A5000/A6000
- **Cloud**: T4, A10G (AWS), A100 (all cloud providers)

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
