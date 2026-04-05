# SPINE Container
# Complete ML stack with PyTorch, MinkowskiEngine, ROOT, and LArCV2
# Extends the larcv2 base image (ROOT + LArCV2)
# Size: ~8-10GB
#
# Features:
# - GPU Support: V100, A100, H100/H200, RTX 20xx/30xx/40xx (compute 7.0-9.0)
# - XRootD client with token authentication for dCache streaming
# - PyTorch 2.5.1, MinkowskiEngine, torch-geometric
# - ROOT 6.30+ and LArCV2 2.3.4

# Allow larcv2 version to be overridden at build time
# Need to use the Ubuntu 22.04 base image to match CUDA 12.1 and PyTorch 2.5.1 requirements
ARG LARCV2_VERSION=2.3.4
FROM --platform=linux/amd64 ghcr.io/deeplearnphysics/larcv2:${LARCV2_VERSION}-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# - libopenblas-dev: BLAS headers for MinkowskiEngine compilation
# - ninja-build: Fast C++ build system for MinkowskiEngine
# - xrootd-*: XRootD client with token authentication for dCache streaming
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev \
        ninja-build \
        xrootd-client \
        xrootd-client-plugins \
        xrootd-scitokens-plugins \
        wget \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA 12.1 toolkit (compiler + headers for MinkowskiEngine build)
# Perfect version match with PyTorch 2.5.1+cu121 runtime
RUN wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-12-1 \
        cuda-cudart-dev-12-1 \
        cuda-libraries-dev-12-1 \
        cuda-nvtx-12-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.5.1 for CUDA 12.1 (perfect version match with toolkit)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1

# Configure CUDA environment
ENV CUDA_HOME=/usr/local/cuda-12.1

# Install torch-geometric ecosystem from pre-built wheels
RUN pip install --no-cache-dir \
    torch-scatter torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Build MinkowskiEngine for multiple GPU architectures
# Use a known-good base commit for torch 2.5.1 + CUDA 12.1 compatibility.
# Apply the upstream CUDA 12.x build fix from MinkowskiEngine PR #607 on top.
# During docker build, torch.cuda.is_available() is typically false, so force
# a CUDA build explicitly.
# V100 (7.0), A100 (8.0), H100/H200 (9.0), RTX 20xx (7.5), RTX 30xx (8.6), RTX 40xx (8.9)
ARG ME_COMMIT=02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
ARG ME_CUDA12_PR=607
ARG ME_MAX_JOBS=2
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine && \
    cd /tmp/MinkowskiEngine && \
    git checkout ${ME_COMMIT} && \
    git fetch origin pull/${ME_CUDA12_PR}/head:cuda12-fix && \
    git cherry-pick --no-commit cuda12-fix && \
    MAX_JOBS=${ME_MAX_JOBS} PIP_BREAK_SYSTEM_PACKAGES=1 python setup.py install --blas=openblas --force_cuda && \
    cd / && \
    rm -rf /tmp/MinkowskiEngine

# Copy SPINE project files
COPY pyproject.toml MANIFEST.in README.md ./
COPY src/ ./src/

# Install SPINE with all dependencies
RUN pip install --no-cache-dir .[all]

# Set working directory for user data
WORKDIR /workspace

# Default command shows help, but users can easily override
# docker run spine bash              -> interactive shell
# docker run spine python script.py  -> run Python script
# docker run spine --config cfg      -> run SPINE training
CMD ["spine", "--help"]
