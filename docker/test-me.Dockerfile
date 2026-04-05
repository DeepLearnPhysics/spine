# Test MinkowskiEngine build only
FROM --platform=linux/amd64 ghcr.io/deeplearnphysics/larcv2:2.3.3

WORKDIR /app

# Install OpenBLAS development headers (required for MinkowskiEngine)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install --no-cache-dir --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1

# Test MinkowskiEngine build
ARG ME_COMMIT=02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine && \
    cd /tmp/MinkowskiEngine && \
    git checkout ${ME_COMMIT} && \
    PIP_BREAK_SYSTEM_PACKAGES=1 python setup.py install --blas=openblas && \
    cd / && \
    rm -rf /tmp/MinkowskiEngine

# Test import
RUN python -c "import MinkowskiEngine as ME; print('MinkowskiEngine version:', ME.__version__)"
