# Dockerfile of qwenllm/qwen3-asr:cu128-sbsa (DGX Spark / GB10 Blackwell, aarch64)
#
# Target platform: NVIDIA DGX Spark
#   - CPU arch : aarch64 (Grace, ARM64 SBSA)
#   - GPU      : NVIDIA GB10 (Blackwell, compute capability 12.1 / sm_121)
#   - Driver   : CUDA 13.0 (nvidia-smi), runs CUDA 12.8 apps via backward compat
#   - OS       : Ubuntu 24.04
#
# Toolkit choice: CUDA 12.8, NOT 13.0. vLLM's prebuilt aarch64 wheel (and its
# torch) are CUDA 12 binaries (vllm._C links libcudart.so.12), so the whole
# stack -- nvcc, torch (cu128), vLLM, flash-attn -- is pinned to CUDA 12.8 for
# ABI consistency. GB10 (sm_121) runs the sm_120 (Blackwell-family) kernels plus
# PTX JIT; the 13.0 driver runs these 12.8 binaries without issue.
#
# NOTE: The base image tag resolves to the arm64/sbsa variant automatically
#       when built on an aarch64 host (multi-arch manifest).

ARG CUDA_VERSION=12.8.1
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    wget \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
    ffmpeg \
    ca-certificates \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN git lfs install

WORKDIR /data/shared/Qwen3-ASR

ENV MAX_JOBS=20
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache

# Build CUDA extensions only for the Blackwell target. CUDA 12.8's nvcc knows
# sm_120 (not sm_121); GB10's sm_121 runs the sm_120 SASS plus PTX JIT.
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV CUDA_HOME=/usr/local/cuda

# flash-attn's setup.py ignores TORCH_CUDA_ARCH_LIST and uses its own list
# (default "80;90;100;110;120" -> 6 arches). Restrict it to Blackwell only:
# with CUDA 12.8 arch "120" emits compute_120/sm_120 SASS (runs on GB10 sm_121)
# plus compute_120 PTX. Building one arch instead of six cuts the flash-attn
# compile time by ~5-6x.
ENV FLASH_ATTN_CUDA_ARCHS=120

# Use an isolated venv so pip fully owns the Python environment. Ubuntu 24.04's
# Debian-managed python3-* packages (pip, wheel, PyJWT, blinker, ...) ship
# without RECORD files, so pip cannot uninstall/upgrade them in the system
# environment; a clean venv sidesteps that entirely (and PEP 668).
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Pull CUDA 12.8 (aarch64/sbsa) PyTorch wheels so torch matches vLLM's cu12
# binary. This index also carries the pinned nvidia-*-cu12 runtime libraries.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128

ARG BUNDLE_FLASH_ATTENTION=true

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel packaging cmake ninja

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U "qwen-asr[vllm]" fastapi uvicorn python-multipart requests soundfile scipy websockets

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        MAX_JOBS=8 pip install -U flash-attn --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git; \
    fi

# Triton (bundled with torch cu128) detects GB10 as sm_121a and JIT-compiles
# vLLM's kernels with its own ptxas -- but the CUDA 12.8 ptxas does not know
# sm_121a (added in CUDA 12.9), so engine startup dies with:
#   ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'
# Install a 12.9 ptxas (which does know sm_121a) and point Triton at it. The
# 13.0 driver loads the resulting sm_121a cubins natively.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U "nvidia-cuda-nvcc-cu12==12.9.86"
ENV TRITON_PTXAS_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas

RUN rm -rf /root/.cache/pip

EXPOSE 80

ENV APP_CONFIG_PATH=/data/shared/Qwen3-ASR/config/runtime.json
ENV PYTHONPATH=/data/shared/Qwen3-ASR/src
CMD ["uvicorn", "fast_qwen_asr.api.http_server:app", "--host", "0.0.0.0", "--port", "8000"]
