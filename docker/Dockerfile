# susgrad -- reproduction image for both experiments.
#
# ONE Dockerfile serves both variants; only the PyTorch wheel index differs:
#
#   CPU (Linux, Windows/WSL2, Intel and Apple-Silicon macOS):
#     docker build -f docker/Dockerfile -t susgrad:cpu .
#
#   GPU (NVIDIA only -- needs the host driver + NVIDIA Container Toolkit):
#     docker build -f docker/Dockerfile -t susgrad:gpu \
#       --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 .
#
# The CUDA runtime comes from the torch wheels themselves (the nvidia-* pip
# packages), so the GPU image needs no CUDA base image -- only `--gpus all` at
# run time. cu130 is the index that carries the SAME torch version as the CPU
# wheels, so both variants compute identically; on a host driver too old for
# CUDA 13, use .../whl/cu128 instead (an older torch, still fine for this code).
#
# Apple-Silicon MPS is NOT available inside Docker (the container cannot reach
# Metal); on macOS the CPU image is the supported route, and `make ...` on the
# host is the way to use MPS.

# 3.12 by default: every torch wheel supports it. 3.11 and 3.14 also have wheels
# on both indexes, so PYTHON_VERSION can be moved either way.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# CPU wheels by default -- the smaller image, and the one that runs everywhere.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    PYTHON=python

# `make` is the documented entry point of the project, so the image ships it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing the source does not re-download torch.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX_URL}" torch torchvision \
    && pip install scikit-learn numpy click tqdm matplotlib pytest

COPY susgrad ./susgrad
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./
COPY docker/entrypoint.sh /usr/local/bin/susgrad-entrypoint

# torch/torchvision are already satisfied above, so this does not pull the
# (much larger) default-index wheels over the top of them.
RUN pip install --no-deps -e . \
    && chmod +x /usr/local/bin/susgrad-entrypoint

# Run as a non-root user so files written into the bind-mounted outputs/ are not
# root-owned on the host. uid 1000 matches the usual first Linux desktop user;
# override with `docker run --user "$(id -u):$(id -g)"` if yours differs.
RUN useradd --create-home --uid 1000 susgrad \
    && mkdir -p /app/outputs /app/data /app/trained_models \
    && chown -R susgrad:susgrad /app
USER susgrad

ENTRYPOINT ["susgrad-entrypoint"]
CMD ["help"]
