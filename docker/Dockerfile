# Development image for the three generative models.
#
# Built by `make init` (which calls docker.py).  PyTorch and the CUDA runtime come
# from the official image, so the build is small and the CUDA version is pinned in
# exactly one place.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG USER_NAME=dev
ARG USER_ID=1000
ARG GROUP_ID=1000

# HOME lives in the mounted CONTAINER_DIR, so caches and dotfiles persist between
# container runs.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/container \
    TORCH_HOME=/container/.cache/torch \
    HF_HOME=/container/.cache/huggingface \
    XDG_CACHE_HOME=/container/.cache

# Small system dependencies: compilers for anything that still needs to build,
# libGL/X11 libraries for RDKit's drawing code, make for the in-container scripts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates make bash-completion \
        libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# An unprivileged user whose UID matches the host user, so that files created in the
# mounted directories are not owned by root on the host.
RUN groupadd -g "${GROUP_ID}" "${USER_NAME}" \
    && useradd -m -u "${USER_ID}" -g "${GROUP_ID}" -d /container -s /bin/bash "${USER_NAME}" \
    && mkdir -p /container /workspace/code /workspace/data /workspace/logs \
    && chown -R "${USER_ID}:${GROUP_ID}" /container /workspace

WORKDIR /workspace/code
USER ${USER_NAME}

# Sanity check at build time: the runtime stack must import.
RUN python -c "import torch, hydra, omegaconf, numpy, rdkit; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

CMD ["bash"]
