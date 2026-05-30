#### BASE STAGE
#### Installs uv and moon.

# k8s NVIDIA L4 GPU Cluster supports CUDA 13.0.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS base
WORKDIR /app

# Default to NVIDIA L4 supported CUDA architecture. Adjust as needed for other GPU targets.
ARG CUDA_ARCH="8.9"

# Ensure compilation is restricted.
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
ENV DEBIAN_FRONTEND=noninteractive

# Install bash, curl, certificates, and git for the build layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ca-certificates \
    git \
    xz-utils \
    tar \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Grab the official precompiled uv binaries directly from Astral's image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install moon binary.
RUN curl -fsSL https://moonrepo.dev/install/moon.sh | bash
ENV PATH="/root/.moon/bin:/root/.proto/bin:$PATH"

# The project requires Python >=3.14, which Ubuntu's apt does not provide, so uv
# downloads a managed standalone CPython. Pin its install location and force
# managed-only so the venv's interpreter lives at a known, copyable path that the
# runner stage can reproduce (the venv's bin/python symlinks point here).
ENV UV_PYTHON_INSTALL_DIR=/opt/python
ENV UV_PYTHON_PREFERENCE=only-managed

#### SKELETON STAGE
#### Scaffolds repository skeleton structures.

FROM base AS skeleton

# Copy entire repository and scaffold.
COPY . .
RUN moon docker scaffold annotate

#### BUILD STAGE
#### Builds the project.

FROM base AS build
ENV UV_NO_DEV=1

# Copy workspace configs.
COPY --from=skeleton /app/.moon/docker/configs .

# Install toolchains and dependencies.
RUN moon docker setup

# Copy project sources.
COPY --from=skeleton /app/.moon/docker/sources .

# Prune extraneous dependencies.
RUN moon docker prune

# vLLM is excluded from the default dependencies (see pyproject.toml) so the Linux
# GitHub CI runners never try to install its multi-GB GPU build. It lives in the
# optional `llm` extra; pull it into the venv here — AFTER `moon docker prune`,
# since prune re-installs prod-only deps and would strip an extra added earlier.
# --frozen installs straight from uv.lock without re-resolving.
RUN uv sync --directory packages/annotate --extra llm --no-dev --frozen

#### FINAL STAGE
#### Runs the application.

# vLLM JIT-compiles kernels at startup: torch.compile/inductor + Triton need a
# host C/C++ compiler, and flashinfer compiles CUDA kernels (e.g. its sampling
# op) which require nvcc PLUS the CUDA dev headers/libs (curand.h, etc.). The
# -runtime base ships none of that, and apt-installing the toolkit piecemeal into
# it hits version conflicts (cuda-toolkit-13-0 wants libcublas>=13.1.1.3 vs the
# base's pinned 13.0.0.x). The -devel base ships a self-consistent full toolchain,
# which is the only reliable way to satisfy the runtime compilers.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS runner
WORKDIR /app

# Re-expose noninteractive and GPU configs for safety.
ENV DEBIAN_FRONTEND=noninteractive

# Ensure nvcc is discoverable by flashinfer's runtime build (it probes CUDA_HOME
# and /usr/local/cuda/bin). The -devel base provides nvcc + dev headers.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:$PATH"

# The CUDA -devel base already includes a host compiler, but install gcc/g++
# explicitly so Triton's launcher build can never regress on "Failed to find C
# compiler" regardless of base-image changes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv-managed CPython interpreter to the same absolute path it was
# installed at in the build stage. The venv created by uv contains symlinks
# (bin/python -> /opt/python/.../bin/python3); without the interpreter present
# here those symlinks dangle and `python` resolves to nothing on $PATH.
COPY --from=build /opt/python /opt/python

# Copy the pruned virtual environment.
COPY --from=build /app/packages/annotate/.venv /app/.venv

# Copy the pruned application sources.
COPY --from=build /app/packages/annotate/src /app/src

# Make sure we use the virtualenv.
ENV PATH="/app/.venv/bin:$PATH"

# Expose standard OpenAI API port for vLLM
EXPOSE 8000
