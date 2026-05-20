#### BASE STAGE
#### Installs moon.

FROM python:3.14-slim AS base
WORKDIR /app

# Install bash, curl, certificates, and git so moon can be fetched safely.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ca-certificates \
    git \
    xz-utils tar \
    && rm -rf /var/lib/apt/lists/*

# Grab the official precompiled uv binaries directly from Astral's image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install moon binary.
RUN curl -fsSL https://moonrepo.dev/install/moon.sh | bash
ENV PATH="/root/.moon/bin:/root/.proto/bin:$PATH"

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

#### FINAL STAGE
#### Runs the application.

FROM python:3.14-slim AS runner
WORKDIR /app

# Copy the pruned virtual environment.
COPY --from=build /app/packages/annotate/.venv /app/.venv

# Copy the pruned application sources.
COPY --from=build /app/packages/annotate/src /app/src

# Make sure we use the virtualenv.
ENV PATH="/app/.venv/bin:$PATH"
