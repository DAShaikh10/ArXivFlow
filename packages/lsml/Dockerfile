#### BASE STAGE
#### Installs moon.

# Image contains python3.14.
FROM python:3.14-slim AS base
WORKDIR /app

# Install bash, curl, certificates, and git so moon can be fetched safely.
# Also install build tools and python3-dev for any dependencies that require compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ca-certificates \
    git \
    xz-utils tar \
    gcc \
    g++ \
    make \
    python3-dev \
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
RUN moon docker scaffold lsml

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
COPY --from=build /app/packages/lsml/.venv /app/.venv

# Copy the pruned application sources.
COPY --from=build /app/packages/lsml/src /app/src

# Make sure we use the virtualenv.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1

EXPOSE 9090

CMD ["sh", "-c", "exec /app/.venv/bin/python -m uvicorn src.lib.models.server:app --host 0.0.0.0 --port $PORT --workers $WORKERS"]
