#### BASE STAGE
#### Installs moon.

# Image contains bash, curl, python3.14 and uv.
FROM python:3.14-alpine AS base
WORKDIR /app

# Install bash, curl, certificates, and git so moon can be fetched safely.
RUN apk add --no-cache bash curl ca-certificates git

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
RUN moon docker scaffold crucible

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

FROM python:3.14-alpine AS runner
WORKDIR /app

# Copy the pruned virtual environment.
COPY --from=build /app/packages/crucible/.venv /app/.venv

# Copy the pruned application sources.
COPY --from=build /app/packages/crucible/src /app/src

# Make sure we use the virtualenv.
ENV PATH="/app/.venv/bin:$PATH"
