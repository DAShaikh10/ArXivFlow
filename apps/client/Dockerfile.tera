#### BASE STAGE
#### Installs moon + Bun.

FROM oven/bun:1 AS base
WORKDIR /app

# Install bash, curl, certificates, and git so moon can be fetched safely.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ca-certificates \
    git \
    xz-utils tar \
    && rm -rf /var/lib/apt/lists/*

# Install moon binary.
RUN curl -fsSL https://moonrepo.dev/install/moon.sh | bash
ENV PATH="/root/.moon/bin:/root/.proto/bin:$PATH"

#### SKELETON STAGE
#### Scaffolds repository skeleton structures.

FROM base AS skeleton

# Copy entire repository and scaffold.
COPY . .
RUN moon docker scaffold client

#### BUILD STAGE
#### Builds the Next.js standalone bundle.

FROM base AS build
ENV NEXT_TELEMETRY_DISABLED=1

# Copy the scaffolded manifests (apps/client/package.json, bun.lock, bunfig.toml).
COPY --from=skeleton /app/.moon/docker/configs .

# Install dependencies with Bun. moon has no Node/Bun toolchain (only Python/uv), so `moon docker
# setup` can't install them — do it directly. Before the source copy so the layer caches on the lock.
WORKDIR /app/apps/client
RUN bun install --frozen-lockfile

# Copy project sources.
WORKDIR /app
COPY --from=skeleton /app/.moon/docker/sources .

# Build the Next.js standalone bundle.
WORKDIR /app/apps/client
RUN bun --bun next build

#### FINAL STAGE
#### Runs the Next.js standalone server on Bun.

FROM oven/bun:1-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Non-root user is already provided by oven/bun:1-slim ("bun"). Switch to it
# after copying so the chowned files land with the right owner.
# With no monorepo root lockfile, Next's tracing root is apps/client, so the standalone bundle is
# flat: `server.js`, `.next/` and `node_modules/` sit at its top. Copied to /app, the server is at
# /app/server.js and static must land at /app/.next/static.
# No `public/` copy — this app ships no static assets under public/. Add one back (plus a `public/`
# dir) if that changes; Next's standalone output does not bundle it.
COPY --from=build --chown=bun:bun /app/apps/client/.next/standalone /app
COPY --from=build --chown=bun:bun /app/apps/client/.next/static /app/.next/static

USER bun
EXPOSE 3000
CMD ["bun", "server.js"]
