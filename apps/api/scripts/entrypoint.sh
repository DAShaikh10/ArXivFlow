#!/bin/sh
# Container entrypoint: ensure the 2D atlas projection exists on the PVC, then serve the API.
#
# The atlas (data/embeddings/atlas.json) powers the client's map view. It is derived from the
# SPECTER2 embeddings and only changes when those change, so we build it once — on the first start
# where it is absent — and skip it on every subsequent start (the PVC persists it). The API serves
# fine without it (no clusters), so a build failure is logged but never blocks startup.
#
# `@author`: DAShaikh10
set -eu

DATA_DIR="${DATA_DIR:-/app/data}"
ATLAS_PATH="${DATA_DIR}/${ATLAS_PROJECTION_FILE:-embeddings/atlas.json}"

if [ ! -f "${ATLAS_PATH}" ]; then
  echo "[entrypoint] atlas not found at ${ATLAS_PATH} - building ..."
  python -m src.build_atlas || echo "[entrypoint] atlas build failed; starting API without it"
else
  echo "[entrypoint] atlas present at ${ATLAS_PATH} - skipping build."
fi

# exec so uvicorn becomes PID 1 and receives SIGTERM directly (clean pod shutdown).
exec uvicorn src.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
