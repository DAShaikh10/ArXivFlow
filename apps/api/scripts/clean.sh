#!/usr/bin/env bash
set -euo pipefail

# Delete .venv, Python cache, and uv lockfiles.
find . -type d \( -name .venv -o -name venv -o -name env -o -name __pycache__ \) -prune -exec rm -rf {} +
find . -type f -name uv.lock -delete
