#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Fish Audio S2-Pro TTS Server — Startup Script
#
# Usage:
#   bash server/run.sh
#   bash server/run.sh --gpu-model 2 --gpu-codec 3
#
# Environment variables (optional overrides):
#   TTS_DEVICE_MODEL  — GPU for the LLM (default: cuda:0)
#   TTS_DEVICE_CODEC  — GPU for the codec (default: cuda:1)
#   TTS_PORT          — Server port (default: 3008)
#   TTS_HOST          — Server bind address (default: 0.0.0.0)
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment
source ~/CollegeX/bin/activate

# Add fish-speech to Python path
export PYTHONPATH="${PROJECT_ROOT}/fish-speech:${PYTHONPATH:-}"

# Default GPU assignments (can be overridden via env vars)
# Model on cuda:0 (~10.5GB), codec on cuda:1 (separate GPU, model is too large to share)
export TTS_DEVICE_MODEL="${TTS_DEVICE_MODEL:-cuda:0}"
export TTS_DEVICE_CODEC="${TTS_DEVICE_CODEC:-cuda:1}"
export TTS_PORT="${TTS_PORT:-3008}"
export TTS_HOST="${TTS_HOST:-0.0.0.0}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Fish Audio S2-Pro TTS Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Model GPU:    ${TTS_DEVICE_MODEL}"
echo "  Codec GPU:    ${TTS_DEVICE_CODEC}"
echo "  Port:         ${TTS_PORT}"
echo "  Host:         ${TTS_HOST}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "${PROJECT_ROOT}"

exec python -m server.main
