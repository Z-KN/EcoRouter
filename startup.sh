#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
PC_SERVER_PID=""

usage() {
  cat <<'EOF'
Usage:
  ./startup.sh [--serve-pc] [--] <ecorouter args>

Examples:
  ./startup.sh route --origin phone --prompt "What's the weather?"
  ./startup.sh --serve-pc run --origin pc --prompt "Summarize this note" --live-pc

Options:
  --serve-pc   Start the local X-Elite server before running EcoRouter.
  --           Stop parsing script options and pass the rest directly to EcoRouter.
EOF
}

cleanup() {
  if [[ -n "$PC_SERVER_PID" ]]; then
    kill "$PC_SERVER_PID" >/dev/null 2>&1 || true
  fi
}

if [[ ! -x "$VENV_PY" ]]; then
  echo "error: missing virtual environment at .venv; run 'python3 -m venv .venv && .venv/bin/pip install -e .' first." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

START_PC_SERVER=0
ECOROUTER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve-pc)
      START_PC_SERVER=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      ECOROUTER_ARGS+=("$@")
      break
      ;;
    *)
      ECOROUTER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ECOROUTER_ARGS[@]} -eq 0 ]]; then
  usage
  exit 1
fi

trap cleanup EXIT

if [[ "$START_PC_SERVER" -eq 1 ]]; then
  export XELITE_SERVER_ENDPOINT="${XELITE_SERVER_ENDPOINT:-http://localhost:8000}"
  "$VENV_PY" "$ROOT_DIR/x_elite_laptop_server/serve_qwen_vl.py" --host 0.0.0.0 --port 8000 &
  PC_SERVER_PID=$!
  sleep 1
fi

"$VENV_PY" -m ecorouter "${ECOROUTER_ARGS[@]}"