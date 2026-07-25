#!/usr/bin/env bash
# Start mooncake_master for Store alignment (RPC :50051).
set -euo pipefail
PORT="${MOONCAKE_MASTER_PORT:-50051}"
if ! command -v mooncake_master >/dev/null 2>&1; then
  echo "mooncake_master not found. Install: pip/uv install mooncake-transfer-engine" >&2
  exit 1
fi
echo "Starting mooncake_master on port ${PORT} (PYTHONHASHSEED=${PYTHONHASHSEED:-unset})"
exec mooncake_master --port "${PORT}"
