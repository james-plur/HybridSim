#!/usr/bin/env bash
# Smoke-check Mooncake / vLLM Store prerequisites. Exit 0 if usable, 1 otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ok=1

echo "== Mooncake alignment env check =="
echo "PYTHONHASHSEED=${PYTHONHASHSEED:-<unset> (set to 0 for reproducible hashes)}"

if command -v mooncake_master >/dev/null 2>&1; then
  echo "[ok] mooncake_master: $(command -v mooncake_master)"
else
  echo "[missing] mooncake_master"
  ok=0
fi

python3 - <<'PY' || ok=0
import importlib.util
mods = ["mooncake", "mooncake.store"]
for m in ("mooncake",):
    if importlib.util.find_spec(m) is None:
        print(f"[missing] python package: {m}")
        raise SystemExit(1)
print("[ok] python package: mooncake")
try:
    import vllm  # noqa: F401
    print("[ok] vllm importable")
except Exception as e:
    print(f"[warn] vllm not importable: {e}")
PY

CFG="${ROOT}/scripts/mooncake_config.tcp.json"
if [[ -f "$CFG" ]]; then
  echo "[ok] sample config: $CFG"
else
  echo "[missing] $CFG"
  ok=0
fi

if [[ "$ok" -eq 1 ]]; then
  echo "Env looks usable for Store alignment (start master, export MOONCAKE_CONFIG_PATH)."
  exit 0
fi
echo "Env incomplete — CI / hosts without Mooncake should skip real Store tests."
exit 1
