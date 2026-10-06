#!/bin/bash
# billion-context (bili): optional context compression for every chat request LiteLLM sends upstream.
# Loopback-only proxy on 127.0.0.1:8787, BEHIND LiteLLM: with ~/.ultron/bili-mode = on, ultron_admit's
# deployment hook points each routed-tier / cloud/* chat request at /bili/openai/<upstream base>; bili
# compresses the history, then forwards to llama-swap or the cloud endpoint. Auth, admission, overflow
# and the hooks all stay LiteLLM's; clients change nothing.
# Not installed or not deployed: exits 0, so launchd doesn't retry it, and requests go direct.
set -euo pipefail

# Overrides (BILI_BIN, extra env for bili) must be loaded before anything reads them.
if [ -f "$HOME/.bili/env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOME/.bili/env"
  set +a
fi

# config.json comes from the repo (bili/config.json, filled in from litellm/tiers.conf by deploy.py).
export BILI_CONFIG_FILE="${BILI_CONFIG_FILE:-$HOME/.bili/config.json}"
BILI_BIN="${BILI_BIN:-$HOME/.billion-context/bin/bili}"

if [ ! -f "$BILI_CONFIG_FILE" ]; then
  echo "bili: config missing: $BILI_CONFIG_FILE (./deploy.py push bili); not starting" >&2
  exit 0
fi

if [ ! -x "$BILI_BIN" ]; then
  echo "bili: billion-context not installed ($BILI_BIN); not starting. Requests go direct." >&2
  echo "  Install: npm i -g --prefix ~/.billion-context billion-context (or set BILI_BIN in ~/.bili/env)," >&2
  echo "  then: launchctl kickstart gui/\$(id -u)/com.billion-context.bili" >&2
  exit 0
fi

# Zero-config routing: LiteLLM calls /bili/openai/http://127.0.0.1:8001/v1/... (or the cloud base);
# the providers in config.json only tell bili each upstream model's context window.
exec "$BILI_BIN" start --config "$BILI_CONFIG_FILE"
