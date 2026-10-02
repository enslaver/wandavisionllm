#!/usr/bin/env bash
# Started by the com.litellm.proxy LaunchAgent. Secrets and runtime settings live in ~/.litellm/env
# (see litellm/env.example); LITELLM_PORT defaults to 4000.
set -a; source "$HOME/.litellm/env"; set +a
# LiteLLM listens on 0.0.0.0: never start it without a real master key.
case "${LITELLM_MASTER_KEY:-}" in
  ""|sk-change-me) echo "start.sh: set LITELLM_MASTER_KEY in ~/.litellm/env; refusing to start without auth" >&2; exit 1 ;;
esac
exec "$HOME/.local/bin/litellm" --config "$HOME/.litellm/config.yaml" --host 0.0.0.0 --port "${LITELLM_PORT:-4000}"
