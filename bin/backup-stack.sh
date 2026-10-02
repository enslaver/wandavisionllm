#!/bin/zsh
# backup-stack.sh [DEST] — copy the stack's live settings, scripts and state to DEST/<date>/.
# DEST defaults to $WANDAVISION_BACKUP_DIR, else ~/wandavision-backup. Rerun any time.
# Configs, scripts and code only: no model weights, caches or logs (manifests/ lists what to reinstall).
# The copy holds secrets (~/.litellm/env, ~/.wanda/token): keep DEST private and out of git.
set -u
DEST="${1:-${WANDAVISION_BACKUP_DIR:-$HOME/wandavision-backup}}/$(date +%Y-%m-%d)"
mkdir -p "$DEST"/{home,system,launchagents,manifests} && chmod 700 "$DEST"
R=(rsync -rlt --delete-excluded --exclude .DS_Store --exclude __pycache__ --exclude '*.pyc' --exclude node_modules --exclude '*.log' --exclude '*.log.[0-9]*' --exclude logs/)

copy() {  # copy <src> <dest-subdir> [rsync excludes...]
  local src=$1 sub=$2; shift 2
  [ -e "$src" ] || { echo "skip (missing) $src"; return; }
  mkdir -p "$DEST/$sub"
  if [ -d "$src" ]; then  # trailing slash: excludes like /models/ anchor at the folder itself
    "${R[@]}" "$@" "$src/" "$DEST/$sub/${src:t}/" && echo "ok   $src" || echo "FAIL $src"
  else
    "${R[@]}" "$@" "$src" "$DEST/$sub/" && echo "ok   $src" || echo "FAIL $src"
  fi
}

# --- routing stack: LiteLLM, llama-swap, hook state, Wanda
copy ~/.litellm        home
copy ~/.llama-swap     home
copy ~/.ultron         home --exclude /media/
copy ~/.wanda          home
copy ~/wanda           home
copy ~/.wandavision    home
# --- model servers: configs, tier scripts, tuning (no weights, caches or venvs)
copy ~/.mtplx          home --exclude /models/ --exclude /session-bank/ --exclude /metrics/
copy ~/.tensorfold     home --exclude /venv/
# --- system-level config
copy /opt/homebrew/etc/Caddyfile system
for p in ~/Library/LaunchAgents/{com.litellm.proxy,com.llama-swap,com.wanda.portal}.plist; do
  [ -e "$p" ] && cp -p "$p" "$DEST/launchagents/"
done
echo "ok   launchagents"

# --- manifests: what to re-download / reinstall
M="$DEST/manifests"
{ echo "# model weights not copied; sizes as of $(date)"; for d in ~/.mtplx/models ~/.cache/huggingface/hub; do
    echo "## $d"; du -sh "$d"/* 2>/dev/null; done; } > "$M/models.txt"
ls ~/.cache/huggingface/hub 2>/dev/null | sed -n 's/^models--//p' | sed 's/--/\//' > "$M/huggingface-repo-ids.txt"
~/.tensorfold/venv/bin/python -m pip freeze > "$M/tensorfold-venv-pip-freeze.txt" 2>/dev/null \
  || uv pip freeze -p ~/.tensorfold/venv/bin/python > "$M/tensorfold-venv-pip-freeze.txt" 2>/dev/null
uv tool list --show-with > "$M/uv-tools.txt" 2>/dev/null
uv pip freeze -p ~/.local/share/uv/tools/litellm/bin/python > "$M/litellm-pip-freeze.txt" 2>/dev/null
brew list --versions > "$M/brew.txt" 2>/dev/null
{ ~/.mtplx/bin/mtplx --version; ~/.local/bin/llama-swap --version; ~/.tensorfold/venv/bin/tensorfold --version; caddy version; sw_vers; } > "$M/versions.txt" 2>&1
launchctl list | grep -iE "litellm|llama|wanda|caddy|mtplx" > "$M/launchctl-running.txt"
echo "ok   manifests"

du -sh "$DEST" | awk '{print "total " $1 " at " $2}'
