#!/usr/bin/env bash
# Runs ON the Mac (./deploy.py push search copies this folder to ~/search/ and runs it). Installs/restarts the LaunchAgent.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
U=$(id -u); LA=~/Library/LaunchAgents/com.ultron.search.plist
mkdir -p ~/.search
sed "s#__HOME__#$HOME#g" ~/search/com.ultron.search.plist > "$LA"
plutil -lint "$LA" >/dev/null
launchctl bootout "gui/$U/com.ultron.search" 2>/dev/null || true
for i in $(seq 1 20); do launchctl print "gui/$U/com.ultron.search" >/dev/null 2>&1 || break; sleep 0.25; done
launchctl bootstrap "gui/$U" "$LA"
for i in $(seq 1 20); do curl -fsS -m 1 http://127.0.0.1:8792/healthz >/dev/null 2>&1 && break; sleep 0.5; done
curl -fsS -m 2 http://127.0.0.1:8792/healthz >/dev/null && echo "search: up on 127.0.0.1:8792" || { echo "search: NOT answering"; tail -20 ~/.search/service.log; exit 1; }
