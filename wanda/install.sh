#!/usr/bin/env bash
# Runs on the Mac after `./deploy.py push wanda` has copied this folder to ~/wanda/: installs the
# LaunchAgent (deploy.py already filled in __HOME__) and restarts Wanda.
set -euo pipefail
U=$(id -u); LA=~/Library/LaunchAgents/com.wanda.portal.plist
mkdir -p ~/.wanda ~/Library/LaunchAgents && chmod 700 ~/.wanda
cp ~/wanda/com.wanda.portal.plist "$LA"
launchctl bootout "gui/$U/com.wanda.portal" 2>/dev/null || true
# bootout returns before the service is gone; bootstrapping too early fails with "5: Input/output error"
for i in $(seq 1 20); do launchctl print "gui/$U/com.wanda.portal" >/dev/null 2>&1 || break; sleep 0.25; done
launchctl bootstrap "gui/$U" "$LA"
for i in $(seq 1 20); do curl -fsS -m 1 http://127.0.0.1:8790/healthz >/dev/null 2>&1 && break; sleep 0.5; done
curl -fsS -m 2 http://127.0.0.1:8790/healthz >/dev/null && echo "wanda: up on 127.0.0.1:8790" || { echo "wanda: NOT answering"; tail -20 ~/.wanda/wanda.log; exit 1; }
