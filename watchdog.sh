#!/usr/bin/env bash
# Top-level overnight watchdog: keeps caffeinate + the auto-switcher alive. The switcher in turn keeps
# the directional-crawler grow loop alive (its own keep-alive). Two levels of self-healing so the crawl
# survives an unattended night even if a process hiccups. Only the drive staying mounted is on the user.
cd "$(dirname "$0")"
LOG=fasrc/logs/watchdog.log
exec >> "$LOG" 2>&1
while true; do
  pgrep -f 'caffeinate -i' >/dev/null || { caffeinate -i -s & echo "$(date '+%m-%d %H:%M:%S') restarted caffeinate"; }
  # disk safety: boundary renders are re-renderable from the drive; prune old ones if disk gets tight so
  # the grow loop never hits its 2GB hard-pause. JSON reads are already folded, so this loses nothing.
  free=$(df -m . | tail -1 | awk '{print $4}')
  if [ "$free" -lt 2800 ]; then
    echo "$(date '+%m-%d %H:%M:%S') disk ${free}MB low -> pruning boundary renders >20min old"
    find export/boundary_grow_renders -name '*.jpg' -type f -mmin +20 -delete 2>/dev/null || true
  fi
  if ! ps -eo args | grep -w 'auto_switch.sh' | grep -qv grep; then
    echo "$(date '+%m-%d %H:%M:%S') switcher down -> restarting"
    : > fasrc/logs/auto_switch.log
    nohup bash auto_switch.sh >/dev/null 2>&1 &
  fi
  sleep 120
done
