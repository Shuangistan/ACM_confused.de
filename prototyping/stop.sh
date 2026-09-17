#!/usr/bin/env bash
# Stop the console server. Leaves Chrome alone — that is the user's window.
#
#   ./stop.sh            port 8000
#   PORT=9000 ./stop.sh

set -uo pipefail
cd "$(dirname "$0")" || exit 1

PORT="${PORT:-8000}"
PIDFILE=".server.pid"
stopped=""

# --- the process we started -------------------------------------------------
if [ -f "$PIDFILE" ]; then
  pid=$(cat "$PIDFILE")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    # Still there after five seconds: it is not going to shut down politely.
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null
      echo "Stopped pid ${pid} (forced)."
    else
      echo "Stopped pid ${pid}."
    fi
    stopped=1
  fi
  rm -f "$PIDFILE"
fi

# --- anything left holding the port -----------------------------------------
# A pid file can be lost — a crash, a machine restart, a copied directory —
# while the server keeps running. Without this the port stays occupied and
# start.sh refuses to run, with no obvious way forward.
leftovers=$(pgrep -f "uvicorn server:app --port ${PORT}" 2>/dev/null)
if [ -n "$leftovers" ]; then
  echo "$leftovers" | while read -r p; do
    kill "$p" 2>/dev/null && echo "Stopped stray pid ${p}."
  done
  stopped=1
fi

if [ -z "$stopped" ]; then
  echo "Nothing running on port ${PORT}."
fi
