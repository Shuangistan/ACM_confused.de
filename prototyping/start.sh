#!/usr/bin/env bash
# Start the console and open it in Chrome.
#
#   ./start.sh            port 8000
#   PORT=9000 ./start.sh  somewhere else
#
# The page must be served rather than opened from disk: it calls /api/turn, and
# a file:// origin cannot reach it.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}/"
PIDFILE=".server.pid"
LOGFILE=".server.log"

# --- already up? ------------------------------------------------------------
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running (pid $(cat "$PIDFILE")) — opening ${URL}"
  google-chrome "$URL" >/dev/null 2>&1 &
  exit 0
fi
rm -f "$PIDFILE"

if curl -sf -m 1 "$URL" -o /dev/null 2>/dev/null; then
  echo "Something is already serving port ${PORT}. Use PORT=… or stop it first." >&2
  exit 1
fi

# --- environment ------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" 2>/dev/null
  conda activate iese 2>/dev/null || echo "note: conda env 'iese' not activated" >&2
fi

# The key is read here rather than by the app, so a missing key fails now with
# an explanation instead of failing later inside a request.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  for f in ../.env .env; do
    [ -f "$f" ] || continue
    key=$(grep -m1 -E '^(ANTHROPIC_API_KEY|API_KEY)=' "$f" | cut -d= -f2- | tr -d "\"' ")
    if [ -n "${key:-}" ]; then export ANTHROPIC_API_KEY="$key"; break; fi
  done
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "No API key. Set ANTHROPIC_API_KEY, or put API_KEY=… in ../.env" >&2
  exit 1
fi

# --- start ------------------------------------------------------------------
python -m uvicorn server:app --port "$PORT" --log-level warning \
  > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

printf 'Starting on %s' "$URL"
ready=""
for _ in $(seq 1 60); do
  if curl -sf -m 1 "$URL" -o /dev/null 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then break; fi
  printf '.'; sleep 0.5
done
echo

if [ -z "$ready" ]; then
  echo "Server did not come up. Last lines of ${LOGFILE}:" >&2
  tail -n 15 "$LOGFILE" >&2
  rm -f "$PIDFILE"
  exit 1
fi

echo "Running  pid $(cat "$PIDFILE")  log ${LOGFILE}"
if command -v google-chrome >/dev/null 2>&1; then
  google-chrome "$URL" >/dev/null 2>&1 &
  echo "Opened Chrome."
else
  echo "Chrome not found — open ${URL} yourself."
fi
echo "Stop with ./stop.sh"
