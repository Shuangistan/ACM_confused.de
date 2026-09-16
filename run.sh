#!/usr/bin/env bash
# Launch the logic database.
#
#   ./run.sh                        recruitment pack  -> http://127.0.0.1:8000/console
#   ./run.sh consumer_credit        real UCI credit data
#   ./run.sh benefits_eligibility   synthetic, with known hidden rules
#
# Two front ends share one governed engine:
#   /console   the control room  — screening, rule review, proofs, live metrics
#   /          the classic pages — dashboard, review queue, rule browser
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate iese
export LOGICDB_PACK="${1:-recruitment}"
export LOGICDB_DB="${2:-data/logicdb.sqlite}"
echo "pack=$LOGICDB_PACK  db=$LOGICDB_DB"
echo "  control room -> http://127.0.0.1:8000/console"
echo "  classic      -> http://127.0.0.1:8000/"
exec uvicorn logicdb.app:app --host 127.0.0.1 --port 8000 --reload
