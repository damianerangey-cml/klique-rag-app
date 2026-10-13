#!/usr/bin/env bash
# Local end-to-end run without Klique, a GPU or an external API.
# Starts the stub model endpoint, publishes corpus v1, runs the ingestion
# pipeline, asks a question, publishes the update batch, re-runs the pipeline
# and asks again. State goes to .local/demo (deleted at the start of each run).
#
#   scripts/run_local_e2e.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WORK=.local/demo
STUB_PORT=${STUB_PORT:-9000}
SERVICE_PORT=${SERVICE_PORT:-8000}
rm -rf "$WORK" && mkdir -p "$WORK"

export CLEARML_OFFLINE_MODE=1                       # task records go to disk
export CLEARML_CONFIG_FILE="$WORK/no-clearml.conf"  # never read a real config
export CLEARML_CACHE_DIR="$WORK/clearml"
export RAG_DATASET_BACKEND=local RAG_LOCAL_DATASET_ROOT="$WORK/datasets"
export QDRANT_PATH="$WORK/qdrant"
export EMBED_BASE_URL="http://127.0.0.1:$STUB_PORT/v1" LLM_BASE_URL="http://127.0.0.1:$STUB_PORT/v1"
unset QDRANT_URL

wait_for() { for _ in $(seq 60); do curl -sf "$1" >/dev/null && return 0; sleep 0.5; done; echo "timeout: $1"; exit 1; }
start_service() { $PY -m uvicorn service.app:app --port "$SERVICE_PORT" --log-level warning >"$WORK/service.log" 2>&1 & SERVICE_PID=$!; wait_for "http://127.0.0.1:$SERVICE_PORT/readyz"; }
ask() { curl -s "http://127.0.0.1:$SERVICE_PORT/query" -H 'content-type: application/json' -d "{\"question\": \"$1\"}" | $PY -c '
import json, sys
d = json.load(sys.stdin)
print("Q:", sys.argv[1]); print("A:", d["answer"])
for s in d["sources"]:
    print("   [%d] %-45s v%s  %s" % (s["n"], s["doc_path"], s["dataset_version"], "cited" if s["cited"] else ""))' "$1"; }
summary() { $PY -c '
import json, sys
d = json.load(open(sys.argv[1])); p, s, m = d["plan"], d["stats"], d["metrics"]
print("dataset version %s: %d added, %d modified, %d removed, %d unchanged" % (p["dataset_version"], len(p["added"]), len(p["modified"]), len(p["removed"]), len(p["unchanged"])))
for k in ("modified", "removed"):
    for path in p[k]: print("  %s: %s" % (k, path))
if len(p["added"]) < 10:
    for path in p["added"]: print("  added: %s" % path)
print("embedded %d chunks from %d documents, skipped %d documents, %d points in the store" % (s["embedded_chunks"], s["embedded_documents"], s["skipped_documents"], s["points_total"]))
print("retrieval check: %d questions, hit@1 %.2f, hit@%d %.2f, MRR %.2f" % (m["questions"], m["hit_at_1"], m["top_k"], m["hit_at_k"], m["mrr"]))' "$1"; }
stub_inputs() { curl -s "http://127.0.0.1:$STUB_PORT/stats" | $PY -c 'import json,sys; print("embedding endpoint received %d inputs since reset" % json.load(sys.stdin)["embedding_inputs"])'; }

$PY -m uvicorn stub.openai_stub:app --port "$STUB_PORT" --log-level warning >"$WORK/stub.log" 2>&1 &
STUB_PID=$!
trap 'kill $STUB_PID ${SERVICE_PID:-} 2>/dev/null || true' EXIT
wait_for "http://127.0.0.1:$STUB_PORT/healthz"

echo "== 1. publish corpus v1 and run the pipeline"
$PY pipeline/publish_docs.py --source corpus/base >"$WORK/publish-v1.log" 2>&1
$PY pipeline/rag_ingest_pipeline.py --local --summary-file "$WORK/ingest-v1.json" >"$WORK/ingest-v1.log" 2>&1
summary "$WORK/ingest-v1.json"; stub_inputs

echo; echo "== 2. run the pipeline again with no new version"
curl -s -X POST "http://127.0.0.1:$STUB_PORT/stats/reset" >/dev/null
$PY pipeline/rag_ingest_pipeline.py --local --summary-file "$WORK/ingest-rerun.json" >"$WORK/ingest-rerun.log" 2>&1
summary "$WORK/ingest-rerun.json"; stub_inputs

echo; echo "== 3. start the middleware and ask"
start_service
ask "How long do the batteries in a Sense S1 last?"
ask "How many API requests per minute are allowed per API key?"
kill $SERVICE_PID; wait $SERVICE_PID 2>/dev/null || true   # embedded Qdrant allows one process

echo; echo "== 4. publish the update batch and run the pipeline"
curl -s -X POST "http://127.0.0.1:$STUB_PORT/stats/reset" >/dev/null
$PY pipeline/publish_docs.py --source corpus/base --overlay corpus/update >"$WORK/publish-v2.log" 2>&1
$PY pipeline/rag_ingest_pipeline.py --local --summary-file "$WORK/ingest-v2.json" >"$WORK/ingest-v2.log" 2>&1
summary "$WORK/ingest-v2.json"; stub_inputs

echo; echo "== 5. ask again"
start_service
ask "How long do the batteries in a Sense S1 last?"
ask "How many API requests per minute are allowed per API key?"
ask "How many times is a failed webhook delivery retried?"
echo; echo "offline task records: $(ls "$WORK"/clearml/offline/*.zip | wc -l | tr -d ' ') in $WORK/clearml/offline"
