# Klique RAG example

A small, complete retrieval-augmented generation (RAG) application, built as a
worked example of end-to-end AI application development on Klique.

It has two halves:

- An **ingestion pipeline** that takes a new version of a document corpus,
  embeds only the documents that changed and updates the vector store, then
  checks retrieval quality.
- A **hosted application**: a vector store, a middleware service (FastAPI +
  LlamaIndex) and a chat UI (Streamlit), running as long-running containers.

The sample corpus is a set of 25 short product and support documents for a
fictional company (Pellwood, greenhouse sensors), plus an update batch that
modifies three documents, adds two and removes one. All of it is synthetic.

Everything runs locally without a Klique server, a GPU or an external API. A
deterministic stub stands in for the embedding and chat model endpoints.

## Layout

```
corpus/
  base/docs/**.md            25 documents (v1 of the corpus)
  base/eval/questions.jsonl  20 retrieval-check questions with the expected document
  update/                    update batch: 3 modified + 2 new documents, REMOVED.txt, new question set
rag/                         shared library (used by the pipeline, the middleware and the tests)
  config.py                  settings from environment variables only
  datasets.py                corpus as immutable dataset versions (Klique datasets, or a local folder backend)
  chunking.py                Markdown chunking with LlamaIndex node parsers, deterministic chunk ids
  clients.py                 embedding client, chat client, Qdrant client
  store.py                   what is indexed (path -> hash), deleting a document's chunks
  ingest.py                  plan_changes, chunk_changed, embed_and_upsert
  retrieval.py               retrieval through a LlamaIndex VectorStoreIndex
  evaluate.py                hit@1, hit@k, MRR
  tracking.py                task creation and metric reporting
pipeline/
  publish_docs.py            register the corpus as a new dataset version
  rag_ingest_pipeline.py     PipelineController pipeline: plan -> chunk -> embed_upsert -> evaluate
  trigger_reembed.py         SDK trigger: re-run the pipeline when a corpus version is published
service/app.py               middleware: /healthz, /readyz, /info, /query
ui/app.py                    Streamlit chat UI with sources
stub/openai_stub.py          deterministic OpenAI-compatible embedding + chat endpoint (local only)
deploy/
  *.json                     one launch spec per Klique catalog app instance
  klique_apply.py            sends a spec to the apps API (--dry-run prints it)
  custom-app/rag-chat.app.conf   optional: the UI packaged as its own catalog app
docker-compose.yml           local stack: Qdrant, stub, middleware, UI, ingest jobs
scripts/run_local_e2e.sh     local end-to-end run with readable output
tests/                       unit tests and the ordered end-to-end test
```

## How it works

### Ingestion

1. `publish_docs.py` registers the corpus folder as a new dataset version. On
   Klique this is a child of the previous version and it is published, which is
   the event the trigger listens for.
2. `rag_ingest_pipeline.py` runs four steps:
   - **plan**: read the version's `{path: content hash}` manifest and compare it
     with the hashes stored in the payload of the chunks in the vector store.
     The result lists added, modified, removed and unchanged documents.
   - **chunk**: split only the added and modified documents (by heading, then by
     sentence for long sections). Chunk ids are derived from path, content hash
     and position.
   - **embed_upsert**: embed those chunks through the OpenAI-compatible
     embedding endpoint, upsert them, then delete the old chunks of modified
     documents and all chunks of removed documents.
   - **evaluate**: run the question set that ships inside the dataset version
     (`eval/questions.jsonl`), report hit@1, hit@k and MRR, and fail the run
     below `min_hit_at_k`.

The vector store is the reference for what is embedded, so a run with no
changes embeds no documents, and a run after a failure picks up where the
store actually is.

### Query

1. The UI posts the question to the middleware (`POST /query`).
2. The middleware embeds the question, retrieves the top chunks from Qdrant
   through LlamaIndex, and builds a prompt with numbered sources.
3. It calls the chat model through an OpenAI-compatible base URL and returns the
   answer, the retrieved sources and which of them the answer cites.

Example response (local run, stub models):

```
Q: How long do the batteries in a Sense S1 last?
A: The Sense S1 runs on two AA lithium batteries. Expected battery life is 18 months
   at the default reporting interval of 5 minutes. [1]
   [1] docs/sensors/s1-battery.md   v1.0.0  cited
```

## Run it locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/python -m pytest          # unit tests + the full local flow
scripts/run_local_e2e.sh            # the same flow with readable output
```

Local mode uses:

| Piece | Local stand-in | Switch |
|---|---|---|
| Embedding and chat models | `stub/openai_stub.py` (hashed bag-of-words vectors, extractive answers) | `EMBED_BASE_URL`, `LLM_BASE_URL` |
| Vector store | Qdrant embedded on-disk mode | `QDRANT_PATH` (no `QDRANT_URL`) |
| Dataset versions | folder backend with SHA-256 manifests | `RAG_DATASET_BACKEND=local` |
| Task tracking | task SDK offline mode, records written to disk | `CLEARML_OFFLINE_MODE=1` |

The embedded Qdrant mode allows one process at a time, so the local scripts
stop the middleware while the pipeline writes. With a Qdrant server (compose or
Klique) the middleware stays up during a re-embed.

### With Docker

```bash
docker compose up -d --build              # Qdrant server, stub, middleware, UI
docker compose run --rm ingest            # corpus v1
docker compose run --rm ingest-update     # update batch; the middleware stays up
# UI: http://localhost:8501   middleware: http://localhost:8000/readyz
```

Host ports can be moved with `RAG_QDRANT_PORT`, `RAG_STUB_PORT`,
`RAG_SERVICE_PORT` and `RAG_UI_PORT`. To use real models, set `EMBED_BASE_URL`,
`EMBED_MODEL`, `LLM_BASE_URL`, `LLM_MODEL` and the two API keys in the
environment before `docker compose up`.

## Configuration

Environment variables only. No endpoint or key is written in code.

| Variable | Meaning | Default |
|---|---|---|
| `EMBED_BASE_URL`, `EMBED_MODEL`, `EMBED_API_KEY` | OpenAI-compatible embedding endpoint | `http://localhost:9000/v1`, `embed`, none |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | OpenAI-compatible chat endpoint | `http://localhost:9000/v1`, `chat`, none |
| `QDRANT_URL` | Qdrant server URL; unset means embedded mode | unset |
| `QDRANT_API_KEY` | Qdrant's own API key | unset |
| `QDRANT_BEARER_TOKEN` | bearer token for an authenticated gateway route | unset |
| `QDRANT_PATH` | folder for embedded mode | `.local/qdrant` |
| `RAG_COLLECTION` | collection name | `pellwood_docs` |
| `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` | chunk size and overlap, in words | 220, 30 |
| `RAG_TOP_K` | chunks retrieved per question | 4 |
| `RAG_DATASET_BACKEND` | `clearml` or `local` | `clearml` |
| `RAG_TRACKING` | `off` skips task creation | on |
| `RAG_API_URL`, `RAG_API_TOKEN` | UI: middleware address and optional bearer token | `http://localhost:8000`, none |

## On Klique

| Component | Runs as | Spec |
|---|---|---|
| Corpus | Klique dataset `pellwood-docs`, one immutable version per publish | `pipeline/publish_docs.py` |
| Ingestion pipeline | pipeline with four steps, each a task on a CPU compute resource | `pipeline/rag_ingest_pipeline.py` |
| Re-embed on new version | Trigger Manager app (dataset trigger, on publish) | `deploy/trigger-manager.json` |
| Nightly retrieval check | Task Scheduler app | `deploy/task-scheduler.json` |
| Embedding model | Embedding Model Deployment app | `deploy/embedding-deployment.json` |
| Chat model | vLLM Model Deployment app, or an existing endpoint behind a gateway alias | `deploy/chat-model-deployment.json` |
| Model access | LLM gateway: one base URL, aliases `embed` and `chat`, a virtual key per caller | configured by the tenant admin |
| Vector store | Qdrant app, or Qdrant on the Containerized Application Launcher | `deploy/qdrant-app.json`, `deploy/qdrant-container.json` |
| Middleware | Containerized Application Launcher (HTTP route to port 8000) | `deploy/middleware.json` |
| Chat UI | Streamlit app (from Git), behind the application gateway with SSO | `deploy/ui-streamlit.json` |

Order of a first deployment:

1. Build and push the middleware image (`service/Dockerfile`) to a registry the
   workers can pull from. Push this repository to a Git host the workers can clone.
2. Launch the embedding model and the chat model, and point the gateway aliases
   `embed` and `chat` at them.
3. Launch Qdrant.
4. Put `EMBED_API_KEY`, `LLM_API_KEY` (gateway virtual keys) and, where a route
   requires it, `QDRANT_BEARER_TOKEN` in a vault. Launch parameters are stored on
   the task, so secrets do not go in the specs.
5. Publish the corpus (`pipeline/publish_docs.py --source corpus/base`) and start
   the pipeline (`pipeline/rag_ingest_pipeline.py --repo <git url> --embed-base-url ... --qdrant-url ...`).
6. Launch the middleware and the UI.
7. Launch the Trigger Manager entry with the pipeline's task id. From then on,
   publishing a new corpus version re-runs the pipeline.

Each spec is applied with:

```bash
export RAG_CPU_QUEUE=... RAG_GPU_QUEUE=... RAG_SERVICE_IMAGE=...   # site-specific values
python deploy/klique_apply.py deploy/middleware.json --dry-run       # print the request
python deploy/klique_apply.py deploy/middleware.json                 # launch
```

`${ENV:NAME}` placeholders in a spec are filled from the environment at apply
time. The field names in every spec are checked against the launch forms of the
official app packages by `tests/test_units.py`.

### Status of the Klique path

The local flow is tested end to end. The Klique path is written against the SDK
and the launch forms but has not been run against a server. Before relying on it:

- `rag/datasets.py` (Klique backend), `pipeline/rag_ingest_pipeline.py` without
  `--local`, `pipeline/trigger_reembed.py` and `deploy/klique_apply.py` without
  `--dry-run` need a first run on a server. The pipeline's step definitions were
  checked by constructing the `PipelineController` in offline mode only.
- Pipeline steps need this code on the worker: `--repo` (Git) or `--docker`
  (an image built from `pipeline/Dockerfile`).
- How a route authenticates service-to-service calls (pipeline and middleware
  to Qdrant, UI to middleware) depends on the gateway configuration. The code
  supports a bearer token and Qdrant's own API key.
- The stub is not a model. Retrieval metrics measured against it show that the
  wiring works; they say nothing about a real embedding model on this corpus.
  Changing the embedding model requires a new collection and a full re-embed.

## Design choices

- **Qdrant** as the vector store: it runs as one container, Klique has a Qdrant
  catalog app, and its embedded mode lets the tests run the real client code
  without a server.
- **LlamaIndex** as the middleware library: the application is retrieval over
  documents, and LlamaIndex's node model (text + metadata + id) maps directly to
  chunks with citations. The same classes are used for chunking, the Qdrant
  store, the embedding client and the chat client, on both the ingestion and
  the query side. Only the core package and three small integrations are installed.
- **Streamlit** for the UI: it is a catalog app, it has chat components, and its
  test harness drives the UI without a browser.
- **Changed-only re-embedding from content hashes**, with the vector store as
  the record of what is indexed, instead of trusting a "last processed version"
  marker kept elsewhere.
- **Evaluation questions inside the dataset version**, so the check always
  matches the corpus it runs against.

All names, figures and model choices in the sample corpus and specs are illustrative.
