#!/usr/bin/env python3
"""The RAG ingestion flow as ONE Klique pipeline.

    plan ──► chunk ──► embed_upsert ──► evaluate

  plan          resolve the dataset version (a given id, or the latest by name)
                and diff its file hashes against the hashes stored with the
                chunks in the vector store
  chunk         chunk only the added and modified documents
  embed_upsert  embed those chunks through the OpenAI-compatible embedding
                endpoint, upsert them, delete stale and removed chunks
  evaluate      run the question set that ships with the corpus and report
                hit@1, hit@k and MRR; fail the run below the threshold

Each step is a function step on its own compute resource. Step outputs travel
to the next step as artifacts.

    python pipeline/rag_ingest_pipeline.py                 # create + start on Klique
    python pipeline/rag_ingest_pipeline.py --local         # same functions, in-process

Running on Klique requires credentials (CLEARML_API_* or ~/clearml.conf), the
compute resources named below, and a way for the agents to get this code:
--repo (a Git URL the agents can clone) or --docker (an image with the code
installed). NOT yet run against a server; verify on first use.

Non-secret settings are pipeline parameters. Secrets (EMBED_API_KEY,
QDRANT_BEARER_TOKEN) are read from the step's environment, which on Klique is
filled from a vault.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROJECT = "RAG Example"
DATASET_NAME = "pellwood-docs"
CPU_QUEUE = "1XCPU"          # compute resource for every step; none needs a GPU

# Pipeline parameters and their defaults (editable after Clone, and the values
# a Trigger Manager entry overrides with "Args/<name>:<value>").
PARAMETERS = {
    "dataset_project": PROJECT,
    "dataset_name": DATASET_NAME,
    "dataset_id": "",                 # empty = latest version by name
    "embed_base_url": "",             # e.g. https://<gateway>/v1 (LLM gateway)
    "embed_model": "embed",           # gateway alias or served model name
    "qdrant_url": "",                 # route of the Qdrant instance
    "collection": "pellwood_docs",
    "chunk_size": "220",
    "chunk_overlap": "30",
    "top_k": "4",
    "min_hit_at_k": "0.9",
}
SETTING_KEYS = ("embed_base_url", "embed_model", "qdrant_url", "collection",
                "chunk_size", "chunk_overlap", "top_k")


# ---------------------------------------------------------------------- steps
# Each step imports what it needs inside the function, which is what a pipeline
# function step requires to be self-contained.

def step_plan(dataset_project, dataset_name, dataset_id,
              embed_base_url, embed_model, qdrant_url, collection, chunk_size, chunk_overlap, top_k):
    from rag.config import Settings
    from rag.datasets import get_version
    from rag.ingest import plan_changes

    settings = Settings.from_env(dict(
        embed_base_url=embed_base_url, embed_model=embed_model, qdrant_url=qdrant_url,
        collection=collection, chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k))
    version = get_version(dataset_project, dataset_name, dataset_id or None)
    plan = plan_changes(version.to_dict(), settings)
    print("dataset %s version %s: %d added, %d modified, %d removed, %d unchanged" % (
        plan["dataset_id"], plan["dataset_version"], len(plan["added"]),
        len(plan["modified"]), len(plan["removed"]), len(plan["unchanged"])))
    return plan


def step_chunk(plan, dataset_project, dataset_name,
               embed_base_url, embed_model, qdrant_url, collection, chunk_size, chunk_overlap, top_k):
    from rag.config import Settings
    from rag.datasets import local_copy
    from rag.ingest import chunk_changed

    settings = Settings.from_env(dict(
        embed_base_url=embed_base_url, embed_model=embed_model, qdrant_url=qdrant_url,
        collection=collection, chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k))
    folder = local_copy(dataset_project, dataset_name, plan["dataset_id"])
    chunks = chunk_changed(folder, plan, settings)
    print("chunked %d documents into %d chunks" % (
        len(plan["added"]) + len(plan["modified"]), len(chunks)))
    return chunks


def step_embed_upsert(chunks, plan,
                      embed_base_url, embed_model, qdrant_url, collection, chunk_size, chunk_overlap, top_k):
    import json
    from rag.config import Settings
    from rag.ingest import embed_and_upsert

    settings = Settings.from_env(dict(
        embed_base_url=embed_base_url, embed_model=embed_model, qdrant_url=qdrant_url,
        collection=collection, chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k))
    stats = embed_and_upsert(chunks, plan, settings)
    print("embed/upsert:", json.dumps(stats))
    return stats


def step_evaluate(plan, stats, dataset_project, dataset_name, min_hit_at_k,
                  embed_base_url, embed_model, qdrant_url, collection, chunk_size, chunk_overlap, top_k):
    import json
    from clearml import Task
    from rag.config import Settings
    from rag.datasets import local_copy
    from rag.evaluate import evaluate_retrieval, load_questions
    from rag.ingest import EVAL_FILE
    from rag.tracking import report_values

    settings = Settings.from_env(dict(
        embed_base_url=embed_base_url, embed_model=embed_model, qdrant_url=qdrant_url,
        collection=collection, chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k))
    folder = local_copy(dataset_project, dataset_name, plan["dataset_id"])
    metrics = evaluate_retrieval(load_questions(folder / EVAL_FILE), settings)
    task = Task.current_task()
    report_values(task, {k: v for k, v in metrics.items() if k != "not_top_ranked"})
    report_values(task, {k: v for k, v in stats.items()})
    for miss in metrics["not_top_ranked"]:
        print("not top ranked:", json.dumps(miss))
    if metrics["questions"] and metrics["hit_at_k"] < float(min_hit_at_k):
        # never let a failed quality check pass as a green step
        raise RuntimeError("retrieval hit@%d %.3f is below the threshold %.3f" % (
            metrics["top_k"], metrics["hit_at_k"], float(min_hit_at_k)))
    return metrics


# ------------------------------------------------------------------ run modes

def run_local(params: dict) -> dict:
    """Run the same four step functions in this process, as one tracked task.
    With CLEARML_OFFLINE_MODE=1 the task is recorded to disk instead of a server."""
    from rag.tracking import close_task, start_task

    settings = {k: params[k] for k in SETTING_KEYS}
    task = start_task(params["dataset_project"], "RAG Ingest (local run)", "data_processing",
                      dict(params))
    try:
        plan = step_plan(params["dataset_project"], params["dataset_name"],
                         params["dataset_id"], **settings)
        chunks = step_chunk(plan, params["dataset_project"], params["dataset_name"], **settings)
        stats = step_embed_upsert(chunks, plan, **settings)
        metrics = step_evaluate(plan, stats, params["dataset_project"], params["dataset_name"],
                                params["min_hit_at_k"], **settings)
    finally:
        close_task(task)
    return {"plan": {k: plan[k] for k in ("dataset_id", "dataset_version", "added",
                                           "modified", "removed", "unchanged")},
            "chunk_ids": [c["id"] for c in chunks], "stats": stats, "metrics": metrics}


def run_on_klique(params: dict, repo: str, branch: str, docker: str, queue: str) -> None:
    from clearml import PipelineController

    pipe = PipelineController(name="RAG Ingest Pipeline", project=PROJECT, version="1.0.0")
    pipe.set_default_execution_queue(queue)
    for name, default in params.items():
        pipe.add_parameter(name, default)

    settings = {k: "${pipeline.%s}" % k for k in SETTING_KEYS}
    common = dict(
        execution_queue=queue,
        repo=repo or None, repo_branch=branch or None, docker=docker or None,
        packages=str(ROOT / "pipeline" / "requirements.txt"), cache_executed_step=False)
    ds = dict(dataset_project="${pipeline.dataset_project}",
              dataset_name="${pipeline.dataset_name}")

    pipe.add_function_step(
        name="plan", function=step_plan, function_return=["plan"],
        function_kwargs=dict(dataset_id="${pipeline.dataset_id}", **ds, **settings), **common)
    pipe.add_function_step(
        name="chunk", function=step_chunk, function_return=["chunks"], parents=["plan"],
        function_kwargs=dict(plan="${plan.plan}", **ds, **settings), **common)
    pipe.add_function_step(
        name="embed_upsert", function=step_embed_upsert, function_return=["stats"],
        parents=["chunk"],
        function_kwargs=dict(chunks="${chunk.chunks}", plan="${plan.plan}", **settings), **common)
    pipe.add_function_step(
        name="evaluate", function=step_evaluate, function_return=["metrics"],
        parents=["embed_upsert"],
        function_kwargs=dict(plan="${plan.plan}", stats="${embed_upsert.stats}",
                             min_hit_at_k="${pipeline.min_hit_at_k}", **ds, **settings),
        monitor_metrics=[("Summary", "hit_at_k"), ("Summary", "mrr")], **common)

    # The controller runs where it is launched; each step is enqueued to its
    # compute resource. With the code in an agent-reachable repository, switch to
    # pipe.start(queue=...) so the controller task itself can be cloned by the
    # Trigger Manager and the Task Scheduler.
    if repo or docker:
        pipe.start(queue=queue)
    else:
        pipe.start_locally(run_pipeline_steps_locally=False)
    print("pipeline started.")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG ingestion pipeline")
    ap.add_argument("--local", action="store_true",
                    help="run the steps in this process instead of on Klique")
    ap.add_argument("--repo", default=os.environ.get("RAG_GIT_REPO", ""))
    ap.add_argument("--branch", default=os.environ.get("RAG_GIT_BRANCH", "main"))
    ap.add_argument("--docker", default=os.environ.get("RAG_PIPELINE_IMAGE", ""))
    ap.add_argument("--queue", default=CPU_QUEUE)
    ap.add_argument("--summary-file", type=Path, default=None)
    for name, default in PARAMETERS.items():
        ap.add_argument("--" + name.replace("_", "-"), default=None,
                        help="default: %r" % default)
    args = ap.parse_args()

    # CLI value, else (for settings) the environment via Settings, else default.
    params = {}
    for name, default in PARAMETERS.items():
        value = getattr(args, name)
        params[name] = default if value is None else value
    if args.local:
        from rag.config import Settings

        env = Settings.from_env().public()
        for key in SETTING_KEYS:
            if getattr(args, key) is None:
                params[key] = str(env[key])
        summary = run_local(params)
        print(json.dumps({k: summary[k] for k in ("plan", "stats", "metrics")}, indent=2))
        if args.summary_file:
            args.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        run_on_klique(params, args.repo, args.branch, args.docker, args.queue)


if __name__ == "__main__":
    main()
