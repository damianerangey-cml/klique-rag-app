#!/usr/bin/env python3
"""Re-run the ingestion pipeline when a new corpus version is published.

SDK equivalent of deploy/trigger-manager.json (the Trigger Manager catalog
app). Use one or the other, not both.

    python pipeline/trigger_reembed.py --pipeline-task-id <id of a RAG Ingest Pipeline run>

The trigger watches the dataset project for a published dataset whose name
matches and that carries the corpus tag, clones the pipeline controller task
and enqueues the clone. The pipeline resolves the latest version by name, so no
dataset id has to be passed. The nightly safety-net run is defined separately in
deploy/task-scheduler.json.

Requires Klique credentials and a pipeline controller task that agents can
re-run (code in a reachable repository or image). NOT yet run against a server.
"""
import argparse

from clearml.automation import TriggerScheduler

PROJECT = "RAG Example"
DATASET_NAME = "pellwood-docs"
DATASET_TAG = "rag-corpus"
SERVICES_QUEUE = "services"   # compute resource that runs pipeline controllers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-task-id", required=True)
    ap.add_argument("--queue", default=SERVICES_QUEUE)
    args = ap.parse_args()

    trigger = TriggerScheduler(pooling_frequency_minutes=3.0)
    trigger.add_dataset_trigger(
        name="re-embed on new corpus version",
        schedule_task_id=args.pipeline_task_id,
        schedule_queue=args.queue,
        trigger_project=PROJECT,
        trigger_name=DATASET_NAME,
        trigger_on_publish=True,
        trigger_required_tags=[DATASET_TAG],
        target_project=PROJECT,
    )
    # The trigger service is itself a long-running task on the services resource.
    trigger.start_remotely(queue=args.queue)


if __name__ == "__main__":
    main()
