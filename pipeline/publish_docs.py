#!/usr/bin/env python3
"""Register the document corpus as a new immutable dataset version.

Runs as a lightweight CPU task, in the same shape as the labs'
prepare_datasets.py. The source folder is the source of truth (the kind of
folder a docs team exports from a wiki or a Git repository). Each run cuts a
NEW version; on Klique the version is a child of the previous one, so the
dataset records which files were added, modified and removed.

    python pipeline/publish_docs.py --source corpus/base
    python pipeline/publish_docs.py --source corpus/base --overlay corpus/update

--overlay applies an update batch on top of the source: files in the overlay
replace or add to the source, and paths listed in the overlay's REMOVED.txt are
dropped. Publishing the version is the event that starts the re-embed pipeline
(see deploy/trigger-manager.json).

Local run without a Klique server:
    RAG_DATASET_BACKEND=local CLEARML_OFFLINE_MODE=1 python pipeline/publish_docs.py ...
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import datasets  # noqa: E402
from rag.tracking import close_task, report_values, start_task  # noqa: E402

DATASET_PROJECT = "RAG Example"
DATASET_NAME = "pellwood-docs"


def stage(source: Path, overlay: Path | None) -> Path:
    """Build the folder that becomes the dataset version."""
    staging = Path(tempfile.mkdtemp(prefix="rag-corpus-")) / "files"
    shutil.copytree(source, staging)
    if overlay:
        removed = overlay / "REMOVED.txt"
        for item in sorted(overlay.rglob("*")):
            if item.is_file() and item != removed:
                target = staging / item.relative_to(overlay)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        if removed.exists():
            for line in removed.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    (staging / line.strip()).unlink(missing_ok=True)
    return staging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--overlay", type=Path, default=None)
    ap.add_argument("--dataset-project", default=DATASET_PROJECT)
    ap.add_argument("--dataset-name", default=DATASET_NAME)
    ap.add_argument("--summary-file", type=Path, default=None)
    args = ap.parse_args()

    staging = stage(args.source, args.overlay)
    # With the Klique backend the dataset version is itself a tracked task, and
    # the SDK allows one open task per process, so a separate run task is only
    # opened for the local backend.
    task = None
    if datasets.backend() == "local":
        task = start_task(args.dataset_project, "Publish Docs", "data_processing",
                          {"source": str(args.source), "overlay": str(args.overlay or "")})

    version = datasets.publish_version(staging, args.dataset_project, args.dataset_name)
    summary = {"dataset_id": version.id, "version": version.version, "files": len(version.files)}
    previous = datasets.previous_version(args.dataset_project, args.dataset_name)
    if previous:
        diff = datasets.diff_manifests(previous.files, version.files)
        summary.update({k: diff[k] for k in ("added", "modified", "removed")})
    report_values(task, {"files": len(version.files)})
    close_task(task)

    print("=" * 60)
    print("published dataset", args.dataset_name, "version", version.version, "id", version.id)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    if args.summary_file:
        args.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
