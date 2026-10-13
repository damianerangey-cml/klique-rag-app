"""Document corpus as immutable dataset versions.

Two backends with the same interface:

  clearml  Klique datasets (clearml.Dataset). Each publish creates a child
           version of the previous one; file hashes come from the dataset's
           own file entries. This is the deployment path.
  local    A folder of versions with a manifest of SHA-256 hashes. Used for the
           local flow and the tests, where no Klique server is reachable.

Select with RAG_DATASET_BACKEND (default: clearml). The pipeline only needs
three things from a version: its id, its {relative path: hash} manifest and a
local copy of the files.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

DATASET_TAG = "rag-corpus"


@dataclass
class DatasetVersion:
    id: str
    name: str
    project: str
    version: str
    files: dict = field(default_factory=dict)   # relative posix path -> content hash

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "project": self.project,
                "version": self.version, "files": self.files}


def backend() -> str:
    return os.environ.get("RAG_DATASET_BACKEND", "clearml").lower()


def _local_root() -> Path:
    return Path(os.environ.get("RAG_LOCAL_DATASET_ROOT", ".local/datasets"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder_manifest(folder: Path) -> dict:
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): sha256_file(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def diff_manifests(old: dict, new: dict) -> dict:
    return {
        "added": sorted(p for p in new if p not in old),
        "modified": sorted(p for p in new if p in old and old[p] != new[p]),
        "removed": sorted(p for p in old if p not in new),
        "unchanged": sorted(p for p in new if p in old and old[p] == new[p]),
    }


# --------------------------------------------------------------------------- local

def _local_versions(project: str, name: str) -> list:
    base = _local_root() / project / name
    if not base.exists():
        return []
    return sorted((p for p in base.iterdir() if (p / "manifest.json").exists()),
                  key=lambda p: int(p.name[1:]))


def _local_load(vdir: Path) -> DatasetVersion:
    meta = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    return DatasetVersion(**meta)


def _local_publish(folder: Path, project: str, name: str) -> DatasetVersion:
    versions = _local_versions(project, name)
    number = len(versions) + 1
    vdir = _local_root() / project / name / ("v%d" % number)
    shutil.copytree(folder, vdir / "files")
    version = DatasetVersion(id="local-%s-v%d" % (name, number), name=name, project=project,
                             version="1.0.%d" % (number - 1), files=folder_manifest(vdir / "files"))
    (vdir / "manifest.json").write_text(json.dumps(version.to_dict(), indent=2), encoding="utf-8")
    return version


def _local_find(project: str, name: str, dataset_id: str | None) -> Path:
    versions = _local_versions(project, name)
    if not versions:
        raise SystemExit("No local dataset version for %s/%s. Run pipeline/publish_docs.py first."
                         % (project, name))
    if not dataset_id:
        return versions[-1]
    for vdir in versions:
        if _local_load(vdir).id == dataset_id:
            return vdir
    raise SystemExit("Local dataset version %r not found" % dataset_id)


# ------------------------------------------------------------------------- clearml

def _clearml_version(ds) -> DatasetVersion:
    files = {path: entry.hash for path, entry in ds.file_entries_dict.items()}
    return DatasetVersion(id=ds.id, name=ds.name, project=ds.project,
                          version=str(ds.version or ""), files=files)


def _clearml_publish(folder: Path, project: str, name: str) -> DatasetVersion:
    from clearml import Dataset

    try:
        parent = Dataset.get(dataset_project=project, dataset_name=name, only_completed=True)
        parents = [parent.id]
    except ValueError:
        parents = None  # first version
    ds = Dataset.create(dataset_name=name, dataset_project=project,
                        parent_datasets=parents, dataset_tags=[DATASET_TAG])
    # sync_folder records additions, modifications and removals against the parent.
    ds.sync_folder(str(folder))
    ds.upload()
    ds.finalize()
    # Publishing is the event the Trigger Manager listens for.
    ds.publish()
    return _clearml_version(ds)


def _clearml_get(project: str, name: str, dataset_id: str | None):
    from clearml import Dataset

    if dataset_id:
        return Dataset.get(dataset_id=dataset_id)
    return Dataset.get(dataset_project=project, dataset_name=name, only_completed=True)


# -------------------------------------------------------------------------- public

def publish_version(folder: Path, project: str, name: str) -> DatasetVersion:
    """Register the folder as a new immutable dataset version."""
    if backend() == "local":
        return _local_publish(Path(folder), project, name)
    return _clearml_publish(Path(folder), project, name)


def get_version(project: str, name: str, dataset_id: str | None = None) -> DatasetVersion:
    """A specific version by id, or the latest version by name."""
    if backend() == "local":
        return _local_load(_local_find(project, name, dataset_id))
    return _clearml_version(_clearml_get(project, name, dataset_id))


def previous_version(project: str, name: str) -> DatasetVersion | None:
    """The version before the latest one (local backend only; used for the
    publish summary)."""
    if backend() != "local":
        return None
    versions = _local_versions(project, name)
    return _local_load(versions[-2]) if len(versions) > 1 else None


def local_copy(project: str, name: str, dataset_id: str | None = None) -> Path:
    """Read-only local folder holding the files of a version."""
    if backend() == "local":
        return _local_find(project, name, dataset_id) / "files"
    return Path(_clearml_get(project, name, dataset_id).get_local_copy())
