"""Shared fixtures for the local end-to-end tests.

Everything runs on this machine: the stub model endpoint and the middleware are
started as uvicorn subprocesses on free ports, the vector store is Qdrant's
embedded on-disk mode in a temporary folder, datasets use the local backend and
the task SDK runs in offline mode. No Klique server, GPU or external API.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except Exception as exc:  # not listening yet
            last = exc
        time.sleep(0.3)
    raise RuntimeError("%s not ready after %ss (%s)" % (url, timeout, last))


class Env:
    """The local environment: paths, URLs and helpers to run the CLI entry points."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.stub_url = "http://127.0.0.1:%d" % free_port()
        self.service_url = "http://127.0.0.1:%d" % free_port()
        self.environ = dict(
            os.environ,
            CLEARML_OFFLINE_MODE="1",
            CLEARML_CONFIG_FILE=str(workdir / "no-clearml.conf"),   # never read a real config
            CLEARML_CACHE_DIR=str(workdir / "clearml"),
            RAG_DATASET_BACKEND="local",
            RAG_LOCAL_DATASET_ROOT=str(workdir / "datasets"),
            QDRANT_PATH=str(workdir / "qdrant"),
            EMBED_BASE_URL=self.stub_url + "/v1",
            LLM_BASE_URL=self.stub_url + "/v1",
            EMBED_MODEL="embed",
            LLM_MODEL="chat",
            RAG_COLLECTION="pellwood_docs_test",
            PYTHONPATH=str(ROOT),
        )
        for name in ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_BEARER_TOKEN",
                     "EMBED_API_KEY", "LLM_API_KEY"):
            self.environ.pop(name, None)
        self._procs = {}

    # -- processes
    def start(self, name: str, module: str, url: str) -> None:
        port = url.rsplit(":", 1)[1]
        log = open(self.workdir / (name + ".log"), "w")
        self._procs[name] = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", port,
             "--log-level", "warning"], cwd=ROOT, env=self.environ, stdout=log, stderr=log)
        wait_for(url + "/healthz")

    def stop(self, name: str) -> None:
        proc = self._procs.pop(name, None)
        if proc:
            proc.terminate()
            proc.wait(timeout=20)

    def stop_all(self) -> None:
        for name in list(self._procs):
            self.stop(name)

    # -- CLI entry points
    def run(self, *args: str) -> None:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=self.environ,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    def publish(self, *args: str) -> dict:
        out = self.workdir / "publish.json"
        self.run("pipeline/publish_docs.py", *args, "--summary-file", str(out))
        return json.loads(out.read_text())

    def ingest(self) -> dict:
        out = self.workdir / "ingest.json"
        self.run("pipeline/rag_ingest_pipeline.py", "--local", "--summary-file", str(out))
        return json.loads(out.read_text())

    # -- stub counters
    def stub_stats(self) -> dict:
        return httpx.get(self.stub_url + "/stats").json()

    def reset_stub(self) -> None:
        httpx.post(self.stub_url + "/stats/reset")

    def ask(self, question: str) -> dict:
        r = httpx.post(self.service_url + "/query", json={"question": question}, timeout=60)
        assert r.status_code == 200, r.text
        return r.json()

    def index_state(self) -> dict:
        """{doc_path: {"hash", "point_ids"}} read straight from the vector store.
        The embedded store allows one process, so the middleware must be stopped."""
        from rag.clients import make_qdrant
        from rag.config import Settings
        from rag.store import indexed_state

        settings = Settings(qdrant_path=self.environ["QDRANT_PATH"],
                            collection=self.environ["RAG_COLLECTION"])
        client = make_qdrant(settings)
        try:
            return indexed_state(client, settings.collection)
        finally:
            client.close()


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("rag"))
    e.start("stub", "stub.openai_stub:app", e.stub_url)
    yield e
    e.stop_all()
