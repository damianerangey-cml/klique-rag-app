"""Unit tests for the pieces that need no running process."""
import json
import os
import re
import zipfile
from pathlib import Path

import pytest

from rag.chunking import chunk_document
from rag.config import SECRET_FIELDS, Settings
from rag.datasets import diff_manifests
from stub.openai_stub import embed, extractive_answer

ROOT = Path(__file__).resolve().parents[1]


def test_chunk_ids_are_deterministic_and_follow_content():
    text = (ROOT / "corpus/base/docs/sensors/s1-battery.md").read_text()
    a = chunk_document("docs/sensors/s1-battery.md", text, "hash-1", "1.0.0")
    b = chunk_document("docs/sensors/s1-battery.md", text, "hash-1", "1.0.0")
    c = chunk_document("docs/sensors/s1-battery.md", text, "hash-2", "1.0.1")
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert not {x["id"] for x in a} & {x["id"] for x in c}
    assert len(a) == 3 and [x["metadata"]["section"] for x in a] == [
        "Battery life", "Low battery warning", "Replacing batteries"]
    assert all(x["metadata"]["title"].startswith("Sense S1 battery") for x in a)
    assert a[0]["embed_text"].startswith("Sense S1 battery life")


def test_long_sections_are_split():
    text = "# Title\n\n## Section\n\n" + " ".join("Sentence number %d is here." % i for i in range(200))
    chunks = chunk_document("docs/x.md", text, "h", "v", chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 5 and all(len(c["text"].split()) <= 110 for c in chunks)


def test_diff_manifests():
    diff = diff_manifests({"a": "1", "b": "1", "c": "1"}, {"a": "1", "b": "2", "d": "1"})
    assert diff == {"added": ["d"], "modified": ["b"], "removed": ["c"], "unchanged": ["a"]}


def test_settings_overrides_never_touch_secrets(monkeypatch):
    monkeypatch.setenv("EMBED_API_KEY", "from-env")
    s = Settings.from_env({"embed_model": "other", "embed_api_key": "from-override",
                           "chunk_size": "100", "qdrant_url": ""})
    assert s.embed_model == "other" and s.chunk_size == 100
    assert s.embed_api_key == "from-env"
    assert not set(SECRET_FIELDS) & set(s.public())


def test_stub_embedding_is_deterministic_and_lexical():
    a, b = embed("battery life of the sensor"), embed("battery life of the sensor")
    assert a == b and abs(sum(x * x for x in a) - 1.0) < 1e-9
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
    assert dot(embed("sensor battery life"), embed("how long does the battery last")) > \
        dot(embed("sensor battery life"), embed("returns and refunds policy"))


def test_stub_chat_answers_from_the_matching_block():
    prompt = ("Sources:\n\n[1] source: docs/a.md (A)\n## Returns\n\nHardware can be returned "
              "within 45 days.\n\n[2] source: docs/b.md (B)\n## Warranty\n\nThe hub has a warranty "
              "of 3 years.\n\nQuestion: How long is the warranty on the hub?")
    assert extractive_answer(prompt) == "The hub has a warranty of 3 years. [2]"


# ---------------------------------------------------------------- deploy specs

# Official app packages are optional: set these to check the deploy specs against them.
APPS_DIR = Path(os.environ.get("KLIQUE_APPS_DIR", "apps-to-upload"))
QDRANT_CONF = Path(os.environ.get("KLIQUE_QDRANT_APP_CONF", "qdrant.app.conf"))
SPEC_APPS = {
    "embedding-deployment.json": "embedding_model_deployment",
    "chat-model-deployment.json": "model_deployment",
    "middleware.json": "container_launcher",
    "qdrant-container.json": "container_launcher",
    "qdrant-app.json": "qdrant",
    "ui-streamlit.json": "streamlit",
    "trigger-manager.json": "trigger_manager",
    "task-scheduler.json": "task_scheduler",
}


def _wizard(app: str):
    """(entry names, required entry names without a default) declared in the
    official app package. The packages are read, never modified."""
    if app == "qdrant":
        if not QDRANT_CONF.exists():
            pytest.skip("Qdrant app source not available on this machine")
        text = QDRANT_CONF.read_text(encoding="utf-8")
    else:
        archives = sorted(APPS_DIR.glob(app + "-*.zip"))
        if not archives:
            pytest.skip("official app packages not available on this machine")
        with zipfile.ZipFile(archives[-1]) as z:
            text = z.read(app + ".app.conf").decode("utf-8")
    from clearml.utilities.pyhocon import ConfigFactory

    names, required = set(), set()

    def walk(entries, top):
        for e in entries:
            if e.get("type", None) == "group":
                walk(e.get("item_template", []), top)
                continue
            names.add(e["name"])
            if top and e.get("required", False) and e.get("default", None) in (None, ""):
                required.add(e["name"])
            if e.get("type", None) == "list":
                walk(e.get("item_template", []), False)

    walk(ConfigFactory.parse_string(text)["wizard"]["entries"], True)
    return names, required


def _keys(value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(value, list):
        for item in value:
            _keys(item, out)
    return out


@pytest.mark.parametrize("spec,app", sorted(SPEC_APPS.items()))
def test_deploy_spec_uses_real_launch_form_fields(spec, app):
    doc = json.loads((ROOT / "deploy" / spec).read_text())
    names, required = _wizard(app)
    params = doc["launch_params"]
    unknown = _keys(params, set()) - names
    assert not unknown, "fields not in the %s launch form: %s" % (app, sorted(unknown))
    missing = {r for r in required if params.get(r) in (None, "")}
    assert not missing, "required launch-form fields not set: %s" % sorted(missing)


def test_deploy_specs_carry_no_secrets():
    for path in (ROOT / "deploy").glob("*.json"):
        doc = json.loads(path.read_text())
        assert doc["launch_params"].get("hf_token") is None
        for key in re.findall(r'"env_key"\s*:\s*"([^"]*)"', path.read_text()):
            assert not re.search(r"KEY|TOKEN|SECRET|PASSWORD", key), (path.name, key)


def test_apply_dry_run_resolves_placeholders(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("klique_apply", ROOT / "deploy/klique_apply.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("RAG_SERVICE_IMAGE", "registry.example/klique-rag-service:1.0.0")
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    request = module.load_spec(ROOT / "deploy/middleware.json", allow_missing=True)
    assert request["app"] == "containerlaunch" and request["task_name"] == "rag-middleware"
    assert request["launch_params"]["container"] == "registry.example/klique-rag-service:1.0.0"
    assert "RAG_QDRANT_URL" in request["unresolved"]
    with pytest.raises(SystemExit):
        module.load_spec(ROOT / "deploy/middleware.json")
