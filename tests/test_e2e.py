"""Local end-to-end flow, in order:

  1. publish corpus v1, run the ingestion pipeline, check the index and metrics
  2. run the pipeline again: no document is embedded
  3. start the middleware, ask questions, check answers and cited sources
  4. drive the Streamlit UI headlessly against the middleware
  5. publish the update batch, run the pipeline: only changed documents are
     embedded, unchanged chunks keep their ids, removed documents disappear
  6. ask again: answers reflect the update

The tests share one environment and depend on this order.
"""
import json
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE_DOCS = sorted(p.relative_to(ROOT / "corpus/base").as_posix()
                   for p in (ROOT / "corpus/base/docs").rglob("*.md"))
MODIFIED = ["docs/cloud/api-rate-limits.md", "docs/cloud/data-retention.md",
            "docs/sensors/s1-battery.md"]
ADDED = ["docs/cloud/webhooks.md", "docs/sensors/c4-co2-sensor.md"]
REMOVED = ["docs/support/legacy-h1-migration.md"]

state = {}  # values carried between the ordered tests


def cited_docs(result: dict) -> list:
    return [s["doc_path"] for s in result["sources"] if s["cited"]]


# ------------------------------------------------------------ 1. first ingest

def test_first_ingest_embeds_every_document(env):
    published = env.publish("--source", "corpus/base")
    assert published["version"] == "1.0.0"

    env.reset_stub()
    summary = env.ingest()
    plan, stats, metrics = summary["plan"], summary["stats"], summary["metrics"]

    assert plan["added"] == BASE_DOCS and len(BASE_DOCS) == 25
    assert plan["modified"] == plan["removed"] == plan["unchanged"] == []
    assert stats["embedded_documents"] == 25
    assert stats["points_total"] == stats["embedded_chunks"] > 25
    # The embedding endpoint saw exactly the chunks plus the evaluation questions.
    assert env.stub_stats()["embedding_inputs"] == stats["embedded_chunks"] + metrics["questions"]
    # Retrieval check (against the lexical stub embeddings).
    assert metrics["questions"] == 20
    assert metrics["hit_at_k"] >= 0.9
    state["v1_index"] = env.index_state()
    assert sorted(state["v1_index"]) == BASE_DOCS


# ------------------------------------------------------- 2. re-run is a no-op

def test_rerun_without_changes_embeds_no_document(env):
    env.reset_stub()
    summary = env.ingest()
    assert summary["plan"]["added"] == summary["plan"]["modified"] == []
    assert len(summary["plan"]["unchanged"]) == 25
    assert summary["stats"]["embedded_chunks"] == 0
    # Only the evaluation questions were embedded.
    assert env.stub_stats()["embedding_inputs"] == summary["metrics"]["questions"]
    assert env.index_state() == state["v1_index"]


# ------------------------------------------------------- 3. middleware, v1

def test_service_health_and_readiness(env):
    env.start("service", "service.app:app", env.service_url)
    assert httpx.get(env.service_url + "/healthz").json() == {"status": "ok"}
    ready = httpx.get(env.service_url + "/readyz").json()
    assert ready["status"] == "ready" and ready["points"] == sum(
        len(e["point_ids"]) for e in state["v1_index"].values())
    info = httpx.get(env.service_url + "/info").json()
    assert "llm_api_key" not in info["settings"] and "embed_api_key" not in info["settings"]


V1_QUESTIONS = [
    ("How long do the batteries in a Sense S1 last?", "docs/sensors/s1-battery.md", "18 months"),
    ("Which firewall ports does the Hub H2 need for outbound connections?",
     "docs/hub/h2-network-requirements.md", "8883"),
    ("How long are raw readings kept on the Standard plan?",
     "docs/cloud/data-retention.md", "90 days"),
    ("How many API requests per minute are allowed per API key?",
     "docs/cloud/api-rate-limits.md", "120 requests"),
    ("Within how many days can unused hardware be returned for a refund?",
     "docs/support/returns.md", "45 days"),
    ("What discount do customers get when migrating from the legacy Hub H1?",
     "docs/support/legacy-h1-migration.md", "20 percent"),
]


@pytest.mark.parametrize("question,expected_doc,expected_fact", V1_QUESTIONS)
def test_answers_cite_the_right_document(env, question, expected_doc, expected_fact):
    result = env.ask(question)
    assert result["sources"][0]["doc_path"] == expected_doc      # top retrieved source
    assert cited_docs(result) == [expected_doc]                  # source cited in the answer
    assert expected_fact in result["answer"]
    assert all(s["dataset_version"] == "1.0.0" for s in result["sources"])


def test_query_validation(env):
    assert httpx.post(env.service_url + "/query", json={"question": ""}).status_code == 422


# ------------------------------------------------------------------ 4. UI

def test_streamlit_ui_shows_answer_and_sources(env, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RAG_API_URL", env.service_url)
    app = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    app.chat_input[0].set_value("How long is the warranty on the Hub H2?").run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "3 years" in text
    assert "docs/support/warranty.md" in text and "cited" in text


# ------------------------------------------------------- 5. update batch

def test_update_embeds_only_changed_documents(env):
    env.stop("service")   # the embedded store allows one process at a time
    published = env.publish("--source", "corpus/base", "--overlay", "corpus/update")
    assert published["version"] == "1.0.1"
    assert [p for p in published["modified"] if p.startswith("docs/")] == MODIFIED
    assert published["added"] == ADDED and published["removed"] == REMOVED

    env.reset_stub()
    summary = env.ingest()
    plan, stats, metrics = summary["plan"], summary["stats"], summary["metrics"]

    # The plan is exactly the difference between the two versions.
    assert plan["added"] == ADDED
    assert plan["modified"] == MODIFIED
    assert plan["removed"] == REMOVED
    assert len(plan["unchanged"]) == 25 - len(MODIFIED) - len(REMOVED) == 21
    assert stats["embedded_documents"] == 5 and stats["skipped_documents"] == 21

    # The embedding endpoint received only the chunks of the 5 changed documents
    # (plus the evaluation questions), not the 21 unchanged ones.
    assert env.stub_stats()["embedding_inputs"] == stats["embedded_chunks"] + metrics["questions"]
    assert stats["embedded_chunks"] < sum(len(e["point_ids"]) for e in state["v1_index"].values()) / 3

    v1, v2 = state["v1_index"], env.index_state()
    for path in plan["unchanged"]:      # untouched: same hash, same point ids
        assert v2[path] == v1[path]
    for path in MODIFIED:               # replaced: new hash, none of the old points left
        assert v2[path]["hash"] != v1[path]["hash"]
        assert not set(v2[path]["point_ids"]) & set(v1[path]["point_ids"])
    for path in ADDED:
        assert path in v2 and path not in v1
    for path in REMOVED:
        assert path in v1 and path not in v2
    assert stats["points_total"] == sum(len(e["point_ids"]) for e in v2.values())
    assert metrics["questions"] == 21 and metrics["hit_at_k"] >= 0.9
    state["update"] = summary


# ------------------------------------------------------- 6. middleware, v2

V2_QUESTIONS = [
    ("How long do the batteries in a Sense S1 last?", "docs/sensors/s1-battery.md", "24 months"),
    ("How long are raw readings kept on the Standard plan?",
     "docs/cloud/data-retention.md", "180 days"),
    ("How many API requests per minute are allowed per API key?",
     "docs/cloud/api-rate-limits.md", "300 requests"),
    ("What range of carbon dioxide concentration does the Sense C4 measure?",
     "docs/sensors/c4-co2-sensor.md", "400 to 5000"),
    ("How many times is a failed webhook delivery retried?", "docs/cloud/webhooks.md", "6 times"),
    # An unchanged document still answers from its v1 chunks.
    ("Within how many days can unused hardware be returned for a refund?",
     "docs/support/returns.md", "45 days"),
]


@pytest.mark.parametrize("question,expected_doc,expected_fact", V2_QUESTIONS)
def test_answers_reflect_the_update(env, question, expected_doc, expected_fact):
    if "service" not in env._procs:
        env.start("service", "service.app:app", env.service_url)
    result = env.ask(question)
    assert result["sources"][0]["doc_path"] == expected_doc
    assert cited_docs(result) == [expected_doc]
    assert expected_fact in result["answer"]


def test_removed_document_is_no_longer_a_source(env):
    result = env.ask("What discount do customers get when migrating from the legacy Hub H1?")
    assert all(s["doc_path"] != REMOVED[0] for s in result["sources"])
    versions = {s["doc_path"]: s["dataset_version"] for s in env.ask(
        "How long do the batteries in a Sense S1 last?")["sources"]}
    assert versions["docs/sensors/s1-battery.md"] == "1.0.1"
