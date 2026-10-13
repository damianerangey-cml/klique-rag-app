"""Streamlit chat front end for the RAG middleware.

The UI holds no model or vector store logic. It posts the question to the
middleware's /query endpoint and shows the answer with its numbered sources.

Middleware address, in order of precedence:
    --api-url <url>          script argument (the Klique Streamlit app's
                             "Arguments" launch-form list passes name/value pairs)
    RAG_API_URL              environment variable
    ARGS_API_URL             set by the launch form of the optional custom app
                             (deploy/custom-app/rag-chat.app.conf)
    http://localhost:8000    default for local runs
RAG_API_TOKEN (optional) is sent as a bearer token, which an authenticated
application gateway route expects.

    streamlit run ui/app.py -- --api-url http://localhost:8000
"""
import argparse
import os

import httpx
import streamlit as st


def _config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--api-url", "--api_url", dest="api_url", default=None)
    args, _ = parser.parse_known_args()
    url = (args.api_url or os.environ.get("RAG_API_URL") or os.environ.get("ARGS_API_URL")
           or "http://localhost:8000")
    return url.rstrip("/"), os.environ.get("RAG_API_TOKEN", "")


API_URL, API_TOKEN = _config()
HEADERS = {"Authorization": "Bearer " + API_TOKEN} if API_TOKEN else {}


def ask(question: str) -> dict:
    response = httpx.post(API_URL + "/query", json={"question": question},
                          headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response.json()


def show_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander("Sources (%d)" % len(sources), expanded=True):
        for s in sources:
            marker = "cited" if s["cited"] else "retrieved"
            st.markdown("**[%d] %s** · `%s` · %s · score %.2f" % (
                s["n"], s["title"], s["doc_path"], marker, s["score"]))
            st.caption(s["snippet"])


st.set_page_config(page_title="Pellwood docs assistant", page_icon=None)
st.title("Pellwood docs assistant")
st.caption("Answers come from the indexed documentation. Each answer lists its sources.")

with st.sidebar:
    st.subheader("Service")
    st.code(API_URL)
    try:
        info = httpx.get(API_URL + "/info", headers=HEADERS, timeout=10).json()
        st.write("Indexed chunks:", info["points"])
        st.write("Chat model:", info["settings"]["llm_model"])
        st.write("Embedding model:", info["settings"]["embed_model"])
    except Exception as exc:
        st.warning("Middleware not reachable: %s" % type(exc).__name__)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

question = st.chat_input("Ask about Pellwood products")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = ask(question)
            answer, sources = result["answer"], result["sources"]
        except Exception as exc:
            answer, sources = "The middleware returned an error: %s" % exc, []
        st.markdown(answer)
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
