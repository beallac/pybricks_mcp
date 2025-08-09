#!/usr/bin/env python3
"""
Pybricks RAG MCP server (stdio transport) backed by ChromaDB.

Tools exposed:
- docs.search(query: str, k: int = 8, filters: dict | None)
- code.search(symbol: str, k: int = 8, version: str | None)
- snippet.suggest(goal: str, hardware: str | None, k: int = 3)
- compare.versions(feature: str, k: int = 6)

Env vars:
- CHROMA_PATH (default: ./chroma_pybricks)
- COLL_DOCS (default: pybricks_docs)
- COLL_SNIPPETS (default: pybricks_snippets)
- EMBED_MODEL (default: all-MiniLM-L6-v2)
- RUN_HTTP=1 to run an HTTP server instead of stdio (useful for manual testing)

Usage (MCP stdio):
    python -m pybricks_mcp_server
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from mcp.server.fastmcp import FastMCP

# --------------------------
# Config (env-overridable)
# --------------------------
CHROMA_DIR = os.getenv("CHROMA_PATH", "./chroma_pybricks")
COLL_DOCS = os.getenv("COLL_DOCS", "pybricks_docs")
COLL_IDENT = f"{COLL_DOCS}_identifiers"  # created by the ingestion script
COLL_SNIPS = os.getenv("COLL_SNIPPETS", "pybricks_snippets")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# --------------------------
# Chroma client/collections
# --------------------------
def _init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    # These must exist (created by ingest_pybricks.py)
    docs = client.get_or_create_collection(name=COLL_DOCS, embedding_function=ef)
    ident = client.get_or_create_collection(name=COLL_IDENT, embedding_function=ef)
    snips = client.get_or_create_collection(name=COLL_SNIPS, embedding_function=ef)
    return client, ef, docs, ident, snips

_client, _ef, _DOCS, _IDENT, _SNIPS = _init_chroma()

# --------------------------
# Helpers
# --------------------------
def _apply_filters(metadatas: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[int]:
    if not filters:
        return list(range(len(metadatas)))
    keep = []
    for i, m in enumerate(metadatas):
        ok = True
        for k, v in filters.items():
            if str(m.get(k)) != str(v):
                ok = False
                break
        if ok:
            keep.append(i)
    return keep

def _format_results(res: Dict[str, Any], limit: int, postfilter_idxs: Optional[List[int]] = None):
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    idxs = list(range(len(docs)))
    if postfilter_idxs is not None:
        idxs = [i for i in idxs if i in postfilter_idxs]
    out = []
    for i in idxs[:limit]:
        out.append({
            "text": docs[i],
            "score": 1 - dists[i] if i < len(dists) and dists[i] is not None else None,
            "meta": metas[i],
        })
    return out

# --------------------------
# MCP server + tools
# --------------------------
mcp = FastMCP("pybricks-rag")

@mcp.tool()
def docs_search(query: str, k: int = 8, filters: Optional[Dict[str, Any]] = None) -> str:
    """Search Pybricks docs/snippets (dense). Optional filters like {"version":"latest","source":"docs.pybricks.com"}."""
    if not query or not query.strip():
        return json.dumps({"results": [], "error": "Empty query"})
    res = _DOCS.query(query_texts=[query], n_results=max(k, 8), include=["documents","metadatas","distances"])
    keep = _apply_filters(res["metadatas"][0], filters) if filters else None
    results = _format_results(res, k, keep)
    return json.dumps({"results": results})

@mcp.tool()
def code_search(symbol: str, k: int = 8, version: Optional[str] = None) -> str:
    """Find symbols/identifiers quickly (identifier view first). Example: 'DriveBase.turn', 'Port.A'."""
    if not symbol or not symbol.strip():
        return json.dumps({"results": [], "error": "Empty symbol"})
    res = _IDENT.query(query_texts=[symbol], n_results=max(k, 8), include=["documents","metadatas","distances"])
    metas = res.get("metadatas", [[]])[0]
    keep = None
    if version:
        keep = [i for i, m in enumerate(metas) if m.get("version") == version]
    out = _format_results(res, k, keep)
    # Fallback to docs if identifier view is empty
    if not out:
        res2 = _DOCS.query(query_texts=[symbol], n_results=max(k, 8), include=["documents","metadatas","distances"])
        out = _format_results(res2, k)
    return json.dumps({"results": out})

@mcp.tool()
def snippet_suggest(goal: str, hardware: Optional[str] = None, k: int = 3) -> str:
    """Return a small task-oriented snippet. goal='drive straight', hardware='PrimeHub'."""
    if not goal or not goal.strip():
        return json.dumps({"results": [], "error": "Empty goal"})
    query = goal if not hardware else f"{goal} {hardware}"
    res = _SNIPS.query(query_texts=[query], n_results=max(k, 3), include=["documents","metadatas","distances"])
    out = _format_results(res, k)
    return json.dumps({"results": out})

@mcp.tool()
def compare_versions(feature: str, k: int = 6) -> str:
    """Search changelog entries related to a feature (looks for GitHub pybricks CHANGELOG hits)."""
    if not feature or not feature.strip():
        return json.dumps({"results": [], "error": "Empty feature"})
    res = _DOCS.query(query_texts=[f"CHANGELOG {feature}"], n_results=max(k, 20), include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits = []
    for i in range(len(docs)):
        url = metas[i].get("url", "")
        if "github.com/pybricks" in url:
            hits.append({
                "text": docs[i],
                "score": 1 - dists[i] if i < len(dists) and dists[i] is not None else None,
                "meta": metas[i]
            })
    return json.dumps({"results": hits[:k]})

# --------------------------
# Entrypoint(s)
# --------------------------
if __name__ == "__main__":
    # Default to stdio for VS Code MCP. Set RUN_HTTP=1 to run an HTTP server for manual testing.
    run_http = os.getenv("RUN_HTTP") == "1"
    if run_http:
        # Simple HTTP wrapper (for curl testing only)
        # NOTE: VS Code uses stdio, so prefer the default run method.
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI(title="Pybricks MCP (HTTP test)")
        class _DocsReq(BaseModel):
            query: str
            k: int = 8
            filters: Optional[Dict[str, Any]] = None
        class _CodeReq(BaseModel):
            symbol: str
            k: int = 8
            version: Optional[str] = None
        class _SnipReq(BaseModel):
            goal: str
            hardware: Optional[str] = None
            k: int = 3
        class _VerReq(BaseModel):
            feature: str
            k: int = 6

        @app.post("/tools/docs.search")
        def _d(req: _DocsReq):
            return json.loads(docs_search(req.query, req.k, req.filters))
        @app.post("/tools/code.search")
        def _c(req: _CodeReq):
            return json.loads(code_search(req.symbol, req.k, req.version))
        @app.post("/tools/snippet.suggest")
        def _s(req: _SnipReq):
            return json.loads(snippet_suggest(req.goal, req.hardware, req.k))
        @app.post("/tools/compare.versions")
        def _v(req: _VerReq):
            return json.loads(compare_versions(req.feature, req.k))

        uvicorn.run(app, host="127.0.0.1", port=8765)
    else:
        mcp.run()
