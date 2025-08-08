#!/usr/bin/env python3
"""
Minimal MCP-style tool server exposing Pybricks RAG functions over ChromaDB.

Run:
    uvicorn mcp_server:app --reload --port 8765

Tools:
- /tools/docs.search
- /tools/code.search
- /tools/snippet.suggest
- /tools/compare.versions
"""

import re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Body, Query
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "./chroma_pybricks"
COLL_DOCS = "pybricks_docs"
COLL_DOCS_IDENT = f"{COLL_DOCS}_identifiers"
COLL_SNIPPETS = "pybricks_snippets"
EMBED_MODEL = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=CHROMA_DIR)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

docs = client.get_or_create_collection(name=COLL_DOCS, embedding_function=ef)
ident = client.get_or_create_collection(name=COLL_DOCS_IDENT, embedding_function=ef)
snips = client.get_or_create_collection(name=COLL_SNIPPETS, embedding_function=ef)

app = FastAPI(title="Pybricks MCP Tools")

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    filters: Optional[Dict[str, Any]] = None  # e.g., {"version":"latest"} or {"source":"docs.pybricks.com"}

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

def apply_filters(metadatas: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[int]:
    idxs = []
    for i, m in enumerate(metadatas):
        ok = True
        for k, v in filters.items():
            if str(m.get(k)) != str(v):
                ok = False
                break
        if ok:
            idxs.append(i)
    return idxs

@app.post("/tools/docs.search", response_model=SearchResponse)
def docs_search(req: SearchRequest):
    # dense search
    res = docs.query(query_texts=[req.query], n_results=max(req.k, 8), include=["documents","metadatas","distances"])
    docs_idxs = list(range(len(res["documents"][0])))

    # optional hard filter by simple metadata
    if req.filters:
        keep = apply_filters(res["metadatas"][0], req.filters)
        docs_idxs = [i for i in docs_idxs if i in keep]

    results = []
    for i in docs_idxs[:req.k]:
        results.append({
            "text": res["documents"][0][i],
            "score": 1 - res["distances"][0][i],
            "meta": res["metadatas"][0][i],
        })
    return {"results": results}

class CodeSearchRequest(BaseModel):
    symbol: str
    k: int = 8
    version: Optional[str] = None

@app.post("/tools/code.search", response_model=SearchResponse)
def code_search(req: CodeSearchRequest):
    # prioritize identifier view collection; fall back to docs
    res = ident.query(query_texts=[req.symbol], n_results=max(req.k, 8), include=["documents","metadatas","distances"])
    out = []
    for i in range(len(res["documents"][0])):
        m = res["metadatas"][0][i]
        if req.version and m.get("version") != req.version:
            continue
        out.append({
            "text": res["documents"][0][i],
            "score": 1 - res["distances"][0][i],
            "meta": m,
        })
    if not out:
        return docs_search(SearchRequest(query=req.symbol, k=req.k, filters={"version": req.version} if req.version else None))
    return {"results": out[:req.k]}

class SnippetSuggestRequest(BaseModel):
    goal: str
    hardware: Optional[str] = None
    k: int = 3

@app.post("/tools/snippet.suggest", response_model=SearchResponse)
def snippet_suggest(req: SnippetSuggestRequest):
    query = req.goal + (" " + req.hardware if req.hardware else "")
    res = snips.query(query_texts=[query], n_results=req.k, include=["documents","metadatas","distances"])
    results = [{
        "text": res["documents"][0][i],
        "score": 1 - res["distances"][0][i],
        "meta": res["metadatas"][0][i],
    } for i in range(len(res["documents"][0]))]
    return {"results": results}

class CompareVersionsRequest(BaseModel):
    feature: str
    k: int = 6

@app.post("/tools/compare.versions", response_model=SearchResponse)
def compare_versions(req: CompareVersionsRequest):
    # Look for the feature in CHANGELOG pages inside docs collection
    res = docs.query(query_texts=[f"CHANGELOG {req.feature}"], n_results=20, include=["documents","metadatas","distances"])
    hits = []
    for i in range(len(res["documents"][0])):
        m = res["metadatas"][0][i]
        if "github.com/pybricks" in m.get("url",""):
            hits.append({
                "text": res["documents"][0][i],
                "score": 1 - res["distances"][0][i],
                "meta": m
            })
    return {"results": hits[:req.k]}
