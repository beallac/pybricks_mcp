Heck yes—let’s get you a working starter kit. Below are **two drop-in files**:

1. `ingest_pybricks.py` — crawls the **official** Pybricks docs/tutorials + key GitHub pages, chunks them nicely, and builds a **ChromaDB** collection.
2. `mcp_server.py` — a minimal **MCP**-style tool server exposing `docs.search`, `code.search`, `snippet.suggest`, and `compare.versions` backed by that Chroma collection.

I’ve limited the crawl to the most useful, stable surfaces for beginners (API reference, parameters/constants, robotics/DriveBase, getting started, tutorials, and changelogs). You can add more later.

# What we’ll ingest (initial allowlist)

- **API docs (stable + latest)**: `docs.pybricks.com` including Robotics/DriveBase and Parameters/Constants. ([Pybricks Documentation](https://docs.pybricks.com/?utm_source=chatgpt.com))
- **Getting started & IDE basics**: `pybricks.com/learn/getting-started/*`, `code.pybricks.com` landing. ([Pybricks](https://pybricks.com/learn/getting-started/pybricks-environment/?utm_source=chatgpt.com), [Pybricks Code](https://code.pybricks.com/?utm_source=chatgpt.com))
- **Beginner-friendly tutorials/examples**: hub-to-PC and hub-to-hub comms, button basics, quick reference. ([Pybricks](https://pybricks.com/projects/tutorials/wireless/hub-to-device/pc-communication/?utm_source=chatgpt.com))
- **Changelogs (for version checks)**: `pybricks-code` and `pybricks-micropython` CHANGELOGs. ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com))
- **pybricksdev** docs (optional coach power tool). ([Pybricks Documentation](https://docs.pybricks.com/projects/pybricksdev/en/latest/api/?utm_source=chatgpt.com), [GitHub](https://github.com/pybricks/pybricksdev?utm_source=chatgpt.com))

------

# 1) `ingest_pybricks.py`

```python
#!/usr/bin/env python3
"""
Build a beginner-friendly Pybricks RAG corpus into ChromaDB.

- Crawls/loads selected Pybricks docs/tutorials + key GitHub changelogs.
- Normalizes & chunks (HTML/Markdown-aware; preserves code blocks).
- Creates multi-view docs (full text + headings/signatures + identifiers).
- Indexes into ChromaDB collections: pybricks_docs and pybricks_snippets.

Requires: requests, beautifulsoup4, lxml, markdownify, tiktoken, chromadb, sentence-transformers
    pip install requests beautifulsoup4 lxml markdownify tiktoken chromadb sentence-transformers
"""

import re
import time
import json
import hashlib
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import tiktoken

import chromadb
from chromadb.utils import embedding_functions

############################
# Config
############################
SEED_URLS = [
    # API docs (latest & versioned)
    "https://docs.pybricks.com/en/latest/robotics.html",       # DriveBase, etc.
    "https://docs.pybricks.com/en/latest/parameters/",         # Port, Direction, Stop, etc.
    "https://docs.pybricks.com/",                              # root (to discover other key pages)
    "https://docs.pybricks.com/en/v3.5.0/robotics.html",       # older ref (version contrast)
    "https://docs.pybricks.com/en/v3.3.0/robotics.html",

    # Tutorials / Getting started
    "https://pybricks.com/learn/getting-started/pybricks-environment/",
    "https://pybricks.com/learn/getting-started/install-pybricks/",
    "https://pybricks.com/projects/tutorials/wireless/hub-to-device/pc-communication/",
    "https://pybricks.com/projects/tutorials/wireless/hub-to-hub/broadcast/",
    "https://pybricks.com/projects/tutorials/wireless/remote-control/button-basics/",
    "https://pybricks.com/projects/sets/mindstorms-robot-inventor/other-models/quick-reference/",
    "https://code.pybricks.com/",

    # Changelogs (version awareness)
    "https://github.com/pybricks/pybricks-micropython/blob/master/CHANGELOG.md",
    "https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md",

    # pybricksdev (optional power tool docs)
    "https://docs.pybricks.com/projects/pybricksdev/en/latest/api/",
]

ALLOWED_DOMAINS = {
    "docs.pybricks.com",
    "pybricks.com",
    "code.pybricks.com",
    "github.com",
}

MAX_PAGES = 80           # keep crawl polite; bump when needed
CRAWL_TIMEOUT = 15
USER_AGENT = "pybricks-rag-ingestor/1.0 (+for educational/FLL use)"

# Chunking defaults
DOC_CHUNK_TOKENS = 600
DOC_OVERLAP_TOKENS = 100

# Chroma config
CHROMA_DIR = "./chroma_pybricks"
COLL_DOCS = "pybricks_docs"
COLL_SNIPPETS = "pybricks_snippets"  # short task-oriented code templates/snips

# Embeddings: sentence-transformers (local)
EMBED_MODEL = "all-MiniLM-L6-v2"

############################
# Helpers
############################
tok = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tok.encode(text))

def hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

def within_domain(url: str) -> bool:
    try:
        host = urlparse(url).netloc
        return any(host.endswith(d) for d in ALLOWED_DOMAINS)
    except Exception:
        return False

def get(url: str) -> requests.Response:
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=CRAWL_TIMEOUT)

def extract_links(url: str, soup: BeautifulSoup):
    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith("#"):
            continue
        u = urljoin(url, href)
        if within_domain(u):
            yield u.split("#")[0]

def html_to_markdown_keep_code(html: str) -> str:
    # Convert HTML to Markdown but keep code blocks and headings clean
    # md() handles most; we'll fix common issues
    text = md(html, heading_style="ATX", strip=["nav","footer","script","style","noscript"])
    # compact extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def normalize_markdown(text: str) -> str:
    # de-dup whitespace, make headings spaced
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()

############################
# Chunkers
############################
HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)", re.MULTILINE)

def split_by_headings(markdown: str):
    """
    Split markdown by H1–H3. Keep fenced code blocks with their section.
    """
    lines = markdown.splitlines()
    sections = []
    current = {"title": None, "content": []}

    def push():
        if current["content"]:
            sections.append({
                "title": current["title"] or "",
                "text": "\n".join(current["content"]).strip()
            })

    for i, line in enumerate(lines):
        if HEADING_PATTERN.match(line):
            # new section
            if current["content"]:
                push()
                current = {"title": None, "content": []}
            current["title"] = HEADING_PATTERN.match(line).group(2)
            current["content"].append(line)
        else:
            current["content"].append(line)

    push()
    return [s for s in sections if s["text"]]

def window_tokens(text: str, size_tokens: int, overlap_tokens: int):
    ids = tok.encode(text)
    n = len(ids)
    if n <= size_tokens:
        yield text
        return
    start = 0
    while start < n:
        end = min(n, start + size_tokens)
        chunk_ids = ids[start:end]
        yield tok.decode(chunk_ids)
        if end == n:
            break
        start = max(0, end - overlap_tokens)

def identifiers_view(text: str) -> str:
    # Pull likely “identifier-ish” tokens: backticked names, UPPER_CASE, CamelCase, Port.X, Direction.X, drivebase methods, HTML ids
    ids = re.findall(r"`([^`]+)`|([A-Z_]{2,})|([A-Za-z_][A-Za-z0-9_]+)|id=\"([^\"]+)\"", text)
    flat = []
    for g in ids:
        flat.extend([x for x in g if x])
    # de-noise trivial words
    flat = [w for w in flat if len(w) > 2]
    return "\n".join(sorted(set(flat)))

############################
# Crawl
############################
def crawl():
    seen = set()
    queue = list(SEED_URLS)
    pages = []

    while queue and len(pages) < MAX_PAGES:
        url = queue.pop(0)
        if url in seen or not within_domain(url):
            continue
        seen.add(url)
        try:
            r = get(url)
            if r.status_code != 200:
                continue
            ct = r.headers.get("Content-Type","")
            html = r.text
            soup = BeautifulSoup(html, "lxml")
            # strip obvious boilerplate
            for tag in soup(["nav","footer","script","style","noscript"]): tag.decompose()
            body = soup.body or soup
            main = body.select_one("main") or body
            # Many docs are Sphinx → good structure + headings
            # Convert to markdown for easier heading split
            markdown = html_to_markdown_keep_code(str(main))
            markdown = normalize_markdown(markdown)

            pages.append({
                "url": url,
                "markdown": markdown
            })

            # discover more links from key roots (docs/tutorials)
            if any(url.startswith(prefix) for prefix in [
                "https://docs.pybricks.com/",
                "https://pybricks.com/learn/",
                "https://pybricks.com/projects/",
            ]):
                for u in set(extract_links(url, soup)):
                    if u not in seen and within_domain(u):
                        # keep within same area
                        if len(pages) + len(queue) < MAX_PAGES:
                            queue.append(u)

            time.sleep(0.3)  # be polite

        except Exception:
            continue

    return pages

############################
# Build records & index
############################
def build_records(pages):
    records = []
    for p in pages:
        url = p["url"]
        mdtext = p["markdown"]
        # version tags from URL
        version = "latest" if "/en/latest/" in url else (
            re.search(r"/en/v?(\d+\.\d+\.\d+)/", url).group(1) if re.search(r"/en/v?(\d+\.\d+\.\d+)/", url) else "stable"
        )
        # sectionize by H1–H3
        sections = split_by_headings(mdtext) or [{"title":"","text":mdtext}]
        for sec in sections:
            # windowing to stay within token budget
            for win in window_tokens(sec["text"], DOC_CHUNK_TOKENS, DOC_OVERLAP_TOKENS):
                rec_id = hash_id(url + "|" + sec["title"] + "|" + win[:200])
                records.append({
                    "id": rec_id,
                    "text": win,
                    "meta": {
                        "url": url,
                        "title": sec["title"],
                        "version": version,
                        "source": urlparse(url).netloc,
                        "breadcrumbs": sec["title"],
                        "type": "doc"
                    },
                    "identifiers": identifiers_view(win)
                })
    return records

############################
# ChromaDB
############################
def upsert_chroma(records):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    docs = client.get_or_create_collection(name=COLL_DOCS, embedding_function=ef, metadata={"hnsw:space":"cosine"})

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    metas = [r["meta"] for r in records]
    # store a second view as metadata field for regex/BM25 on your side, but we can also make a tiny aux collection:
    docs.add(ids=ids, documents=texts, metadatas=metas)

    # Optional: create a lightweight identifiers collection to help rerank exact symbol hits
    ident = client.get_or_create_collection(name=f"{COLL_DOCS}_identifiers", embedding_function=ef, metadata={"hnsw:space":"cosine"})
    ident.add(
        ids=[f"I_{i}" for i in ids],
        documents=[r["identifiers"] for r in records],
        metadatas=[{**r["meta"], "type":"identifiers"} for r in records]
    )

    # Starter snippet/templates (task-oriented)
    snippets = [
        {
            "id": "snip_drivebase_basic",
            "text": """# Drive straight and turn
from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.robotics import DriveBase

hub = PrimeHub()
left = Motor(Port.A)
right = Motor(Port.B)
bot = DriveBase(left, right, wheel_diameter=56, axle_track=114)

bot.straight(300)   # mm
bot.turn(90)        # + is clockwise/right
""",
            "meta": {"topic":"drivebase","version":"stable","url":"https://docs.pybricks.com/en/latest/robotics.html"}
        },
    ]
    sn = client.get_or_create_collection(name=COLL_SNIPPETS, embedding_function=ef, metadata={"hnsw:space":"cosine"})
    sn.add(
        ids=[s["id"] for s in snippets],
        documents=[s["text"] for s in snippets],
        metadatas=[s["meta"] for s in snippets]
    )

    return client

def main():
    print("Crawling…")
    pages = crawl()
    print(f"Crawled {len(pages)} pages")
    records = build_records(pages)
    print(f"Built {len(records)} records")
    client = upsert_chroma(records)
    print("ChromaDB ready at", CHROMA_DIR)

if __name__ == "__main__":
    main()
```

**Why these sources?** They’re the official, beginner-relevant places your team actually needs: API reference & constants (for method/port names) ([Pybricks Documentation](https://docs.pybricks.com/en/latest/robotics.html?utm_source=chatgpt.com)), how to run programs & pair hubs (classroom logistics) ([Pybricks](https://pybricks.com/learn/getting-started/pybricks-environment/?utm_source=chatgpt.com)), ready-to-use tutorials and examples (frequent FLL tasks) ([Pybricks](https://pybricks.com/projects/tutorials/wireless/hub-to-device/pc-communication/?utm_source=chatgpt.com)), and changelogs for feature/version hints ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com)).

------

# 2) `mcp_server.py` (simple MCP-style tool server over Chroma)

This is a minimal Python FastAPI server emulating “tools” you can register with your MCP agent:

- `docs.search(query, filters)` — hybrid-ish dense search with optional version/source filters.
- `code.search(symbol)` — prioritizes identifier hits.
- `snippet.suggest(goal, hardware)` — returns a template (seeded in the index).
- `compare.versions(feature)` — checks changelogs for mentions.

```python
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
```

------

# How to run it

1. **Ingest once** (or on a schedule):

```bash
python ingest_pybricks.py
```

1. **Start the tool server**:

```bash
pip install fastapi uvicorn
uvicorn mcp_server:app --reload --port 8765
```

1. **Hook into your MCP agent** by registering these tool endpoints (your agent would call `/tools/docs.search`, etc., with JSON requests).

------

# Next upgrades (when you have time)

- **Add AST-aware chunking for Python examples** you keep locally (the docs mainly contain snippets, so heading chunking works well to start).
- **Reranking**: after dense search, locally boost results with exact matches of tokens like `DriveBase.turn`, `Port.A`, `Direction.COUNTERCLOCKWISE`.
- **Version banner** in answers using the changelog lookups so you can say “added in v3.5” (see pybricks-code / micropython changelogs). ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com))

If you want, I can wire this directly into your MCP agent config (whatever framework you’re using) and add a couple of **few-shot answer templates** tuned for FLL beginners.







Awesome goal. Here’s a lean-but-solid blueprint for an **MCP-based RAG assistant** tuned for beginner Python + Pybricks devs (FLL kids + coaches). I’ll pull in Pybricks-specific details from their docs/site so your assistant gives *grounded, version-aware* answers.

# What “beginner-friendly” means here

- Speak Python, not firmware internals—explain *what to import, which port to use, what the units are*.
- Prefer **official API calls and examples** over clever hacks.
- Detect **MicroPython vs CPython** mismatches and nudge correctly (“this runs on the hub only”). ([Pybricks Documentation](https://docs.pybricks.com/?utm_source=chatgpt.com), [GitHub](https://github.com/orgs/pybricks/discussions/1607?utm_source=chatgpt.com))

------

# 1) Source of truth to ingest (and keep fresh)

Minimum viable corpus:

- **Official API docs (stable + latest):**
  - Hubs & sensors/motors, parameters/constants, robotics helpers (DriveBase). ([Pybricks Documentation](https://docs.pybricks.com/en/latest/hubs/primehub.html?utm_source=chatgpt.com))
- **Pybricks Code (IDE) basics**: pairing, running, limits. ([Pybricks Code](https://code.pybricks.com/?utm_source=chatgpt.com))
- **Getting started / “other editors” posts** (great for setup FAQs). ([Pybricks](https://pybricks.com/project/pybricks-other-editors/?utm_source=chatgpt.com))
- **Release/Changelog snippets** for breaking changes or new helpers. ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com))
- **Selected project tutorials** (BLE comms, etc.) for worked examples. ([Pybricks](https://pybricks.com/project/micropython-ble-communication/?utm_source=chatgpt.com))
- **Pybricks API repo/PyPI** as a reference label for versions. ([GitHub](https://github.com/pybricks/pybricks-api?utm_source=chatgpt.com), [PyPI](https://pypi.org/project/pybricks/?utm_source=chatgpt.com))
- **Landing page** for a short “what is Pybricks” explainer. ([Pybricks](https://pybricks.com/?utm_source=chatgpt.com))

Practical tip: keep **stable** docs as default grounding, but store **latest** docs in a separate collection and *surface a gentle version banner* if answers use features newer than the team’s firmware. ([Pybricks Documentation](https://docs.pybricks.com/en/latest/hubs/primehub.html?utm_source=chatgpt.com))

------

# 2) Preprocessing & chunking (works great for Markdown/HTML/Python)

## Markdown/HTML docs

- **Split by headings (H2/H3)**. Keep fenced code, tables with their captions, and any note/admonition together.
- **Overlap** ~80–120 tokens so definitions carry to the next chunk.
- **Metadata**: page URL, version (stable/latest), breadcrumbs (page → H2 → H3), any HTML `id` anchors.

## Python examples/snippets

- **AST or Tree-sitter per function/class**, include **signature + docstring + imports used**.
- If a unit is long, window with **~80–120 token overlap** and repeat the signature.
- Create a 2nd “interface” vector: **signature + docstring only** (perfect for “what does `DriveBase.turn` do?” style queries).

Why this matters for Pybricks:

- Kids search exact tokens like `Port.A`, `Direction.COUNTERCLOCKWISE`, `DriveBase.turn`, `gears=[12,36]`. Keep those intact for keyword/rerank. ([Pybricks Documentation](https://docs.pybricks.com/en/v2.0/parameters.html?utm_source=chatgpt.com))

------

# 3) Indexing & retrieval

- **Hybrid search**: dense embeddings **+** BM25/regex for symbol hits (`PrimeHub`, `run_angle`, `stalled()`). ([Pybricks Documentation](https://docs.pybricks.com/en/latest/hubs/primehub.html?utm_source=chatgpt.com))
- **Parent–child**: embed small chunks; keep a “parent” section/file to optionally expand.
- **Multi-vector per chunk**:
  1. full text, 2) headings/signature view, 3) identifiers-only (symbols, constants, error names).
- **Version filter**: tag each chunk `version: 3.6.1`, `channel: stable/latest`, `source: docs/blog/discussion`.
- **Rerank** top-K by: exact symbol match > version proximity > doc section type (API reference > blog).

Good starting knobs:

- Chunk size **300–800 tokens** for docs, **200–600** for code; overlap as above.

------

# 4) MCP server design (tools & routes)

Expose a few simple tools your agent can call:

1. **docs.search(query, filters)** → returns passages + metadata
   - Filters: `symbol`, `hub=PrimeHub`, `version=stable|latest`, `category=motors|robotics|parameters`.
2. **code.search(symbol)** → exact/regex search across examples.
3. **explain.error(message, context)** → maps common runtime messages or gotchas to fixes (e.g., wrong port, MicroPython import).
4. **snippet.suggest(goal, hardware)** → templates (see §7).
5. **compare.versions(feature)** → show when something was added/changed (reads changelogs). ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com))

Optional power-ups:

- **pybricksdev bridge** for scripted flashing/log retrieval (coach use only). ([GitHub](https://github.com/pybricks/pybricksdev?utm_source=chatgpt.com))

------

# 5) Guardrails & beginner heuristics

- **MicroPython ≠ CPython**: if user tries `pip install`/desktop-only libs, return a helpful correction and suggest the Pybricks way or hub-to-PC comms. ([GitHub](https://github.com/orgs/pybricks/discussions/1607?utm_source=chatgpt.com))
- **Units & sign conventions**: explain degrees vs deg/s, positive means clockwise/right, etc., with short examples. ([Pybricks Documentation](https://docs.pybricks.com/en/stable/pupdevices/motor.html?utm_source=chatgpt.com))
- **Hardware sanity**: warn if method requires certain sensors or hubs; confirm **which hub** (Prime/Inventor/Technic). ([Pybricks Documentation](https://docs.pybricks.com/en/latest/hubs/primehub.html?utm_source=chatgpt.com))
- **Version banner**: “This answer uses feature added in v3.2.0… consider updating” (e.g., `DriveBase.stalled()`mention). ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com))

------

# 6) Answer patterns (prompt templates for your agent)

## A) “How do I…?” (action recipe)

**System hint:** prefer official API + short code + 2 gotcha bullets.

**Template (few-shot)**

> **Goal:** Drive straight 300 mm, then turn right 90°.
> **Hardware:** PrimeHub, 2 motors on A/B, wheel_diameter=56, axle_track=114.
> **Answer:**
>
> 1. Imports + setup; 2) Code; 3) Notes on signs/units; 4) Link to docs.

**Generated answer should resemble:**

```python
from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.robotics import DriveBase

hub = PrimeHub()
left = Motor(Port.A)
right = Motor(Port.B)
bot = DriveBase(left, right, wheel_diameter=56, axle_track=114)

bot.straight(300)   # millimeters
bot.turn(90)        # + is clockwise (right)
```

Notes: positive = forward/right; measure diameter/track for accuracy. ([Pybricks Documentation](https://docs.pybricks.com/en/latest/robotics.html?utm_source=chatgpt.com))

## B) “What does this error mean?”

- Search error text; map to common causes.
- Offer fix + 1-line explanation + tiny example.

## C) “Why isn’t my motor moving?”

Checklist: correct **Port**, `Direction`, gear list shape, stall/hold behavior, power limits. Provide a `run_angle` demo and link to Motor docs. ([Pybricks Documentation](https://docs.pybricks.com/en/stable/pupdevices/motor.html?utm_source=chatgpt.com))

## D) “Which port constant do I use?”

Answer with `Port.A/B/C/D` and mention sensor vs motor ports on their hub; link to **parameters/constants**. ([Pybricks Documentation](https://docs.pybricks.com/en/v2.0/parameters.html?utm_source=chatgpt.com))

------

# 7) Prebuilt snippet templates (your `snippet.suggest`tool)

Seed a few common tasks (return parameterized code):

- **DriveBase skeleton** (takes wheel_diameter, axle_track). ([Pybricks Documentation](https://docs.pybricks.com/en/latest/robotics.html?utm_source=chatgpt.com))
- **Single motor move by angle/speed** (with `Direction` & `gears` examples). ([Pybricks Documentation](https://docs.pybricks.com/en/stable/pupdevices/motor.html?utm_source=chatgpt.com))
- **Read hub orientation / display light / buttons** (Prime Hub basics). ([Pybricks Documentation](https://docs.pybricks.com/en/latest/hubs/primehub.html?utm_source=chatgpt.com))
- **Broadcast between hubs / BLE example** for advanced kids. ([Pybricks Documentation](https://docs.pybricks.com/en/v3.4.1/hubs/primehub.html?utm_source=chatgpt.com), [Pybricks](https://pybricks.com/project/micropython-ble-communication/?utm_source=chatgpt.com))

Each template should embed short comments like “units: deg/s, mm; +angle = right turn”.

------

# 8) Evaluation plan (coach-friendly)

Make a tiny test set of **25 real questions** your kids ask. Examples:

1. “How do I set up `DriveBase` and turn 45°?” ([Pybricks Documentation](https://docs.pybricks.com/en/latest/robotics.html?utm_source=chatgpt.com))
2. “What does `Direction.COUNTERCLOCKWISE` change?” ([Pybricks Documentation](https://docs.pybricks.com/en/v2.0/parameters.html?utm_source=chatgpt.com))
3. “Why does +90 turn right on our robot?” ([Pybricks Documentation](https://docs.pybricks.com/en/latest/robotics.html?utm_source=chatgpt.com))
4. “How do I gear down a motor in code?” (`gears=[12,36]` example). ([Pybricks Documentation](https://docs.pybricks.com/en/stable/pupdevices/motor.html?utm_source=chatgpt.com))
5. “Which ports are valid on Prime Hub?” (A–F for motors, etc.). ([Pybricks Documentation](https://docs.pybricks.com/en/latest/hubs/primehub.html?utm_source=chatgpt.com))
6. “Can I `pip install` numpy?” (explain hub MicroPython limits). ([GitHub](https://github.com/orgs/pybricks/discussions/1607?utm_source=chatgpt.com))
7. “How to check if the robot is stalled?” (`DriveBase.stalled()` availability). ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com))

Score each answer for **correctness, actionable code, cites present, beginner clarity**. Iterate chunk sizes and reranker.

------

# 9) Ops: keeping it current

- **Weekly crawl** of “stable” docs; **monthly** for “latest”.
- Watch **pybricks-code** and **pybricks-micropython** changelogs for features that change beginner UX (e.g., new `DriveBase` helpers). ([GitHub](https://github.com/pybricks/pybricks-code/blob/master/CHANGELOG.md?utm_source=chatgpt.com))
- If a change conflicts with an existing snippet, mark snippet with a **version guard**.

------

# 10) Quick pitfalls your assistant should proactively catch

- Suggesting desktop-only libraries or features not on the hub. ([GitHub](https://github.com/orgs/pybricks/discussions/1607?utm_source=chatgpt.com))
- Mixing SPIKE/Inventor vs Technic hub specifics without checking the hub. ([Pybricks Documentation](https://docs.pybricks.com/en/latest/hubs/primehub.html?utm_source=chatgpt.com))
- Getting sign conventions wrong for turns/speeds. ([Pybricks Documentation](https://docs.pybricks.com/en/latest/robotics.html?utm_source=chatgpt.com))
- Forgetting to include `gears` or misunderstand its list-of-lists format. ([Pybricks Documentation](https://docs.pybricks.com/en/stable/pupdevices/motor.html?utm_source=chatgpt.com))

------

# 11) Day-one stack recommendation

- **Embedder**: any strong general code/text embedding works; you don’t need a code-only model for doc QA.
- **Index**: pgvector/FAISS (two collections: “api_docs” + “examples/snippets”).
- **Rerank**: small BM25 over the top-50 dense hits + symbol regex bonus.
- **Caching**: store resolved answers (code + citations) per question for classroom speed.

------

If you want, I can:

- generate a **starter ingestion script** (scrapes the specific doc sections above, chunks them with the rules here, and builds a pgvector index), and
- stub **MCP tool definitions** (`docs.search`, `snippet.suggest`, etc.) you can drop into your server.

Tell me your preferred DB (FAISS vs pgvector) and I’ll hand you the code.