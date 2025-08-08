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
import argparse
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
def crawl(max_pages: int = MAX_PAGES):
    seen = set()
    queue = list(SEED_URLS)
    pages = []

    while queue and len(pages) < max_pages:
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
                        if len(pages) + len(queue) < max_pages:
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
        for sec_idx, sec in enumerate(sections):
            # windowing to stay within token budget
            for win_idx, win in enumerate(window_tokens(sec["text"], DOC_CHUNK_TOKENS, DOC_OVERLAP_TOKENS)):
                # Include indices to avoid duplicates when titles and leading text repeat
                rec_key = f"{url}|sec={sec_idx}|win={win_idx}|title={sec['title']}"
                rec_id = hash_id(rec_key)
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
    # Write in small batches and use upsert for idempotency
    BATCH = 128
    for i in range(0, len(ids), BATCH):
        sl = slice(i, i+BATCH)
        try:
            docs.upsert(ids=ids[sl], documents=texts[sl], metadatas=metas[sl])
        except Exception:
            # Fallback for older chromadb without upsert or if a validation error occurs
            try:
                docs.delete(ids=ids[sl])
            except Exception:
                pass
            docs.add(ids=ids[sl], documents=texts[sl], metadatas=metas[sl])

    # Optional: create a lightweight identifiers collection to help rerank exact symbol hits
    ident = client.get_or_create_collection(name=f"{COLL_DOCS}_identifiers", embedding_function=ef, metadata={"hnsw:space":"cosine"})
    ident_ids = [f"I_{i}" for i in ids]
    for i in range(0, len(ident_ids), BATCH):
        sl = slice(i, i+BATCH)
        try:
            ident.upsert(
                ids=ident_ids[sl],
                documents=[r["identifiers"] for r in records][sl],
                metadatas=[{**r["meta"], "type":"identifiers"} for r in records][sl]
            )
        except Exception:
            try:
                ident.delete(ids=ident_ids[sl])
            except Exception:
                pass
            ident.add(
                ids=ident_ids[sl],
                documents=[r["identifiers"] for r in records][sl],
                metadatas=[{**r["meta"], "type":"identifiers"} for r in records][sl]
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
    try:
        sn.upsert(
            ids=[s["id"] for s in snippets],
            documents=[s["text"] for s in snippets],
            metadatas=[s["meta"] for s in snippets]
        )
    except Exception:
        try:
            sn.delete(ids=[s["id"] for s in snippets])
        except Exception:
            pass
        sn.add(
            ids=[s["id"] for s in snippets],
            documents=[s["text"] for s in snippets],
            metadatas=[s["meta"] for s in snippets]
        )

    return client

def main():
    parser = argparse.ArgumentParser(description="Build Pybricks RAG corpus into ChromaDB")
    parser.add_argument("--test", action="store_true", help="Enable quick test mode: limit crawl pages for fast debug")
    parser.add_argument("--max-pages", type=int, default=None, help="Override max pages to crawl")
    args = parser.parse_args()

    # Determine crawl size
    max_pages = args.max_pages if args.max_pages is not None else (8 if args.test else MAX_PAGES)
    if args.test:
        print(f"[TEST MODE] Limiting crawl to {max_pages} pages for rapid debugging")

    print("Crawling…")
    pages = crawl(max_pages=max_pages)
    print(f"Crawled {len(pages)} pages")
    records = build_records(pages)
    print(f"Built {len(records)} records")
    client = upsert_chroma(records)
    print("ChromaDB ready at", CHROMA_DIR)

if __name__ == "__main__":
    main()
