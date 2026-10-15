"""
One-time build of the vector database.

Steps:
  1. Load every .mdx page from the downloaded LangChain docs
  2. Clean out the website-only markup (JSX components, JavaScript examples)
  3. Split each page into overlapping chunks
  4. Embed the chunks with sentence-transformers
  5. Store them in Chroma, saved to rag_server/chroma_db/

After this runs once, every assistant session reuses the saved database.

    python rag_server/ingest.py            # build if it doesn't exist yet
    python rag_server/ingest.py --rebuild  # throw it away and build again
"""

import argparse
import re

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, SKIP_FOLDERS
from store import delete_collection, embed, get_collection, log

BATCH_SIZE = 128  # chunks embedded and stored per batch


# ---------- Step 1 and 2: load and clean ----------

def split_frontmatter(raw):
    """Pull the title out of the YAML block at the top of an .mdx page."""
    title = ""
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            header = raw[3:end]
            match = re.search(r"^title:\s*(.+)$", header, re.MULTILINE)
            if match:
                title = match.group(1).strip().strip("\"'")
            raw = raw[end + 4:]
    return title, raw


def clean_mdx(text):
    """Turn website markup into plain markdown.

    The docs are written for both Python and JavaScript. Python sections are
    wrapped in :::python and JS sections in :::js. We keep Python and drop JS.
    JSX components like <Tip> or <Icon /> are removed, but code blocks are
    left untouched.
    """
    lines_out = []
    skipping_js = False
    in_code = False

    for line in text.splitlines():
        stripped = line.strip()

        # Track code fences so we never edit code
        if stripped.startswith("```"):
            in_code = not in_code
            if not skipping_js:
                lines_out.append(line)
            continue

        if not in_code:
            if stripped == ":::js":
                skipping_js = True
                continue
            if stripped == ":::python":
                continue
            if stripped == ":::":
                skipping_js = False
                continue

        if skipping_js:
            continue

        if not in_code:
            # Drop import/export lines used by the docs website
            if re.match(r"^(import|export)\s", stripped):
                continue
            # Remove self-closing components: <Icon ... />, <ChatModelTabsPy />
            line = re.sub(r"<[A-Z][A-Za-z]*\b[^>]*/>", "", line)
            # Remove opening/closing components but keep their text: <Tip>, </Tip>
            line = re.sub(r"</?[A-Z][A-Za-z]*\b[^>]*>", "", line)
            # Docs shorthand @[`name`][ref] becomes just `name`
            line = re.sub(r"@\[([^\]]+)\](\[[^\]]*\])?", r"\1", line)

        lines_out.append(line)

    cleaned = "\n".join(lines_out)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def load_pages():
    """Read and clean every page. Returns a list of dicts."""
    pages = []
    for path in sorted(DOCS_DIR.rglob("*.mdx")):
        if SKIP_FOLDERS & set(path.relative_to(DOCS_DIR).parts):
            continue
        title, body = split_frontmatter(path.read_text(encoding="utf-8"))
        body = clean_mdx(body)
        if len(body) < 50:  # skip empty or stub pages
            continue
        rel = path.relative_to(DOCS_DIR).with_suffix("").as_posix()
        pages.append({
            "source": rel,
            "title": title or rel,
            "url": f"https://docs.langchain.com/oss/python/langchain/{rel}",
            "text": body,
        })
    return pages


# ---------- Step 3: chunk ----------

def chunk_pages(pages):
    """Split pages into chunks, trying to break at markdown headings first."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "id": f"{page['source']}::{i}",
                # Putting the page title on every chunk helps retrieval,
                # since a chunk from the middle of a page may not name its topic
                "text": f"Page: {page['title']}\n\n{piece}",
                "metadata": {
                    "source": page["source"],
                    "title": page["title"],
                    "url": page["url"],
                    "chunk": i,
                },
            })
    return chunks


# ---------- Step 4 and 5: embed and store ----------

def store_chunks(chunks):
    collection = get_collection()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embed([c["text"] for c in batch]),
        )
        log(f"  stored {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
    return collection.count()


def main():
    parser = argparse.ArgumentParser(description="Build the LangChain docs vector DB")
    parser.add_argument("--rebuild", action="store_true",
                        help="delete the existing database and build it again")
    args = parser.parse_args()

    if not DOCS_DIR.exists():
        raise SystemExit("Docs not found. Run: python rag_server/download_docs.py")

    if args.rebuild:
        log("Deleting existing database...")
        delete_collection()
    elif get_collection().count() > 0:
        log("Vector database already exists, nothing to do. Use --rebuild to start over.")
        return

    pages = load_pages()
    log(f"Loaded {len(pages)} pages")
    chunks = chunk_pages(pages)
    log(f"Split into {len(chunks)} chunks, embedding now...")
    total = store_chunks(chunks)
    log(f"Done. {total} chunks saved.")


if __name__ == "__main__":
    main()
