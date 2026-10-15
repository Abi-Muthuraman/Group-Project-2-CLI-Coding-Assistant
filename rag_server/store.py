"""
Shared helpers for the embedding model and the Chroma vector database.

Both ingest.py (writes to the DB) and crag.py (reads from it) use these,
so they always agree on the model, the DB location, and the collection.
"""

import sys

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

_model = None  # loaded once and reused


def log(message):
    """Print to stderr. The MCP server talks to the agent over stdout, so
    anything printed to stdout would break the connection."""
    print(message, file=sys.stderr, flush=True)


def get_embedder():
    """Load the embedding model the first time it's needed.

    This is the same all-MiniLM-L6-v2 sentence-transformers model, but the
    ONNX version that ships with Chroma. It runs without PyTorch, which
    avoids version problems on Intel Macs. The model downloads once
    (about 80 MB) and is cached after that.
    """
    global _model
    if _model is None:
        log(f"Loading embedding model: {EMBED_MODEL} (ONNX)")
        _model = ONNXMiniLM_L6_V2()
    return _model


def embed(texts):
    """Turn a list of strings into a list of vectors."""
    return [vector.tolist() for vector in get_embedder()(texts)]


def get_collection():
    """Open (or create) the Chroma collection saved on disk.

    Cosine distance means distance = 1 - similarity, which makes scores
    easy to read: 0 is a perfect match, 1 is unrelated.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def delete_collection():
    """Remove the saved collection so it can be rebuilt from scratch."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete yet
