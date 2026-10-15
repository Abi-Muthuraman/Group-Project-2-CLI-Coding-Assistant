"""
Settings for the custom RAG server.

Everything lives in one place so the other scripts don't hardcode paths or
model names. Values can be overridden with environment variables in the
repo's .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Folder that holds this file (rag_server/) and the repo root above it
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# Load API keys and overrides from the repo's .env file
load_dotenv(REPO_ROOT / ".env")

# ---- Where things are stored ----
# Raw LangChain docs downloaded by download_docs.py
DOCS_DIR = BASE_DIR / "data" / "langchain_docs"
# Code examples live in separate snippet files that the doc pages import
SNIPPETS_DIR = BASE_DIR / "data" / "snippets"
# Chroma saves the vector database here so we only build it once
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "langchain_docs"

# ---- Where the docs come from ----
DOCS_REPO_URL = "https://github.com/langchain-ai/docs.git"
# Only the core LangChain Python docs, not every integration page
DOCS_SUBFOLDER = "src/oss/langchain"
SNIPPETS_SUBFOLDER = "src/snippets"
# Subfolders to skip: the frontend pages are React/TypeScript, not Python
SKIP_FOLDERS = {"frontend"}

# ---- Chunking ----
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks

# ---- Embeddings ----
# Small, fast sentence-transformers model (ONNX build, no PyTorch needed)
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---- Retrieval ----
TOP_K = 5  # how many chunks to pull from Chroma per search

# ---- CRAG grading ----
# Which LLM grades the retrieved chunks: "groq" (cloud) or "ollama" (local)
GRADER_PROVIDER = os.getenv("CRAG_PROVIDER", "groq").lower()
GRADER_MODEL = os.getenv(
    "CRAG_MODEL",
    "openai/gpt-oss-20b" if GRADER_PROVIDER == "groq" else "llama3.1",
)

# A chunk scoring at or above this counts as relevant
RELEVANT_THRESHOLD = 0.7
# If every chunk scores below this, retrieval failed
IRRELEVANT_THRESHOLD = 0.3
