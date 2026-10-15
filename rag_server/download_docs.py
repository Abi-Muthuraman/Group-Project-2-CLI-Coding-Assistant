"""
Download the LangChain docs we want to search.

Uses a sparse git clone so we only pull the folder we need instead of the
whole docs repo. Run this once before ingest.py:

    python rag_server/download_docs.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from config import DOCS_DIR, DOCS_REPO_URL, DOCS_SUBFOLDER


def run(cmd, cwd=None):
    """Run a shell command and stop the script if it fails."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(f"Command failed: {' '.join(cmd)}")


def main():
    if DOCS_DIR.exists() and any(DOCS_DIR.rglob("*.mdx")):
        print(f"Docs already downloaded at {DOCS_DIR}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "docs"
        print("Cloning LangChain docs (this only grabs one folder)...")
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             DOCS_REPO_URL, str(clone_dir)])
        run(["git", "sparse-checkout", "set", DOCS_SUBFOLDER], cwd=clone_dir)

        # Copy just the docs folder into rag_server/data/
        DOCS_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(clone_dir / DOCS_SUBFOLDER, DOCS_DIR, dirs_exist_ok=True)

    count = len(list(DOCS_DIR.rglob("*.mdx")))
    print(f"Saved {count} doc pages to {DOCS_DIR}")


if __name__ == "__main__":
    main()
