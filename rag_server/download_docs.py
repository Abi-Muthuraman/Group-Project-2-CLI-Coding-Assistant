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

from config import (DOCS_DIR, DOCS_REPO_URL, DOCS_SUBFOLDER, SNIPPETS_DIR,
                    SNIPPETS_SUBFOLDER)


def run(cmd, cwd=None):
    """Run a shell command and stop the script if it fails."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(f"Command failed: {' '.join(cmd)}")


def main():
    if any(DOCS_DIR.rglob("*.mdx")) and any(SNIPPETS_DIR.rglob("*.mdx")):
        print(f"Docs already downloaded at {DOCS_DIR.parent}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "docs"
        print("Cloning LangChain docs (only the folders we need)...")
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             DOCS_REPO_URL, str(clone_dir)])
        run(["git", "sparse-checkout", "set", DOCS_SUBFOLDER, SNIPPETS_SUBFOLDER], cwd=clone_dir)

        # Copy the doc pages and the code snippets they import into rag_server/data/
        DOCS_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(clone_dir / DOCS_SUBFOLDER, DOCS_DIR, dirs_exist_ok=True)
        shutil.copytree(clone_dir / SNIPPETS_SUBFOLDER, SNIPPETS_DIR, dirs_exist_ok=True)

    pages = len(list(DOCS_DIR.rglob("*.mdx")))
    snippets = len(list(SNIPPETS_DIR.rglob("*.mdx")))
    print(f"Saved {pages} doc pages and {snippets} code snippets to {DOCS_DIR.parent}")


if __name__ == "__main__":
    main()
