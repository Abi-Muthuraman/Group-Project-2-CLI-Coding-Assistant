# Custom RAG MCP Server

A local MCP server that answers questions from the LangChain Python docs. It uses **Corrective RAG (CRAG)**, so it checks whether the retrieved chunks are actually relevant before passing them to the agent.

## How it works

**Build once (`ingest.py`):** load the docs, clean out website markup, split them into chunks of about 1000 characters, embed them with `all-MiniLM-L6-v2`, and save everything to Chroma in `rag_server/chroma_db/`. Every later session reuses this database.

**Every search (`crag.py`):**

1. Pull the top 5 chunks from Chroma.
2. Ask an LLM to score each chunk from 0 to 1 for relevance.
3. Decide based on the best score:
   - **CORRECT** (0.7 or higher): keep only the relevant chunks.
   - **AMBIGUOUS** (between 0.3 and 0.7): rewrite the query, search again, and merge the useful chunks from both searches.
   - **INCORRECT** (below 0.3): rewrite the query and try once more. If that still finds nothing, tell the agent to use web search.

If the grading LLM isn't available, it falls back to embedding similarity, so the tool never breaks.

## Setup

From the repo root:

```bash
pip install -r rag_server/requirements.txt

# Add GROQ_API_KEY to .env at the repo root (see rag_server/.env.example)

python rag_server/download_docs.py   # grabs the LangChain docs from GitHub
python rag_server/ingest.py          # builds the vector DB (one time only)
```

The first run of `ingest.py` downloads the embedding model, which takes a minute. To rebuild from scratch, run `python rag_server/ingest.py --rebuild`.

## Testing without the agent

```bash
# CRAG
python rag_server/try_search.py "how do I add memory to an agent"

# Basic RAG, for comparison
python rag_server/try_search.py "how do I add memory to an agent" --basic

# Call the MCP tools in a browser UI
npx @modelcontextprotocol/inspector python rag_server/server.py
```

## Connecting it to the agent

The MCP client launches it over stdio with:

```
command: python
args:    ["rag_server/server.py"]
```

Use an absolute path to `server.py` if the client runs from a different folder.

**Tools it exposes:**

| Tool | Input | What it returns |
|---|---|---|
| `search_langchain_docs` | `query` (string) | CRAG status, plus the relevant chunks with titles, scores, and source links |
| `langchain_docs_info` | none | Number of chunks loaded and the grader model in use |

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, chunk size, model names, CRAG thresholds |
| `download_docs.py` | Sparse-clones only the LangChain docs folder from GitHub |
| `ingest.py` | One-time build of the vector DB |
| `store.py` | Shared embedding model and Chroma helpers |
| `crag.py` | Retrieval, grading, query rewriting, and the CRAG decision |
| `server.py` | The MCP server |
| `try_search.py` | Command line test script, with a `--basic` mode for comparison |
