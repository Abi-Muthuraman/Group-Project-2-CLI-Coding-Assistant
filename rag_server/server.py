"""
Custom RAG MCP server for the LangChain docs.

The coding assistant launches this as a subprocess and talks to it over
stdio. It exposes two tools:

  search_langchain_docs  ask a question, get back relevant doc chunks (CRAG)
  langchain_docs_info    check that the vector database is loaded

Run ingest.py once before starting this server.

Launch command for the MCP client:
    python rag_server/server.py
"""

try:
    from mcp.server.fastmcp import FastMCP  # mcp 1.x
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp 2.x renamed it

from config import CHROMA_DIR, GRADER_MODEL, GRADER_PROVIDER
from crag import crag_search
from store import get_collection, get_embedder, log

mcp = FastMCP("langchain-docs")


def format_result(question, result):
    """Turn the CRAG result into plain text the LLM can read."""
    lines = [f"CRAG status: {result['action']}"]
    if len(result["queries"]) > 1:
        lines.append(f"Query was rewritten to: {result['queries'][1]}")

    if not result["chunks"]:
        lines.append(
            f"No relevant LangChain documentation found for: {question}\n"
            "Try the web search tool instead."
        )
        return "\n".join(lines)

    lines.append("")
    for i, chunk in enumerate(result["chunks"], 1):
        lines.append(f"--- Result {i}: {chunk['title']} (relevance {chunk['score']:.2f})")
        lines.append(f"Source: {chunk['url']}")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def search_langchain_docs(query: str) -> str:
    """Search the official LangChain Python documentation.

    Use this when you need to know how a LangChain feature works: agents,
    tools, chat models, retrieval, memory, middleware, structured output,
    streaming, and similar topics. Pass a clear natural-language question.
    Returns the most relevant documentation sections with their source links.
    """
    log(f"search_langchain_docs: {query}")
    return format_result(query, crag_search(query))


@mcp.tool()
def langchain_docs_info() -> str:
    """Show how many documentation chunks are loaded and which models are used."""
    count = get_collection().count()
    return (
        f"Chunks in vector DB: {count}\n"
        f"DB location: {CHROMA_DIR}\n"
        f"Grader: {GRADER_PROVIDER} / {GRADER_MODEL}"
    )


if __name__ == "__main__":
    count = get_collection().count()
    if count == 0:
        log("Warning: vector DB is empty. Run: python rag_server/ingest.py")
    else:
        log(f"Loaded vector DB with {count} chunks")
    # Load the embedding model now so the first search isn't slow
    get_embedder()
    mcp.run(transport="stdio")
