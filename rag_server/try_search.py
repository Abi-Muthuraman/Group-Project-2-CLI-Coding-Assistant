"""
Try the retriever from the terminal, without the agent.

Handy for testing and for the report's basic RAG vs CRAG comparison.

    python rag_server/try_search.py "how do I add memory to an agent"
    python rag_server/try_search.py "how do I add memory to an agent" --basic
"""

import argparse

from crag import crag_search, retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--basic", action="store_true",
                        help="plain top-k retrieval with no CRAG")
    args = parser.parse_args()

    if args.basic:
        print("Mode: basic RAG\n")
        for i, c in enumerate(retrieve(args.question), 1):
            print(f"{i}. {c['title']}  (similarity {c['similarity']:.2f})  {c['url']}")
    else:
        result = crag_search(args.question)
        print(f"Mode: CRAG  |  status: {result['action']}")
        if len(result["queries"]) > 1:
            print(f"Rewritten query: {result['queries'][1]}")
        print()
        for i, c in enumerate(result["chunks"], 1):
            print(f"{i}. {c['title']}  (relevance {c['score']:.2f})  {c['url']}")
        if not result["chunks"]:
            print("No relevant chunks found.")


if __name__ == "__main__":
    main()
