"""
Corrective RAG (CRAG).

Basic RAG grabs the top chunks and trusts them blindly. CRAG checks them first:

  1. Retrieve the top chunks from Chroma
  2. Ask an LLM to score how relevant each chunk is to the question (0 to 1)
  3. Decide what to do based on the best score:
       CORRECT    good match found -> keep only the relevant chunks
       AMBIGUOUS  partly relevant  -> keep the partial ones, also rewrite the
                                      query and search again, then merge
       INCORRECT  nothing useful   -> rewrite the query and try once more.
                                      If that also fails, tell the agent to
                                      use web search instead.

Based on the CRAG notebook in the NirDiamant RAG_Techniques repo. In the
original paper the INCORRECT case falls back to web search directly. We leave
that to the agent, since Tavily is already one of its tools.
"""

import json
import re

from config import (GRADER_MODEL, GRADER_PROVIDER, IRRELEVANT_THRESHOLD,
                    RELEVANT_THRESHOLD, TOP_K)
from store import embed, get_collection, log

_llm = None


# ---------- LLM used for grading and rewriting ----------

def get_llm():
    """Create the grading LLM once. Groq by default, Ollama if set in .env."""
    global _llm
    if _llm is None:
        if GRADER_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama
            _llm = ChatOllama(model=GRADER_MODEL, temperature=0)
        else:
            from langchain_groq import ChatGroq
            _llm = ChatGroq(model=GRADER_MODEL, temperature=0)
        log(f"CRAG grader: {GRADER_PROVIDER} / {GRADER_MODEL}")
    return _llm


def ask_llm(prompt):
    return get_llm().invoke(prompt).content.strip()


# ---------- Step 1: plain retrieval (this alone is basic RAG) ----------

def retrieve(query, k=TOP_K):
    """Return the k closest chunks from Chroma, best first."""
    result = get_collection().query(query_embeddings=embed([query]), n_results=k)
    chunks = []
    for cid, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                     result["metadatas"][0], result["distances"][0]):
        chunks.append({
            "id": cid,
            "text": text,
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "similarity": round(1 - dist, 3),  # cosine similarity
        })
    return chunks


# ---------- Step 2: grade ----------

GRADE_PROMPT = """You are checking search results for a question about the LangChain library.

Question: {question}

Here are {n} documentation chunks. For each one, give a relevance score from 0 to 1:
1.0 = directly answers the question
0.5 = related but only partly helpful
0.0 = unrelated

{chunks}

Reply with only JSON in this exact format, one score per chunk in order:
{{"scores": [0.0, 0.0]}}"""


def grade(question, chunks):
    """Score every chunk in one LLM call. Adds a 'score' key to each chunk.

    If the LLM can't be reached (no API key, Ollama not running), fall back
    to the embedding similarity so the tool still works.
    """
    if not chunks:
        return chunks

    listed = "\n\n".join(
        f"[Chunk {i + 1}]\n{c['text'][:800]}" for i, c in enumerate(chunks)
    )
    try:
        reply = ask_llm(GRADE_PROMPT.format(question=question, n=len(chunks), chunks=listed))
        scores = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))["scores"]
        if len(scores) != len(chunks):
            raise ValueError(f"expected {len(chunks)} scores, got {len(scores)}")
        for chunk, score in zip(chunks, scores):
            chunk["score"] = max(0.0, min(1.0, float(score)))
    except Exception as err:
        log(f"Grading failed ({err}), using embedding similarity instead")
        for chunk in chunks:
            chunk["score"] = chunk["similarity"]
    return chunks


# ---------- Step 3: rewrite the query ----------

REWRITE_PROMPT = """Rewrite this question so it is more likely to match the official LangChain Python documentation.
Use the terms the docs would use (class names, function names, concepts). Keep it to one line.

Question: {question}

Rewritten question:"""


def rewrite(question):
    try:
        new = ask_llm(REWRITE_PROMPT.format(question=question)).splitlines()[0]
        return new.strip().strip('"') or question
    except Exception as err:
        log(f"Rewrite failed ({err}), keeping the original question")
        return question


# ---------- The full CRAG flow ----------

def classify(chunks):
    """Label a set of graded chunks by their best score."""
    best = max((c["score"] for c in chunks), default=0.0)
    if best >= RELEVANT_THRESHOLD:
        return "CORRECT"
    if best < IRRELEVANT_THRESHOLD:
        return "INCORRECT"
    return "AMBIGUOUS"


def merge(*chunk_lists):
    """Combine chunk lists, dropping duplicates and keeping the higher score."""
    by_id = {}
    for chunk in (c for lst in chunk_lists for c in lst):
        if chunk["id"] not in by_id or chunk["score"] > by_id[chunk["id"]]["score"]:
            by_id[chunk["id"]] = chunk
    return sorted(by_id.values(), key=lambda c: c["score"], reverse=True)


def crag_search(question):
    """Run CRAG and return a dict the server can format for the agent."""
    queries = [question]
    first = grade(question, retrieve(question))
    action = classify(first)
    log(f"CRAG first pass: {action}")

    if action == "CORRECT":
        kept = [c for c in first if c["score"] >= RELEVANT_THRESHOLD]

    else:
        # Both AMBIGUOUS and INCORRECT try a second search with a better query
        better = rewrite(question)
        if better != question:
            queries.append(better)
            # Grade against the original question, since that's what the user asked
            second = grade(question, retrieve(better))
            log(f"CRAG second pass with '{better}': {classify(second)}")
        else:
            second = []  # rewrite didn't change anything, no point searching again

        if action == "AMBIGUOUS":
            kept = [c for c in merge(first, second) if c["score"] >= IRRELEVANT_THRESHOLD]
        else:  # INCORRECT
            kept = [c for c in second if c["score"] >= RELEVANT_THRESHOLD]
            action = "CORRECT_AFTER_REWRITE" if kept else "INCORRECT"

    return {
        "action": action,
        "queries": queries,
        "chunks": kept[:TOP_K],
    }
