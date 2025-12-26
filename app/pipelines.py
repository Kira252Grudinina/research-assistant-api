from typing import List, Optional
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from .llm_client import generate_answer, critique_question
from .vector_store import search_papers
from .logging_utils import log_query



class Paper(BaseModel):
    id: str          # arXiv id, e.g. 1706.03762v1
    title: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    papers: List[Paper]


class CritiqueResponse(BaseModel):
    critique: str


_rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _rerank(
    query: str,
    candidates: List[tuple[str, str, str]],
) -> List[tuple[str, str, str]]:
    if not candidates:
        return candidates

    contents = [c[2] for c in candidates]
    pairs = [(query, text) for text in contents]
    scores = _rerank_model.predict(pairs)

    indexed = list(zip(candidates, scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [c for (c, s) in indexed]


def run_research_pipeline(user_query: str, max_papers: Optional[int]) -> QueryResponse:
    k = max_papers or 3
    retrieved = search_papers(user_query, k=10)
    reranked = _rerank(user_query, retrieved)
    top_k = reranked[:k]

    context_chunks: List[str] = []
    papers_for_response: List[Paper] = []

    for paper_id, title, content in top_k:
        context_chunks.append(f"Title: {title}\nContent:\n{content}\n---")
        url = f"https://arxiv.org/abs/{paper_id}"
        papers_for_response.append(
            Paper(
                id=paper_id,
                title=title,
                url=url,
            )
        )

    paper_ids_used = [p.id for p in papers_for_response]
    log_query(user_query, max_papers, paper_ids_used)

    context_text = (
        "\n\n".join(context_chunks)
        if context_chunks
        else "No relevant papers found in the store."
    )

    prompt = (
        "You are a research assistant. Use ONLY the context below to answer.\n\n"
        "=== CONTEXT START ===\n"
        f"{context_text}\n"
        "=== CONTEXT END ===\n\n"
        f"User question: {user_query}\n\n"
        "Give a concise answer grounded in the context. "
        "If the context is insufficient, say that clearly."
    )

    answer_text = generate_answer(prompt)

    return QueryResponse(answer=answer_text, papers=papers_for_response)




def run_critique_pipeline(user_query: str) -> CritiqueResponse:
    critique_text = critique_question(
        f"Critique this research question: {user_query}"
    )
    return CritiqueResponse(critique=critique_text)
