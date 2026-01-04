from typing import List, Optional
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from enum import Enum
from datetime import datetime, timezone


from .llm_client import generate_answer
from .vector_store import search_papers
from .logging_utils import log_query


class Paper(BaseModel):
    title: str
    url: str
    published: str






class QueryResponse(BaseModel):
    answer: str
    papers: List[Paper]





_rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def recency_score(published_iso: str) -> float:
    published = datetime.fromisoformat(published_iso)
    now = datetime.now(timezone.utc)
    age_days = (now - published).days
    return 1.0 / (1.0 + age_days / 30.0)  


def _rerank(
    query: str,
    candidates: List[tuple[str, str, str, str]],
    recency_weight: float = 0.2,
    use_recency: bool = False,
) -> List[tuple[str, str, str, str]]:
    if not candidates:
        return candidates

    # 1) clean content for semantic scoring
    contents = [
        c[2]
        .replace("Year:", "")
        .replace("Title:", "")
        .replace("Abstract:", "")
        for c in candidates
    ]

    pairs = [(query, text) for text in contents]
    sem_scores = _rerank_model.predict(pairs)

    # 2) combine semantic + recency
    final = []
    for (pid, title, content, published), sem_score in zip(candidates, sem_scores):
        score = sem_score
        if use_recency:
            score += recency_weight * recency_score(published)
        final.append(((pid, title, content, published), score))

    # 3) sort by combined score
    final.sort(key=lambda x: x[1], reverse=True)

    return [c for (c, _) in final]

def wants_recent_papers(query: str) -> bool:
    q = query.lower()
    keywords = [
        "recent",
        "latest",
        "state of the art",
        "sota",
        "new",
        "current",
    ]
    return any(k in q for k in keywords)


async def run_research_pipeline(
    user_query: str,
    max_papers: Optional[int],
    history: Optional[List[str]] = None,
) -> QueryResponse:
    k = max_papers or 3

    retrieved = search_papers(user_query, k=30)
    use_recency = wants_recent_papers(user_query)
    reranked = _rerank(user_query, retrieved, use_recency=use_recency)

    # deduplicate by paper (not chunk)
    seen_papers = set()
    deduped = []

    for pid, title, content, published in reranked:
        base_pid = pid.split("_chunk_")[0]
        if base_pid not in seen_papers:
            seen_papers.add(base_pid)
            deduped.append((pid, title, content, published))

    top_k = deduped[:k]

    print("\n[DEBUG] Top retrieved papers:")
    for i, (pid, title, _, published) in enumerate(top_k[:10]):
        print(f"{i+1}. {published} — {title} — {pid}")
    print()

    context_chunks: List[str] = []
    papers_for_response: List[Paper] = []
    seen_paper_ids = set()

    for chunk_id, title, content, published in top_k:
        paper_id = chunk_id.split("_chunk_")[0]

        context_chunks.append(
    f"Title: {title}\nPublished: {published[:4]}\nContent:\n{content}\n---"
)


        if paper_id not in seen_paper_ids:
            seen_paper_ids.add(paper_id)
            papers_for_response.append(
                Paper(
                    title=title,
                    url=f"https://arxiv.org/abs/{paper_id}",
                    published=published[:10],
                )
            )


    # logging
    paper_ids_used = [
    p.url.split("/")[-1] for p in papers_for_response
]
    log_query(user_query, max_papers, paper_ids_used)

    # prompt assembly
    context_text = (
        "\n\n".join(context_chunks)
        if context_chunks
        else "No relevant papers found in the store."
    )

    history_text = ""
    if history:
        joined = "\n".join(history[-6:])
        history_text = f"Conversation so far:\n{joined}\n\n"

    prompt = (
        "Answer the user's question using ONLY the information in the context below.\n"
        "If the context does not contain enough information to answer fully, say so explicitly.\n"
        "Answer using only the provided context.\n"
        "When relevant, attribute claims by mentioning the paper title (not IDs).\n"
        "Attribute claims implicitly by phrasing (e.g. 'One paper shows that…').\n"
        "Do not introduce external papers or claims.\n"

        "=== CONTEXT START ===\n"
        f"{context_text}\n"
        "=== CONTEXT END ===\n\n"
        f"{history_text}"
        f"User question: {user_query}\n\n"
        "Answer in a neutral academic tone. Try to simplify the answer as much as possible to make it more readable."
    )

    answer_text = await generate_answer(prompt)

    return QueryResponse(answer=answer_text, papers=papers_for_response)





    # 3) SEARCH (default): run the normal RAG pipeline (history-aware)
    rag_response = await run_research_pipeline(user_query, max_papers, history)
    return rag_response



