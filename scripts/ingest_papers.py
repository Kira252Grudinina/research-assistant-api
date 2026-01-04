# scripts/ingest_papers.py
import arxiv
from app.vector_store import add_papers, reset_collection


def fetch_arxiv_papers(
    query: str,
    max_results: int = 5000,
):
    ids, titles, contents, published_dates = [], [], [], []

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    for result in arxiv.Client().results(search):
        published = result.published

        # keep ONLY 2025
        if published.year != 2025:
            continue

        arxiv_id = result.get_short_id()
        title = result.title.strip()
        abstract = result.summary.strip()

        ids.append(arxiv_id)
        titles.append(title)
        contents.append(f"Title: {title}\nAbstract: {abstract}")
        published_dates.append(published.isoformat())

    return ids, titles, contents, published_dates


def chunk_text(text, chunk_size=280, overlap=50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if len(chunk_words) < 50:
            break
        chunks.append(" ".join(chunk_words))

    return chunks


if __name__ == "__main__":
    CATEGORY_QUERY = (
        "cat:cs.AI OR cat:cs.LG OR cat:stat.ML OR cat:cs.MA OR cat:cs.AR"
    )

    paper_ids, titles, contents, published_dates = fetch_arxiv_papers(
        CATEGORY_QUERY,
        max_results=5000,
    )

    reset_collection()

    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []

    for paper_id, title, content, published in zip(
        paper_ids, titles, contents, published_dates
    ):
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_ids.append(f"{paper_id}_chunk_{i}")
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                "paper_id": paper_id,
                "title": title,
                "published": published,
                "chunk_id": i,
            })

    add_papers(
        ids=chunk_ids,
        documents=chunk_texts,
        metadatas=chunk_metadatas,
    )

    print(f"Ingested {len(chunk_ids)} chunks into Chroma.")
