# scripts/ingest_papers.py

import arxiv
from app.vector_store import add_papers, reset_collection


def fetch_arxiv_papers(query: str, max_results: int = 200, min_year: int = 2021):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    ids = []
    titles = []
    contents = []
    years = []

    for result in arxiv.Client().results(search):
        year = result.published.year
        if year < min_year:
            continue

        arxiv_id = result.get_short_id()
        title = result.title.strip()
        abstract = result.summary.strip()

        ids.append(arxiv_id)
        titles.append(title)
        contents.append(f"Year: {year}\nTitle: {title}\nAbstract: {abstract}")
        years.append(year)

    return ids, titles, contents, years



if __name__ == "__main__":
    query = "cat:cs.AI OR cat:cs.LG OR cat:stat.ML"
    paper_ids, titles, contents, years = fetch_arxiv_papers(query, max_results=200)

    reset_collection()
    add_papers(paper_ids, titles, contents, years)

    print(f"Ingested {len(paper_ids)} arXiv papers into Chroma.")
