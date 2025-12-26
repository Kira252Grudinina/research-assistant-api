# app/vector_store.py

import os
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="papers")

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _embed(texts: List[str]) -> List[List[float]]:
    vectors = _embedding_model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def reset_collection() -> None:
    """
    Delete the existing 'papers' collection and recreate it empty.
    Call this once before re-ingesting from arXiv.
    """
    client.delete_collection(name="papers")
    global collection
    collection = client.get_or_create_collection(name="papers")


def add_papers(paper_ids, titles, contents, years):
    embeddings = _embed(contents)
    collection.add(
        ids=paper_ids,
        embeddings=embeddings,
        metadatas=[{"title": t, "year": y} for t, y in zip(titles, years)],
        documents=contents,
    )




def search_papers(query: str, k: int = 3) -> List[Tuple[str, str, str]]:
    query_emb = _embed([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
    )
    ids = results["ids"][0]                         # these are your arxiv_ids
    titles = [m.get("title", "") for m in results["metadatas"][0]]
    docs = results["documents"][0]
    return list(zip(ids, titles, docs))

