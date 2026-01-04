# app/vector_store.py
import os
from typing import List, Tuple, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi
import hashlib
import pickle

db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")

client = chromadb.PersistentClient(path=db_dir)
collection = client.get_or_create_collection(name="papers")

model_name = "all-MiniLM-L6-v2"
_embedding_model = SentenceTransformer(model_name)
cache_path = os.path.join(db_dir, f"{model_name.replace('/','_')} embedding_cache.pkl")

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# load cache once
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        _embedding_cache = pickle.load(f)
else:
    _embedding_cache = {}

def _embed(texts: List[str]) -> List[List[float]]:
    embeddings: List[Optional[List[float]]] = [None] * len(texts)

    texts_to_embed = []
    indices_to_embed = []

    for i, text in enumerate(texts):
        h = _text_hash(text)
        if h in _embedding_cache:
            embeddings[i] = _embedding_cache[h]
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    if texts_to_embed:
        new_vectors = _embedding_model.encode(
            texts_to_embed,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        for idx, vec in zip(indices_to_embed, new_vectors):
            vec_list = vec.tolist()
            embeddings[idx] = vec_list
            _embedding_cache[_text_hash(texts[idx])] = vec_list

        # persist cache to disk
        with open(cache_path, "wb") as f:
            pickle.dump(_embedding_cache, f)

    return embeddings




def reset_collection() -> None:
    client.delete_collection(name="papers")

    global collection
    collection = client.get_or_create_collection(name="papers")

    # reset BM25 state
    global _bm25_index, _bm25_ids, _bm25_titles, _bm25_docs
    _bm25_ids = []
    _bm25_titles = []
    _bm25_docs = []
    _bm25_published = []




def add_papers(
    ids: List[str],
    documents: List[str],
    metadatas: List[dict],
) -> None:
    embeddings = _embed(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


# dense search (semantic search)

def semantic_search(query: str, k: int = 20):
    query_emb = _embed([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
    )
<<<<<<< HEAD
    ids = results["ids"][0]                         
    titles = [m.get("title", "") for m in results["metadatas"][0]]
    docs = results["documents"][0]
    return list(zip(ids, titles, docs))
=======
>>>>>>> 7017d2e (Refactor RAG pipeline and update README documentation)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return [
        (
            pid,
            m.get("title", ""),
            doc,
            m.get("published"),  # ← THIS is the key line
        )
        for pid, doc, m in zip(ids, docs, metas)
    ]



# sparse search (BM25)

_bm25_index: Optional[BM25Okapi] = None
_bm25_ids: List[str] = []
_bm25_titles: List[str] = []
_bm25_docs: List[str] = []
_bm25_published: List[str] = []


def _init_bm25() -> None:
    """Build BM25 index once from all stored documents."""
    global _bm25_index, _bm25_ids, _bm25_titles, _bm25_docs, _bm25_published

    if _bm25_index is not None:
        return

    results = collection.get(include=["documents", "metadatas"])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    ids = results.get("ids", [])

    if not docs:
        return

    _bm25_index = BM25Okapi([doc.split() for doc in docs])
    _bm25_ids = ids
    _bm25_titles = [m.get("title", "") for m in metas]
    _bm25_docs = docs
    _bm25_published = [m.get("published") for m in metas]



def keyword_search(query: str, k: int = 20):
    _init_bm25()
    if _bm25_index is None:
        return []

    scores = _bm25_index.get_scores(query.split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        (
            _bm25_ids[i],
            _bm25_titles[i],
            _bm25_docs[i],
            _bm25_published[i],
        )
        for i in ranked
    ]



# hybrid search

def hybrid_search(query: str, k_semantic: int = 20, k_keyword: int = 20) -> List[Tuple[str, str, str, int]]:
    """Merge semantic and keyword results, deduplicate by id."""
    dense = semantic_search(query, k=k_semantic)
    sparse = keyword_search(query, k=k_keyword)

    merged = {}
    for pid, title, doc, published in dense + sparse:
        merged[pid] = (pid, title, doc, published)



    return list(merged.values())
def search_papers(query: str, k: int = 3) -> List[Tuple[str, str, str, int]]:
    """Default search used by pipelines: hybrid dense + sparse, then cut to k."""
    merged = hybrid_search(query, k_semantic=20, k_keyword=20)
    return merged[:k]
