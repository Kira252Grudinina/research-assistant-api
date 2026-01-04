# Research Assistant 
## Retrieval-Augmented Generation (RAG) over arXiv. 

A research-oriented Retrieval-Augmented Generation (RAG) API for querying recent machine learning and AI papers from arXiv, with a strong focus on grounded answers, transparent retrieval, and reproducibility.

This project is designed as a technical portfolio piece showcasing end-to-end RAG system design, including ingestion, hybrid retrieval, cross-encoder reranking, and recency-aware ranking.

---

## Overview 

The system ingests recent arXiv papers, indexes them at chunk level, and answers user queries by:

- retrieving relevant paper chunks
- re-ranking results using a cross-encoder
- optionally favoring recent publications
- generating answers strictly grounded in retrieved context


## Key Features

### arXiv Ingestion Pipeline

- Fetches recent arXiv papers (currently focused on 2025)
- Supports ML- and AI-relevant categories:
  - cs.LG — Machine Learning
  - stat.ML — Statistical Machine Learning
  - cs.AI — Artificial Intelligence
  - cs.MA — Multi-Agent Systems
- Stores publication timestamps for recency-aware ranking

### Hybrid Retrieval

- Dense semantic search using Sentence Transformers
- Sparse keyword search using BM25

### Cross-Encoder Re-ranking

- Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (can run on CPU)
- Re-ranks retrieved chunks based on query–content

### Recency-Aware Ranking

- Detects recency intent in queries (using keywords: e.g, latest, recent, state of the art)
- Applies a time-decay score based on publication date during re-ranking

### Grounded Answer Generation

- Answers generated only from retrieved context
- Explicit citation of paper titles
- No hallucinated references

#### Fully runnable via Docker <img width="18" height="18" alt="image" src="https://github.com/user-attachments/assets/38862f22-d565-49f7-8d86-3601a31e421f" />


## API Usage

### Requirements

To run the API, you must provide a Groq API key, used for LLM-based answer generation.

You can obtain an API key from:

https://console.groq.com

**Set the key as an environment variable**:

```bash
export GROQ_API_KEY=your_api_key_here
```
**Run Locally with Docker**

- Build the Docker image:

```bash
docker build -t research-assistant .
```
- Run the container:
```bash
docker run -p 8000:8000 \
  -e GROQ_API_KEY=YOUR_API_KEY \
  research-assistant
```
- The API will be available at:

http://localhost:8000/docs 

## Intended Use

### This project is intended as:

- a research assistant prototype

- a portfolio project demonstrating RAG system design

- a foundation for experimentation in literature-aware NLP systems

It is not intended to be a production-grade scholarly search engine.

## Technical Stack

Python

FastAPI

Sentence Transformers

BM25

Cross-Encoders

Docker

arXiv API

Groq LLM API
