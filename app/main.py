from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from .pipelines import (
    Paper,
    QueryResponse,
    CritiqueResponse,
    run_research_pipeline,
    run_critique_pipeline,
)

app = FastAPI(title="Research Assistant API")


@app.get("/")
def read_root():
    return {"message": "API is running"}


class QueryRequest(BaseModel):
    query: str
    max_papers: Optional[int] = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    return run_research_pipeline(req.query, req.max_papers)


@app.post("/critique", response_model=CritiqueResponse)
def critique_endpoint(req: QueryRequest):
    return run_critique_pipeline(req.query)
