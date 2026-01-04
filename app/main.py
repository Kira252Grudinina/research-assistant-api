from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from pydantic import BaseModel

from .pipelines import (
    Paper,
    QueryResponse,
    run_research_pipeline
)


app = FastAPI(title="Research Assistant API")


@app.get("/")
async def read_root():
    return {"message": "API is running"}


class QueryRequest(BaseModel):
    query: str
    max_papers: Optional[int] = 3
    history: Optional[List[str]] = None   


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    return await run_research_pipeline(req.query, req.max_papers)




