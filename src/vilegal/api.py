"""REST API and search page: ``vilegal serve`` (builds the BM25 index at start-up, ~1 min)."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from vilegal import __version__
from vilegal.search import SearchEngine


class HitModel(BaseModel):
    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    query: str
    hits: list[HitModel]
    elapsed_ms: float


def create_app(engine: SearchEngine | None = None, data_dir: str = "data") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            from vilegal import data

            ds = data.load(data_dir)
            app.state.engine = SearchEngine(ds.doc_ids, ds.doc_texts)
        else:
            app.state.engine = engine
        yield

    app = FastAPI(
        title="vilegal",
        version=__version__,
        lifespan=lifespan,
        description="BM25 search over 61k articles of Vietnamese legal documents.",
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return (files("vilegal") / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: Annotated[str, Query(min_length=2, max_length=500)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
    ):
        t0 = time.perf_counter()
        hits = request.app.state.engine.search(q, k)
        return {
            "query": q,
            "hits": [h.__dict__ for h in hits],
            "elapsed_ms": round(1000 * (time.perf_counter() - t0), 1),
        }

    @app.get("/health")
    def health(request: Request) -> dict:
        e = request.app.state.engine
        return {"status": "ok", "documents": len(e.doc_ids), "vocabulary": len(e.bm25.vocab)}

    return app


app = create_app()
