# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import Settings
from app.rag_engine import ContractRAGEngine
from app.schemas import (AnalyzeRequest, AnalyzeResponse, DocumentMeta,
                         UploadResponse)

settings = Settings.from_env()
engine = ContractRAGEngine(settings)

app = FastAPI(
    title="Legal Contract Analyzer",
    version="0.1.0",
    description=("RAG-style legal contract analysis with Hugging Face + "
                 "PyTorch + JAX + FAISS."),
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentMeta])
async def list_documents() -> list[DocumentMeta]:
    docs = await asyncio.to_thread(engine.list_documents)
    return [DocumentMeta(**d) for d in docs]


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or "uploaded_contract.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        meta = await asyncio.to_thread(engine.ingest_pdf, filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Failed to process upload: {exc}") from exc
    return UploadResponse(**meta)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = await asyncio.to_thread(engine.answer_query, payload.document_id,
                                         payload.query, payload.top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Analysis failed: {exc}") from exc
    return AnalyzeResponse(**result)


@app.get("/summary/{document_id}", response_model=AnalyzeResponse)
async def summary(document_id: str) -> AnalyzeResponse:
    try:
        result = await asyncio.to_thread(engine.summarize_contract, document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Summary failed: {exc}") from exc
    return AnalyzeResponse(**result)
