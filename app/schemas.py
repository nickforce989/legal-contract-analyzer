# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int


class AnalyzeRequest(BaseModel):
    document_id: str = Field(..., min_length=3)
    query: str = Field(..., min_length=5, max_length=3000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_id: int
    score: float
    text_excerpt: str


class AnalyzeResponse(BaseModel):
    document_id: str
    query: str
    answer: str
    sources: list[SourceChunk]


class DocumentMeta(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
