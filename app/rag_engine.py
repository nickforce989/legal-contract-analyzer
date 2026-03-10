# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import json
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

# macOS can load duplicate OpenMP runtimes when mixing some prebuilt wheels.
# Allow process startup instead of hard-aborting with OMP Error #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import jax
import jax.numpy as jnp
import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

from app.config import Settings
from app.pdf_utils import chunk_text, extract_text_from_pdf

_WORD_RE = re.compile(r"[A-Za-z0-9]{3,}")
_TIME_PHRASE_RE = re.compile(
    r"\b(?:\d+\s*(?:day|days|week|weeks|month|months|year|years)"
    r"|within\s+\d+|not less than|no later than|before the end of"
    r"|starting with the day)\b",
    re.IGNORECASE,
)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```",
                            re.IGNORECASE | re.DOTALL)
_CITATION_ID_RE = re.compile(r"[Cc](\d+)")
_NAME_LINE_RE = re.compile(r"Name:\s*([^\n\r]+)")
_PARTY_QUERY_RE = re.compile(
    r"\b(who|name|names|identify|list)\b.*\b(contract[\s-]?holders?|tenant|"
    r"landlord|parties?|agents?|principal\s+contact)\b"
    r"|\b(contract[\s-]?holders?|tenant|landlord|parties?|agents?|principal\s+"
    r"contact)\b.*\b(who|name|names|identify|list)\b",
    re.IGNORECASE,
)
_PARTY_IDENTITY_BLOCKLIST = {
    "obligation",
    "obligations",
    "responsibility",
    "responsibilities",
    "must",
    "shall",
    "rent",
    "payment",
    "notice",
    "term",
    "deposit",
    "rights",
    "deadline",
    "penalty",
    "default",
    "termination",
}
_ROLE_LABELS = {
    "contract_holder": ("the contract-holder",),
    "landlord": ("the landlord",),
    "agent": ("the landlord's agent", "the landlord’s agent"),
}
_OCR_HEADER_RE = re.compile(
    r"^(?:\d+\s+[A-Z]\+?\s*(?:FT\s*&\s*PT|FT|PT)?\s*){1,4}",
)
_MULTISPACE_RE = re.compile(r"\s+")


@jax.jit
def _jax_cosine_similarity(query: jnp.ndarray,
                           docs: jnp.ndarray) -> jnp.ndarray:
    query = query / (jnp.linalg.norm(query) + 1e-8)
    docs = docs / (jnp.linalg.norm(docs, axis=1, keepdims=True) + 1e-8)
    return jnp.sum(docs * query, axis=1)


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str


@dataclass
class ExtractedFact:
    clause_type: str
    statement: str
    obligation_party: str
    timeline: str
    risk_level: str
    citations: list[int]


class HFEmbeddingEncoder:

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @staticmethod
    def _mean_pool(hidden_state: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        summed = torch.sum(hidden_state * mask, dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int = 12) -> np.ndarray:
        if not texts:
            raise ValueError("encode received empty input")

        batches: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(batch,
                                     padding=True,
                                     truncation=True,
                                     max_length=512,
                                     return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
                pooled = self._mean_pool(output.last_hidden_state,
                                         encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
            batches.append(normalized.cpu().numpy().astype("float32"))
        return np.concatenate(batches, axis=0)


class HFPairReranker:

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name).to(device)
        self.model.eval()

    def score_pairs(self, query: str, docs: list[str],
                    batch_size: int) -> np.ndarray:
        if not docs:
            return np.array([], dtype="float32")
        outputs: list[np.ndarray] = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_queries = [query] * len(batch_docs)
            encoded = self.tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            if logits.ndim == 2:
                logits = logits[:, 0]
            outputs.append(logits.detach().cpu().numpy().astype("float32"))
        return np.concatenate(outputs, axis=0)


class HFTextGenerator:

    def __init__(self, model_name: str, device: str, dtype: torch.dtype,
                 max_new_tokens: int, temperature: float, top_p: float):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.max_context_tokens = self._resolve_context_window()
        reserve = min(self.max_new_tokens, max(64, self.max_context_tokens // 2))
        self.max_input_tokens = max(256, self.max_context_tokens - reserve - 16)

    def _resolve_context_window(self) -> int:
        candidates: list[int] = []
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100_000:
            candidates.append(tokenizer_limit)
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            candidates.append(model_limit)
        return min(candidates) if candidates else 2048

    def _build_model_prompt(self, user_prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{
                "role":
                "system",
                "content":
                "You are a senior contract review assistant. Use only the "
                "provided evidence and never invent missing facts.",
            }, {
                "role": "user",
                "content": user_prompt
            }]
            return self.tokenizer.apply_chat_template(messages,
                                                      tokenize=False,
                                                      add_generation_prompt=True)
        return (
            "System: You analyze legal contracts and explain clauses in plain "
            "English. Be precise and avoid legal hallucinations.\n"
            f"User: {user_prompt}\nAssistant:")

    def generate(self, user_prompt: str) -> str:
        prompt = self._build_model_prompt(user_prompt)
        inputs = self.tokenizer(prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.max_input_tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "repetition_penalty": 1.08,
                "no_repeat_ngram_size": 3,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
                generation_kwargs["top_p"] = self.top_p
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if generated.strip():
            return generated.strip()

        # Fallback for tokenizers/models that may not preserve prompt boundaries.
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if full_text.startswith(prompt):
            full_text = full_text[len(prompt):]
        return full_text.strip()


class TextGenerator(Protocol):
    model_name: str

    def generate(self, user_prompt: str) -> str:
        ...


class OpenAICompatibleRemoteGenerator:

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        timeout_seconds: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        parsed = urlparse(self.base_url)
        host = (parsed.hostname or "").lower()
        is_local_endpoint = host in {"localhost", "127.0.0.1", "::1"}
        if not self.api_key and not is_local_endpoint:
            raise RuntimeError(
                "Remote LLM API key missing. Set LEGAL_ANALYZER_REMOTE_LLM_API_KEY "
                "(or OPENROUTER_API_KEY / OPENAI_API_KEY).")

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    # Common OpenAI-compatible multimodal schema variants.
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        return ""

    def generate(self, user_prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "legal-contract-analyzer/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["x-api-key"] = self.api_key
            headers["api-key"] = self.api_key

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{
                "role":
                "system",
                "content":
                "You are a senior contract review assistant. Use only the "
                "provided evidence and never invent missing facts.",
            }, {
                "role": "user",
                "content": user_prompt
            }],
            "max_tokens": self.max_new_tokens,
        }
        # Some providers reject these when not sampling.
        if self.temperature > 0:
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Remote LLM request failed: {exc}") from exc

        if not response.ok:
            detail = response.text[:500]
            raise RuntimeError(
                "Remote LLM request failed with "
                f"status {response.status_code}: {detail}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Remote LLM returned non-JSON response") from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Remote LLM returned no choices")

        message = choices[0].get("message", {})
        content = self._extract_text_content(message.get("content", ""))
        if content:
            return content

        # Some compatible providers can return plain text directly.
        fallback_text = choices[0].get("text", "")
        if isinstance(fallback_text, str) and fallback_text.strip():
            return fallback_text.strip()
        raise RuntimeError("Remote LLM response contained no text content")


class ContractRAGEngine:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)

        self._embedder: HFEmbeddingEncoder | None = None
        self._generator: TextGenerator | None = None
        self._reranker: HFPairReranker | None = None
        self._reranker_load_attempted = False
        self._active_llm_model_name: str | None = None
        self._model_lock = threading.Lock()

    def _ensure_embedder(self) -> None:
        with self._model_lock:
            if self._embedder is None:
                self._embedder = HFEmbeddingEncoder(
                    self.settings.embedding_model_name, self.settings.device)

    def _ensure_generator(self) -> None:
        self._ensure_embedder()
        with self._model_lock:
            if self._generator is None:
                if self.settings.llm_mode == "remote":
                    try:
                        self._generator = OpenAICompatibleRemoteGenerator(
                            base_url=self.settings.remote_llm_base_url,
                            api_key=self.settings.remote_llm_api_key,
                            model_name=self.settings.remote_llm_model_name,
                            max_new_tokens=self.settings.max_new_tokens,
                            temperature=self.settings.temperature,
                            top_p=self.settings.top_p,
                            timeout_seconds=self.settings.
                            remote_llm_timeout_seconds,
                        )
                        self._active_llm_model_name = (
                            self.settings.remote_llm_model_name)
                        return
                    except Exception as remote_exc:
                        fallback = (self.settings.fallback_llm_model_name
                                    or self.settings.llm_model_name)
                        print(
                            "Remote LLM init failed; falling back to local model "
                            f"'{fallback}'. Reason: {remote_exc}")
                        self._generator = HFTextGenerator(
                            model_name=fallback,
                            device=self.settings.device,
                            dtype=self.settings.torch_dtype,
                            max_new_tokens=self.settings.max_new_tokens,
                            temperature=self.settings.temperature,
                            top_p=self.settings.top_p,
                        )
                        self._active_llm_model_name = fallback
                        return

                primary = self.settings.llm_model_name
                try:
                    self._generator = HFTextGenerator(
                        model_name=primary,
                        device=self.settings.device,
                        dtype=self.settings.torch_dtype,
                        max_new_tokens=self.settings.max_new_tokens,
                        temperature=self.settings.temperature,
                        top_p=self.settings.top_p,
                    )
                    self._active_llm_model_name = primary
                except Exception as primary_exc:
                    fallback = self.settings.fallback_llm_model_name
                    if not fallback or fallback == primary:
                        raise RuntimeError(
                            f"Failed to load LLM model '{primary}': {primary_exc}"
                        ) from primary_exc
                    print(
                        "Primary LLM load failed "
                        f"('{primary}'). Falling back to '{fallback}'. "
                        f"Reason: {primary_exc}"
                    )
                    self._generator = HFTextGenerator(
                        model_name=fallback,
                        device=self.settings.device,
                        dtype=self.settings.torch_dtype,
                        max_new_tokens=self.settings.max_new_tokens,
                        temperature=self.settings.temperature,
                        top_p=self.settings.top_p,
                    )
                    self._active_llm_model_name = fallback

    def _ensure_reranker(self) -> None:
        if not self.settings.use_cross_encoder_rerank:
            return
        with self._model_lock:
            if self._reranker is not None or self._reranker_load_attempted:
                return
            self._reranker_load_attempted = True
            try:
                self._reranker = HFPairReranker(
                    model_name=self.settings.cross_encoder_model_name,
                    device=self.settings.device,
                )
            except Exception as exc:
                self._reranker = None
                print("Cross-encoder reranker unavailable; continuing without it. "
                      f"Reason: {exc}")

    def _fallback_to_local_generator(self, reason: Exception | str) -> None:
        reason_text = str(reason)
        with self._model_lock:
            if isinstance(self._generator, HFTextGenerator):
                return
            fallback = (self.settings.fallback_llm_model_name
                        or self.settings.llm_model_name)
            print(
                "Remote LLM failed, falling back to local model "
                f"'{fallback}'. Reason: {reason_text}")
            self._generator = HFTextGenerator(
                model_name=fallback,
                device=self.settings.device,
                dtype=self.settings.torch_dtype,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
            )
            self._active_llm_model_name = fallback

    def _safe_generate(self, prompt: str) -> str:
        self._ensure_generator()
        if self._generator is None:
            raise RuntimeError("LLM is not initialized")
        try:
            return self._generator.generate(prompt)
        except Exception as exc:
            if self.settings.llm_mode != "remote":
                raise
            self._fallback_to_local_generator(exc)
            if self._generator is None:
                raise RuntimeError("Fallback LLM is not initialized") from exc
            return self._generator.generate(prompt)

    @staticmethod
    def _doc_dir(root: Path, document_id: str) -> Path:
        return root / document_id

    def _save_document(self, document_id: str, metadata: dict[str, Any],
                       chunks: list[str], embeddings: np.ndarray,
                       index: faiss.Index) -> None:
        doc_dir = self._doc_dir(self.settings.data_dir, document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        with (doc_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)
        with (doc_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=True, indent=2)
        np.save(doc_dir / "embeddings.npy", embeddings)
        faiss.write_index(index, str(doc_dir / "index.faiss"))

    def _load_document(self, document_id: str) -> dict[str, Any]:
        doc_dir = self._doc_dir(self.settings.data_dir, document_id)
        if not doc_dir.exists():
            raise KeyError(f"document_id {document_id} not found")

        with (doc_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        with (doc_dir / "chunks.json").open("r", encoding="utf-8") as f:
            chunks: list[str] = json.load(f)
        embeddings = np.load(doc_dir / "embeddings.npy").astype("float32")
        index = faiss.read_index(str(doc_dir / "index.faiss"))

        return {
            "meta": meta,
            "chunks": chunks,
            "embeddings": embeddings,
            "index": index,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        if not self.settings.data_dir.exists():
            return docs
        for doc_dir in sorted(self.settings.data_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            meta_path = doc_dir / "meta.json"
            if not meta_path.exists():
                continue
            with meta_path.open("r", encoding="utf-8") as f:
                docs.append(json.load(f))
        return docs

    def ingest_pdf(self, filename: str, pdf_bytes: bytes) -> dict[str, Any]:
        self._ensure_embedder()
        if self._embedder is None:
            raise RuntimeError("Embedding model is not initialized")

        text, num_pages = extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            raise ValueError("No extractable text found in PDF")

        chunks = chunk_text(text, self.settings.chunk_size,
                            self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("Failed to build text chunks from the PDF")

        embeddings = self._embedder.encode(
            chunks, batch_size=self.settings.embedding_batch_size)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise RuntimeError("Embedding model returned invalid shape")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        document_id = uuid.uuid4().hex
        metadata = {
            "document_id": document_id,
            "filename": filename,
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "embedding_model_name": self.settings.embedding_model_name,
            "model_profile": self.settings.model_profile,
        }
        self._save_document(document_id, metadata, chunks, embeddings, index)
        return metadata

    @staticmethod
    def _jax_rerank(query_embedding: np.ndarray,
                    candidate_embeddings: np.ndarray) -> np.ndarray:
        query_jax = jnp.asarray(query_embedding, dtype=jnp.float32)
        docs_jax = jnp.asarray(candidate_embeddings, dtype=jnp.float32)
        return np.asarray(_jax_cosine_similarity(query_jax, docs_jax))

    def _retrieve(self, document_id: str, query: str, top_k: int,
                  candidate_factor: int | None = None) -> list[RetrievedChunk]:
        self._ensure_embedder()
        if self._embedder is None:
            raise RuntimeError("Embedding model is not initialized")

        doc = self._load_document(document_id)
        stored_embed_model = str(doc["meta"].get("embedding_model_name", ""))
        if (stored_embed_model
                and stored_embed_model != self.settings.embedding_model_name):
            raise ValueError(
                "This document was indexed with embedding model "
                f"'{stored_embed_model}', but server is using "
                f"'{self.settings.embedding_model_name}'. Re-upload the PDF or "
                "set LEGAL_ANALYZER_EMBED_MODEL to the original value.")

        chunks: list[str] = doc["chunks"]
        embeddings: np.ndarray = doc["embeddings"]
        index: faiss.Index = doc["index"]

        query_embedding = self._embedder.encode(
            [query], batch_size=1)[0].astype("float32")
        if query_embedding.shape[0] != embeddings.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch between stored document vectors "
                f"({embeddings.shape[1]}) and active embedding model "
                f"({query_embedding.shape[0]}). Re-upload the document with the "
                "current embedding model or restart with the original model.")

        resolved_factor = candidate_factor
        if resolved_factor is None:
            resolved_factor = self.settings.retrieval_candidate_factor
        resolved_factor = max(1, int(resolved_factor))

        fetch_k = min(len(chunks), max(top_k, top_k * resolved_factor))
        _, indices = index.search(query_embedding.reshape(1, -1), fetch_k)
        candidate_ids = [int(i) for i in indices[0] if i >= 0]
        if not candidate_ids:
            return []

        candidate_embeddings = embeddings[candidate_ids]
        dense_scores = self._jax_rerank(query_embedding, candidate_embeddings)
        query_tokens = self._tokenize_legal_words(query)
        lexical_scores = np.asarray([
            self._lexical_overlap_score(query_tokens, chunks[cid])
            for cid in candidate_ids
        ],
                                   dtype="float32")

        lexical_weight = float(
            np.clip(self.settings.lexical_rerank_weight, 0.0, 1.0))
        dense_norm = self._normalize_scores(dense_scores)
        lexical_norm = self._normalize_scores(lexical_scores)
        fused_scores = ((1.0 - lexical_weight) * dense_norm) + (lexical_weight *
                                                                 lexical_norm)

        cross_weight = float(np.clip(self.settings.cross_encoder_weight, 0.0, 1.0))
        if self.settings.use_cross_encoder_rerank and cross_weight > 0.0:
            self._ensure_reranker()
            if self._reranker is not None:
                try:
                    cross_scores = self._reranker.score_pairs(
                        query,
                        [chunks[cid] for cid in candidate_ids],
                        batch_size=self.settings.cross_encoder_batch_size,
                    )
                    cross_norm = self._normalize_scores(cross_scores)
                    fused_norm = self._normalize_scores(fused_scores)
                    fused_scores = ((1.0 - cross_weight) * fused_norm +
                                   (cross_weight * cross_norm))
                except Exception as exc:
                    print(
                        "Cross-encoder scoring failed; using dense+lexical scores. "
                        f"Reason: {exc}")

        reranked_order = np.argsort(-fused_scores)

        retrieved: list[RetrievedChunk] = []
        for idx in reranked_order[:top_k]:
            chunk_id = candidate_ids[int(idx)]
            retrieved.append(
                RetrievedChunk(chunk_id=chunk_id,
                               score=float(fused_scores[int(idx)]),
                               text=chunks[chunk_id]))
        return retrieved

    @staticmethod
    def _tokenize_legal_words(text: str) -> set[str]:
        return {tok.lower() for tok in _WORD_RE.findall(text)}

    @staticmethod
    def _lexical_overlap_score(query_tokens: set[str], chunk_text: str) -> float:
        if not query_tokens:
            return 0.0
        chunk_tokens = ContractRAGEngine._tokenize_legal_words(chunk_text)
        if not chunk_tokens:
            return 0.0
        overlap = len(query_tokens & chunk_tokens)
        norm = np.sqrt(len(query_tokens) * len(chunk_tokens))
        return float(overlap / norm) if norm > 0 else 0.0

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if (max_score - min_score) < 1e-9:
            return np.full_like(scores, 0.5, dtype="float32")
        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def _build_context_block(retrieved: list[RetrievedChunk]) -> str:
        context_blocks = []
        for item in retrieved:
            context_blocks.append(f"[C{item.chunk_id}]\n{item.text[:950]}")
        return "\n\n".join(context_blocks)

    @staticmethod
    def _build_qa_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
        highlights = ContractRAGEngine._extract_evidence_highlights(
            query, retrieved)
        context = ContractRAGEngine._build_context_block(retrieved)
        highlight_text = "\n".join(highlights) if highlights else "- None"

        return (
            "Task: answer a legal contract question using ONLY the evidence "
            "below.\n"
            "Rules:\n"
            "- Do not invent facts.\n"
            "- If evidence is missing, say: 'Insufficient evidence in provided "
            "chunks.'\n"
            "- Add citations at the end of each bullet using [C<number>].\n"
            "- Keep wording plain-English and concrete.\n"
            "- Prefer evidence in the 'Priority Evidence Snippets' section.\n"
            "- Include exact timeline values when present (days/weeks/months).\n"
            "- Never fabricate dates, durations, or section numbers.\n"
            "Return Markdown with headings exactly:\n"
            "## Direct Answer\n"
            "## Key Clauses\n"
            "## Obligations & Deadlines\n"
            "## Red Flags\n"
            "## Missing Information\n"
            "## Evidence Quotes\n\n"
            "In ## Evidence Quotes include 3-8 bullets where each bullet is:\n"
            "- <short quote copied verbatim> [C<number>]\n\n"
            f"Question: {query}\n\n"
            f"Priority Evidence Snippets:\n{highlight_text}\n\n"
            f"Evidence:\n{context}\n")

    @staticmethod
    def _build_summary_prompt(retrieved: list[RetrievedChunk]) -> str:
        context = ContractRAGEngine._build_context_block(retrieved)

        return (
            "Task: produce a contract summary using ONLY the evidence below.\n"
            "Rules:\n"
            "- No invented facts.\n"
            "- Use citations [C<number>] for each bullet.\n"
            "- Include explicit durations and notice periods exactly as written.\n"
            "- If missing, write 'Not found in provided evidence.'\n"
            "Return Markdown with headings exactly:\n"
            "## Parties & Purpose\n"
            "## Term & Termination\n"
            "## Payment / Compensation\n"
            "## Confidentiality & IP\n"
            "## Liability / Indemnity\n"
            "## Governing Law / Disputes\n"
            "## Critical Obligations\n"
            "## Red Flags\n"
            "## Evidence Quotes\n\n"
            "In ## Evidence Quotes include 5-12 short verbatim quotes with "
            "citations [C<number>].\n\n"
            f"Evidence:\n{context}\n")

    @staticmethod
    def _build_fact_extraction_prompt(query: str, retrieved: list[RetrievedChunk],
                                      max_facts: int) -> str:
        context = ContractRAGEngine._build_context_block(retrieved)
        allowed_chunk_ids = ", ".join(
            f"C{item.chunk_id}" for item in retrieved) or "None"
        highlights = ContractRAGEngine._extract_evidence_highlights(
            query, retrieved, max_items=12)
        highlight_text = "\n".join(highlights) if highlights else "- None"

        return (
            "Task: extract grounded contract facts from the evidence.\n"
            "Return JSON only (no markdown) with this schema:\n"
            "{\n"
            '  "facts": [\n'
            "    {\n"
            '      "clause_type": "<short label>",\n'
            '      "statement": "<plain-English fact>",\n'
            '      "obligation_party": "<party or empty>",\n'
            '      "timeline": "<deadline/term or empty>",\n'
            '      "risk_level": "low|medium|high",\n'
            '      "citations": [<chunk_id_int>, ...]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Use only facts directly supported by evidence.\n"
            "- Use only citation IDs from the allowed list.\n"
            f"- Return at most {max_facts} facts.\n"
            "- Skip unsupported facts instead of guessing.\n"
            "- Keep each statement concise (max ~220 chars).\n\n"
            f"Question: {query}\n"
            f"Allowed citation chunk IDs: {allowed_chunk_ids}\n\n"
            f"Priority Evidence Snippets:\n{highlight_text}\n\n"
            f"Evidence:\n{context}\n")

    @staticmethod
    def _extract_json_payload(text: str) -> Any:
        cleaned = text.strip()
        fenced = _JSON_BLOCK_RE.search(cleaned)
        if fenced:
            cleaned = fenced.group(1).strip()

        parse_candidates = [cleaned]
        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")
        if object_start >= 0 and object_end > object_start:
            parse_candidates.append(cleaned[object_start:object_end + 1])
        list_start = cleaned.find("[")
        list_end = cleaned.rfind("]")
        if list_start >= 0 and list_end > list_start:
            parse_candidates.append(cleaned[list_start:list_end + 1])

        for candidate in parse_candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise ValueError("Could not parse JSON from fact extraction output")

    @staticmethod
    def _parse_citations(raw: Any, allowed_ids: set[int],
                         fallback_text: str) -> list[int]:
        candidates: list[int] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, int):
                    candidates.append(item)
                elif isinstance(item, str):
                    candidates.extend(
                        int(match) for match in _CITATION_ID_RE.findall(item))
                    if item.isdigit():
                        candidates.append(int(item))
        elif isinstance(raw, int):
            candidates.append(raw)
        elif isinstance(raw, str):
            candidates.extend(int(match) for match in _CITATION_ID_RE.findall(raw))
            if raw.isdigit():
                candidates.append(int(raw))

        if fallback_text:
            candidates.extend(
                int(match) for match in _CITATION_ID_RE.findall(fallback_text))

        deduped: list[int] = []
        seen: set[int] = set()
        for cid in candidates:
            if cid not in allowed_ids or cid in seen:
                continue
            deduped.append(cid)
            seen.add(cid)
        return deduped

    @staticmethod
    def _facts_to_json_blob(facts: list[ExtractedFact]) -> str:
        payload = {
            "facts": [{
                "clause_type": fact.clause_type,
                "statement": fact.statement,
                "obligation_party": fact.obligation_party,
                "timeline": fact.timeline,
                "risk_level": fact.risk_level,
                "citations": fact.citations,
            } for fact in facts]
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    @staticmethod
    def _infer_clause_type(sentence: str) -> str:
        s = sentence.lower()
        if any(tok in s for tok in ("termination", "terminate", "possession", "end")):
            return "Termination"
        if any(tok in s for tok in ("notice", "days", "weeks", "months", "years")):
            return "Notice"
        if any(tok in s for tok in ("rent", "payment", "fee", "deposit", "compensation")):
            return "Payment"
        if any(tok in s for tok in ("confidential", "non-disclosure", "nda")):
            return "Confidentiality"
        if any(tok in s for tok in ("liability", "indemn", "damages", "loss")):
            return "Liability"
        if any(tok in s for tok in
               ("governing law", "jurisdiction", "arbitration", "dispute", "court")):
            return "Disputes"
        if any(tok in s for tok in ("intellectual property", "ip", "license")):
            return "IP"
        if any(tok in s for tok in ("shall", "must", "required", "obligation")):
            return "Obligations"
        return "General"

    @staticmethod
    def _infer_obligation_party(sentence: str) -> str:
        s = sentence.lower()
        if "landlord" in s:
            return "Landlord"
        if "tenant" in s:
            return "Tenant"
        if "contract-holder" in s:
            return "Contract holder"
        if "party" in s or "parties" in s:
            return "Both parties"
        return ""

    @staticmethod
    def _infer_risk_level(sentence: str) -> str:
        s = sentence.lower()
        high_markers = {
            "terminate",
            "immediately",
            "penalty",
            "forfeit",
            "breach",
            "indemnify",
            "liable",
            "evict",
            "possession claim",
        }
        medium_markers = {"notice", "must", "shall", "required", "deadline"}
        if any(tok in s for tok in high_markers):
            return "high"
        if any(tok in s for tok in medium_markers):
            return "medium"
        return "low"

    @staticmethod
    def _extract_timeline(sentence: str) -> str:
        match = re.search(
            r"\b(?:within\s+\d+\s*(?:business\s+)?(?:day|days|week|weeks|month|months|year|years)"
            r"|\d+\s*(?:business\s+)?(?:day|days|week|weeks|month|months|year|years)"
            r"|no later than [^.;,\n]{1,80}"
            r"|before the end of [^.;,\n]{1,80})\b",
            sentence,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        return match.group(0).strip()

    @staticmethod
    def _citation_text(citations: list[int]) -> str:
        return " ".join(f"[C{cid}]" for cid in citations)

    @staticmethod
    def _normalize_timeline_text(timeline: str) -> str:
        cleaned = ContractRAGEngine._clean_display_text(timeline, max_len=80)
        cleaned = re.sub(r"\b(of|the|a|an)\.?$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" ,;:.")

    @staticmethod
    def _rewrite_fact_statement(statement: str, fact: ExtractedFact) -> str:
        s = statement.lower()
        timeline = ContractRAGEngine._normalize_timeline_text(fact.timeline)
        party = fact.obligation_party.strip()

        if "may not give" in s and "notice" in s:
            actor = party or "Landlord"
            if timeline:
                if timeline.lower().startswith(
                    ("before ", "after ", "within ", "no later than ")):
                    return f"{actor} cannot give this notice {timeline}."
                return f"{actor} cannot give this notice within {timeline}."
            return f"{actor} cannot give this notice in certain restricted periods."

        if "may give one more notice" in s:
            actor = party or "Landlord"
            if timeline:
                return f"{actor} may give one additional notice within {timeline}."
            return f"{actor} may give one additional notice only once in the allowed window."

        if "ceases to have effect" in s or "cease to have effect" in s:
            if timeline:
                return f"The notice can become invalid in specific situations within {timeline}."
            return "The notice can become invalid in specific situations."

        if "minimum notice period" in s or "may not be less than" in s:
            if timeline:
                return f"Minimum notice period is {timeline}."
            return "A minimum notice period applies before termination can proceed."

        if "possession claim" in s and "landlord" in s:
            return "Landlord may make a possession claim if notice requirements are met."

        return statement

    @staticmethod
    def _format_fact_bullet(fact: ExtractedFact) -> str:
        statement = ContractRAGEngine._clean_display_text(fact.statement,
                                                          max_len=180)
        statement = ContractRAGEngine._rewrite_fact_statement(statement, fact)
        if statement and statement[-1] not in ".!?":
            statement = statement + "."
        extras: list[str] = []
        if fact.obligation_party:
            extras.append(f"Party: {fact.obligation_party}")
        if fact.timeline:
            timeline = ContractRAGEngine._normalize_timeline_text(fact.timeline)
            extras.append(f"Timeline: {timeline}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        citation = ContractRAGEngine._citation_text(fact.citations)
        return f"- {statement}{suffix} {citation}".strip()

    @staticmethod
    def _fact_dedup_key(fact: ExtractedFact) -> str:
        base = ContractRAGEngine._clean_display_text(fact.statement, max_len=220)
        rewritten = ContractRAGEngine._rewrite_fact_statement(base, fact)
        rewritten = rewritten.replace("...", "").strip().lower()
        rewritten = re.sub(r"[^a-z0-9 ]", " ", rewritten)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if len(rewritten) > 120:
            rewritten = rewritten[:120].strip()
        return rewritten

    @staticmethod
    def _dedupe_facts(facts: list[ExtractedFact]) -> list[ExtractedFact]:
        out: list[ExtractedFact] = []
        seen: list[str] = []
        for fact in facts:
            key = ContractRAGEngine._fact_dedup_key(fact)
            if not key:
                continue
            duplicate = False
            for existing in seen:
                if key == existing or key in existing or existing in key:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append(key)
            out.append(fact)
        return out

    def _extract_heuristic_facts(self, query: str, retrieved: list[RetrievedChunk],
                                 max_facts: int) -> list[ExtractedFact]:
        query_tokens = self._tokenize_legal_words(query)
        candidates: list[tuple[float, int, str]] = []
        for item in retrieved:
            sentences = re.split(r"(?<=[.!?;])\s+", item.text.replace("\n", " "))
            for sent in sentences:
                sentence = self._clean_display_text(sent, max_len=240)
                if len(sentence) < 40 or not self._is_readable_sentence(sentence):
                    continue
                overlap = self._lexical_overlap_score(query_tokens, sentence)
                has_time = bool(_TIME_PHRASE_RE.search(sentence))
                score = overlap + (0.25 if has_time else 0.0) + (item.score * 0.12)
                if score <= 0.08 and not has_time:
                    continue
                candidates.append((score, item.chunk_id, sentence[:260]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        facts: list[ExtractedFact] = []
        seen_sentence: set[str] = set()
        for _, chunk_id, sentence in candidates:
            key = sentence.lower()
            if key in seen_sentence:
                continue
            seen_sentence.add(key)
            facts.append(
                ExtractedFact(
                    clause_type=self._infer_clause_type(sentence),
                    statement=sentence,
                    obligation_party=self._infer_obligation_party(sentence),
                    timeline=self._extract_timeline(sentence),
                    risk_level=self._infer_risk_level(sentence),
                    citations=[chunk_id],
                ))
            if len(facts) >= max_facts:
                break
        return facts

    def _prefer_structured_renderer(self) -> bool:
        model_name = (self._active_llm_model_name or "").lower()
        if not model_name:
            return False
        # Deterministic rendering works better than free-form generation for
        # tiny local models that often produce unstable legal prose.
        small_markers = ("tinyllama", "0.5b", "1.1b", "1.5b")
        return any(marker in model_name for marker in small_markers)

    @staticmethod
    def _build_evidence_quotes(retrieved: list[RetrievedChunk],
                               max_quotes: int) -> list[str]:
        quotes: list[str] = []
        seen: set[str] = set()
        for item in retrieved:
            sentences = re.split(r"(?<=[.!?;])\s+", item.text.replace("\n", " "))
            chosen = ""
            for sent in sentences:
                s = ContractRAGEngine._clean_display_text(sent, max_len=200)
                if ContractRAGEngine._is_readable_sentence(s):
                    chosen = s
                    break
            if not chosen:
                chosen = ContractRAGEngine._clean_display_text(item.text,
                                                               max_len=200)
            key = chosen.lower()
            if not chosen or key in seen:
                continue
            seen.add(key)
            quotes.append(f'- "{chosen}" [C{item.chunk_id}]')
            if len(quotes) >= max_quotes:
                break
        return quotes

    @staticmethod
    def _unique_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = item.strip()
            if not key:
                continue
            lower = key.lower()
            if lower in seen:
                continue
            seen.add(lower)
            out.append(key)
        return out

    @staticmethod
    def _clean_display_text(text: str, max_len: int = 220) -> str:
        cleaned = text.replace("\n", " ").replace("\t", " ")
        cleaned = cleaned.replace("•", " ").replace("●", " ")
        cleaned = cleaned.replace("contract- holder", "contract-holder")
        cleaned = cleaned.replace("landlord PRINCIPAL CONTACT", "landlord")
        cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip(" -;,:")
        # Strip repeated leading OCR headers (e.g., "172 I FT & PT").
        for _ in range(3):
            updated = _OCR_HEADER_RE.sub("", cleaned).strip(" -;,:")
            if updated == cleaned:
                break
            cleaned = updated
        cleaned = re.sub(r"^[^A-Za-z0-9(]+", "", cleaned).strip(" -;,:")
        if len(cleaned) > max_len:
            cut = cleaned[:max_len]
            last_break = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "))
            if last_break >= int(max_len * 0.6):
                cleaned = cut[:last_break + 1].strip()
            else:
                word_break = cut.rfind(" ")
                if word_break >= int(max_len * 0.6):
                    cleaned = cut[:word_break].strip(" ,;:")
                else:
                    cleaned = cut.rstrip(" ,;:")
        return cleaned

    @staticmethod
    def _is_readable_sentence(text: str) -> bool:
        if not text:
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'`.-]*", text)
        if len(words) < 7:
            return False
        # Skip heavily clipped fragments that often start mid-word.
        if text and text[0].islower():
            return False
        short_ratio = sum(1 for w in words if len(w) <= 2) / max(1, len(words))
        if short_ratio > 0.45:
            return False
        legal_terms = (
            "landlord",
            "contract-holder",
            "tenant",
            "notice",
            "must",
            "shall",
            "may",
            "terminate",
            "term",
            "rent",
            "liability",
        )
        lowered = text.lower()
        if not any(term in lowered for term in legal_terms):
            return False
        return True

    @staticmethod
    def _is_party_identity_query(query: str) -> bool:
        if _PARTY_QUERY_RE.search(query):
            return True
        lowered = query.lower()
        if any(term in lowered for term in _PARTY_IDENTITY_BLOCKLIST):
            return False
        party_terms = (
            "contract-holder",
            "contract holder",
            "tenant",
            "landlord",
            "parties",
            "party",
            "agent",
            "principal contact",
        )
        if any(term in lowered for term in party_terms):
            return True
        return False

    @staticmethod
    def _split_name_field(raw: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", raw).strip(" .,:;")
        if not cleaned:
            return []
        if cleaned.upper() in {"N/A", "NA", "~"}:
            return []

        parts = re.split(r"\s*&\s*|\s+and\s+|,|;|/", cleaned, flags=re.IGNORECASE)
        names: list[str] = []
        for part in parts:
            candidate = part.strip(" .,:;")
            if not candidate:
                continue
            if any(ch.isdigit() for ch in candidate):
                continue
            if "@" in candidate:
                continue
            words = candidate.split()
            if len(words) < 2 or len(words) > 4:
                continue
            if not all(re.match(r"^[A-Za-z][A-Za-z'`.-]*$", word)
                       for word in words):
                continue
            if sum(1 for word in words if word[0].isupper()) < 2:
                continue
            names.append(" ".join(words))
        return ContractRAGEngine._unique_preserve_order(names)

    def _extract_role_names_from_chunk(self, chunk: str,
                                       role: str) -> list[str]:
        labels = _ROLE_LABELS.get(role, ())
        if not labels:
            return []

        lower_chunk = chunk.lower()
        names: list[str] = []
        for label in labels:
            idx = lower_chunk.find(label.lower())
            if idx < 0:
                continue
            # Party blocks are near the beginning of the document and short.
            window = chunk[idx:idx + 850]
            for line in _NAME_LINE_RE.findall(window):
                parsed = self._split_name_field(line)
                if parsed:
                    # Use the closest meaningful Name: line after role header.
                    return parsed
        return self._unique_preserve_order(names)

    def _extract_role_name_line(self, chunk: str, role: str) -> str:
        labels = _ROLE_LABELS.get(role, ())
        if not labels:
            return ""
        lower_chunk = chunk.lower()
        for label in labels:
            idx = lower_chunk.find(label.lower())
            if idx < 0:
                continue
            window = chunk[idx:idx + 850]
            match = _NAME_LINE_RE.search(window)
            if match:
                return match.group(0).strip()
        return ""

    def _extract_known_parties(
            self, chunks: list[str]) -> dict[str, dict[str, list[Any]]]:
        result: dict[str, dict[str, list[Any]]] = {
            "contract_holders": {
                "names": [],
                "citations": []
            },
            "landlords": {
                "names": [],
                "citations": []
            },
            "agents": {
                "names": [],
                "citations": []
            },
        }

        scan_limit = min(len(chunks), 30)
        found_role = {
            "contract_holder": False,
            "landlord": False,
            "agent": False,
        }
        for chunk_id in range(scan_limit):
            chunk = chunks[chunk_id]
            for role, key in (("contract_holder", "contract_holders"),
                              ("landlord", "landlords"), ("agent", "agents")):
                if found_role[role]:
                    continue
                names = self._extract_role_names_from_chunk(chunk, role)
                if not names:
                    continue
                result[key]["names"].extend(names)
                result[key]["citations"].append(chunk_id)
                found_role[role] = True

        for key in result:
            names = [str(n) for n in result[key]["names"]]
            citations = [int(c) for c in result[key]["citations"]]
            result[key]["names"] = self._unique_preserve_order(names)
            result[key]["citations"] = sorted(set(citations))

        return result

    @staticmethod
    def _snippet_for_citation(chunk_text: str,
                              max_len: int = 220,
                              prefer_terms: list[str] | None = None) -> str:
        lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        if prefer_terms:
            terms = [t.lower() for t in prefer_terms if t]
            priority_terms = [t for t in terms if (" " in t and len(t) >= 6)]
            if priority_terms:
                for line in lines:
                    low = line.lower()
                    if any(term in low for term in priority_terms):
                        return ContractRAGEngine._clean_display_text(line, max_len)
            for line in lines:
                low = line.lower()
                if any(term in low for term in terms):
                    return ContractRAGEngine._clean_display_text(line, max_len)
        for line in lines:
            cleaned = ContractRAGEngine._clean_display_text(line, max_len)
            if ContractRAGEngine._is_readable_sentence(cleaned):
                return cleaned
        return ContractRAGEngine._clean_display_text(chunk_text, max_len)

    @staticmethod
    def _source_excerpt(chunk_text: str, max_len: int = 240) -> str:
        lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        for line in lines:
            cleaned = ContractRAGEngine._clean_display_text(line, max_len=max_len)
            if ContractRAGEngine._is_readable_sentence(cleaned):
                return cleaned

        sentences = re.split(r"(?<=[.!?;])\s+", chunk_text.replace("\n", " "))
        for sent in sentences:
            cleaned = ContractRAGEngine._clean_display_text(sent, max_len=max_len)
            if ContractRAGEngine._is_readable_sentence(cleaned):
                return cleaned
        return ContractRAGEngine._clean_display_text(chunk_text, max_len=max_len)

    def _build_party_identity_answer(self, document_id: str, query: str,
                                     chunks: list[str],
                                     parties: dict[str, dict[str, list[Any]]],
                                     top_k: int) -> dict[str, Any]:
        contract_names = [str(x) for x in parties["contract_holders"]["names"]]
        landlord_names = [str(x) for x in parties["landlords"]["names"]]
        agent_names = [str(x) for x in parties["agents"]["names"]]
        contract_cites = [int(x) for x in parties["contract_holders"]["citations"]]
        landlord_cites = [int(x) for x in parties["landlords"]["citations"]]
        agent_cites = [int(x) for x in parties["agents"]["citations"]]

        contract_line = ""
        if contract_cites:
            contract_line = self._extract_role_name_line(
                chunks[contract_cites[0]], "contract_holder")
        landlord_line = ""
        if landlord_cites:
            landlord_line = self._extract_role_name_line(
                chunks[landlord_cites[0]], "landlord")
        agent_line = ""
        if agent_cites:
            agent_line = self._extract_role_name_line(
                chunks[agent_cites[0]], "agent")

        if not contract_names and not landlord_names and not agent_names:
            raise ValueError("No party names found in extracted text")

        direct_lines: list[str] = []
        key_lines: list[str] = []
        evidence_lines: list[str] = []
        missing_lines: list[str] = []

        if contract_names or contract_line:
            citation = f"[C{contract_cites[0]}]" if contract_cites else ""
            if contract_names:
                names_text = ", ".join(contract_names)
            else:
                names_text = contract_line.split(":", 1)[-1].strip()
            direct_lines.append(f"- Contract-holder(s): {names_text} {citation}".strip())
            key_lines.append(
                f"- The parties section lists contract-holder names as: {names_text} {citation}"
                .strip())
            if contract_cites:
                quote = contract_line or self._snippet_for_citation(
                    chunks[contract_cites[0]],
                    prefer_terms=contract_names + ["contract-holder"])
                evidence_lines.append(f'- "{quote}" [C{contract_cites[0]}]')
        else:
            direct_lines.append("- Contract-holder names were not found in the extracted text.")
            missing_lines.append(
                "- Contract-holder names are missing or unreadable in extracted PDF text.")

        if landlord_names or landlord_line:
            citation = f"[C{landlord_cites[0]}]" if landlord_cites else ""
            if landlord_names:
                names_text = ", ".join(landlord_names)
            else:
                names_text = landlord_line.split(":", 1)[-1].strip()
            direct_lines.append(f"- Landlord: {names_text} {citation}".strip())
            key_lines.append(
                f"- The parties section lists the landlord as: {names_text} {citation}"
                .strip())
            if landlord_cites:
                quote = landlord_line or self._snippet_for_citation(
                    chunks[landlord_cites[0]],
                    prefer_terms=landlord_names + ["landlord"])
                evidence_lines.append(f'- "{quote}" [C{landlord_cites[0]}]')
        else:
            missing_lines.append("- Landlord name was not clearly found.")

        if agent_names or agent_line:
            citation = f"[C{agent_cites[0]}]" if agent_cites else ""
            if agent_names:
                agent_text = ", ".join(agent_names)
            else:
                agent_text = agent_line.split(":", 1)[-1].strip()
            key_lines.append(
                f"- Landlord's agent appears as: {agent_text} {citation}"
                .strip())
            if agent_cites:
                quote = agent_line or self._snippet_for_citation(
                    chunks[agent_cites[0]],
                    prefer_terms=agent_names + ["agent"])
                evidence_lines.append(f'- "{quote}" [C{agent_cites[0]}]')

        obligations = [
            "- No payment/termination obligations are inferred from this identity-only query.",
        ]
        red_flags = [
            "- None specific to identity extraction; verify against the signed parties page if available."
        ]
        if not missing_lines:
            missing_lines = [
                "- No critical party-identity fields missing in extracted text."
            ]

        key_lines = self._unique_preserve_order(key_lines)
        evidence_lines = self._unique_preserve_order(evidence_lines)

        cited_ids = sorted(set(contract_cites + landlord_cites + agent_cites))
        if not cited_ids:
            cited_ids = [0]
        source_ids = cited_ids[:max(1, top_k)]
        sources = [{
            "chunk_id": cid,
            "score": 1.0 - (rank * 0.01),
            "text_excerpt": self._source_excerpt(chunks[cid], max_len=240),
        } for rank, cid in enumerate(source_ids)]

        answer = (
            "## Direct Answer\n"
            f"{chr(10).join(direct_lines)}\n\n"
            "## Key Clauses\n"
            f"{chr(10).join(key_lines)}\n\n"
            "## Obligations & Deadlines\n"
            f"{chr(10).join(obligations)}\n\n"
            "## Red Flags\n"
            f"{chr(10).join(red_flags)}\n\n"
            "## Missing Information\n"
            f"{chr(10).join(missing_lines)}\n\n"
            "## Evidence Quotes\n"
            f"{chr(10).join(evidence_lines)}\n")

        return {
            "document_id": document_id,
            "query": query,
            "answer": answer,
            "sources": sources,
        }

    def _try_answer_party_identity_query(self, document_id: str, query: str,
                                         top_k: int) -> dict[str, Any] | None:
        if not self._is_party_identity_query(query):
            return None
        doc = self._load_document(document_id)
        chunks: list[str] = doc["chunks"]
        parties = self._extract_known_parties(chunks)
        return self._build_party_identity_answer(document_id, query, chunks,
                                                 parties, top_k)

    @staticmethod
    def _render_answer_from_facts(query: str, facts: list[ExtractedFact],
                                  retrieved: list[RetrievedChunk]) -> str:
        key_clauses = ContractRAGEngine._dedupe_facts(facts)[:6]
        direct_points = key_clauses[:2]
        direct_keys = {ContractRAGEngine._fact_dedup_key(f) for f in direct_points}
        key_clause_points = [
            f for f in key_clauses[2:6]
            if ContractRAGEngine._fact_dedup_key(f) not in direct_keys
        ]

        obligation_facts = ContractRAGEngine._dedupe_facts([
            f for f in key_clauses
            if f.timeline or f.obligation_party or f.clause_type in
            {"Obligations", "Termination", "Notice", "Payment"}
        ])[:4]
        used_keys = direct_keys | {
            ContractRAGEngine._fact_dedup_key(f) for f in key_clause_points
        }
        obligation_facts = [
            f for f in obligation_facts
            if ContractRAGEngine._fact_dedup_key(f) not in used_keys
        ][:4]
        if not obligation_facts:
            obligation_facts = [
                f for f in ContractRAGEngine._dedupe_facts([
                    x for x in key_clauses if x.timeline or x.obligation_party
                ]) if ContractRAGEngine._fact_dedup_key(f) not in direct_keys
            ][:3]
        red_flags = ContractRAGEngine._dedupe_facts(
            [f for f in key_clauses if f.risk_level == "high"])[:3]
        if not red_flags:
            red_flags = ContractRAGEngine._dedupe_facts([
                f for f in key_clauses
                if any(tok in f.statement.lower()
                       for tok in ("terminate", "breach", "liable", "penalty"))
            ])[:3]

        missing_info: list[str] = []
        if not any(f.timeline for f in facts):
            missing_info.append(
                "- Exact notice periods/deadlines are not consistently explicit in retrieved chunks."
            )
        if not any(f.clause_type == "Liability" for f in facts):
            missing_info.append(
                "- Liability / indemnity terms are not clearly supported in retrieved evidence."
            )
        if not any(f.clause_type == "Disputes" for f in facts):
            missing_info.append(
                "- Governing law / dispute resolution terms were not clearly found in retrieved chunks."
            )
        if not missing_info:
            missing_info.append("- No major evidence gaps detected in top retrieved chunks.")

        direct_answer = "\n".join(ContractRAGEngine._format_fact_bullet(f)
                                   for f in direct_points)
        if not direct_answer:
            direct_answer = "- Insufficient evidence in provided chunks."

        key_clause_text = "\n".join(ContractRAGEngine._format_fact_bullet(f)
                                     for f in key_clause_points)
        if not key_clause_text:
            key_clause_text = "- No additional key clauses beyond the direct answer."

        obligations_text = "\n".join(ContractRAGEngine._format_fact_bullet(f)
                                      for f in obligation_facts)
        if not obligations_text:
            obligations_text = "- No explicit obligations/deadlines found in retrieved chunks."
        red_flag_text = ("\n".join(
            ContractRAGEngine._format_fact_bullet(f) for f in red_flags)
                         if red_flags else "- No high-risk clauses detected in retrieved chunks.")
        evidence_quotes = "\n".join(
            ContractRAGEngine._build_evidence_quotes(retrieved, max_quotes=4))

        return (
            "## Direct Answer\n"
            f"{direct_answer}\n\n"
            "## Key Clauses\n"
            f"{key_clause_text}\n\n"
            "## Obligations & Deadlines\n"
            f"{obligations_text}\n\n"
            "## Red Flags\n"
            f"{red_flag_text}\n\n"
            "## Missing Information\n"
            f"{chr(10).join(missing_info)}\n\n"
            "## Evidence Quotes\n"
            f"{evidence_quotes}\n")

    @staticmethod
    def _render_summary_from_facts(facts: list[ExtractedFact],
                                   retrieved: list[RetrievedChunk]) -> str:
        def select(max_items: int, clause_types: set[str]) -> list[ExtractedFact]:
            return [
                f for f in facts if f.clause_type in clause_types
            ][:max_items]

        parties = select(3, {"General", "Obligations"})
        term = select(4, {"Termination", "Notice"})
        payment = select(3, {"Payment"})
        conf_ip = select(3, {"Confidentiality", "IP"})
        liability = select(3, {"Liability"})
        disputes = select(3, {"Disputes"})
        obligations = [
            f for f in facts if f.obligation_party or f.timeline
        ][:5]
        red_flags = [f for f in facts if f.risk_level == "high"][:4]
        if not red_flags:
            red_flags = [
                f for f in facts if "terminate" in f.statement.lower()
                or "breach" in f.statement.lower()
            ][:4]

        def render_section(rows: list[ExtractedFact], empty_text: str) -> str:
            if not rows:
                return f"- {empty_text}"
            return "\n".join(ContractRAGEngine._format_fact_bullet(f) for f in rows)

        evidence_quotes = "\n".join(
            ContractRAGEngine._build_evidence_quotes(retrieved, max_quotes=6))

        return (
            "## Parties & Purpose\n"
            f"{render_section(parties, 'Not found in provided evidence.')}\n\n"
            "## Term & Termination\n"
            f"{render_section(term, 'Not found in provided evidence.')}\n\n"
            "## Payment / Compensation\n"
            f"{render_section(payment, 'Not found in provided evidence.')}\n\n"
            "## Confidentiality & IP\n"
            f"{render_section(conf_ip, 'Not found in provided evidence.')}\n\n"
            "## Liability / Indemnity\n"
            f"{render_section(liability, 'Not found in provided evidence.')}\n\n"
            "## Governing Law / Disputes\n"
            f"{render_section(disputes, 'Not found in provided evidence.')}\n\n"
            "## Critical Obligations\n"
            f"{render_section(obligations, 'Not found in provided evidence.')}\n\n"
            "## Red Flags\n"
            f"{render_section(red_flags, 'No high-risk clauses detected in retrieved chunks.')}\n\n"
            "## Evidence Quotes\n"
            f"{evidence_quotes}\n")

    def _extract_structured_facts(self, query: str,
                                  retrieved: list[RetrievedChunk]
                                  ) -> list[ExtractedFact]:
        max_facts = max(4, int(self.settings.fact_extract_max_facts))
        prompt = self._build_fact_extraction_prompt(query, retrieved, max_facts)
        raw = self._safe_generate(prompt)

        try:
            payload = self._extract_json_payload(raw)
        except ValueError as exc:
            print(f"Fact extraction JSON parse failed: {exc}")
            return self._extract_heuristic_facts(query, retrieved, max_facts)

        facts_raw: Any
        if isinstance(payload, dict):
            facts_raw = payload.get("facts", [])
        elif isinstance(payload, list):
            facts_raw = payload
        else:
            return []

        if not isinstance(facts_raw, list):
            return self._extract_heuristic_facts(query, retrieved, max_facts)

        allowed_ids = {item.chunk_id for item in retrieved}
        parsed: list[ExtractedFact] = []
        for item in facts_raw:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            clause_type = str(item.get("clause_type", "General")).strip()
            if not clause_type:
                clause_type = "General"
            obligation_party = str(item.get("obligation_party", "")).strip()
            timeline = str(item.get("timeline", "")).strip()
            risk = str(item.get("risk_level", "medium")).strip().lower()
            if risk not in {"low", "medium", "high"}:
                risk = "medium"
            citations = self._parse_citations(item.get("citations"), allowed_ids,
                                              statement)
            if not citations:
                continue
            parsed.append(
                ExtractedFact(
                    clause_type=clause_type[:80],
                    statement=statement[:260],
                    obligation_party=obligation_party[:100],
                    timeline=timeline[:120],
                    risk_level=risk,
                    citations=citations,
                ))
            if len(parsed) >= max_facts:
                break

        if len(parsed) >= max_facts:
            return parsed[:max_facts]

        heuristic_facts = self._extract_heuristic_facts(query, retrieved, max_facts)
        if not parsed:
            return heuristic_facts

        existing = {fact.statement.lower() for fact in parsed}
        for fact in heuristic_facts:
            key = fact.statement.lower()
            if key in existing:
                continue
            parsed.append(fact)
            existing.add(key)
            if len(parsed) >= max_facts:
                break
        return parsed

    @staticmethod
    def _build_two_stage_qa_prompt(query: str, facts: list[ExtractedFact],
                                   retrieved: list[RetrievedChunk]) -> str:
        facts_blob = ContractRAGEngine._facts_to_json_blob(facts)
        context = ContractRAGEngine._build_context_block(retrieved)
        return (
            "Task: produce the final legal Q&A in plain English.\n"
            "Use only the structured facts and evidence below.\n"
            "Rules:\n"
            "- Do not invent facts.\n"
            "- Keep every bullet grounded with citations [C<number>].\n"
            "- If a section has no support, write: 'Insufficient evidence in "
            "provided chunks.'\n"
            "Return Markdown with headings exactly:\n"
            "## Direct Answer\n"
            "## Key Clauses\n"
            "## Obligations & Deadlines\n"
            "## Red Flags\n"
            "## Missing Information\n"
            "## Evidence Quotes\n\n"
            f"Question: {query}\n\n"
            f"Structured Facts JSON:\n{facts_blob}\n\n"
            f"Evidence:\n{context}\n")

    @staticmethod
    def _build_two_stage_summary_prompt(facts: list[ExtractedFact],
                                        retrieved: list[RetrievedChunk]) -> str:
        facts_blob = ContractRAGEngine._facts_to_json_blob(facts)
        context = ContractRAGEngine._build_context_block(retrieved)
        return (
            "Task: produce a contract summary in plain English.\n"
            "Use only the structured facts and evidence below.\n"
            "Rules:\n"
            "- No invented facts.\n"
            "- Every bullet must include citations [C<number>].\n"
            "- If a section is unsupported, write 'Not found in provided "
            "evidence.'\n"
            "Return Markdown with headings exactly:\n"
            "## Parties & Purpose\n"
            "## Term & Termination\n"
            "## Payment / Compensation\n"
            "## Confidentiality & IP\n"
            "## Liability / Indemnity\n"
            "## Governing Law / Disputes\n"
            "## Critical Obligations\n"
            "## Red Flags\n"
            "## Evidence Quotes\n\n"
            f"Structured Facts JSON:\n{facts_blob}\n\n"
            f"Evidence:\n{context}\n")

    @staticmethod
    def _extract_evidence_highlights(query: str,
                                     retrieved: list[RetrievedChunk],
                                     max_items: int = 10) -> list[str]:
        query_tokens = ContractRAGEngine._tokenize_legal_words(query)
        candidates: list[tuple[float, int, str]] = []
        for item in retrieved:
            sentences = re.split(r"(?<=[.!?;])\s+", item.text.replace("\n", " "))
            for sent in sentences:
                sentence = sent.strip()
                if len(sentence) < 40:
                    continue
                overlap = ContractRAGEngine._lexical_overlap_score(
                    query_tokens, sentence)
                has_time_marker = bool(_TIME_PHRASE_RE.search(sentence))
                if overlap <= 0.0 and not has_time_marker:
                    continue
                bonus = 0.2 if has_time_marker else 0.0
                score = overlap + bonus
                candidates.append((score, item.chunk_id, sentence[:280]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        highlights: list[str] = []
        seen_text: set[str] = set()
        for _, chunk_id, sentence in candidates:
            key = sentence.lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            highlights.append(f"- [C{chunk_id}] {sentence}")
            if len(highlights) >= max_items:
                break
        return highlights

    def answer_query(self, document_id: str, query: str,
                     top_k: int) -> dict[str, Any]:
        identity_answer = self._try_answer_party_identity_query(
            document_id, query, top_k)
        if identity_answer is not None:
            return identity_answer

        retrieved = self._retrieve(document_id, query, top_k)
        if not retrieved:
            raise ValueError("No matching chunks found for query")

        if self.settings.two_stage_generation:
            facts = self._extract_structured_facts(query, retrieved)
            if facts:
                if self._prefer_structured_renderer():
                    answer = self._render_answer_from_facts(
                        query, facts, retrieved)
                else:
                    prompt = self._build_two_stage_qa_prompt(
                        query, facts, retrieved)
                    answer = self._safe_generate(prompt)
            else:
                prompt = self._build_qa_prompt(query, retrieved)
                answer = self._safe_generate(prompt)
        else:
            prompt = self._build_qa_prompt(query, retrieved)
            answer = self._safe_generate(prompt)

        return {
            "document_id": document_id,
            "query": query,
            "answer": answer,
            "sources": [{
                "chunk_id": item.chunk_id,
                "score": round(item.score, 6),
                "text_excerpt": self._source_excerpt(item.text, max_len=240),
            } for item in retrieved],
        }

    def summarize_contract(self, document_id: str) -> dict[str, Any]:
        summary_query = (
            "Generate a complete contract summary including obligations, "
            "termination rights, liability, and red flags.")
        retrieved = self._retrieve(document_id, summary_query, top_k=8)
        if not retrieved:
            raise ValueError("No chunks available for summary")

        if self.settings.two_stage_generation:
            facts = self._extract_structured_facts(summary_query, retrieved)
            if facts:
                if self._prefer_structured_renderer():
                    answer = self._render_summary_from_facts(facts, retrieved)
                else:
                    prompt = self._build_two_stage_summary_prompt(
                        facts, retrieved)
                    answer = self._safe_generate(prompt)
            else:
                prompt = self._build_summary_prompt(retrieved)
                answer = self._safe_generate(prompt)
        else:
            prompt = self._build_summary_prompt(retrieved)
            answer = self._safe_generate(prompt)

        return {
            "document_id": document_id,
            "query": "Auto-summary",
            "answer": answer,
            "sources": [{
                "chunk_id": item.chunk_id,
                "score": round(item.score, 6),
                "text_excerpt": self._source_excerpt(item.text, max_len=240),
            } for item in retrieved],
        }
