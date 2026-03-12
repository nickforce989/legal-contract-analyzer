# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Settings:
    llm_mode: str
    model_profile: str
    data_dir: Path
    notebook_mode: bool
    prompt_style: str
    simple_retrieval: bool
    default_top_k: int
    use_4bit: bool
    enable_party_identity_shortcut: bool
    hf_token: str
    embedding_model_name: str
    llm_model_name: str
    fallback_llm_model_name: str
    remote_llm_base_url: str
    remote_llm_api_key: str
    remote_llm_model_name: str
    remote_llm_timeout_seconds: int
    two_stage_generation: bool
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    retrieval_candidate_factor: int
    use_cross_encoder_rerank: bool
    cross_encoder_model_name: str
    cross_encoder_batch_size: int
    cross_encoder_weight: float
    max_new_tokens: int
    temperature: float
    top_p: float
    lexical_rerank_weight: float
    fact_extract_max_facts: int
    device: str
    torch_dtype: torch.dtype

    @staticmethod
    def from_env() -> "Settings":
        def env_bool(name: str, default: str) -> bool:
            return os.getenv(name, default).strip().lower() in {
                "1", "true", "yes", "on"
            }

        llm_mode = os.getenv("LEGAL_ANALYZER_LLM_MODE", "local").lower()
        if llm_mode not in {"local", "remote"}:
            llm_mode = "local"

        notebook_mode = env_bool("LEGAL_ANALYZER_NOTEBOOK_MODE", "false")
        raw_prompt_style = os.getenv("LEGAL_ANALYZER_PROMPT_STYLE")
        if raw_prompt_style:
            prompt_style = raw_prompt_style.strip().lower()
        else:
            prompt_style = "notebook" if notebook_mode or llm_mode == "local" else "structured"
        if prompt_style not in {"structured", "notebook"}:
            prompt_style = "structured"

        simple_retrieval = env_bool(
            "LEGAL_ANALYZER_SIMPLE_RETRIEVAL",
            "true" if notebook_mode else "false",
        )
        default_top_k = int(
            os.getenv("LEGAL_ANALYZER_DEFAULT_TOP_K",
                      "5" if notebook_mode else "0"))
        if default_top_k < 0:
            default_top_k = 0

        profile = os.getenv("LEGAL_ANALYZER_MODEL_PROFILE", "quality").lower()
        default_embed_by_profile = {
            "fast": "BAAI/bge-small-en-v1.5",
            "balanced": "BAAI/bge-base-en-v1.5",
            "quality": "BAAI/bge-base-en-v1.5",
        }
        default_llm_by_profile = {
            "fast": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "balanced": "Qwen/Qwen2.5-0.5B-Instruct",
            "quality": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }
        if profile not in default_llm_by_profile:
            profile = "balanced"

        base_dir = Path(__file__).resolve().parent.parent
        data_dir = Path(
            os.getenv("LEGAL_ANALYZER_DATA_DIR", str(base_dir / "data")))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        use_4bit = env_bool("LEGAL_ANALYZER_USE_4BIT",
                            "true" if device == "cuda" else "false")
        enable_party_identity_shortcut = env_bool(
            "LEGAL_ANALYZER_PARTY_IDENTITY_SHORTCUT", "false")

        return Settings(
            llm_mode=llm_mode,
            model_profile=profile,
            data_dir=data_dir,
            notebook_mode=notebook_mode,
            prompt_style=prompt_style,
            simple_retrieval=simple_retrieval,
            default_top_k=default_top_k,
            use_4bit=use_4bit,
            enable_party_identity_shortcut=enable_party_identity_shortcut,
            hf_token=os.getenv("LEGAL_ANALYZER_HF_TOKEN",
                               os.getenv("HF_TOKEN", "")).strip(),
            embedding_model_name=os.getenv("LEGAL_ANALYZER_EMBED_MODEL",
                                           default_embed_by_profile[profile]),
            llm_model_name=os.getenv("LEGAL_ANALYZER_LLM_MODEL",
                                     default_llm_by_profile[profile]),
            fallback_llm_model_name=os.getenv(
                "LEGAL_ANALYZER_FALLBACK_LLM_MODEL",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            remote_llm_base_url=os.getenv("LEGAL_ANALYZER_REMOTE_LLM_BASE_URL",
                                          "https://openrouter.ai/api/v1"),
            remote_llm_api_key=os.getenv(
                "LEGAL_ANALYZER_REMOTE_LLM_API_KEY",
                os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY",
                                                          ""))),
            remote_llm_model_name=os.getenv(
                "LEGAL_ANALYZER_REMOTE_LLM_MODEL",
                "Qwen/Qwen2.5-7B-Instruct"),
            remote_llm_timeout_seconds=int(
                os.getenv("LEGAL_ANALYZER_REMOTE_LLM_TIMEOUT_SECONDS", "180")),
            two_stage_generation=env_bool("LEGAL_ANALYZER_TWO_STAGE_GENERATION",
                                          "false"),
            chunk_size=int(
                os.getenv("LEGAL_ANALYZER_CHUNK_SIZE",
                          "1500" if notebook_mode else "1200")),
            chunk_overlap=int(
                os.getenv("LEGAL_ANALYZER_CHUNK_OVERLAP",
                          "200" if notebook_mode else "200")),
            embedding_batch_size=int(
                os.getenv("LEGAL_ANALYZER_EMBED_BATCH_SIZE", "12")),
            retrieval_candidate_factor=int(
                os.getenv("LEGAL_ANALYZER_RETRIEVAL_CANDIDATE_FACTOR",
                          "1" if notebook_mode else "5")),
            use_cross_encoder_rerank=env_bool(
                "LEGAL_ANALYZER_USE_CROSS_ENCODER_RERANK",
                "false"),
            cross_encoder_model_name=os.getenv(
                "LEGAL_ANALYZER_CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            cross_encoder_batch_size=int(
                os.getenv("LEGAL_ANALYZER_CROSS_ENCODER_BATCH_SIZE", "8")),
            cross_encoder_weight=float(
                os.getenv("LEGAL_ANALYZER_CROSS_ENCODER_WEIGHT", "0.65")),
            max_new_tokens=int(
                os.getenv("LEGAL_ANALYZER_MAX_NEW_TOKENS",
                          "256" if notebook_mode else "280")),
            temperature=float(os.getenv("LEGAL_ANALYZER_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("LEGAL_ANALYZER_TOP_P", "0.9")),
            lexical_rerank_weight=float(
                os.getenv("LEGAL_ANALYZER_LEXICAL_RERANK_WEIGHT",
                          "0.0" if notebook_mode else "0.22")),
            fact_extract_max_facts=int(
                os.getenv("LEGAL_ANALYZER_FACT_EXTRACT_MAX_FACTS", "16")),
            device=device,
            torch_dtype=torch_dtype,
        )
