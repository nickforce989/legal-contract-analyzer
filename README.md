# Legal Contract Analyzer (FastAPI + Hugging Face + PyTorch + JAX + FAISS)

This is a practical RAG app for legal contracts (NDA, lease, employment, MSA, etc.).

It provides:
- `POST /upload`: upload a PDF contract and build a vector index.
- `POST /analyze`: ask contract-specific questions and get plain-English answers with citations.
- `GET /summary/{document_id}`: generate a structured contract summary.

## Architecture

PDF Upload -> Text Extraction -> Chunking -> Embedding (Hugging Face + PyTorch)
-> FAISS Vector Index -> Retrieval (Dense + Lexical + Cross-Encoder Rerank)
-> Two-Stage Generation (Fact Extraction -> Final Answer)
-> FastAPI Response -> Optional Gradio Frontend

## What’s Included

- Hybrid retrieval: dense embeddings + lexical scoring + cross-encoder rerank.
- Two-stage generation: structured fact extraction then grounded synthesis.
- Deterministic structured renderer for small local models (less hallucination).
- Party identity extraction (contract-holders, landlord, agent) from the “Parties” section.
- Readability cleanup: OCR noise removal, deduped bullets, shorter evidence quotes.
- Remote LLM support (OpenAI-compatible) with automatic local fallback on auth/network errors.

## Folder Layout

```text
examples/legal_contract_analyzer/
  app/
    config.py
    main.py
    pdf_utils.py
    rag_engine.py
    schemas.py
  data/                    # created automatically
  frontend_gradio.py
  requirements.txt
  Dockerfile
```

## Quick Start (Local)

1. Create and activate a virtual environment.
2. Install dependencies.
3. Start the API.
4. (Optional) Start Gradio frontend.

```bash
cd /Users/niccoloforzano/Downloads/legal-contract-analyzer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd /Users/niccoloforzano/Downloads/legal-contract-analyzer
source .venv/bin/activate
python frontend_gradio.py
```

## Readable Output (CLI)

Use `jq` to render the answer with preserved newlines:

```bash
curl -sS -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"document_id":"YOUR_DOC_ID","query":"Who are the contract-holders?","top_k":6}' \
| jq -r '.answer'
```

## Config (Environment Variables)

- `LEGAL_ANALYZER_LLM_MODE` (default: `local`; options: `local`, `remote`)
- `LEGAL_ANALYZER_MODEL_PROFILE` (default: `quality`; options: `fast`, `balanced`, `quality`)
- `LEGAL_ANALYZER_DATA_DIR` (default: `./data`)
- `LEGAL_ANALYZER_EMBED_MODEL` (default by profile)
- `LEGAL_ANALYZER_LLM_MODEL` (default by profile)
- `LEGAL_ANALYZER_FALLBACK_LLM_MODEL` (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `LEGAL_ANALYZER_REMOTE_LLM_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `LEGAL_ANALYZER_REMOTE_LLM_API_KEY` (default: empty)
- `LEGAL_ANALYZER_REMOTE_LLM_MODEL` (default: `Qwen/Qwen2.5-7B-Instruct`)
- `LEGAL_ANALYZER_REMOTE_LLM_TIMEOUT_SECONDS` (default: `180`)
- `LEGAL_ANALYZER_CHUNK_SIZE` (default: `1200`)
- `LEGAL_ANALYZER_CHUNK_OVERLAP` (default: `200`)
- `LEGAL_ANALYZER_EMBED_BATCH_SIZE` (default: `12`)
- `LEGAL_ANALYZER_RETRIEVAL_CANDIDATE_FACTOR` (default: `5`)
- `LEGAL_ANALYZER_USE_CROSS_ENCODER_RERANK` (default: `true`)
- `LEGAL_ANALYZER_CROSS_ENCODER_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `LEGAL_ANALYZER_CROSS_ENCODER_BATCH_SIZE` (default: `8`)
- `LEGAL_ANALYZER_CROSS_ENCODER_WEIGHT` (default: `0.65`)
- `LEGAL_ANALYZER_TWO_STAGE_GENERATION` (default: `true`)
- `LEGAL_ANALYZER_FACT_EXTRACT_MAX_FACTS` (default: `16`)
- `LEGAL_ANALYZER_MAX_NEW_TOKENS` (default: `280`)
- `LEGAL_ANALYZER_TEMPERATURE` (default: `0.0`)
- `LEGAL_ANALYZER_TOP_P` (default: `0.9`)
- `LEGAL_ANALYZER_LEXICAL_RERANK_WEIGHT` (default: `0.22`)
- `LEGAL_ANALYZER_API_URL` (for Gradio, default: `http://localhost:8000`)

### Quality Tuning

- Increase `top_k` in `/analyze` requests to retrieve more chunks (usually `6-10` works well).
- Increase `LEGAL_ANALYZER_RETRIEVAL_CANDIDATE_FACTOR` to widen the reranking pool.
- Increase `LEGAL_ANALYZER_CROSS_ENCODER_WEIGHT` to rely more on cross-encoder reranking.
- Keep `LEGAL_ANALYZER_TWO_STAGE_GENERATION=true` for fact-extraction + grounded final synthesis.
- For small local models, keep `LEGAL_ANALYZER_MODEL_PROFILE=balanced` and rely on the structured renderer.

### Recommended Profiles

- `fast`: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (quickest, lowest quality).
- `balanced`: `Qwen/Qwen2.5-0.5B-Instruct` (good speed/quality on CPU).
- `quality`: `Qwen/Qwen2.5-1.5B-Instruct` (best local quality, slower startup).

### Remote 7B+ Example

Use an OpenAI-compatible provider (or your own vLLM/TGI endpoint) without changing API/UI:

```bash
export LEGAL_ANALYZER_LLM_MODE=remote
export LEGAL_ANALYZER_REMOTE_LLM_BASE_URL=https://openrouter.ai/api/v1
export LEGAL_ANALYZER_REMOTE_LLM_API_KEY=YOUR_KEY
export LEGAL_ANALYZER_REMOTE_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# or: export LEGAL_ANALYZER_REMOTE_LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If remote calls fail (auth/network/provider), the backend now auto-falls back to a local model so `/analyze` and `/summary` still return responses.

## API Examples

Upload:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/absolute/path/to/contract.pdf"
```

Analyze:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "YOUR_DOC_ID",
    "query": "What are the termination rights and notice periods?",
    "top_k": 5
  }'
```

Summary:

```bash
curl "http://localhost:8000/summary/YOUR_DOC_ID"
```

## CUAD Extension Path

This MVP is RAG-first. To add CUAD:
- Add a clause classifier head (41 labels).
- Fine-tune with Hugging Face `Trainer`.
- Store predicted clause tags with each chunk for retrieval filtering and better explainability.

## Fine-Tuning (Colab)

A ready-made QLoRA notebook is included:

- `qwen2_5_1_5b_legal_qlora.ipynb`

It fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` on `nhankins/legal_contracts` (SQuAD-style QA).

## License

MIT (see `LICENSE`).

## Notes

- This app is for informational assistance, not legal advice.
- Model downloads happen the first time the app runs.
- For larger local models, use GPU and adjust `LEGAL_ANALYZER_LLM_MODEL`. For hosted models, set `LEGAL_ANALYZER_LLM_MODE=remote` and `LEGAL_ANALYZER_REMOTE_LLM_MODEL`.
