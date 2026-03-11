# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import requests

API_URL = os.getenv("LEGAL_ANALYZER_API_URL", "http://localhost:8000").rstrip("/")
NOTEBOOK_MODE = os.getenv("LEGAL_ANALYZER_NOTEBOOK_MODE",
                          "false").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_TOP_K = _env_int("LEGAL_ANALYZER_DEFAULT_TOP_K", 0)
if NOTEBOOK_MODE and DEFAULT_TOP_K <= 0:
    DEFAULT_TOP_K = 5


def _safe_json(resp: requests.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"detail": resp.text}


def upload_contract(file_path: str | None) -> tuple[str, str]:
    if not file_path:
        return "", "Please upload a PDF file first."

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return "", "Only .pdf files are supported."

    with path.open("rb") as f:
        files = {"file": (path.name, f, "application/pdf")}
        resp = requests.post(f"{API_URL}/upload", files=files, timeout=600)

    data = _safe_json(resp)
    if not resp.ok:
        return "", f"Upload failed: {data.get('detail', 'Unknown error')}"

    status = (f"Uploaded `{data['filename']}` as document `{data['document_id']}` "
              f"({data['num_pages']} pages, {data['num_chunks']} chunks).")
    return data["document_id"], status


def analyze_contract(document_id: str, question: str,
                     top_k: int) -> tuple[str, str]:
    if not document_id:
        return "Please upload a contract first.", ""
    if not question.strip():
        return "Please enter a question.", ""

    payload = {
        "document_id": document_id.strip(),
        "query": question.strip(),
        "top_k": int(top_k),
    }
    resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=600)
    data = _safe_json(resp)
    if not resp.ok:
        return f"Analyze failed: {data.get('detail', 'Unknown error')}", ""

    citations = []
    for src in data.get("sources", []):
        citations.append(
            f"[Chunk {src['chunk_id']}] score={src['score']}\n{src['text_excerpt']}"
        )
    return data["answer"], "\n\n".join(citations)


def summarize_contract(document_id: str) -> tuple[str, str]:
    if not document_id:
        return "Please upload a contract first.", ""

    resp = requests.get(f"{API_URL}/summary/{document_id.strip()}", timeout=600)
    data = _safe_json(resp)
    if not resp.ok:
        return f"Summary failed: {data.get('detail', 'Unknown error')}", ""

    citations = []
    for src in data.get("sources", []):
        citations.append(
            f"[Chunk {src['chunk_id']}] score={src['score']}\n{src['text_excerpt']}"
        )
    return data["answer"], "\n\n".join(citations)


with gr.Blocks(title="Legal Contract Analyzer") as demo:
    gr.Markdown("# Legal Contract Analyzer")
    gr.Markdown(
        "Upload a contract PDF, ask questions, and get plain-English explanations "
        "with source chunks."
    )

    doc_id_state = gr.State(value="")

    with gr.Row():
        contract_file = gr.File(label="Contract PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Upload Contract")

    upload_status = gr.Textbox(label="Upload Status", lines=2)
    doc_id_box = gr.Textbox(label="Document ID", interactive=False)

    with gr.Row():
        question = gr.Textbox(label="Question",
                              placeholder="What are the termination obligations?",
                              lines=3)
        if DEFAULT_TOP_K > 0:
            top_k = gr.State(value=DEFAULT_TOP_K)
        else:
            top_k = gr.Slider(label="Top K Chunks",
                              minimum=1,
                              maximum=12,
                              value=5)

    with gr.Row():
        analyze_btn = gr.Button("Analyze Question")
        summary_btn = gr.Button("Generate Full Summary")

    answer_box = gr.Textbox(label="Answer", lines=16)
    sources_box = gr.Textbox(label="Source Chunks", lines=14)

    upload_btn.click(fn=upload_contract,
                     inputs=[contract_file],
                     outputs=[doc_id_state, upload_status]).then(
                         lambda x: x,
                         inputs=[doc_id_state],
                         outputs=[doc_id_box],
                     )

    analyze_btn.click(fn=analyze_contract,
                      inputs=[doc_id_state, question, top_k],
                      outputs=[answer_box, sources_box])

    summary_btn.click(fn=summarize_contract,
                      inputs=[doc_id_state],
                      outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
