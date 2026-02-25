# Legal RAG System

Production-grade Retrieval-Augmented Generation system for legal and financial documents.

## What It Does
Upload any legal or financial PDF. Ask questions in natural language. 
Get accurate, cited, grounded answers with full observability.

## Architecture — 7 Layers
```
Query → Input Guard → Memory → Hybrid Retrieval → LLM Synthesis → Output Guard → Traced Response
```

1. **Input Guardrails** — blocks prompt injection and off-topic queries before spending any compute
2. **Conversation Memory** — sliding window of last 5 turns, enriches retrieval query with context
3. **Hybrid Retrieval** — combines semantic search + BM25 keyword search via Reciprocal Rank Fusion
4. **Grounded Synthesis** — LLM answers ONLY from retrieved chunks, cites every source
5. **Output Guardrails** — detects hallucination signals, validates citation presence
6. **Observability** — every request traced: latency per layer, token count, cost, similarity scores
7. **Evaluation** — RAGAS metrics: faithfulness, context recall, context precision, answer relevancy

## Key Design Decisions

**Why recursive chunking?** Legal documents have natural paragraph structure. 
Recursive splitting respects that structure instead of blindly cutting mid-sentence.

**Why hybrid search?** Legal users ask both conceptual questions (semantic search wins) 
and exact clause references like "Section 4.2" (BM25 wins). RRF combines both without 
score normalization issues.

**Why local LLM?** Entire system runs on Ollama + Mistral — no API keys, no data leaving 
your machine. Critical for legal documents with confidentiality requirements. 
Swap to GPT-4 with one config line change.

**Why guardrails before retrieval?** Blocking bad queries before any compute is spent 
means zero token cost and zero latency for attacks.

## Tech Stack
- **LLM**: Mistral 7B via Ollama (local, free, swappable)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: ChromaDB (persistent)
- **Keyword Search**: BM25 via rank-bm25
- **API**: FastAPI
- **Evaluation**: RAGAS
- **Observability**: Loguru + custom request tracer

## API Endpoints
- `POST /ingest/upload` — upload and process a PDF
- `POST /query` — ask a question with full traced response
- `DELETE /memory/{session_id}` — clear conversation history
- `POST /evaluate` — run RAGAS evaluation suite
- `GET /health` — health check
- `GET /config` — current system configuration

## Results On Apple 10-Q Filing
- Context Recall: 1.0 (retrieval finds all required information)
- Similarity scores: 0.69–0.72 on financial domain queries
- Guardrails: prompt injection blocked in 0.001s
- Grounding: LLM correctly refused off-topic questions

## Run Locally
```bash
ollama pull mistral
ollama pull nomic-embed-text
pip install -r requirements.txt
python main.py
# Open http://localhost:8000/docs
