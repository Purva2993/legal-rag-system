# config.py
from pydantic import BaseModel
from typing import Literal

class RAGConfig(BaseModel):
    
    # ── Chunking ──────────────────────────────────────────
    chunk_strategy: Literal["recursive", "fixed", "sentence"] = "recursive"
    chunk_size: int = 600          # characters per chunk
    chunk_overlap: int = 100       # overlap between chunks
    
    # ── Embedding ─────────────────────────────────────────
    embedding_model: str = "nomic-embed-text"   # via Ollama (free)
    embedding_backend: Literal["ollama", "huggingface"] = "ollama"
    
    # ── Vector Store ──────────────────────────────────────
    vectorstore_path: str = "./chroma_db"
    collection_name: str = "legal_documents"
    
    # ── Retrieval ─────────────────────────────────────────
    retrieval_strategy: Literal["similarity", "mmr", "hybrid"] = "hybrid"
    top_k: int = 5                 # how many chunks to retrieve
    mmr_diversity_score: float = 0.3
    
    # ── LLM ───────────────────────────────────────────────
    primary_llm: str = "mistral"
    fallback_llm: str = "mistral"  # same for now; swap when you add more
    llm_backend: Literal["ollama"] = "ollama"
    llm_temperature: float = 0.0   # 0 = deterministic, important for legal
    max_tokens: int = 1024
    
    # ── Memory ────────────────────────────────────────────
    memory_enabled: bool = True
    max_history_turns: int = 5     # how many Q&A pairs to remember
    
    # ── Guardrails ────────────────────────────────────────
    input_guard_enabled: bool = True
    output_guard_enabled: bool = True
    min_faithfulness_score: float = 0.7   # below this = flag the answer
    
    # ── Observability ─────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "./logs/rag_system.log"
    track_costs: bool = True
    
    # ── Approximate token cost (for local = $0, but we track anyway) ──
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0

# Single global config instance — import this everywhere
config = RAGConfig()