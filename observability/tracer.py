# observability/tracer.py
import time
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
import sys

# Configure logger — writes to console AND file
logger.remove()
logger.add(sys.stdout, level="INFO", 
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add("./logs/rag_system.log", level="DEBUG", rotation="10 MB")


@dataclass
class RequestTrace:
    """
    Think of this as a receipt for one RAG request.
    Every layer stamps it with how long it took.
    At the end you have a complete breakdown.
    """
    request_id: str
    query: str
    
    # Timing per layer (in seconds)
    ingestion_time: float = 0.0
    embedding_time: float = 0.0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    llm_time: float = 0.0
    guardrail_time: float = 0.0
    total_time: float = 0.0
    
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Retrieval quality
    chunks_retrieved: int = 0
    top_similarity_score: float = 0.0
    
    # Cost (zero for local, but tracked for when you swap to paid)
    estimated_cost_usd: float = 0.0
    
    # Outcome
    answer_length: int = 0
    faithfulness_score: Optional[float] = None
    guardrail_triggered: bool = False

    def calculate_cost(self, cost_per_1k_input: float, cost_per_1k_output: float):
        self.estimated_cost_usd = (
            (self.input_tokens / 1000) * cost_per_1k_input +
            (self.output_tokens / 1000) * cost_per_1k_output
        )

    def log_summary(self):
        logger.info(f"""
╔══════════════════════════════════════════════╗
║           REQUEST TRACE SUMMARY              ║
╠══════════════════════════════════════════════╣
║ Query     : {self.query[:45]:<45}║
║ Request ID: {self.request_id:<45}║
╠══════════════════════════════════════════════╣
║ LATENCY BREAKDOWN                            ║
║  Embedding    : {self.embedding_time:.3f}s                         ║
║  Retrieval    : {self.retrieval_time:.3f}s                         ║
║  Reranking    : {self.rerank_time:.3f}s                         ║
║  LLM          : {self.llm_time:.3f}s                         ║
║  Guardrails   : {self.guardrail_time:.3f}s                         ║
║  TOTAL        : {self.total_time:.3f}s                         ║
╠══════════════════════════════════════════════╣
║ TOKENS & COST                                ║
║  Input tokens : {self.input_tokens:<45}║
║  Output tokens: {self.output_tokens:<45}║
║  Est. cost    : ${self.estimated_cost_usd:.6f}                    ║
╠══════════════════════════════════════════════╣
║ QUALITY                                      ║
║  Chunks used  : {self.chunks_retrieved:<45}║
║  Top score    : {self.top_similarity_score:<45.4f}║
║  Faithfulness : {str(self.faithfulness_score):<45}║
╚══════════════════════════════════════════════╝
        """)


class Timer:
    """Simple context manager. Use it to time any block of code."""
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        logger.debug(f"[{self.name}] took {self.elapsed:.3f}s")