# rag_pipeline.py  (create this in root folder)
import uuid
import time
from typing import Optional
from loguru import logger

from config import config
from ingestion.loader import DocumentLoader
from ingestion.chunker import Chunker
from embeddings.embedder import EmbeddingModel
from vectorstore.store import VectorStore
from retrieval.retriever import HybridRetriever
from memory.memory_manager import ConversationMemory
from llm.llm_router import LLMRouter
from guardrails.input_guard import InputGuard
from guardrails.output_guard import OutputGuard
from observability.tracer import RequestTrace, Timer, logger


class RAGPipeline:
    """
    The orchestrator. Connects every layer in the right order.
    
    This is what you explain in interviews:
    'I built a pipeline where each layer has a single responsibility.
    If something breaks, I can isolate it to the exact layer,
    look at the trace, and fix it without touching other components.'
    """
    
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        self.embedding_model = EmbeddingModel()
        self.vectorstore = VectorStore(self.embedding_model)
        self.llm_router = LLMRouter()
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.memory = ConversationMemory()
        self.all_chunks = []
        self.retriever: Optional[HybridRetriever] = None
        
        # Try to load existing index
        if self.vectorstore.load_existing():
            logger.info("Existing vector store loaded. Ready to query.")
            # Automatically rebuild retriever from saved chunks
            try:
                saved_chunks = self.vectorstore.vectorstore.get()
                if saved_chunks and saved_chunks['documents']:
                    from langchain_core.documents import Document
                    docs = [
                        Document(
                            page_content=text,
                            metadata=meta
                        )
                        for text, meta in zip(
                            saved_chunks['documents'],
                            saved_chunks['metadatas']
                        )
                    ]
                    self.all_chunks = docs
                    self.retriever = HybridRetriever(self.vectorstore, self.all_chunks)
                    logger.info(f"Retriever rebuilt with {len(docs)} chunks")
            except Exception as e:
                logger.warning(f"Could not rebuild retriever automatically: {e}")
    
    def ingest(self, file_path: str) -> dict:
        """
        Full ingestion pipeline:
        File → Load → Chunk → Embed → Store → Build retriever
        """
        loader = DocumentLoader()
        chunker = Chunker()
        
        documents = loader.load(file_path)
        chunks = chunker.chunk(documents)
        chunker.inspect_chunks(chunks, n=2)  # Always inspect
        
        self.all_chunks = chunks
        self.vectorstore.ingest(chunks)
        self.retriever = HybridRetriever(self.vectorstore, self.all_chunks)
        
        return {
            "status": "success",
            "document": file_path,
            "chunks_created": len(chunks),
            "chunk_strategy": config.chunk_strategy,
        }
    
    def query(self, question: str, session_id: str = "default") -> dict:
        """
        Full query pipeline with observability:
        Question → Guard → Memory → Retrieve → LLM → Guard → Trace → Answer
        """
        request_id = str(uuid.uuid4())[:8]
        trace = RequestTrace(request_id=request_id, query=question)
        total_start = time.time()
        
        # ── 1. INPUT GUARDRAIL ─────────────────────────────────
        with Timer("guardrail_input") as t:
            is_valid, reason = self.input_guard.validate(question)
        trace.guardrail_time += t.elapsed
        
        if not is_valid:
            trace.guardrail_triggered = True
            trace.total_time = time.time() - total_start
            trace.log_summary()
            return {
                "answer": f"Query rejected: {reason}",
                "guardrail_triggered": True,
                "confidence": 0.0,
                "sources": [],
                "trace": self._trace_to_dict(trace),
            }
        
        # ── 2. MEMORY: Build contextualized query ───────────────
        contextualized_query = self.memory.build_contextualized_query(question)
        memory_context = self.memory.get_context_string()
        
        # ── 3. RETRIEVAL ────────────────────────────────────────
        if not self.retriever:
            return {
                "answer": "No documents ingested yet. Please upload a document first.",
                "sources": [],
                "confidence": 0.0,
                "guardrail_triggered": False,
                "trace": {}
            }
        
        with Timer("retrieval") as t:
            chunks, top_score = self.retriever.retrieve(contextualized_query)
        trace.retrieval_time = t.elapsed
        trace.chunks_retrieved = len(chunks)
        trace.top_similarity_score = top_score
        
        # Low similarity score = retrieval likely failed
        if top_score < 0.3:
            logger.warning(
                f"Low similarity score ({top_score:.3f}). "
                "Retrieved chunks may not be relevant. "
                "Possible causes: query outside document scope, chunking issue, or embedding mismatch."
            )
        
        # ── 4. LLM SYNTHESIS ────────────────────────────────────
        with Timer("llm") as t:
            answer, input_tokens, output_tokens = self.llm_router.generate(
                question=question,
                chunks=chunks,
                memory_context=memory_context,
            )
        trace.llm_time = t.elapsed
        trace.input_tokens = input_tokens
        trace.output_tokens = output_tokens
        
        # ── 5. OUTPUT GUARDRAIL ─────────────────────────────────
        with Timer("guardrail_output") as t:
            is_valid_output, warning, confidence = self.output_guard.validate(
                answer, question
            )
        trace.guardrail_time += t.elapsed
        trace.faithfulness_score = confidence
        
        if not is_valid_output:
            trace.guardrail_triggered = True
            logger.warning(f"Output guardrail triggered: {warning}")
            answer = (
                f"[Warning: Answer quality check failed — {warning}]\n\n"
                + answer
            )
        
        # ── 6. MEMORY: Store this turn ───────────────────────────
        self.memory.add_turn(question, answer)
        
        # ── 7. OBSERVABILITY: Cost + trace ──────────────────────
        trace.calculate_cost(
            config.cost_per_1k_input_tokens,
            config.cost_per_1k_output_tokens
        )
        trace.total_time = time.time() - total_start
        trace.answer_length = len(answer)
        trace.log_summary()
        
        # Build source citations
        sources = [
            {
                "source": chunk.metadata.get("source_file", "unknown"),
                "preview": chunk.page_content[:200],
                "page": str(chunk.metadata.get("page", "N/A")),
            }
            for chunk in chunks
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "guardrail_triggered": trace.guardrail_triggered,
            "trace": self._trace_to_dict(trace),
        }
    
    def _trace_to_dict(self, trace: RequestTrace) -> dict:
        return {
            "request_id": trace.request_id,
            "latency": {
                "embedding_s": trace.embedding_time,
                "retrieval_s": trace.retrieval_time,
                "llm_s": trace.llm_time,
                "guardrails_s": trace.guardrail_time,
                "total_s": trace.total_time,
            },
            "tokens": {
                "input": trace.input_tokens,
                "output": trace.output_tokens,
            },
            "cost_usd": trace.estimated_cost_usd,
            "retrieval": {
                "chunks_used": trace.chunks_retrieved,
                "top_similarity_score": trace.top_similarity_score,
            },
            "quality": {
                "faithfulness_score": trace.faithfulness_score,
                "guardrail_triggered": trace.guardrail_triggered,
            },
        }