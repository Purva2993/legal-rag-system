# llm/llm_router.py
import time
from typing import Optional
from loguru import logger
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from config import config


GROUNDING_PROMPT = """You are a precise legal and financial document analyst.

STRICT RULES — you must follow these without exception:
1. Answer ONLY using the provided document context below.
2. If the answer is not found in the context, respond with exactly: 
   "I cannot find this information in the provided documents."
3. ALWAYS cite your source by referencing the document name and relevant section.
4. Never add information from your general knowledge. 
5. Be precise. Legal and financial accuracy is critical.
6. If numbers, dates, or names appear in your answer, they must come directly from the context.

{memory_context}

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""


class LLMRouter:
    """
    WHAT IS LLM ROUTING?
    You might have multiple LLMs available.
    Primary: best quality (Mistral 7B)
    Fallback: if primary fails or is too slow, use backup
    
    WHY FALLBACKS MATTER IN PRODUCTION:
    - Local model crashes → fall back to smaller model
    - Response takes > 30s → timeout and fall back
    - Model returns garbage → retry with different temperature
    
    For interviews: "I designed fallback chains so the system
    degrades gracefully rather than failing completely."
    """
    
    def __init__(self):
        self.primary_llm = self._load_llm(config.primary_llm)
        logger.info(f"LLM Router initialized | Primary: {config.primary_llm}")
    
    def _load_llm(self, model_name: str) -> OllamaLLM:
        return OllamaLLM(
            model=model_name,
            temperature=config.llm_temperature,
            num_predict=config.max_tokens,
        )
    
    def generate(
        self,
        question: str,
        chunks: list[Document],
        memory_context: str = "",
        timeout: int = 120,
    ) -> tuple[str, int, int]:
        """
        Generate answer from retrieved chunks.
        Returns (answer, input_tokens, output_tokens)
        
        GROUNDING: Notice the prompt forces the LLM to use ONLY the context.
        This is the most important prompt engineering decision in a RAG system.
        Without it, the LLM will happily fill gaps with hallucinations.
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source_file", "unknown")
            context_parts.append(
                f"[Source {i+1}: {source}]\n{chunk.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)
        
        # Build the full prompt
        prompt = GROUNDING_PROMPT.format(
            memory_context=memory_context,
            context=context,
            question=question,
        )
        
        # Count input tokens (approximate: 1 token ≈ 4 characters)
        input_tokens = len(prompt) // 4
        
        # Try primary LLM with fallback
        answer = self._call_with_fallback(prompt)
        output_tokens = len(answer) // 4
        
        return answer, input_tokens, output_tokens
    
    def _call_with_fallback(self, prompt: str) -> str:
        """Try primary, fall back if it fails."""
        try:
            logger.debug("Calling primary LLM...")
            start = time.time()
            response = self.primary_llm.invoke(prompt)
            elapsed = time.time() - start
            logger.info(f"Primary LLM responded in {elapsed:.2f}s")
            return response
        except Exception as e:
            logger.error(f"Primary LLM failed: {e}")
            logger.warning("Attempting fallback LLM...")
            try:
                fallback = self._load_llm(config.fallback_llm)
                return fallback.invoke(prompt)
            except Exception as e2:
                logger.error(f"Fallback LLM also failed: {e2}")
                return "System error: Unable to generate response. Please try again."