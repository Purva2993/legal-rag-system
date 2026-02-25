# ingestion/chunker.py
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from loguru import logger
from config import config


class Chunker:
    """
    Takes full documents and cuts them into digestible pieces.
    
    WHY CHUNKING EXISTS:
    LLMs have a limited reading window (context window).
    You can't feed a 100-page legal contract to an LLM.
    You find the relevant pages first, then feed only those.
    
    THE CORE TRADEOFF:
    - Too small: a chunk might not have enough context to be useful
    - Too large: you stuff irrelevant content into the LLM's window
    - Overlap: prevents answers from being cut in half at chunk boundaries
    """
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        strategy = config.chunk_strategy
        logger.info(f"Chunking {len(documents)} documents using '{strategy}' strategy")
        
        if strategy == "recursive":
            return self._recursive_chunk(documents)
        elif strategy == "fixed":
            return self._fixed_chunk(documents)
        elif strategy == "sentence":
            return self._sentence_chunk(documents)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _recursive_chunk(self, documents: List[Document]) -> List[Document]:
        """
        RECURSIVE CHUNKING — Best for legal/financial documents.
        
        How it works: It tries to split on paragraphs first (\n\n).
        If the chunk is still too big, it splits on sentences (\n).
        If still too big, it splits on words.
        
        WHY IT'S GOOD FOR LEGAL TEXT:
        Legal documents have natural structure — clauses, paragraphs, sections.
        Recursive splitting respects that structure instead of blindly cutting.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        self._log_chunk_stats(chunks, "recursive")
        return chunks
    
    def _fixed_chunk(self, documents: List[Document]) -> List[Document]:
        """
        FIXED SIZE CHUNKING — Simplest approach.
        
        Cuts every N characters regardless of sentence or paragraph boundaries.
        
        WEAKNESS: Can cut mid-sentence.
        Example: "The defendant shall pay..." cut to "The defendant shall pa"
        and next chunk starts with "y damages of $500,000"
        The meaning is split. Retrieval will find neither chunk useful.
        
        USE WHEN: Documents have no natural structure (raw OCR output).
        """
        splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator=" ",
        )
        chunks = splitter.split_documents(documents)
        self._log_chunk_stats(chunks, "fixed")
        return chunks
    
    def _sentence_chunk(self, documents: List[Document]) -> List[Document]:
        """
        SENTENCE-BASED CHUNKING — Groups complete sentences together.
        
        Ensures no sentence is ever split across chunks.
        
        STRENGTH: Every chunk is semantically self-contained.
        WEAKNESS: Chunk sizes vary wildly. A one-word sentence is a tiny chunk.
        
        USE WHEN: Document quality and sentence integrity matters more than
        uniform chunk sizes (academic papers, legal opinions).
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=[". ", "? ", "! ", "\n\n", "\n"],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        self._log_chunk_stats(chunks, "sentence")
        return chunks
    
    def _log_chunk_stats(self, chunks: List[Document], strategy: str):
        """Always inspect your chunks. Silent failures start here."""
        sizes = [len(c.page_content) for c in chunks]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        logger.info(
            f"[{strategy}] Created {len(chunks)} chunks | "
            f"Avg size: {avg_size:.0f} chars | "
            f"Min: {min(sizes)} | Max: {max(sizes)}"
        )
    
    def inspect_chunks(self, chunks: List[Document], n: int = 3):
        """
        Call this during development to manually verify chunk quality.
        In interviews: 'I always visually inspect chunks because 
        chunk quality issues are silent — the system won't crash,
        it just gives bad answers.'
        """
        logger.info(f"=== CHUNK INSPECTION (first {n}) ===")
        for i, chunk in enumerate(chunks[:n]):
            logger.info(f"\n--- Chunk {i} ---\n{chunk.page_content}\n")
            logger.info(f"Metadata: {chunk.metadata}")