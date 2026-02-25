# retrieval/retriever.py
from typing import List, Tuple
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi
from config import config
from vectorstore.store import VectorStore
import time


class HybridRetriever:
    """
    HYBRID RETRIEVAL = Semantic Search + Keyword Search combined.
    
    WHY HYBRID FOR LEGAL/FINANCIAL DOCUMENTS?
    
    Semantic search: finds meaning matches
    - Query: "what happens if payment is late?"
    - Finds: "consequences of delayed remittance" ← different words, same meaning
    
    Keyword search (BM25): finds exact term matches
    - Query: "Section 4.2 indemnification clause"
    - Finds: chunks containing exactly "Section 4.2" and "indemnification"
    - Semantic search might miss this because section numbers have no semantic meaning
    
    Legal documents need BOTH:
    - Specific clause references (keyword)
    - Conceptual questions (semantic)
    
    WHAT IS BM25?
    A classic information retrieval algorithm. 
    It scores documents based on: how often your search terms appear,
    and penalizes very long documents (term frequency normalization).
    It's what Google used before neural search. Still excellent for exact terms.
    """
    
    def __init__(self, vectorstore: VectorStore, all_chunks: List[Document]):
        self.vectorstore = vectorstore
        self.all_chunks = all_chunks
        self.bm25 = self._build_bm25_index(all_chunks)
    
    def _build_bm25_index(self, chunks: List[Document]) -> BM25Okapi:
        """Build keyword search index from all chunks."""
        tokenized = [chunk.page_content.lower().split() for chunk in chunks]
        logger.info(f"Built BM25 index over {len(tokenized)} chunks")
        return BM25Okapi(tokenized)
    
    def retrieve(self, query: str) -> Tuple[List[Document], float]:
        """
        Retrieve using configured strategy.
        Returns (chunks, top_similarity_score)
        """
        strategy = config.retrieval_strategy
        start = time.time()
        
        if strategy == "hybrid":
            chunks = self._hybrid_retrieve(query)
        elif strategy == "mmr":
            chunks = self.vectorstore.mmr_search(query, k=config.top_k)
        else:
            results = self.vectorstore.similarity_search_with_scores(
                query, k=config.top_k
            )
            chunks = [doc for doc, _ in results]
        
        elapsed = time.time() - start
        logger.info(f"Retrieval ({strategy}): {len(chunks)} chunks in {elapsed:.3f}s")
        
        # Get top score for observability
        scored = self.vectorstore.similarity_search_with_scores(query, k=1)
        top_score = scored[0][1] if scored else 0.0
        
        return chunks, top_score
    
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """
        Combine semantic + keyword retrieval with score fusion.
        
        RECIPROCAL RANK FUSION (RRF):
        Rather than trying to normalize scores from two different systems
        (which is hard — semantic scores and BM25 scores have different scales),
        we use the RANK instead of the score.
        
        If a chunk is ranked #1 by semantic AND #1 by BM25, 
        it gets a very high combined score.
        If it's #1 semantic but #50 BM25, it gets a moderate score.
        
        This is robust and simple — used in production systems.
        """
        k = config.top_k
        
        # Semantic results
        semantic_results = self.vectorstore.similarity_search_with_scores(
            query, k=k * 2  # fetch more, then merge
        )
        semantic_docs = [doc for doc, _ in semantic_results]
        
        # BM25 keyword results
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:k * 2]
        bm25_docs = [self.all_chunks[i] for i in bm25_top_indices]
        
        # Reciprocal Rank Fusion
        doc_scores = {}
        
        for rank, doc in enumerate(semantic_docs):
            key = doc.page_content[:100]  # use content as key
            doc_scores[key] = doc_scores.get(key, {"doc": doc, "score": 0})
            doc_scores[key]["score"] += 1 / (rank + 60)  # RRF formula
        
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0}
            doc_scores[key]["score"] += 1 / (rank + 60)
        
        # Sort by fused score and return top k
        sorted_results = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        
        final_chunks = [item["doc"] for item in sorted_results[:k]]
        logger.info(f"Hybrid retrieval: merged {len(semantic_docs)} semantic + {len(bm25_docs)} BM25 → {len(final_chunks)} final chunks")
        return final_chunks