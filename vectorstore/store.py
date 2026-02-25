# vectorstore/store.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from loguru import logger
from config import config
from embeddings.embedder import EmbeddingModel


class VectorStore:
    """
    The indexed library where all chunks live as vectors.
    
    WHAT CHROMADB DOES:
    When you search, it doesn't read every chunk linearly.
    It uses a structure called HNSW (Hierarchical Navigable Small World)
    to find nearest neighbors in milliseconds even across millions of vectors.
    
    Think of it like: instead of checking every book in a library,
    you follow a smart map that takes you to the right shelf immediately.
    
    HNSW TRADEOFF:
    It finds approximate nearest neighbors, not guaranteed exact ones.
    For RAG this is fine — we care about "good enough" retrieval speed,
    not mathematically perfect retrieval at 10x the latency.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.vectorstore: Optional[Chroma] = None
    
    def ingest(self, chunks: List[Document]) -> None:
        """
        Store chunks in the vector database.
        This happens OFFLINE — before any user query.
        """
        logger.info(f"Ingesting {len(chunks)} chunks into vector store...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model.get_langchain_embeddings(),
            persist_directory=config.vectorstore_path,
            collection_name=config.collection_name,
        )
        logger.info(f"Vector store created at: {config.vectorstore_path}")
    
    def load_existing(self) -> bool:
        """Load a previously built vector store from disk."""
        try:
            self.vectorstore = Chroma(
                persist_directory=config.vectorstore_path,
                embedding_function=self.embedding_model.get_langchain_embeddings(),
                collection_name=config.collection_name,
            )
            count = self.vectorstore._collection.count()
            logger.info(f"Loaded existing vector store | {count} chunks indexed")
            return True
        except Exception as e:
            logger.warning(f"No existing vector store found: {e}")
            return False
    
    def similarity_search_with_scores(
        self, query: str, k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        Core retrieval: find the k most similar chunks to the query.
        Returns chunks WITH their similarity scores.
        
        ALWAYS log scores. A score of 0.3 when your good answers 
        score 0.8 tells you immediately: retrieval failed, not the LLM.
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Run ingest() first.")
        
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=k
        )
        
        for doc, score in results:
            logger.debug(
                f"Score: {score:.4f} | "
                f"Source: {doc.metadata.get('source_file', 'unknown')} | "
                f"Preview: {doc.page_content[:80]}..."
            )
        
        return results
    
    def mmr_search(self, query: str, k: int = 5) -> List[Document]:
        """
        MMR = Maximal Marginal Relevance.
        
        PROBLEM IT SOLVES:
        Basic similarity search can return 5 chunks that all say 
        the same thing (e.g., the same clause copied 5 times in a contract).
        That wastes your entire context window on redundant information.
        
        HOW MMR WORKS:
        It balances relevance (is this chunk useful?) against 
        diversity (does this chunk add NEW information?).
        
        lambda_mult=0.7 means: 70% care about relevance, 30% care about diversity.
        For legal docs with repetitive clauses, lower lambda = more diversity.
        """
        return self.vectorstore.max_marginal_relevance_search(
            query=query, k=k, lambda_mult=1 - config.mmr_diversity_score
        )
    
    def get_retriever(self, strategy: str = "similarity"):
        """Return a LangChain-compatible retriever for pipeline use."""
        if strategy == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": config.top_k, "lambda_mult": 0.7}
            )
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.top_k}
        )