import time
from typing import List
from loguru import logger
from langchain_ollama import OllamaEmbeddings
from config import config


class EmbeddingModel:
    """
    Converts text into vectors using Ollama locally.
    100% free, no API key, no dependency conflicts.
    
    WHAT IS AN EMBEDDING?
    Text goes in → list of numbers comes out.
    Similar meaning = similar numbers = close in vector space.
    """
    
    def __init__(self):
        self.model = OllamaEmbeddings(model=config.embedding_model)
        logger.info(f"Embedding model loaded: {config.embedding_model} via Ollama")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start = time.time()
        vectors = self.model.embed_documents(texts)
        elapsed = time.time() - start
        logger.debug(f"Embedded {len(texts)} texts in {elapsed:.3f}s | Dim: {len(vectors[0])}")
        return vectors
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)
    
    def get_langchain_embeddings(self):
        return self.model
