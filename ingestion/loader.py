# ingestion/loader.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from loguru import logger


class DocumentLoader:
    """
    Loads documents from disk.
    Think of this as the librarian who picks up books 
    and prepares them for cataloguing.
    """
    
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
    
    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        logger.info(f"Loading document: {path.name}")
        
        if path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        
        documents = loader.load()
        
        # Attach metadata — this is critical for citations later
        for doc in documents:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = path.suffix
        
        logger.info(f"Loaded {len(documents)} pages from {path.name}")
        return documents
    
    def load_directory(self, dir_path: str) -> List[Document]:
        """Load all supported documents from a folder."""
        all_docs = []
        directory = Path(dir_path)
        
        for file_path in directory.rglob("*"):
            if file_path.suffix in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self.load(str(file_path))
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs