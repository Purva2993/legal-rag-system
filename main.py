# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from loguru import logger

from rag_pipeline import RAGPipeline
from api.schemas import QueryRequest, QueryResponse, IngestRequest

app = FastAPI(
    title="Legal RAG System",
    description="Production-grade RAG for legal and financial documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single pipeline instance shared across requests
pipeline = RAGPipeline()


@app.get("/health")
def health_check():
    """Always have a health endpoint. Load balancers need this."""
    return {"status": "healthy", "service": "Legal RAG System"}


@app.post("/ingest/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """Upload a document and ingest it into the vector store."""
    save_path = f"./data/documents/{file.filename}"
    
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    logger.info(f"File uploaded: {file.filename}")
    
    try:
        result = pipeline.ingest(save_path)
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question about ingested documents."""
    try:
        result = pipeline.query(
            question=request.question,
            session_id=request.session_id
        )
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{session_id}")
def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    pipeline.memory.clear()
    return {"status": "memory cleared", "session_id": session_id}


@app.post("/evaluate")
async def run_evaluation():
    """Run RAGAS evaluation in a separate thread to avoid async conflicts."""
    from evaluation.evaluator import run_ragas_evaluation
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, run_ragas_evaluation, pipeline
        )
        return {"status": "complete", "scores": str(scores)}
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
def get_config():
    """Expose current config — useful for debugging in production."""
    from config import config
    return config.dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)