"""Embedding service for generating vector embeddings."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.logging_config import configure_logging, get_logger
from shared.metrics import get_metrics, set_system_info
from shared.sentry_config import configure_sentry
from shared.embedding_service import EmbeddingService

# Configure structured logging
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_FORMAT", "json") == "json",
    service_name="embedding-service"
)
logger = get_logger(__name__)

app = FastAPI(title="Embedding Service")

embedding_service: EmbeddingService = None


@app.on_event("startup")
async def startup_event():
    global embedding_service
    logger.info("Starting Embedding Service...")

    configure_sentry(
        service_name="embedding-service",
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("RELEASE", "1.0.0")
    )
    set_system_info(
        service_name="embedding-service",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )

    model_name = os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
    embedding_service = EmbeddingService(model_name=model_name)

    logger.info("Embedding Service started successfully")


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str]


class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]]
    dimension: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "embedding"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for a list of texts."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list must be non-empty")

    try:
        embeddings = embedding_service.embed_batch(request.texts)
        return EmbedResponse(
            embeddings=[e.tolist() for e in embeddings],
            dimension=embedding_service.get_embedding_dimension(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")
