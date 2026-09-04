"""LLM service for response generation."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.logging_config import configure_logging, get_logger
from shared.metrics import get_metrics, set_system_info
from shared.sentry_config import configure_sentry

# Configure structured logging
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_FORMAT", "json") == "json",
    service_name="llm-service"
)
logger = get_logger(__name__)

# Placeholder - will be implemented in later tasks
app = FastAPI(title="LLM Service")


@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on startup."""
    logger.info("Starting LLM Service...")
    
    # Configure Sentry
    configure_sentry(
        service_name="llm-service",
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("RELEASE", "1.0.0")
    )
    
    # Set system info for metrics
    set_system_info(
        service_name="llm-service",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )
    
    logger.info("LLM Service started successfully")


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str
    tokens_used: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "llm"}


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes metrics for monitoring LLM generation performance.
    """
    from fastapi.responses import Response
    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from prompt."""
    # Placeholder - will be implemented in later tasks
    raise HTTPException(status_code=501, detail="Not implemented yet")
