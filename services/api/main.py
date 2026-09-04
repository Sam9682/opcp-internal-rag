"""
API Gateway for RAG application.

This module provides the FastAPI application that exposes REST and WebSocket
endpoints for the RAG system. It orchestrates calls to the RAGQueryEngine
and other services.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 10.2, 15.1, 15.3
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, AsyncIterator
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import sys
import os

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.rag_query_engine import RAGQueryEngine, SecurityException
from shared.embedding_client import EmbeddingClient
from shared.vector_search_service import VectorSearchService
from shared.llm_client import LLMClient
from shared.llm_guard_service import LLMGuardService
from shared.conversation_memory_service import ConversationMemoryService
from shared.document_ingestion_service import DocumentIngestionService
from shared.text_preprocessor import TextPreprocessor
from shared.cache_client import CacheClient
from shared.database import init_db, get_session, get_db_manager
from shared.config import get_settings
from shared.logging_config import configure_logging, get_logger, log_security_event
from shared.metrics import get_metrics, track_api_request, record_query_success, record_query_error, set_system_info
from shared.sentry_config import configure_sentry, capture_exception, add_breadcrumb, set_user_context
from shared.auth import (
    get_current_user, get_user_id_from_request, check_conversation_access,
    AuthorizationError, log_access_attempt
)

# Configure structured logging
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("LOG_FORMAT", "json") == "json",
    service_name="api-gateway"
)
logger = get_logger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="RAG API Gateway",
    description="API Gateway for Generic RAG Web Application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting storage (in-memory for simplicity)
# In production, use Redis or similar
rate_limit_storage: Dict[str, List[float]] = defaultdict(list)


# Pydantic Models

class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source citations")
    conversation_id: str = Field(..., description="Conversation ID")


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    directory_path: Optional[str] = Field(None, description="Directory path to ingest (defaults to configured docs path)")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    status: str = Field(..., description="Ingestion status")
    message: str = Field(..., description="Status message")
    jobs_queued: int = Field(..., description="Number of jobs queued")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ConversationMessage(BaseModel):
    """Model for conversation message."""
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source citations for assistant messages")


class ConversationResponse(BaseModel):
    """Response model for conversation history."""
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[ConversationMessage] = Field(..., description="Conversation messages")
    total_messages: int = Field(..., description="Total number of messages")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


# Global service instances (initialized on startup)
rag_engine: Optional[RAGQueryEngine] = None
ingestion_service: Optional[DocumentIngestionService] = None

# Security scheme for JWT bearer tokens
security = HTTPBearer(auto_error=False)


# Dependency functions

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Use IP address as client ID (in production, use authenticated user ID)
    return request.client.host if request.client else "unknown"


def check_rate_limit(client_id: str, limit: int = None) -> bool:
    """
    Check if client has exceeded rate limit.
    
    Args:
        client_id: Client identifier
        limit: Rate limit (requests per hour), defaults to anonymous limit
        
    Returns:
        True if within limit, False if exceeded
    """
    if limit is None:
        limit = settings.rate_limit_anonymous
    
    now = time.time()
    hour_ago = now - 3600
    
    # Clean old entries
    rate_limit_storage[client_id] = [
        timestamp for timestamp in rate_limit_storage[client_id]
        if timestamp > hour_ago
    ]
    
    # Check limit
    if len(rate_limit_storage[client_id]) >= limit:
        return False
    
    # Record request
    rate_limit_storage[client_id].append(now)
    return True


async def rate_limit_dependency(request: Request):
    """Dependency for rate limiting."""
    client_id = get_client_id(request)
    
    # TODO: Check if user is authenticated and use higher limit
    # For now, use anonymous limit for all requests
    
    if not check_rate_limit(client_id, settings.rate_limit_anonymous):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.rate_limit_anonymous} requests per hour."
        )


# Exception handlers

@app.exception_handler(AuthorizationError)
async def authorization_exception_handler(request: Request, exc: AuthorizationError):
    """Handle authorization exceptions."""
    logger.warning(f"Authorization error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=ErrorResponse(
            error="AuthorizationError",
            message=str(exc)
        ).dict()
    )


@app.exception_handler(SecurityException)
async def security_exception_handler(request: Request, exc: SecurityException):
    """Handle security exceptions from LLM Guard."""
    logger.warning(f"Security exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="SecurityError",
            message="Request rejected due to safety concerns",
            detail=None  # Don't expose details to client
        ).dict()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    logger.warning(f"Value error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred"
        ).dict()
    )


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_engine, ingestion_service
    
    logger.info("Starting API Gateway...")
    
    try:
        # Configure Sentry
        configure_sentry(
            service_name="api-gateway",
            environment=os.getenv("ENVIRONMENT", "development"),
            release=os.getenv("RELEASE", "1.0.0")
        )
        
        # Set system info for metrics
        set_system_info(
            service_name="api-gateway",
            version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Initialize services
        logger.info("Initializing services...")
        
        # Create cache client (Requirement 14.4)
        cache_client = CacheClient()
        
        # Create embedding client (HTTP proxy to embedding-service container)
        embedding_service = EmbeddingClient(
            service_url=settings.embedding_service_url,
            cache_client=cache_client
        )
        
        # Create vector search service
        vector_search = VectorSearchService()
        
        # Create LLM client (HTTP proxy to llm-service container)
        llm_service = LLMClient(
            service_url=settings.llm_service_url
        )
        
        # Create LLM Guard service
        llm_guard = LLMGuardService()
        
        # Create conversation memory service
        conversation_memory = ConversationMemoryService(db_manager=get_db_manager())
        
        # Create RAG Query Engine with cache
        rag_engine = RAGQueryEngine(
            embedding_service=embedding_service,
            vector_search=vector_search,
            llm_service=llm_service,
            llm_guard=llm_guard,
            conversation_memory=conversation_memory,
            cache_client=cache_client,
            max_prompt_tokens=settings.max_prompt_tokens,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        # Create text preprocessor
        text_preprocessor = TextPreprocessor()
        
        # Create document ingestion service
        ingestion_service = DocumentIngestionService(
            embedding_service=embedding_service,
            vector_search_service=vector_search,
            text_preprocessor=text_preprocessor
        )
        
        logger.info("API Gateway started successfully")
        
    except Exception as e:
        logger.error("Failed to start API Gateway", error=str(e), exc_info=True)
        capture_exception(e, level="critical", tags={"component": "startup"})
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global ingestion_service
    
    logger.info("Shutting down API Gateway...")
    
    if ingestion_service and ingestion_service.is_watching():
        logger.info("Stopping document watcher...")
        ingestion_service.stop_watching()
    
    logger.info("API Gateway shut down")


# API Endpoints

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and component health.
    """
    return {
        "status": "healthy",
        "service": "api-gateway",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "rag_engine": rag_engine is not None,
            "ingestion_service": ingestion_service is not None
        }
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes metrics for monitoring query latency, throughput, error rates,
    and custom application metrics.
    
    Requirements: 17.2, 17.4
    """
    from fastapi.responses import Response
    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@app.post("/api/query", response_model=QueryResponse, dependencies=[Depends(rate_limit_dependency)])
async def query_endpoint(request: QueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Process user query through RAG pipeline.
    
    This endpoint handles synchronous query processing. For streaming responses,
    use the WebSocket endpoint /ws/query.
    
    Validates Requirement 15.3: Track user_id for conversation access control
    
    Requirements: 11.1, 11.5, 15.1, 15.3
    
    Args:
        request: Query request with query text, optional conversation ID, and top_k
        credentials: Optional JWT credentials for authentication
        
    Returns:
        QueryResponse with answer, sources, and conversation ID
        
    Raises:
        HTTPException: If RAG engine not initialized or query processing fails
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    # Get user ID from credentials (or 'anonymous')
    user_id = get_current_user(credentials) or "anonymous"
    
    start_time = time.time()
    
    with track_api_request('POST', '/api/query') as tracker:
        logger.info("Processing query", query_preview=request.query[:100], user_id=user_id)
        add_breadcrumb(message="Query received", category="query", data={"top_k": request.top_k, "user_id": user_id})
        
        try:
            # If conversation_id is provided, verify user has access
            if request.conversation_id:
                # Get conversation owner
                conversation = rag_engine.conversation_memory.get_conversation(request.conversation_id)
                if conversation:
                    conversation_user_id = conversation.get('user_id', 'anonymous')
                    try:
                        check_conversation_access(user_id, conversation_user_id)
                        log_access_attempt(
                            user_id=user_id,
                            resource_type="conversation",
                            resource_id=request.conversation_id,
                            action="query",
                            allowed=True
                        )
                    except AuthorizationError as e:
                        log_access_attempt(
                            user_id=user_id,
                            resource_type="conversation",
                            resource_id=request.conversation_id,
                            action="query",
                            allowed=False
                        )
                        raise
            
            # Process query through RAG pipeline with user_id
            result = rag_engine.process_query(
                query=request.query,
                conversation_id=request.conversation_id,
                top_k=request.top_k,
                user_id=user_id  # Pass user_id for conversation creation
            )
            
            duration = time.time() - start_time
            record_query_success(duration)
            
            logger.info(
                "Query processed successfully",
                conversation_id=result['conversation_id'],
                duration_ms=duration * 1000,
                sources_count=len(result['sources']),
                user_id=user_id
            )
            
            tracker.set_status(200)
            
            return QueryResponse(
                answer=result['answer'],
                sources=result['sources'],
                conversation_id=result['conversation_id']
            )
            
        except AuthorizationError:
            # Re-raise authorization errors
            tracker.set_status(403)
            raise
        except SecurityException as e:
            # Log security event
            log_security_event(
                event_type="input_rejected",
                severity="warning",
                description="Query rejected by LLM Guard",
                query_preview=request.query[:100],
                user_id=user_id
            )
            record_query_error("security_exception")
            tracker.set_status(400)
            raise
        except ValueError as e:
            record_query_error("validation_error")
            tracker.set_status(400)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error("Error processing query", error=str(e), exc_info=True)
            capture_exception(e, level="error", tags={"endpoint": "query"})
            record_query_error("internal_error")
            tracker.set_status(500)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process query"
            )


@app.websocket("/ws/query")
async def websocket_query_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming query responses.
    
    Client sends query as text message, server streams response chunks.
    Server sends "[DONE]" when streaming is complete.
    
    Requirements: 10.2
    
    Message format:
    - Client -> Server: JSON string with query, conversation_id (optional), top_k (optional)
    - Server -> Client: Text chunks as they are generated
    - Server -> Client: "[DONE]" when complete
    - Server -> Client: "[ERROR]: <message>" on error
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    if rag_engine is None:
        await websocket.send_text("[ERROR]: RAG engine not initialized")
        await websocket.close()
        return
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket query: {data[:100]}...")
            
            try:
                # Parse query data
                import json
                query_data = json.loads(data)
                query = query_data.get('query', '')
                conversation_id = query_data.get('conversation_id')
                top_k = query_data.get('top_k', 5)
                
                if not query or not query.strip():
                    await websocket.send_text("[ERROR]: Query cannot be empty")
                    continue
                
                # Stream response (stream_response is a synchronous generator)
                for chunk in rag_engine.stream_response(
                    query=query,
                    conversation_id=conversation_id,
                    top_k=top_k
                ):
                    await websocket.send_text(chunk)
                
                # Send completion marker
                await websocket.send_text("[DONE]")
                logger.info("WebSocket query completed")
                
            except SecurityException as e:
                logger.warning(f"Security exception in WebSocket: {e}")
                await websocket.send_text("[ERROR]: Request rejected due to safety concerns")
            except ValueError as e:
                logger.warning(f"Validation error in WebSocket: {e}")
                await websocket.send_text(f"[ERROR]: {str(e)}")
            except json.JSONDecodeError:
                await websocket.send_text("[ERROR]: Invalid JSON format")
            except Exception as e:
                logger.error(f"Error processing WebSocket query: {e}", exc_info=True)
                await websocket.send_text("[ERROR]: Failed to process query")
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest = IngestRequest()):
    """
    Trigger manual document ingestion.
    
    Starts batch ingestion of documents from the specified directory.
    Returns immediately with job status.
    
    Requirements: 11.2
    
    Args:
        request: Ingestion request with optional directory path
        
    Returns:
        IngestResponse with status and job information
        
    Raises:
        HTTPException: If ingestion service not initialized or ingestion fails
    """
    if ingestion_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion service not initialized"
        )
    
    # Use configured docs path if not specified
    directory_path = request.directory_path or settings.docs_path
    docs_dir = Path(directory_path)
    
    if not docs_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory not found: {directory_path}"
        )
    
    if not docs_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {directory_path}"
        )
    
    logger.info(f"Starting batch ingestion from: {directory_path}")
    
    try:
        # Start batch ingestion
        result = ingestion_service.batch_ingest(docs_dir)
        
        logger.info(f"Batch ingestion started: {result['total_files']} files found")
        
        return IngestResponse(
            status="started",
            message=f"Batch ingestion started for directory: {directory_path}",
            jobs_queued=result['total_files'],
            details={
                "directory": directory_path,
                "files_found": result.get('total_files', 0),
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start ingestion: {str(e)}"
        )


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Retrieve conversation history.
    
    Returns all messages in the conversation with pagination support.
    Access control ensures users can only access their own conversations.
    
    Validates Requirement 15.3: Users can only access their own conversation history
    
    Requirements: 11.3, 15.3
    
    Args:
        conversation_id: Conversation ID to retrieve
        credentials: Optional JWT credentials for authentication
        
    Returns:
        ConversationResponse with messages and metadata
        
    Raises:
        HTTPException: If conversation not found or access denied
    """
    if rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    # Get user ID from credentials
    user_id = get_current_user(credentials) or "anonymous"
    
    logger.info(f"Retrieving conversation: {conversation_id} for user: {user_id}")
    
    try:
        # Check if conversation exists
        if not rag_engine.conversation_memory.conversation_exists(conversation_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}"
            )
        
        # Get conversation to check ownership
        conversation = rag_engine.conversation_memory.get_conversation(conversation_id)
        conversation_user_id = conversation.get('user_id', 'anonymous')
        
        # Check access control (Requirement 15.3)
        try:
            check_conversation_access(user_id, conversation_user_id)
            log_access_attempt(
                user_id=user_id,
                resource_type="conversation",
                resource_id=conversation_id,
                action="read",
                allowed=True
            )
        except AuthorizationError as e:
            log_access_attempt(
                user_id=user_id,
                resource_type="conversation",
                resource_id=conversation_id,
                action="read",
                allowed=False
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(e)
            )
        
        # Retrieve conversation history
        messages = rag_engine.conversation_memory.get_recent_messages(
            conversation_id=conversation_id,
            limit=100  # Get up to 100 messages
        )
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(ConversationMessage(
                role=msg['role'],
                content=msg['content'],
                timestamp=msg['timestamp'],
                sources=msg.get('sources')
            ))
        
        logger.info(f"Retrieved {len(formatted_messages)} messages for conversation {conversation_id}")
        
        return ConversationResponse(
            conversation_id=conversation_id,
            messages=formatted_messages,
            total_messages=len(formatted_messages)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )


# Additional utility endpoints

@app.get("/api/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "rate_limits": {
            "active_clients": len(rate_limit_storage),
            "anonymous_limit": settings.rate_limit_anonymous,
            "authenticated_limit": settings.rate_limit_authenticated
        },
        "services": {
            "rag_engine_initialized": rag_engine is not None,
            "ingestion_service_initialized": ingestion_service is not None,
            "ingestion_watching": ingestion_service.is_watching() if ingestion_service else False
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # TLS configuration (Requirement 15.2)
    settings = get_settings()
    
    if settings.enable_tls:
        logger.info("Starting API Gateway with TLS enabled")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            ssl_keyfile=settings.ssl_keyfile,
            ssl_certfile=settings.ssl_certfile,
            ssl_version=3,  # TLS 1.3
            ssl_cert_reqs=0,  # No client certificate required
        )
    else:
        logger.info("Starting API Gateway without TLS (development mode)")
        uvicorn.run(app, host="0.0.0.0", port=8080)
