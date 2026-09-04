# Implementation Plan: Generic RAG Web Application

## Overview

This plan implements a sovereign, self-hosted RAG web application with Docker Compose microservices architecture. The system includes PostgreSQL with pgvector for vector storage, document ingestion pipeline with BGE-M3 embeddings, RAG query engine with LLM Guard safety checks, L'Oracle chat interface, and conversation memory management.

Implementation follows a bottom-up approach: infrastructure and data layer first, then core services, then orchestration, and finally user-facing components.

## Tasks

- [x] 1. Set up project structure and infrastructure foundation
  - Create directory structure for microservices architecture
  - Set up Docker Compose configuration with PostgreSQL + pgvector
  - Create database initialization scripts with schema and indexes
  - Configure environment variables and secrets management
  - Set up shared Python package for common utilities and models
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_

- [ ]* 1.1 Write property test for database schema validation
  - **Property 7: Transaction Atomicity**
  - **Validates: Requirements 4.4, 13.2, 18.1, 18.5**

- [x] 2. Implement data models and database layer
  - [x] 2.1 Create SQLAlchemy models for Document, TextChunk, Conversation, Message, IngestionJob
    - Define all models with proper types, constraints, and relationships
    - Add pgvector column type for embeddings
    - Implement validation methods for each model
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

  - [ ]* 2.2 Write property test for data model validation
    - **Property 9: Chunk Storage Round-Trip**
    - **Validates: Requirements 4.1, 4.5**

  - [x] 2.3 Implement database connection management and session handling
    - Create connection pool configuration
    - Implement context managers for transactions
    - Add health check functions
    - _Requirements: 16.2_

  - [ ]* 2.4 Write unit tests for database operations
    - Test CRUD operations for all models
    - Test transaction rollback scenarios
    - Test concurrent access handling
    - _Requirements: 13.2, 18.5_

- [x] 3. Implement Text Preprocessor component
  - [x] 3.1 Create TextPreprocessor class with markdown cleaning functionality
    - Implement clean_markdown() to remove formatting while preserving structure
    - Handle code blocks, headers, links, and special characters
    - Normalize whitespace and extract semantic content
    - _Requirements: 2.1, 2.3_

  - [x] 3.2 Implement text chunking with overlap
    - Create chunk_text() with configurable chunk_size and overlap
    - Use tiktoken for accurate token counting
    - Ensure no text loss and proper boundary handling
    - _Requirements: 2.2, 2.5_

  - [x] 3.3 Write property test for chunking coverage
    - **Property 4: Document Coverage**
    - **Validates: Requirements 2.2, 2.5**

  - [x] 3.4 Implement metadata extraction from markdown
    - Extract title from H1 or filename
    - Parse headers, tags, and document structure
    - Create extract_metadata() function
    - _Requirements: 2.3_

  - [ ]* 3.5 Write unit tests for text preprocessing
    - Test markdown cleaning with various formats
    - Test edge cases (empty docs, single sentence, very long docs)
    - Test metadata extraction accuracy
    - _Requirements: 2.1, 2.3, 2.4_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Embedding Service component
  - [x] 5.1 Create EmbeddingService class with BGE-M3 model loading
  - [x] 5.1 Create EmbeddingService class with BGE-M3 model loading
    - Initialize model with caching and device selection (CPU/GPU)
    - Implement model loading with error handling
    - Add configuration for model parameters
    - _Requirements: 20.1, 20.2_

  - [x] 5.2 Implement embed_text() for single text embedding
    - Generate embeddings with proper normalization
    - Validate output dimensions and values
    - Handle edge cases (empty text, very long text)
    - _Requirements: 3.1, 3.3, 3.5_

  - [x] 5.3 Implement embed_batch() for efficient batch processing
    - Process multiple texts in single forward pass
    - Optimize batch size for memory constraints
    - _Requirements: 3.4, 20.3_

  - [ ]* 5.4 Write property test for embedding consistency
    - **Property 1: Embedding Consistency**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5, 18.2**

  - [ ]* 5.5 Write property test for batch consistency
    - **Property 20: Embedding Batch Consistency**
    - **Validates: Requirements 20.3**

  - [ ]* 5.6 Write unit tests for embedding service
    - Test deterministic output for same input
    - Test correct embedding dimensions
    - Test error handling for invalid inputs
    - _Requirements: 3.1, 3.2, 3.5_

- [x] 6. Implement Vector Search Service component
  - [x] 6.1 Create VectorSearchService class with pgvector integration
    - Initialize database connection and query builders
    - Implement connection to PostgreSQL with pgvector
    - _Requirements: 4.1, 4.2_

  - [x] 6.2 Implement store_embedding() for storing chunks with vectors
    - Store text, embedding, and metadata in single transaction
    - Handle HNSW index updates
    - Implement upsert logic for re-ingestion
    - _Requirements: 4.1, 4.3, 4.5_

  - [x] 6.3 Implement search_similar() for vector similarity search
    - Use pgvector cosine distance operator
    - Apply similarity threshold filtering
    - Return top-k results sorted by similarity
    - _Requirements: 5.1, 5.2, 5.3, 5.5_

  - [ ]* 6.4 Write property test for vector search correctness
    - **Property 2: Vector Search Correctness**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5**

  - [ ]* 6.5 Write unit tests for vector search
    - Test similarity threshold filtering
    - Test top-k result limiting
    - Test empty result handling
    - _Requirements: 5.1, 5.2, 5.4_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Document Ingestion Service component
  - [x] 8.1 Create DocumentIngestionService class with file watching
    - Implement watch_directory() using watchdog library
    - Handle file system events (create, modify, delete)
    - Set up event queue and processing loop
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 8.2 Implement ingest_document() pipeline orchestration
    - Coordinate TextPreprocessor, EmbeddingService, and VectorSearchService
    - Create IngestionJob tracking
    - Implement transaction management for atomic operations
    - Handle errors with logging and status updates
    - _Requirements: 1.4, 1.5, 18.1_

  - [x] 8.3 Implement batch_ingest() for startup processing
    - Scan directory for existing markdown files
    - Process multiple documents with progress tracking
    - _Requirements: 1.4_

  - [x] 8.4 Implement retry logic with exponential backoff
    - Add retry mechanism for failed ingestion jobs
    - Implement exponential backoff strategy
    - _Requirements: 1.5, 13.3_

  - [ ]* 8.5 Write property test for idempotent ingestion
    - **Property 6: Idempotent Ingestion**
    - **Validates: Requirements 1.2, 4.3, 18.4**

  - [ ]* 8.6 Write property test for document deletion cleanup
    - **Property 10: Document Deletion Cleanup**
    - **Validates: Requirements 1.3**

  - [ ]* 8.7 Write integration tests for ingestion pipeline
    - Test complete ingestion flow from file to database
    - Test re-ingestion and update scenarios
    - Test deletion handling
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 9. Implement LLM Guard Service component
  - [x] 9.1 Create LLMGuardService class with safety check initialization
    - Initialize llm-guard library with safety scanners
    - Configure prompt injection, toxicity, and PII detectors
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 9.2 Implement check_input() for input validation
    - Check for prompt injection patterns
    - Detect toxic or harmful content
    - Identify and sanitize PII
    - Return safety status and reason
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.3 Implement check_output() for output validation
    - Validate LLM responses for harmful content
    - Detect system prompt leakage
    - Check for inappropriate responses
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 9.4 Write property test for safety guarantees
    - **Property 5: Safety Guarantees**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.4, 8.5**

  - [ ]* 9.5 Write unit tests for LLM Guard
    - Test known attack pattern detection
    - Test toxic content filtering
    - Test PII detection and masking
    - _Requirements: 7.1, 7.2, 7.3, 8.1_

- [x] 10. Implement Conversation Memory Service component
  - [x] 10.1 Create ConversationMemoryService class with database integration
    - Initialize with database session management
    - Implement conversation CRUD operations
    - _Requirements: 9.1, 9.2_

  - [x] 10.2 Implement create_conversation() and add_message()
    - Create new conversation sessions with unique IDs
    - Store messages with timestamps and metadata
    - Maintain chronological order
    - _Requirements: 9.1, 9.2_

  - [x] 10.3 Implement get_history() with pagination
    - Retrieve conversation messages in chronological order
    - Support pagination for long conversations
    - _Requirements: 9.3_

  - [x] 10.4 Implement conversation expiration and summarization
    - Add expiration logic based on retention period
    - Implement summarization for long conversations
    - _Requirements: 9.4, 9.5_

  - [ ]* 10.5 Write property test for conversation integrity
    - **Property 3: Conversation Integrity**
    - **Validates: Requirements 9.2, 9.3, 18.3**

  - [ ]* 10.6 Write unit tests for conversation memory
    - Test message ordering and retrieval
    - Test conversation expiration
    - Test concurrent access handling
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement LLM Service component
  - [x] 12.1 Create LLMService class with model loading
    - Initialize LLM model (Mistral-7B-Instruct or configured model)
    - Support both local models and API-based models
    - Handle GPU/CPU device selection
    - Implement model caching
    - _Requirements: 10.4, 10.5_

  - [x] 12.2 Implement generate() for response generation
    - Generate responses from prompts with configurable parameters
    - Support temperature, max_tokens, and other parameters
    - Handle errors and timeouts
    - _Requirements: 10.1, 10.3_

  - [x] 12.3 Implement generate_stream() for streaming responses
    - Stream response tokens in real-time
    - Use async generators for efficient streaming
    - _Requirements: 10.2_

  - [ ]* 12.4 Write property test for LLM response generation
    - **Property 14: LLM Response Generation**
    - **Validates: Requirements 10.1, 10.2**

  - [ ]* 12.5 Write unit tests for LLM service
    - Test response generation with various prompts
    - Test streaming consistency
    - Test parameter handling
    - _Requirements: 10.1, 10.2, 10.3_

- [x] 13. Implement RAG Query Engine component
  - [x] 13.1 Create RAGQueryEngine class with service orchestration
    - Initialize with all required services (Embedding, VectorSearch, LLM, LLMGuard, ConversationMemory)
    - Set up configuration for top_k, thresholds, token limits
    - _Requirements: 6.1, 6.2_

  - [x] 13.2 Implement build_prompt() for prompt construction
    - Build prompts with system instructions, context, history, and query
    - Prioritize chunks by similarity score
    - Include recent conversation history
    - Implement token counting and truncation logic
    - _Requirements: 6.3, 19.1, 19.2, 19.3, 19.4, 19.5_

  - [ ]* 13.3 Write property test for prompt token limit
    - **Property 8: Prompt Token Limit**
    - **Validates: Requirements 6.3, 19.1, 19.4**

  - [ ]* 13.4 Write property test for context prioritization
    - **Property 18: Context Prioritization**
    - **Validates: Requirements 19.2, 19.3**

  - [x] 13.5 Implement process_query() for complete RAG pipeline
    - Embed user query
    - Perform vector similarity search
    - Retrieve conversation history
    - Build prompt with context
    - Generate LLM response
    - Apply input and output safety checks
    - Store interaction in conversation history
    - Return structured response with sources
    - _Requirements: 6.1, 6.2, 6.4, 6.5_

  - [ ]* 13.6 Write property test for query processing completeness
    - **Property 12: Query Processing Completeness**
    - **Validates: Requirements 6.1, 6.2, 6.4, 6.5**

  - [x] 13.7 Implement stream_response() for streaming queries
    - Stream LLM responses in real-time
    - Handle errors during streaming
    - _Requirements: 10.2_

  - [ ]* 13.8 Write integration tests for RAG query engine
    - Test end-to-end query processing
    - Test conversation context handling
    - Test safety check integration
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 14. Implement API Gateway/Backend service
  - [x] 14.1 Create FastAPI application with endpoint structure
    - Set up FastAPI app with CORS, middleware
    - Define Pydantic request/response models
    - Configure error handlers
    - _Requirements: 11.1, 11.5_

  - [x] 14.2 Implement POST /api/query endpoint
    - Handle query requests with conversation context
    - Call RAGQueryEngine.process_query()
    - Return structured response with answer and sources
    - Handle errors with appropriate HTTP status codes
    - _Requirements: 11.1, 11.5_

  - [x] 14.3 Implement WebSocket /ws/query endpoint for streaming
    - Set up WebSocket connection handling
    - Stream responses from RAGQueryEngine
    - Handle connection errors and cleanup
    - _Requirements: 10.2_

  - [x] 14.4 Implement POST /api/ingest endpoint
    - Trigger document ingestion manually
    - Return job status and tracking information
    - _Requirements: 11.2_

  - [x] 14.5 Implement GET /api/conversations/{id} endpoint
    - Retrieve conversation history
    - Apply access control checks
    - _Requirements: 11.3, 15.3_

  - [x] 14.6 Implement authentication and rate limiting middleware
    - Add JWT token validation
    - Implement rate limiting (100 queries/hour for anonymous)
    - _Requirements: 11.4, 15.1_

  - [ ]* 14.7 Write unit tests for API endpoints
    - Test all endpoints with valid and invalid inputs
    - Test error handling and status codes
    - Test authentication and authorization
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 15. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement monitoring and observability
  - [x] 16.1 Add structured logging with structlog
    - Configure structured logging for all services
    - Add context and timestamps to all log entries
    - Separate security event logging
    - _Requirements: 17.1, 17.5_

  - [ ]* 16.2 Write property test for structured logging
    - **Property 16: Structured Logging**
    - **Validates: Requirements 17.1, 17.5**

  - [x] 16.3 Add Prometheus metrics endpoints
    - Expose metrics for query latency, throughput, error rates
    - Add custom metrics for embedding generation, vector search
    - _Requirements: 17.2, 17.4_

  - [ ]* 16.4 Write property test for metrics tracking
    - **Property 17: Metrics Tracking**
    - **Validates: Requirements 17.4**

  - [x] 16.5 Integrate Sentry for error tracking
    - Configure Sentry SDK for all services
    - Add error context and user information
    - _Requirements: 17.3_

  - [ ]* 16.6 Write unit tests for monitoring integration
    - Test logging output format
    - Test metrics collection
    - Test error reporting
    - _Requirements: 17.1, 17.2, 17.3_

- [x] 17. Implement security features
  - [x] 17.1 Add TLS configuration for all services
    - Configure TLS 1.3 for API gateway
    - Set up certificate management
    - _Requirements: 15.2_

  - [x] 17.2 Implement access control for conversations
    - Add user ID tracking to conversations
    - Implement authorization checks in API endpoints
    - _Requirements: 15.3_

  - [ ]* 17.3 Write property test for access control isolation
    - **Property 15: Access Control Isolation**
    - **Validates: Requirements 15.3**

  - [x] 17.4 Add database encryption at rest
    - Configure PostgreSQL encryption
    - Encrypt sensitive fields
    - _Requirements: 15.4_

  - [ ]* 17.5 Write unit tests for security features
    - Test TLS configuration
    - Test access control enforcement
    - Test rate limiting
    - _Requirements: 15.1, 15.2, 15.3_

- [x] 18. Implement Web UI (L'Oracle Interface)
  - [x] 18.1 Set up React + TypeScript project with Vite
    - Initialize project with Vite
    - Configure TypeScript and Tailwind CSS
    - Set up project structure
    - _Requirements: 12.1_

  - [x] 18.2 Create chat interface components
    - Build MessageList component for displaying conversation
    - Build MessageInput component for user queries
    - Build SourceCard component for displaying citations
    - _Requirements: 12.1, 12.3_

  - [x] 18.3 Implement API client with streaming support
    - Create axios client for REST API calls
    - Implement WebSocket client for streaming responses
    - Handle connection errors and retries
    - _Requirements: 12.2_

  - [x] 18.4 Implement conversation management UI
    - Add "New Conversation" button
    - Add "Clear Conversation" functionality
    - Display conversation history
    - _Requirements: 12.4_

  - [x] 18.5 Add responsive design for mobile and desktop
    - Implement responsive layouts with Tailwind
    - Test on various screen sizes
    - _Requirements: 12.5_

  - [ ]* 18.6 Write unit tests for UI components
    - Test component rendering
    - Test user interactions
    - Test API integration
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [x] 19. Implement caching layer
  - [x] 19.1 Add Redis for embedding and response caching
    - Set up Redis container in Docker Compose
    - Implement cache client wrapper
    - _Requirements: 14.4_

  - [x] 19.2 Implement embedding cache in EmbeddingService
    - Cache embeddings with text hash as key
    - Set appropriate TTL
    - _Requirements: 14.4_

  - [x] 19.3 Implement response cache in RAGQueryEngine
    - Cache LLM responses for identical queries
    - Set TTL to 1 hour
    - _Requirements: 14.4_

  - [ ]* 19.4 Write property test for cache correctness
    - **Property 21: Cache Correctness**
    - **Validates: Requirements 14.4**

  - [ ]* 19.5 Write unit tests for caching
    - Test cache hit and miss scenarios
    - Test TTL expiration
    - Test cache invalidation
    - _Requirements: 14.4_

- [x] 20. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 21. Create Docker Compose orchestration
  - [x] 21.1 Complete docker-compose.yml with all services
    - Define all service containers (postgres, embedding-service, ingestion-service, api-backend, llm-service, web-ui, redis)
    - Configure service dependencies and health checks
    - Set up internal networks for isolation
    - Configure volumes for persistence
    - _Requirements: 16.1, 16.2, 16.5_

  - [x] 21.2 Create Dockerfiles for each service
    - Write Dockerfile for embedding-service
    - Write Dockerfile for ingestion-service
    - Write Dockerfile for api-backend
    - Write Dockerfile for llm-service
    - Write Dockerfile for web-ui
    - _Requirements: 16.1_

  - [x] 21.3 Create environment configuration files
    - Create .env.example with all required variables
    - Document configuration options
    - _Requirements: 16.3_

  - [x] 21.4 Create database initialization scripts
    - Write SQL schema creation script
    - Create pgvector extension setup
    - Add HNSW index creation
    - _Requirements: 4.2, 16.2_

  - [ ]* 21.5 Write integration tests for Docker Compose deployment
    - Test service startup order
    - Test health checks
    - Test inter-service communication
    - _Requirements: 16.1, 16.2, 16.5_

- [x] 22. Create documentation and deployment guides
  - [x] 22.1 Write README.md with setup instructions
    - Document system requirements
    - Provide quick start guide
    - Explain configuration options
    - _Requirements: 16.1, 16.3, 16.4_

  - [x] 22.2 Create API documentation
    - Document all REST API endpoints
    - Provide request/response examples
    - Document WebSocket protocol
    - _Requirements: 11.1, 11.2, 11.3_

  - [x] 22.3 Write deployment guide
    - Document deployment steps
    - Explain GPU vs CPU configuration
    - Provide troubleshooting tips
    - _Requirements: 16.1, 16.4_

- [x] 23. Final integration testing and validation
  - [ ]* 23.1 Run end-to-end integration tests
    - Test complete document ingestion to query flow
    - Test conversation context across multiple queries
    - Test safety checks with malicious inputs
    - Test performance under load
    - _Requirements: All requirements_

  - [x] 23.2 Validate all correctness properties
    - Verify all 21 correctness properties hold
    - Run property-based tests with large input ranges
    - _Requirements: All requirements_

  - [x] 23.3 Performance testing and optimization
    - Measure query latency and throughput
    - Verify vector search performance targets
    - Test with 100K+ chunks
    - _Requirements: 14.1, 14.2, 14.3, 14.5_

- [x] 24. Final checkpoint - System ready for deployment
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout implementation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- Implementation uses Python for backend services and TypeScript/React for frontend
- Docker Compose orchestrates all microservices
- System supports both CPU-only and GPU-accelerated configurations
