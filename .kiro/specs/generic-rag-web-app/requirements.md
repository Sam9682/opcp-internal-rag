# Requirements Document: Generic RAG Web Application

## Introduction

This document specifies the functional and non-functional requirements for a sovereign, self-hosted Retrieval-Augmented Generation (RAG) web application. The system enables users to query markdown documentation through an AI-powered chatbot interface called "L'Oracle". The application ingests documentation, vectorizes content using embedding models, stores vectors in PostgreSQL with pgvector, and provides contextually-aware responses using large language models with safety guardrails.

The system is designed for deployment on sovereign infrastructure, ensuring data privacy and control while providing intelligent documentation search and question-answering capabilities.

## Glossary

- **System**: The complete RAG web application including all microservices
- **Document_Ingestion_Service**: Component that monitors and processes markdown files
- **Text_Preprocessor**: Component that cleans and chunks markdown content
- **Embedding_Service**: Component that generates vector embeddings using BGE-M3
- **Vector_Search_Service**: Component that performs similarity search in pgvector
- **RAG_Query_Engine**: Component that orchestrates the query processing pipeline
- **LLM_Guard**: Component that validates input/output safety
- **LLM_Service**: Component that generates natural language responses
- **Conversation_Memory**: Component that manages conversation history
- **API_Gateway**: REST API backend for frontend communication
- **Web_UI**: L'Oracle chat interface for user interaction
- **Text_Chunk**: Segment of document text with associated embedding vector
- **Embedding_Vector**: Dense numerical representation of text (1024 dimensions for BGE-M3)
- **Conversation**: Session containing user-assistant message exchanges
- **Source**: Reference to document chunk used to generate response
- **Similarity_Score**: Cosine similarity value between 0.0 and 1.0
- **Safety_Check**: Validation process for detecting harmful content or attacks

## Requirements

### Requirement 1: Document Ingestion

**User Story:** As a system administrator, I want the system to automatically ingest markdown documentation, so that users can query the latest content without manual intervention.

#### Acceptance Criteria

1. WHEN a markdown file is added to the documentation directory, THEN THE Document_Ingestion_Service SHALL detect the new file within 10 seconds
2. WHEN a markdown file is modified, THEN THE Document_Ingestion_Service SHALL re-process the file and update existing chunks
3. WHEN a markdown file is deleted, THEN THE Document_Ingestion_Service SHALL remove all associated chunks from the database
4. WHEN the system starts, THEN THE Document_Ingestion_Service SHALL perform batch ingestion of all existing markdown files
5. WHEN an ingestion job fails, THEN THE Document_Ingestion_Service SHALL log the error and retry with exponential backoff

### Requirement 2: Text Preprocessing

**User Story:** As a developer, I want markdown content to be cleaned and chunked appropriately, so that embeddings capture semantic meaning effectively.

#### Acceptance Criteria

1. WHEN markdown content is processed, THEN THE Text_Preprocessor SHALL remove formatting while preserving semantic structure
2. WHEN text is chunked, THEN THE Text_Preprocessor SHALL create chunks of maximum 512 tokens with 50 token overlap
3. WHEN a document is processed, THEN THE Text_Preprocessor SHALL extract metadata including title, headers, and tags
4. WHEN code blocks are encountered, THEN THE Text_Preprocessor SHALL preserve them with language tags
5. WHEN chunking is complete, THEN THE Text_Preprocessor SHALL ensure all document content is covered without loss

### Requirement 3: Vector Embedding Generation

**User Story:** As a system operator, I want text to be converted into vector embeddings, so that semantic similarity search can be performed.

#### Acceptance Criteria

1. WHEN text is provided for embedding, THEN THE Embedding_Service SHALL generate a vector of dimension 1024 using BGE-M3 model
2. WHEN the same text is embedded multiple times, THEN THE Embedding_Service SHALL produce identical vectors (deterministic output)
3. WHEN embeddings are generated, THEN THE Embedding_Service SHALL normalize vectors for cosine similarity computation
4. WHEN multiple texts are provided, THEN THE Embedding_Service SHALL process them in batches for efficiency
5. WHEN an embedding is generated, THEN THE Embedding_Service SHALL ensure no NaN or Inf values are present

### Requirement 4: Vector Storage and Indexing

**User Story:** As a system architect, I want embeddings stored efficiently in PostgreSQL with pgvector, so that fast similarity search is possible.

#### Acceptance Criteria

1. WHEN a text chunk with embedding is stored, THEN THE Vector_Search_Service SHALL save both text and vector in PostgreSQL
2. WHEN chunks are stored, THEN THE Vector_Search_Service SHALL maintain HNSW index for fast similarity search
3. WHEN a document is re-ingested, THEN THE Vector_Search_Service SHALL replace old chunks with new ones without duplication
4. WHEN storage operations fail, THEN THE Vector_Search_Service SHALL rollback the transaction to maintain consistency
5. WHEN chunks are stored, THEN THE Vector_Search_Service SHALL associate metadata including document ID and chunk index

### Requirement 5: Vector Similarity Search

**User Story:** As a user, I want the system to find relevant documentation chunks for my query, so that I receive accurate contextual information.

#### Acceptance Criteria

1. WHEN a query vector is provided, THEN THE Vector_Search_Service SHALL return top-k most similar chunks sorted by similarity score
2. WHEN a similarity threshold is specified, THEN THE Vector_Search_Service SHALL return only chunks with similarity greater than or equal to the threshold
3. WHEN search is performed, THEN THE Vector_Search_Service SHALL use cosine similarity for distance calculation
4. WHEN no chunks meet the threshold, THEN THE Vector_Search_Service SHALL return an empty result set
5. WHEN search completes, THEN THE Vector_Search_Service SHALL include similarity scores with each result

### Requirement 6: RAG Query Processing

**User Story:** As a user, I want to ask questions about documentation and receive accurate answers with source citations, so that I can quickly find information.

#### Acceptance Criteria

1. WHEN a user submits a query, THEN THE RAG_Query_Engine SHALL embed the query and retrieve relevant context chunks
2. WHEN context is retrieved, THEN THE RAG_Query_Engine SHALL build a prompt including query, context, and conversation history
3. WHEN a prompt is built, THEN THE RAG_Query_Engine SHALL ensure total token count does not exceed MAX_PROMPT_TOKENS
4. WHEN an LLM response is generated, THEN THE RAG_Query_Engine SHALL return the answer with source citations
5. WHEN query processing completes, THEN THE RAG_Query_Engine SHALL store the interaction in conversation history

### Requirement 7: Input Safety Validation

**User Story:** As a security administrator, I want all user inputs validated for safety, so that prompt injection and malicious content are blocked.

#### Acceptance Criteria

1. WHEN a user query is received, THEN THE LLM_Guard SHALL check for prompt injection patterns before processing
2. WHEN toxic or harmful content is detected in input, THEN THE LLM_Guard SHALL reject the query and return an error
3. WHEN personally identifiable information is detected in input, THEN THE LLM_Guard SHALL sanitize or reject the query
4. WHEN input passes safety checks, THEN THE LLM_Guard SHALL allow the query to proceed to the RAG pipeline
5. WHEN input is rejected, THEN THE LLM_Guard SHALL log the security event with details

### Requirement 8: Output Safety Validation

**User Story:** As a security administrator, I want all LLM outputs validated for safety, so that harmful or inappropriate responses are not shown to users.

#### Acceptance Criteria

1. WHEN an LLM generates a response, THEN THE LLM_Guard SHALL validate the output before returning to the user
2. WHEN harmful content is detected in output, THEN THE LLM_Guard SHALL replace it with a generic safe message
3. WHEN system prompts or internal information leak in output, THEN THE LLM_Guard SHALL block the response
4. WHEN output passes safety checks, THEN THE LLM_Guard SHALL allow the response to be returned to the user
5. WHEN output is blocked, THEN THE LLM_Guard SHALL log the security event and return sources without the answer

### Requirement 9: Conversation Memory Management

**User Story:** As a user, I want the system to remember our conversation context, so that I can ask follow-up questions naturally.

#### Acceptance Criteria

1. WHEN a new query is submitted without a conversation ID, THEN THE Conversation_Memory SHALL create a new conversation session
2. WHEN messages are added to a conversation, THEN THE Conversation_Memory SHALL store them with timestamps in chronological order
3. WHEN conversation history is retrieved, THEN THE Conversation_Memory SHALL return messages in chronological order with pagination support
4. WHEN a conversation exceeds the configured retention period, THEN THE Conversation_Memory SHALL expire and delete the conversation
5. WHEN a conversation becomes too long, THEN THE Conversation_Memory SHALL summarize older messages to compress context

### Requirement 10: LLM Response Generation

**User Story:** As a user, I want natural language responses generated from my queries and relevant context, so that I receive helpful answers.

#### Acceptance Criteria

1. WHEN a prompt is provided, THEN THE LLM_Service SHALL generate a response using the configured language model
2. WHEN generating responses, THEN THE LLM_Service SHALL support streaming output for real-time display
3. WHEN response generation is requested, THEN THE LLM_Service SHALL respect temperature and max_tokens parameters
4. WHEN the model is first used, THEN THE LLM_Service SHALL load and cache the model in memory for subsequent requests
5. WHERE GPU acceleration is available, THE LLM_Service SHALL use CUDA for faster inference

### Requirement 11: API Gateway Functionality

**User Story:** As a frontend developer, I want a REST API to interact with the RAG system, so that I can build user interfaces.

#### Acceptance Criteria

1. WHEN a POST request is sent to /api/query, THEN THE API_Gateway SHALL process the query and return a response with answer and sources
2. WHEN a POST request is sent to /api/ingest, THEN THE API_Gateway SHALL trigger document ingestion and return job status
3. WHEN a GET request is sent to /api/conversations/{id}, THEN THE API_Gateway SHALL return the conversation history
4. WHEN authentication is required, THEN THE API_Gateway SHALL validate JWT tokens before processing requests
5. WHEN an error occurs, THEN THE API_Gateway SHALL return appropriate HTTP status codes and error messages

### Requirement 12: Web User Interface

**User Story:** As a user, I want a clean chat interface to interact with L'Oracle, so that I can easily query documentation.

#### Acceptance Criteria

1. WHEN the user opens the application, THEN THE Web_UI SHALL display a chat interface with message history
2. WHEN the user submits a query, THEN THE Web_UI SHALL send the request to the API and display the streaming response
3. WHEN a response includes sources, THEN THE Web_UI SHALL display source citations with titles and similarity scores
4. WHEN the user wants to start fresh, THEN THE Web_UI SHALL provide a button to clear the conversation
5. WHEN the interface is accessed from mobile devices, THEN THE Web_UI SHALL provide a responsive design

### Requirement 13: Error Handling and Recovery

**User Story:** As a system operator, I want robust error handling, so that failures are managed gracefully without data loss.

#### Acceptance Criteria

1. WHEN a service becomes unavailable, THEN THE System SHALL use circuit breaker patterns to prevent cascading failures
2. WHEN a database transaction fails, THEN THE System SHALL rollback changes to maintain data consistency
3. WHEN an ingestion job fails, THEN THE System SHALL log the error and queue the job for retry with exponential backoff
4. WHEN the embedding service is unavailable, THEN THE System SHALL queue requests and process them when service recovers
5. WHEN concurrent document updates occur, THEN THE System SHALL detect conflicts and re-ingest with the latest version

### Requirement 14: Performance and Scalability

**User Story:** As a system operator, I want the system to handle multiple concurrent users efficiently, so that response times remain acceptable under load.

#### Acceptance Criteria

1. WHEN vector search is performed on 100K chunks, THEN THE System SHALL return results within 50 milliseconds
2. WHEN embeddings are generated with GPU acceleration, THEN THE System SHALL process 100-500 chunks per second
3. WHEN multiple queries arrive concurrently, THEN THE System SHALL handle at least 100 concurrent requests
4. WHEN the same query is repeated, THEN THE System SHALL use caching to reduce response time by at least 50%
5. WHEN the database grows beyond 1M chunks, THEN THE System SHALL maintain query performance through proper indexing

### Requirement 15: Security and Privacy

**User Story:** As a security administrator, I want comprehensive security controls, so that data and users are protected from threats.

#### Acceptance Criteria

1. WHEN API requests are made, THEN THE System SHALL enforce rate limiting of 100 queries per hour for anonymous users
2. WHEN data is transmitted, THEN THE System SHALL use TLS 1.3 encryption for all network communication
3. WHEN users access conversations, THEN THE System SHALL ensure users can only access their own conversation history
4. WHEN sensitive data is stored, THEN THE System SHALL encrypt it at rest in the database
5. WHEN security events occur, THEN THE System SHALL log them with timestamps and relevant details for audit purposes

### Requirement 16: Deployment and Configuration

**User Story:** As a system administrator, I want easy deployment using Docker Compose, so that I can run the system on sovereign infrastructure.

#### Acceptance Criteria

1. WHEN docker-compose up is executed, THEN THE System SHALL start all required services in the correct order
2. WHEN services start, THEN THE System SHALL wait for PostgreSQL health checks before initializing dependent services
3. WHEN configuration is needed, THEN THE System SHALL read settings from environment variables or .env files
4. WHEN the system is deployed, THEN THE System SHALL support both CPU-only and GPU-accelerated configurations
5. WHEN services communicate, THEN THE System SHALL use internal Docker networks for isolation

### Requirement 17: Monitoring and Observability

**User Story:** As a system operator, I want comprehensive logging and metrics, so that I can monitor system health and troubleshoot issues.

#### Acceptance Criteria

1. WHEN operations are performed, THEN THE System SHALL log events using structured logging with timestamps and context
2. WHEN metrics are collected, THEN THE System SHALL expose Prometheus-compatible metrics endpoints
3. WHEN errors occur, THEN THE System SHALL send error reports to the configured error tracking service
4. WHEN queries are processed, THEN THE System SHALL track latency, throughput, and error rates
5. WHEN security events occur, THEN THE System SHALL log them separately for security audit purposes

### Requirement 18: Data Integrity and Consistency

**User Story:** As a data administrator, I want guarantees of data integrity, so that information remains accurate and consistent.

#### Acceptance Criteria

1. WHEN document ingestion completes, THEN THE System SHALL ensure all chunks are stored or none are (atomic transactions)
2. WHEN embeddings are stored, THEN THE System SHALL ensure the embedding dimension matches the configured model dimension
3. WHEN conversations are stored, THEN THE System SHALL ensure messages maintain chronological order with monotonically increasing timestamps
4. WHEN documents are re-ingested, THEN THE System SHALL ensure no duplicate chunks exist for the same document
5. WHEN database operations fail, THEN THE System SHALL maintain referential integrity between documents, chunks, and conversations

### Requirement 19: Prompt Construction

**User Story:** As a developer, I want prompts built correctly with context and history, so that LLM responses are relevant and accurate.

#### Acceptance Criteria

1. WHEN a prompt is constructed, THEN THE System SHALL include system instructions, retrieved context chunks, conversation history, and the user query
2. WHEN context chunks are added, THEN THE System SHALL prioritize chunks with highest similarity scores
3. WHEN conversation history is included, THEN THE System SHALL include the most recent messages up to the token limit
4. WHEN the token limit is approached, THEN THE System SHALL truncate context or history to stay within MAX_PROMPT_TOKENS
5. WHEN truncation occurs, THEN THE System SHALL log the event for monitoring purposes

### Requirement 20: Embedding Model Management

**User Story:** As a system operator, I want efficient embedding model management, so that resources are used optimally.

#### Acceptance Criteria

1. WHEN the embedding service starts, THEN THE System SHALL load the BGE-M3 model into memory once and cache it
2. WHEN GPU is available, THEN THE System SHALL load the embedding model onto GPU for accelerated inference
3. WHEN multiple embedding requests arrive, THEN THE System SHALL batch them for efficient processing
4. WHEN the model cache directory exists, THEN THE System SHALL reuse cached model files to avoid re-downloading
5. WHEN embedding requests exceed memory capacity, THEN THE System SHALL process them in manageable batches

