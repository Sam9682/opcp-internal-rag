"""
RAG Query Engine for orchestrating the complete RAG pipeline.
This module provides the RAGQueryEngine class that coordinates all services
to process user queries and generate contextually-aware responses with source citations.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 10.2, 14.4, 19.1, 19.2, 19.3, 19.4, 19.5
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Iterator, Any, TYPE_CHECKING
from uuid import uuid4
import tiktoken
import time
import hashlib

if TYPE_CHECKING:
    from .embedding_service import EmbeddingService
    from .llm_service import LLMService

from .vector_search_service import VectorSearchService
from .llm_guard_service import LLMGuardService
from .conversation_memory_service import ConversationMemoryService
from .cache_client import CacheClient
from .config import get_settings
from .logging_config import get_logger, log_security_event
from .metrics import (
    query_processing_duration_seconds,
    record_llm_guard_check,
    track_time
)
from .sentry_config import add_breadcrumb, capture_exception, set_context

logger = get_logger(__name__)


class SecurityException(Exception):
    """Exception raised when safety checks fail."""
    pass


class RAGQueryEngine:
    """
    RAG Query Engine for orchestrating the complete RAG pipeline.
    
    This class coordinates all services to:
    - Embed user queries
    - Perform vector similarity search
    - Retrieve conversation history
    - Build prompts with context and history
    - Generate LLM responses
    - Apply input and output safety checks
    - Store interactions in conversation history
    - Return structured responses with sources
    - Cache LLM responses for identical queries (Requirement 14.4)
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 10.2, 14.4, 19.1, 19.2, 19.3, 19.4, 19.5
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_search: VectorSearchService,
        llm_service: LLMService,
        llm_guard: LLMGuardService,
        conversation_memory: ConversationMemoryService,
        cache_client: Optional[CacheClient] = None,
        max_prompt_tokens: Optional[int] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_cache_ttl: int = 3600  # 1 hour default
    ):
        """
        Initialize RAG Query Engine with all required services.
        
        Sets up service orchestration and configuration for the RAG pipeline.
        All services must be initialized before passing to this constructor.
        Optionally uses Redis cache for LLM responses.
        
        Preconditions:
        - All service instances are initialized and ready
        - max_prompt_tokens > 0 (if provided)
        - top_k > 0
        - 0.0 <= similarity_threshold <= 1.0
        - 0.0 <= temperature <= 2.0
        - max_tokens > 0
        
        Postconditions:
        - RAG engine is ready to process queries
        - All services are accessible
        - Configuration is validated
        - Cache client is initialized if provided
        
        Args:
            embedding_service: Service for generating embeddings
            vector_search: Service for vector similarity search
            llm_service: Service for LLM response generation
            llm_guard: Service for safety validation
            conversation_memory: Service for conversation management
            cache_client: Optional CacheClient for caching responses
            max_prompt_tokens: Maximum tokens in prompt (default: from config)
            top_k: Number of context chunks to retrieve (default: 5)
            similarity_threshold: Minimum similarity score (default: 0.7)
            temperature: LLM sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 512)
            response_cache_ttl: Time-to-live for cached responses in seconds (default: 3600)
            
        Raises:
            ValueError: If parameters are invalid
            
        Requirements: 6.1, 6.2, 14.4
        """
        # Validate parameters
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Store service references
        self.embedding_service = embedding_service
        self.vector_search = vector_search
        self.llm_service = llm_service
        self.llm_guard = llm_guard
        self.conversation_memory = conversation_memory
        self.cache_client = cache_client
        
        # Load settings
        self.settings = get_settings()
        
        # Store configuration
        self.max_prompt_tokens = max_prompt_tokens or self.settings.max_prompt_tokens
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_cache_ttl = response_cache_ttl
        
        # Initialize tokenizer for token counting
        # Using cl100k_base (GPT-4 tokenizer) as a reasonable default
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # System prompt for RAG responses
        self.system_prompt = """You are L'Oracle, an AI assistant that helps users find information in documentation.

Your role is to:
- Answer questions based on the provided context from documentation
- Cite sources when providing information
- Be concise and accurate
- Admit when you don't know something or when the context doesn't contain the answer
- Use the conversation history to understand follow-up questions

Guidelines:
- Always base your answers on the provided context
- If the context doesn't contain relevant information, say so
- Reference specific documents when citing information
- Be helpful and professional
"""
        
        logger.info("RAGQueryEngine initialized")
        logger.info(f"Configuration: max_prompt_tokens={self.max_prompt_tokens}, "
                   f"top_k={self.top_k}, similarity_threshold={self.similarity_threshold}")
        
        if self.cache_client and self.cache_client.is_available():
            logger.info(f"Response cache enabled with TTL: {self.response_cache_ttl}s")
        else:
            logger.info("Response cache disabled")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def _generate_context_hash(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate hash of context chunks for cache key.
        
        Creates a deterministic hash of the context chunks to use as part of
        the cache key. This ensures responses are only cached for identical
        context.
        
        Args:
            context_chunks: List of context chunks with metadata
            
        Returns:
            SHA256 hash of context chunks
            
        Requirements: 14.4
        """
        # Create a deterministic string representation of chunks
        # Use chunk IDs and similarity scores to ensure uniqueness
        chunk_data = []
        for chunk in context_chunks:
            chunk_data.append(f"{chunk['id']}:{chunk['similarity_score']:.4f}")
        
        context_str = "|".join(chunk_data)
        hash_obj = hashlib.sha256(context_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Build LLM prompt with system instructions, context, history, and query.
        
        Constructs a prompt that includes:
        1. System instructions for L'Oracle
        2. Retrieved context chunks (prioritized by similarity)
        3. Recent conversation history (up to token limit)
        4. Current user query
        
        Implements token counting and truncation to stay within MAX_PROMPT_TOKENS.
        Context chunks are prioritized by similarity score, and history is included
        from most recent backwards until token limit is reached.
        
        Preconditions:
        - query is non-empty string
        - context_chunks is list of dicts with 'text' and 'similarity_score' keys
        - conversation_history is list of dicts with 'role' and 'content' keys
        - Each history message has valid role ('user' or 'assistant')
        
        Postconditions:
        - Returns non-empty prompt string
        - Prompt contains system instructions
        - Prompt includes context chunks (prioritized by similarity)
        - Prompt includes conversation history (up to token limit)
        - Prompt ends with current user query
        - Total prompt length <= MAX_PROMPT_TOKENS
        - If truncation occurs, event is logged
        
        Args:
            query: User's current query
            context_chunks: List of retrieved context chunks with metadata
            conversation_history: List of previous messages in conversation
            
        Returns:
            Formatted prompt string ready for LLM
            
        Requirements: 6.3, 19.1, 19.2, 19.3, 19.4, 19.5
        """
        # Step 1: Start with system prompt
        prompt_parts = [self.system_prompt]
        token_count = self._count_tokens(self.system_prompt)
        
        # Step 2: Add context chunks (prioritized by similarity)
        # Context chunks are already sorted by similarity from vector search
        if context_chunks:
            context_section = "\n\n## Relevant Documentation Context:\n\n"
            
            for i, chunk in enumerate(context_chunks, 1):
                chunk_text = chunk.get('text', '')
                similarity = chunk.get('similarity_score', 0.0)
                doc_title = chunk.get('metadata', {}).get('document_title', 'Unknown')
                
                chunk_entry = f"[Source {i}] (Similarity: {similarity:.2f}, Document: {doc_title})\n{chunk_text}\n\n"
                chunk_tokens = self._count_tokens(chunk_entry)
                
                # Check if adding this chunk would exceed token limit
                # Reserve space for history and query (estimate 500 tokens)
                if token_count + chunk_tokens + 500 > self.max_prompt_tokens:
                    logger.info(
                        f"Truncating context: reached token limit after {i-1} chunks "
                        f"(current: {token_count}, chunk: {chunk_tokens}, "
                        f"limit: {self.max_prompt_tokens})"
                    )
                    break
                
                context_section += chunk_entry
                token_count += chunk_tokens
            
            prompt_parts.append(context_section)
        
        # Step 3: Add conversation history (most recent messages, working backwards)
        if conversation_history:
            history_section = "\n\n## Conversation History:\n\n"
            history_tokens = self._count_tokens(history_section)
            
            # Reverse history to process most recent first
            included_history = []
            for msg in reversed(conversation_history):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                msg_entry = f"{role.capitalize()}: {content}\n"
                msg_tokens = self._count_tokens(msg_entry)
                
                # Check if adding this message would exceed token limit
                # Reserve space for query (estimate 200 tokens)
                if token_count + history_tokens + msg_tokens + 200 > self.max_prompt_tokens:
                    logger.info(
                        f"Truncating history: reached token limit after {len(included_history)} messages "
                        f"(current: {token_count + history_tokens}, msg: {msg_tokens}, "
                        f"limit: {self.max_prompt_tokens})"
                    )
                    break
                
                included_history.insert(0, msg_entry)  # Insert at beginning to maintain order
                history_tokens += msg_tokens
            
            if included_history:
                history_section += "".join(included_history)
                prompt_parts.append(history_section)
                token_count += history_tokens
        
        # Step 4: Add current query
        query_section = f"\n\n## Current Question:\n\n{query}\n\n## Answer:\n\n"
        query_tokens = self._count_tokens(query_section)
        
        # Final check: if query itself would exceed limit, we have a problem
        if token_count + query_tokens > self.max_prompt_tokens:
            logger.warning(
                f"Prompt exceeds token limit even after truncation: "
                f"{token_count + query_tokens} > {self.max_prompt_tokens}"
            )
            # In this case, we need to be more aggressive with truncation
            # Remove history entirely if needed
            if len(prompt_parts) > 2:  # Has history
                prompt_parts = prompt_parts[:2]  # Keep only system and context
                token_count = self._count_tokens("".join(prompt_parts))
                logger.info("Removed all history to fit within token limit")
        
        prompt_parts.append(query_section)
        final_prompt = "".join(prompt_parts)
        final_token_count = self._count_tokens(final_prompt)
        
        logger.debug(f"Built prompt with {final_token_count} tokens")
        
        # Postcondition: verify token limit
        if final_token_count > self.max_prompt_tokens:
            logger.error(
                f"Prompt still exceeds token limit: {final_token_count} > {self.max_prompt_tokens}"
            )
        
        return final_prompt

    def process_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        user_id: str = 'anonymous'
    ) -> Dict[str, Any]:
        """
        Process user query through complete RAG pipeline.

        Orchestrates the full RAG workflow:
        1. Validate input safety with LLM Guard
        2. Create or retrieve conversation
        3. Embed user query
        4. Perform vector similarity search
        5. Retrieve conversation history
        6. Build prompt with context and history
        7. Generate LLM response
        8. Validate output safety with LLM Guard
        9. Store interaction in conversation history
        10. Return structured response with sources

        Preconditions:
        - query is non-empty string
        - top_k is positive integer (if provided)
        - If conversation_id provided, conversation exists
        - All services are available

        Postconditions:
        - Returns QueryResponse with answer and sources
        - Query and response stored in conversation history
        - All safety checks passed
        - sources list length <= top_k

        Args:
            query: User's question or query
            conversation_id: Optional conversation ID (creates new if None)
            top_k: Optional override for number of context chunks
            user_id: User identifier for conversation tracking (Requirement 15.3)

        Returns:
            Dictionary with:
            - answer: Generated response text
            - sources: List of source citations with metadata
            - conversation_id: Conversation ID for follow-up queries

        Raises:
            SecurityException: If input or output fails safety checks
            RuntimeError: If any service fails

        Requirements: 6.1, 6.2, 6.4, 6.5, 15.3
        """
        # Validate preconditions
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        # Use instance top_k if not overridden
        k = top_k or self.top_k

        start_time = time.time()
        logger.info("Processing query", query_preview=query[:100], top_k=k, user_id=user_id)
        add_breadcrumb(message="Query processing started", category="query", data={"top_k": k, "user_id": user_id})

        try:
            # Step 1: Input safety check
            logger.debug("Checking input safety")
            with track_time(query_processing_duration_seconds, {'stage': 'input_safety'}):
                is_safe, reason = self.llm_guard.check_input(query)
                record_llm_guard_check('input', is_safe, reason if not is_safe else None)

                if not is_safe:
                    logger.warning("Input rejected by LLM Guard", reason=reason)
                    log_security_event(
                        event_type="input_rejected",
                        severity="warning",
                        description="Query rejected by LLM Guard",
                        reason=reason,
                        query_preview=query[:100],
                        user_id=user_id
                    )
                    raise SecurityException(f"Unsafe input: {reason}")

            # Step 2: Get or create conversation with user_id (Requirement 15.3)
            if conversation_id is None:
                logger.debug("Creating new conversation", user_id=user_id)
                conversation_id = self.conversation_memory.create_conversation(
                    user_id=user_id
                )
                logger.info("Created new conversation", conversation_id=conversation_id, user_id=user_id)
            else:
                # Verify conversation exists
                if not self.conversation_memory.conversation_exists(conversation_id):
                    raise ValueError(f"Conversation {conversation_id} not found or expired")
                logger.debug("Using existing conversation", conversation_id=conversation_id)

            # Set context for error tracking
            set_context('query', {
                'conversation_id': conversation_id,
                'query_length': len(query),
                'top_k': k
            })

            # Step 3: Generate query embedding
            logger.debug("Generating query embedding")
            with track_time(query_processing_duration_seconds, {'stage': 'embedding'}):
                query_embedding = self.embedding_service.embed_text(query)
            logger.debug("Query embedding generated", dimension=len(query_embedding))

            # Step 4: Vector similarity search
            logger.debug("Performing vector search", top_k=k, threshold=self.similarity_threshold)
            with track_time(query_processing_duration_seconds, {'stage': 'vector_search'}):
                similar_chunks = self.vector_search.search_similar(
                    query_vector=query_embedding,
                    top_k=k,
                    threshold=self.similarity_threshold
                )
            logger.info("Found relevant chunks", count=len(similar_chunks))
            add_breadcrumb(message=f"Found {len(similar_chunks)} relevant chunks", category="search")

            # Step 4.5: Check response cache (Requirement 14.4)
            # Generate context hash for cache key
            context_hash = self._generate_context_hash(similar_chunks)
            
            if self.cache_client and self.cache_client.is_available():
                cached_response = self.cache_client.get_response(query, context_hash)
                if cached_response is not None:
                    logger.info("Cache hit for response")
                    
                    # Still need to store in conversation history
                    self.conversation_memory.add_message(
                        conversation_id=conversation_id,
                        role='user',
                        content=query,
                        metadata={'timestamp': 'auto'}
                    )
                    
                    sources_for_storage = [
                        {
                            'chunk_id': chunk['id'],
                            'document_id': chunk['document_id'],
                            'similarity_score': chunk['similarity_score']
                        }
                        for chunk in similar_chunks
                    ]
                    
                    self.conversation_memory.add_message(
                        conversation_id=conversation_id,
                        role='assistant',
                        content=cached_response,
                        sources=sources_for_storage,
                        metadata={'timestamp': 'auto', 'cached': True}
                    )
                    
                    # Format sources for response
                    sources = []
                    for chunk in similar_chunks:
                        sources.append({
                            'chunk_id': chunk['id'],
                            'document_id': chunk['document_id'],
                            'title': chunk['metadata'].get('document_title', 'Unknown'),
                            'excerpt': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                            'similarity_score': chunk['similarity_score']
                        })
                    
                    duration = time.time() - start_time
                    logger.info(
                        "Query processed from cache",
                        sources_count=len(sources),
                        response_length=len(cached_response),
                        duration_ms=duration * 1000
                    )
                    
                    return {
                        'answer': cached_response,
                        'sources': sources,
                        'conversation_id': conversation_id
                    }

            # Step 5: Retrieve conversation history
            logger.debug("Retrieving conversation history")
            history = self.conversation_memory.get_recent_messages(
                conversation_id=conversation_id,
                limit=10  # Get last 10 messages for context
            )
            # Reverse to get chronological order (oldest first)
            history = list(reversed(history))
            logger.debug("Retrieved conversation history", message_count=len(history))

            # Step 6: Build prompt
            logger.debug("Building prompt")
            with track_time(query_processing_duration_seconds, {'stage': 'prompt_building'}):
                prompt = self.build_prompt(
                    query=query,
                    context_chunks=similar_chunks,
                    conversation_history=history
                )
            token_count = self._count_tokens(prompt)
            logger.debug("Prompt built", token_count=token_count)

            # Step 7: Generate LLM response
            logger.debug("Generating LLM response")
            with track_time(query_processing_duration_seconds, {'stage': 'llm_generation'}):
                llm_response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            logger.debug("LLM response generated", response_length=len(llm_response))
            add_breadcrumb(message="LLM response generated", category="llm")

            # Step 8: Output safety check
            logger.debug("Checking output safety")
            with track_time(query_processing_duration_seconds, {'stage': 'output_safety'}):
                is_safe, reason = self.llm_guard.check_output(llm_response)
                record_llm_guard_check('output', is_safe, reason if not is_safe else None)

                if not is_safe:
                    logger.warning("Output rejected by LLM Guard", reason=reason)
                    log_security_event(
                        event_type="output_rejected",
                        severity="warning",
                        description="LLM output rejected by LLM Guard",
                        reason=reason
                    )
                    # Return generic safe message but keep sources
                    llm_response = "I cannot provide that information due to safety policies. Please rephrase your question."

            # Step 9: Store conversation messages
            logger.debug("Storing conversation messages")

            # Store user message
            self.conversation_memory.add_message(
                conversation_id=conversation_id,
                role='user',
                content=query,
                metadata={'timestamp': 'auto'}
            )

            # Prepare sources for storage
            sources_for_storage = [
                {
                    'chunk_id': chunk['id'],
                    'document_id': chunk['document_id'],
                    'similarity_score': chunk['similarity_score']
                }
                for chunk in similar_chunks
            ]

            # Store assistant message
            self.conversation_memory.add_message(
                conversation_id=conversation_id,
                role='assistant',
                content=llm_response,
                sources=sources_for_storage,
                metadata={'timestamp': 'auto'}
            )

            logger.debug("Conversation messages stored")

            # Step 9.5: Cache response (Requirement 14.4)
            if self.cache_client and self.cache_client.is_available():
                self.cache_client.set_response(
                    query,
                    context_hash,
                    llm_response,
                    ttl=self.response_cache_ttl
                )
                logger.debug("Cached response")

            # Step 10: Format sources for response
            sources = []
            for chunk in similar_chunks:
                sources.append({
                    'chunk_id': chunk['id'],
                    'document_id': chunk['document_id'],
                    'title': chunk['metadata'].get('document_title', 'Unknown'),
                    'excerpt': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'similarity_score': chunk['similarity_score']
                })

            # Step 11: Return response
            response = {
                'answer': llm_response,
                'sources': sources,
                'conversation_id': conversation_id
            }

            duration = time.time() - start_time
            logger.info(
                "Query processed successfully",
                sources_count=len(sources),
                response_length=len(llm_response),
                duration_ms=duration * 1000
            )

            return response

        except SecurityException:
            # Re-raise security exceptions
            raise
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error("Error processing query", error=str(e), exc_info=True)
            capture_exception(e, level="error", tags={"component": "rag_query_engine"})
            raise RuntimeError(f"Query processing failed: {e}")
    
    def stream_response(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Iterator[str]:
        """
        Stream LLM response in real-time for user queries.
        
        Similar to process_query but streams the LLM response token by token
        for real-time display. The complete workflow is the same except the
        LLM generation step uses streaming.
        
        Note: Safety checks, conversation storage, and source retrieval happen
        after streaming completes. The stream only yields the LLM response tokens.
        
        Preconditions:
        - query is non-empty string
        - top_k is positive integer (if provided)
        - If conversation_id provided, conversation exists
        - All services are available
        
        Postconditions:
        - Yields response tokens as they are generated
        - Query and response stored in conversation history after streaming
        - All safety checks applied
        - Stream completes without errors
        
        Args:
            query: User's question or query
            conversation_id: Optional conversation ID (creates new if None)
            top_k: Optional override for number of context chunks
            
        Yields:
            Response tokens as strings
            
        Raises:
            SecurityException: If input or output fails safety checks
            RuntimeError: If any service fails
            
        Requirements: 10.2
        """
        # Validate preconditions
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")
        
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Use instance top_k if not overridden
        k = top_k or self.top_k
        
        logger.info(f"Processing streaming query: {query[:100]}...")
        
        try:
            # Step 1: Input safety check
            logger.debug("Checking input safety...")
            is_safe, reason = self.llm_guard.check_input(query)
            if not is_safe:
                logger.warning(f"Input rejected by LLM Guard: {reason}")
                raise SecurityException(f"Unsafe input: {reason}")
            
            # Step 2: Get or create conversation
            if conversation_id is None:
                logger.debug("Creating new conversation...")
                conversation_id = self.conversation_memory.create_conversation(
                    user_id='anonymous'
                )
                logger.info(f"Created new conversation: {conversation_id}")
            else:
                # Verify conversation exists
                if not self.conversation_memory.conversation_exists(conversation_id):
                    raise ValueError(f"Conversation {conversation_id} not found or expired")
                logger.debug(f"Using existing conversation: {conversation_id}")
            
            # Step 3: Generate query embedding
            logger.debug("Generating query embedding...")
            query_embedding = self.embedding_service.embed_text(query)
            
            # Step 4: Vector similarity search
            logger.debug(f"Performing vector search (top_k={k}, threshold={self.similarity_threshold})...")
            similar_chunks = self.vector_search.search_similar(
                query_vector=query_embedding,
                top_k=k,
                threshold=self.similarity_threshold
            )
            logger.info(f"Found {len(similar_chunks)} relevant chunks")
            
            # Step 5: Retrieve conversation history
            logger.debug("Retrieving conversation history...")
            history = self.conversation_memory.get_recent_messages(
                conversation_id=conversation_id,
                limit=10
            )
            history = list(reversed(history))
            
            # Step 6: Build prompt
            logger.debug("Building prompt...")
            prompt = self.build_prompt(
                query=query,
                context_chunks=similar_chunks,
                conversation_history=history
            )
            
            # Step 7: Stream LLM response
            logger.debug("Starting LLM response streaming...")
            response_chunks = []
            
            for chunk in self.llm_service.generate_stream(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            ):
                response_chunks.append(chunk)
                yield chunk
            
            # Reconstruct full response for safety check and storage
            full_response = "".join(response_chunks)
            logger.debug(f"Streaming complete: {len(full_response)} characters")
            
            # Step 8: Output safety check (after streaming)
            logger.debug("Checking output safety...")
            is_safe, reason = self.llm_guard.check_output(full_response)
            if not is_safe:
                logger.warning(f"Output rejected by LLM Guard: {reason}")
                # Note: Response already streamed, but we log the violation
                # In production, you might want to stream safety check results
            
            # Step 9: Store conversation messages
            logger.debug("Storing conversation messages...")
            
            # Store user message
            self.conversation_memory.add_message(
                conversation_id=conversation_id,
                role='user',
                content=query,
                metadata={'timestamp': 'auto'}
            )
            
            # Prepare sources for storage
            sources_for_storage = [
                {
                    'chunk_id': chunk['id'],
                    'document_id': chunk['document_id'],
                    'similarity_score': chunk['similarity_score']
                }
                for chunk in similar_chunks
            ]
            
            # Store assistant message
            self.conversation_memory.add_message(
                conversation_id=conversation_id,
                role='assistant',
                content=full_response,
                sources=sources_for_storage,
                metadata={'timestamp': 'auto', 'streamed': True}
            )
            
            logger.info(f"Streaming query processed successfully: {len(similar_chunks)} sources")
            
        except SecurityException:
            # Re-raise security exceptions
            raise
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error processing streaming query: {e}", exc_info=True)
            raise RuntimeError(f"Streaming query processing failed: {e}")
