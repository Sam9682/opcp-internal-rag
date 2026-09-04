"""
Cache Client for Redis operations.

This module provides a wrapper around Redis for caching embeddings and LLM responses
to improve performance and reduce redundant computations.

Requirements: 14.4
"""

import redis
import json
import hashlib
from typing import Optional, Any
import logging
import os
import time

from .logging_config import get_logger
from .sentry_config import capture_exception
from .metrics import record_cache_hit, record_cache_miss, record_cache_operation

logger = get_logger(__name__)


class CacheClient:
    """
    Redis cache client for storing embeddings and LLM responses.
    
    This class provides:
    - Connection management to Redis
    - Key generation with hashing
    - Get/set operations with TTL support
    - JSON serialization for complex objects
    - Graceful error handling (cache misses don't break the app)
    
    Requirements: 14.4
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        decode_responses: bool = True
    ):
        """
        Initialize Redis cache client.
        
        Connects to Redis server and validates the connection. If Redis is
        unavailable, logs a warning but doesn't fail - cache operations will
        gracefully degrade.
        
        Args:
            host: Redis host (default: from REDIS_HOST env or 'localhost')
            port: Redis port (default: from REDIS_PORT env or 6379)
            db: Redis database number (default: 0)
            decode_responses: Whether to decode responses to strings (default: True)
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = int(port or os.getenv('REDIS_PORT', 6379))
        self.db = db
        self.decode_responses = decode_responses
        
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            self._available = True
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis unavailable: {e}. Cache will be disabled.")
            self.client = None
            self._available = False
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            capture_exception(e, level="warning", tags={"component": "cache_client"})
            self.client = None
            self._available = False
    
    def is_available(self) -> bool:
        """
        Check if Redis cache is available.
        
        Returns:
            True if Redis is connected and operational, False otherwise
        """
        return self._available and self.client is not None
    
    def _generate_key(self, prefix: str, data: str) -> str:
        """
        Generate cache key with hash.
        
        Creates a cache key by hashing the data to ensure consistent key length
        and avoid special characters.
        
        Args:
            prefix: Key prefix (e.g., 'embedding', 'response')
            data: Data to hash (e.g., text content, query)
            
        Returns:
            Cache key in format: prefix:hash
        """
        # Use SHA256 for consistent hashing
        hash_obj = hashlib.sha256(data.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        return f"{prefix}:{hash_hex}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Retrieves a value from Redis cache. Returns None if key doesn't exist
        or if Redis is unavailable. Handles JSON deserialization automatically.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or error occurred
        """
        if not self.is_available():
            return None
        
        start_time = time.time()
        
        try:
            value = self.client.get(key)
            
            duration = time.time() - start_time
            record_cache_operation('get', 'generic', duration)
            
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Return raw value if not JSON
                return value
                
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with optional TTL.
        
        Stores a value in Redis cache with optional expiration time.
        Automatically serializes complex objects to JSON.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON-serialized if not string)
            ttl: Time-to-live in seconds (None for no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        start_time = time.time()
        
        try:
            # Serialize to JSON if not a string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            if ttl is not None:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
            
            duration = time.time() - start_time
            record_cache_operation('set', 'generic', duration)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[list]:
        """
        Get cached embedding for text.
        
        Retrieves a cached embedding vector for the given text. Uses text hash
        as the cache key to ensure consistent lookups.
        
        Args:
            text: Text to look up embedding for
            
        Returns:
            Embedding vector as list or None if not cached
        """
        key = self._generate_key('embedding', text)
        result = self.get(key)
        
        if result is not None:
            record_cache_hit('embedding')
        else:
            record_cache_miss('embedding')
        
        return result
    
    def set_embedding(
        self,
        text: str,
        embedding: list,
        ttl: int = 86400  # 24 hours default
    ) -> bool:
        """
        Cache embedding for text.
        
        Stores an embedding vector in cache with the text hash as key.
        Default TTL is 24 hours since embeddings are deterministic and
        don't change frequently.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector as list
            ttl: Time-to-live in seconds (default: 86400 = 24 hours)
            
        Returns:
            True if successful, False otherwise
        """
        key = self._generate_key('embedding', text)
        return self.set(key, embedding, ttl)
    
    def get_response(self, query: str, context_hash: str) -> Optional[str]:
        """
        Get cached LLM response for query and context.
        
        Retrieves a cached LLM response. The cache key includes both the query
        and a hash of the context to ensure responses are only reused when the
        context is identical.
        
        Args:
            query: User query
            context_hash: Hash of the context chunks used
            
        Returns:
            Cached response text or None if not cached
        """
        # Combine query and context hash for key
        cache_data = f"{query}|{context_hash}"
        key = self._generate_key('response', cache_data)
        result = self.get(key)
        
        if result is not None:
            record_cache_hit('response')
        else:
            record_cache_miss('response')
        
        return result
    
    def set_response(
        self,
        query: str,
        context_hash: str,
        response: str,
        ttl: int = 3600  # 1 hour default
    ) -> bool:
        """
        Cache LLM response for query and context.
        
        Stores an LLM response in cache. Default TTL is 1 hour since responses
        may become stale as documentation is updated.
        
        Args:
            query: User query
            context_hash: Hash of the context chunks used
            response: LLM response text
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        # Combine query and context hash for key
        cache_data = f"{query}|{context_hash}"
        key = self._generate_key('response', cache_data)
        return self.set(key, response, ttl)
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all keys from current database.
        
        WARNING: This deletes all cached data. Use with caution.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.client.flushdb()
            logger.info("Cleared all cache data")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_stats(self) -> Optional[dict]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats or None if unavailable
        """
        if not self.is_available():
            return None
        
        try:
            info = self.client.info('stats')
            return {
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'keys': self.client.dbsize()
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return None
