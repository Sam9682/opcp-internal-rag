"""
HTTP client for the Embedding Service.

Provides the same interface as EmbeddingService but delegates to the
remote embedding-service container via HTTP. This avoids pulling torch
and sentence-transformers into the API container.
"""

import numpy as np
import requests
import logging
import os
import time
from typing import List, Optional

from .logging_config import get_logger
from .metrics import record_embedding_generation
from .cache_client import CacheClient

logger = get_logger(__name__)


class EmbeddingClient:
    """HTTP client that mirrors the EmbeddingService interface."""

    def __init__(
        self,
        service_url: Optional[str] = None,
        cache_client: Optional[CacheClient] = None,
        cache_ttl: int = 86400,
    ):
        self.service_url = (
            service_url
            or os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8000")
        )
        self.cache_client = cache_client
        self.cache_ttl = cache_ttl
        self._embedding_dimension = 1024
        logger.info(f"EmbeddingClient targeting {self.service_url}")

    def get_embedding_dimension(self) -> int:
        return self._embedding_dimension

    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text must be non-empty")

        # Check cache
        if self.cache_client and self.cache_client.is_available():
            cached = self.cache_client.get_embedding(text)
            if cached is not None:
                return np.array(cached, dtype=np.float32)

        start = time.time()
        resp = requests.post(
            f"{self.service_url}/embed",
            json={"texts": [text]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = np.array(data["embeddings"][0], dtype=np.float32)

        # Normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache
        if self.cache_client and self.cache_client.is_available():
            self.cache_client.set_embedding(text, embedding.tolist(), ttl=self.cache_ttl)

        record_embedding_generation(time.time() - start, batch_size=1)
        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            raise ValueError("texts list must be non-empty")

        resp = requests.post(
            f"{self.service_url}/embed",
            json={"texts": texts},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for vec in data["embeddings"]:
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            results.append(arr)
        return results
