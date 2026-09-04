"""
HTTP client for the LLM Service.

Provides the same interface as LLMService but delegates to the
remote llm-service container via HTTP. This avoids pulling torch
and transformers into the API container.
"""

import requests
import logging
import os
import time
from typing import Iterator, Optional

from .logging_config import get_logger
from .metrics import record_llm_generation

logger = get_logger(__name__)


class LLMClient:
    """HTTP client that mirrors the LLMService interface."""

    def __init__(self, service_url: Optional[str] = None):
        self.service_url = (
            service_url
            or os.getenv("LLM_SERVICE_URL", "http://llm-service:8000")
        )
        logger.info(f"LLMClient targeting {self.service_url}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be non-empty")

        start = time.time()
        resp = requests.post(
            f"{self.service_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["text"]

        duration = time.time() - start
        record_llm_generation(duration, data.get("tokens_used", len(text.split())), success=True)
        return text

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        # Fallback: non-streaming call, yield full response
        yield self.generate(prompt, **kwargs)
