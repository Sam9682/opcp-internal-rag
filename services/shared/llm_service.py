"""
LLM Service for RAG Application

This module provides language model functionality for generating natural language
responses from prompts. Supports both local models (like Mistral-7B) and API-based
models with streaming capabilities.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import logging
from typing import Iterator, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import os
import time
import tiktoken

from .logging_config import get_logger
from .metrics import record_llm_generation, track_time, llm_generation_duration_seconds
from .sentry_config import capture_exception

logger = get_logger(__name__)


class LLMService:
    """
    LLM service for generating natural language responses.
    
    This class handles:
    - Loading and caching language models (local or API-based)
    - GPU/CPU device selection
    - Response generation with configurable parameters
    - Streaming response generation for real-time display
    - Model caching for efficient subsequent requests
    - Error handling and timeout management
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the LLMService with specified model.
        
        Loads the language model into memory and caches it for subsequent
        requests. Supports both local models and API-based models.
        
        Preconditions:
        - model_name is valid model identifier or API model name
        - device is None, 'cpu', 'cuda', or 'cuda:N'
        - If use_api is True, api_key must be provided
        
        Postconditions:
        - Model is loaded and cached in memory (if local)
        - Device is set (GPU if available, else CPU)
        - Model is ready for response generation
        - Tokenizer is initialized
        
        Args:
            model_name: Model identifier (Hugging Face or API model name)
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            use_api: Whether to use API-based model instead of local
            api_key: API key for API-based models
            api_base_url: Base URL for API endpoint
            cache_dir: Directory for caching model files
            
        Requirements: 10.4, 10.5
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", "/root/.cache/huggingface")
        
        # Step 1: Determine device (GPU if available, else CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing LLMService with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"API mode: {use_api}")
        
        if use_api:
            # API-based model initialization
            self._init_api_model()
        else:
            # Local model initialization
            self._init_local_model()
    
    def _init_api_model(self):
        """
        Initialize API-based model client.
        
        Postconditions:
        - API client is configured
        - API key is validated
        """
        if not self.api_key:
            raise ValueError("API key required for API-based models")
        
        # Import API client libraries as needed
        try:
            import openai
            self.api_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
            logger.info("API client initialized successfully")
        except ImportError:
            logger.warning("OpenAI library not installed, API mode may not work")
            self.api_client = None
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            raise RuntimeError(f"API client initialization failed: {e}")
        
        self.model = None
        self.tokenizer = None
    
    def _init_local_model(self):
        """
        Initialize local language model.
        
        Postconditions:
        - Model is loaded into memory
        - Tokenizer is initialized
        - Model is moved to appropriate device
        
        Requirements: 10.4, 10.5
        """
        try:
            # Step 1: Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Tokenizer loaded successfully")
            
            # Step 2: Load model with appropriate settings
            logger.info(f"Loading model {self.model_name} (this may take a while...)")
            
            # Determine dtype based on device
            if self.device == "cuda":
                # Use float16 for GPU to save memory
                torch_dtype = torch.float16
            else:
                # Use float32 for CPU
                torch_dtype = torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move model to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model dtype: {self.model.dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
        
        self.api_client = None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        Generate response from prompt.

        Generates a natural language response from the given prompt using
        the configured language model. Supports configurable parameters
        for controlling generation behavior.

        Preconditions:
        - prompt is non-empty string
        - max_tokens > 0
        - 0.0 <= temperature <= 2.0
        - 0.0 <= top_p <= 1.0
        - Model is loaded and initialized

        Postconditions:
        - Returns non-empty response string
        - Response is generated according to parameters
        - No errors or timeouts occurred

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 50)
            repetition_penalty: Penalty for repetition (default: 1.1)
            **kwargs: Additional generation parameters

        Returns:
            Generated response text

        Raises:
            ValueError: If prompt is empty or parameters are invalid
            RuntimeError: If generation fails

        Requirements: 10.1, 10.3
        """
        # Validate preconditions
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be non-empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")

        start_time = time.time()

        try:
            with track_time(llm_generation_duration_seconds):
                if self.use_api:
                    response = self._generate_api(
                        prompt, max_tokens, temperature, top_p, **kwargs
                    )
                else:
                    response = self._generate_local(
                        prompt, max_tokens, temperature, top_p, top_k,
                        repetition_penalty, **kwargs
                    )

            # Estimate tokens generated (rough approximation)
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens_generated = len(encoding.encode(response))
            except:
                # Fallback: rough estimate
                tokens_generated = len(response.split())

            # Record metrics
            duration = time.time() - start_time
            record_llm_generation(duration, tokens_generated, success=True)

            logger.debug(
                "LLM generation completed",
                response_length=len(response),
                tokens_generated=tokens_generated,
                duration_ms=duration * 1000
            )

            return response

        except Exception as e:
            duration = time.time() - start_time
            record_llm_generation(duration, 0, success=False)
            logger.error("Generation failed", error=str(e), exc_info=True)
            capture_exception(e, level="error", tags={"component": "llm_service"})
            raise RuntimeError(f"Response generation failed: {e}")
    
    def _generate_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> str:
        """
        Generate response using API-based model.
        
        Postconditions:
        - Returns generated response text
        - API call completed successfully
        """
        if not self.api_client:
            raise RuntimeError("API client not initialized")
        
        try:
            response = self.api_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            raise RuntimeError(f"API generation failed: {e}")
    
    def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        **kwargs
    ) -> str:
        """
        Generate response using local model.
        
        Postconditions:
        - Returns generated response text
        - Model inference completed successfully
        
        Requirements: 10.1, 10.3
        """
        # Step 1: Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Step 2: Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Step 3: Decode output
        # Remove input tokens from output
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        # Postcondition: verify non-empty response
        if not response.strip():
            logger.warning("Generated empty response")
            return "I apologize, but I couldn't generate a proper response."
        
        return response.strip()
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream response tokens in real-time.
        
        Generates response tokens one at a time using async generators
        for efficient streaming. Allows real-time display of responses
        as they are generated.
        
        Preconditions:
        - prompt is non-empty string
        - max_tokens > 0
        - 0.0 <= temperature <= 2.0
        - Model is loaded and initialized
        
        Postconditions:
        - Yields response tokens as they are generated
        - Final concatenated output matches non-streaming generation
        - Stream completes without errors
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 50)
            repetition_penalty: Penalty for repetition (default: 1.1)
            **kwargs: Additional generation parameters
            
        Yields:
            Generated response tokens as strings
            
        Raises:
            ValueError: If prompt is empty or parameters are invalid
            RuntimeError: If streaming generation fails
            
        Requirements: 10.2
        """
        # Validate preconditions
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be non-empty")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        try:
            if self.use_api:
                yield from self._generate_stream_api(
                    prompt, max_tokens, temperature, top_p, **kwargs
                )
            else:
                yield from self._generate_stream_local(
                    prompt, max_tokens, temperature, top_p, top_k,
                    repetition_penalty, **kwargs
                )
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}")
    
    def _generate_stream_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream response using API-based model.
        
        Yields:
            Generated response tokens
        """
        if not self.api_client:
            raise RuntimeError("API client not initialized")
        
        try:
            stream = self.api_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"API streaming failed: {e}")
            raise RuntimeError(f"API streaming failed: {e}")
    
    def _generate_stream_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream response using local model with TextIteratorStreamer.
        
        Yields:
            Generated response tokens
            
        Requirements: 10.2
        """
        # Step 1: Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Step 2: Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Step 3: Set up generation kwargs
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            **kwargs
        )
        
        # Step 4: Start generation in separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Step 5: Yield tokens as they are generated
        for text in streamer:
            yield text
        
        # Wait for generation to complete
        thread.join()
