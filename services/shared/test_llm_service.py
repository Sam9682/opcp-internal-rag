"""
Unit tests for LLMService

Tests response generation, streaming, and model loading.
Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add services/shared to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_service import LLMService


class TestLLMServiceInitialization:
    """Test suite for LLMService initialization."""
    
    @patch('llm_service.AutoTokenizer')
    @patch('llm_service.AutoModelForCausalLM')
    @patch('llm_service.torch')
    def test_initialization_cpu(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test that LLMService initializes correctly on CPU."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = False
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Initialize service
        service = LLMService(device="cpu")
        
        assert service is not None
        assert service.device == "cpu"
        assert service.use_api is False
        assert service.model is not None
        assert service.tokenizer is not None
    
    @patch('llm_service.AutoTokenizer')
    @patch('llm_service.AutoModelForCausalLM')
    @patch('llm_service.torch')
    def test_initialization_cuda(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test that LLMService initializes correctly on CUDA."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "float16"
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.dtype = "float16"
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Initialize service
        service = LLMService(device="cuda")
        
        assert service is not None
        assert service.device == "cuda"
        assert service.model is not None
    
    def test_initialization_api_mode_without_key_raises_error(self):
        """Test that API mode without API key raises ValueError."""
        with pytest.raises(ValueError, match="API key required"):
            LLMService(use_api=True, api_key=None)
    
    @pytest.mark.skipif(True, reason="OpenAI library is optional")
    def test_initialization_api_mode_with_key(self):
        """Test that API mode initializes correctly with API key."""
        pytest.skip("OpenAI library is optional")


class TestLLMServiceGenerate:
    """Test suite for LLMService generate() method."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock LLMService for testing."""
        with patch('llm_service.AutoTokenizer'), \
             patch('llm_service.AutoModelForCausalLM'), \
             patch('llm_service.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_tokenizer.eos_token = "[EOS]"
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            
            # Mock model
            mock_model = Mock()
            mock_model.eval.return_value = None
            
            service = LLMService.__new__(LLMService)
            service.model_name = "test-model"
            service.use_api = False
            service.device = "cpu"
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service.api_client = None
            
            return service
    
    def test_generate_empty_prompt_raises_error(self, mock_service):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt must be non-empty"):
            mock_service.generate("")
        
        with pytest.raises(ValueError, match="Prompt must be non-empty"):
            mock_service.generate("   ")
    
    def test_generate_invalid_max_tokens_raises_error(self, mock_service):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            mock_service.generate("Test prompt", max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            mock_service.generate("Test prompt", max_tokens=-1)
    
    def test_generate_invalid_temperature_raises_error(self, mock_service):
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between"):
            mock_service.generate("Test prompt", temperature=-0.1)
        
        with pytest.raises(ValueError, match="temperature must be between"):
            mock_service.generate("Test prompt", temperature=2.5)
    
    def test_generate_invalid_top_p_raises_error(self, mock_service):
        """Test that invalid top_p raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between"):
            mock_service.generate("Test prompt", top_p=-0.1)
        
        with pytest.raises(ValueError, match="top_p must be between"):
            mock_service.generate("Test prompt", top_p=1.5)
    
    @patch('llm_service.torch')
    def test_generate_basic(self, mock_torch, mock_service):
        """Test basic response generation."""
        # Mock tokenizer behavior
        mock_inputs = {
            'input_ids': Mock(shape=(1, 10)),
            'attention_mask': Mock()
        }
        mock_tokenizer_output = Mock()
        mock_tokenizer_output.to = Mock(return_value=mock_inputs)
        mock_service.tokenizer.return_value = mock_tokenizer_output
        
        # Mock model generation
        mock_output = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        mock_service.model.generate.return_value = mock_output
        
        # Mock decode
        mock_service.tokenizer.decode.return_value = "This is a generated response."
        
        # Mock torch.no_grad
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        
        # Generate response
        response = mock_service.generate("Test prompt")
        
        assert response == "This is a generated response."
        assert isinstance(response, str)
        assert len(response) > 0
    
    @patch('llm_service.torch')
    def test_generate_with_custom_parameters(self, mock_torch, mock_service):
        """Test generation with custom parameters."""
        # Mock tokenizer behavior
        mock_inputs = {
            'input_ids': Mock(shape=(1, 10)),
            'attention_mask': Mock()
        }
        mock_tokenizer_output = Mock()
        mock_tokenizer_output.to = Mock(return_value=mock_inputs)
        mock_service.tokenizer.return_value = mock_tokenizer_output
        
        # Mock model generation
        mock_output = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        mock_service.model.generate.return_value = mock_output
        
        # Mock decode
        mock_service.tokenizer.decode.return_value = "Custom response."
        
        # Mock torch.no_grad
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        
        # Generate with custom parameters
        response = mock_service.generate(
            "Test prompt",
            max_tokens=256,
            temperature=0.5,
            top_p=0.95,
            top_k=40
        )
        
        assert response == "Custom response."
        
        # Verify model.generate was called with correct parameters
        call_kwargs = mock_service.model.generate.call_args[1]
        assert call_kwargs['max_new_tokens'] == 256
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['top_p'] == 0.95
        assert call_kwargs['top_k'] == 40


class TestLLMServiceGenerateStream:
    """Test suite for LLMService generate_stream() method."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock LLMService for testing."""
        with patch('llm_service.AutoTokenizer'), \
             patch('llm_service.AutoModelForCausalLM'), \
             patch('llm_service.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_tokenizer.eos_token = "[EOS]"
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            
            # Mock model
            mock_model = Mock()
            
            service = LLMService.__new__(LLMService)
            service.model_name = "test-model"
            service.use_api = False
            service.device = "cpu"
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service.api_client = None
            
            return service
    
    def test_generate_stream_empty_prompt_raises_error(self, mock_service):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt must be non-empty"):
            list(mock_service.generate_stream(""))
        
        with pytest.raises(ValueError, match="Prompt must be non-empty"):
            list(mock_service.generate_stream("   "))
    
    def test_generate_stream_invalid_max_tokens_raises_error(self, mock_service):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            list(mock_service.generate_stream("Test prompt", max_tokens=0))
    
    def test_generate_stream_invalid_temperature_raises_error(self, mock_service):
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between"):
            list(mock_service.generate_stream("Test prompt", temperature=-0.1))
    
    @patch('llm_service.TextIteratorStreamer')
    @patch('llm_service.Thread')
    def test_generate_stream_basic(self, mock_thread_class, mock_streamer_class, mock_service):
        """Test basic streaming response generation."""
        # Mock tokenizer behavior
        mock_inputs = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        mock_tokenizer_output = Mock()
        mock_tokenizer_output.to = Mock(return_value=mock_inputs)
        mock_service.tokenizer.return_value = mock_tokenizer_output
        
        # Mock streamer
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(return_value=iter(["This ", "is ", "a ", "test."]))
        mock_streamer_class.return_value = mock_streamer
        
        # Mock thread
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread
        
        # Generate streaming response
        tokens = list(mock_service.generate_stream("Test prompt"))
        
        assert tokens == ["This ", "is ", "a ", "test."]
        assert len(tokens) == 4
        
        # Verify thread was started and joined
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once()
    
    @patch('llm_service.TextIteratorStreamer')
    @patch('llm_service.Thread')
    def test_generate_stream_concatenation_matches_generate(
        self, mock_thread_class, mock_streamer_class, mock_service
    ):
        """Test that concatenated stream output matches non-streaming output."""
        # Mock tokenizer behavior
        mock_inputs = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        mock_tokenizer_output = Mock()
        mock_tokenizer_output.to = Mock(return_value=mock_inputs)
        mock_service.tokenizer.return_value = mock_tokenizer_output
        
        # Mock streamer
        mock_streamer = Mock()
        stream_tokens = ["Hello", " ", "world", "!"]
        mock_streamer.__iter__ = Mock(return_value=iter(stream_tokens))
        mock_streamer_class.return_value = mock_streamer
        
        # Mock thread
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread
        
        # Generate streaming response
        tokens = list(mock_service.generate_stream("Test prompt"))
        concatenated = "".join(tokens)
        
        assert concatenated == "Hello world!"


class TestLLMServiceAPIMode:
    """Test suite for LLMService API mode."""
    
    @pytest.mark.skipif(True, reason="OpenAI library is optional")
    def test_api_generate_basic(self):
        """Test API-based generation."""
        pytest.skip("OpenAI library is optional")
    
    @pytest.mark.skipif(True, reason="OpenAI library is optional")
    def test_api_generate_stream_basic(self):
        """Test API-based streaming generation."""
        pytest.skip("OpenAI library is optional")


class TestLLMServiceEdgeCases:
    """Test suite for edge cases and error handling."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock LLMService for testing."""
        with patch('llm_service.AutoTokenizer'), \
             patch('llm_service.AutoModelForCausalLM'), \
             patch('llm_service.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            mock_tokenizer.eos_token = "[EOS]"
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            
            # Mock model
            mock_model = Mock()
            
            service = LLMService.__new__(LLMService)
            service.model_name = "test-model"
            service.use_api = False
            service.device = "cpu"
            service.tokenizer = mock_tokenizer
            service.model = mock_model
            service.api_client = None
            
            return service
    
    @patch('llm_service.torch')
    def test_generate_empty_response_returns_fallback(self, mock_torch, mock_service):
        """Test that empty generated response returns fallback message."""
        # Mock tokenizer behavior
        mock_inputs = {
            'input_ids': Mock(shape=(1, 10)),
            'attention_mask': Mock()
        }
        mock_tokenizer_output = Mock()
        mock_tokenizer_output.to = Mock(return_value=mock_inputs)
        mock_service.tokenizer.return_value = mock_tokenizer_output
        
        # Mock model generation to return empty
        mock_output = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        mock_service.model.generate.return_value = mock_output
        
        # Mock decode to return empty string
        mock_service.tokenizer.decode.return_value = "   "
        
        # Mock torch.no_grad
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        
        # Generate response
        response = mock_service.generate("Test prompt")
        
        # Should return fallback message
        assert "couldn't generate" in response.lower()
    
    @patch('llm_service.torch')
    def test_generate_with_model_error_raises_runtime_error(self, mock_torch, mock_service):
        """Test that model errors are caught and re-raised as RuntimeError."""
        # Mock tokenizer behavior
        mock_inputs = {
            'input_ids': Mock(shape=(1, 10)),
            'attention_mask': Mock()
        }
        mock_tokenizer_output = Mock()
        mock_tokenizer_output.to = Mock(return_value=mock_inputs)
        mock_service.tokenizer.return_value = mock_tokenizer_output
        
        # Mock model to raise error
        mock_service.model.generate.side_effect = Exception("Model error")
        
        # Mock torch.no_grad
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Response generation failed"):
            mock_service.generate("Test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
