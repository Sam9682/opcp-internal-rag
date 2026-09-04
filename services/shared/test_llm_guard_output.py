"""
Unit tests for LLMGuardService check_output() method

Tests cover:
- Harmful content detection in output (Requirement 8.1, 8.2)
- System prompt leakage detection (Requirement 8.3)
- Safe output validation (Requirement 8.4)
- Security event logging (Requirement 8.5)
"""

import pytest
import logging
from llm_guard_service import LLMGuardService


class TestCheckOutputToxicContent:
    """Test toxic content detection in output (Requirements 8.1, 8.2)"""
    
    def test_hate_speech_in_output(self):
        """Should detect hate speech in LLM output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("I think we should hate all developers")
        
        assert is_safe is False
        assert "toxic" in reason.lower()
    
    def test_violent_language_in_output(self):
        """Should detect violent language in output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("You should kill all the bugs in your code")
        
        assert is_safe is False
        assert "toxic" in reason.lower()
    
    def test_profanity_in_output(self):
        """Should detect profanity in output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("This is shit and broken")
        
        assert is_safe is False
        assert "toxic" in reason.lower()
    
    def test_insults_in_output(self):
        """Should detect insults in output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("That's a stupid question")
        
        assert is_safe is False
        assert "toxic" in reason.lower()
    
    def test_multiple_toxic_patterns_in_output(self):
        """Should detect multiple toxic patterns in output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("You idiot, go to hell")
        
        assert is_safe is False
        assert "toxic" in reason.lower()


class TestCheckOutputSystemLeakage:
    """Test system prompt leakage detection (Requirement 8.3)"""
    
    def test_ai_identity_leak(self):
        """Should detect AI identity disclosure"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("I am an AI assistant designed to help you")
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()
    
    def test_chatbot_identity_leak(self):
        """Should detect chatbot identity disclosure"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("I am a chatbot designed to help you")
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()
    
    def test_language_model_identity_leak(self):
        """Should detect language model identity disclosure"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("You are a language model trained by researchers")
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()
    
    def test_instructions_reference_leak(self):
        """Should detect references to system instructions"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("According to your instructions, I should not reveal this")
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()
    
    def test_system_prompt_reference_leak(self):
        """Should detect references to system prompt"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("Your system prompt tells me to be helpful")
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()
    
    def test_guidelines_reference_leak(self):
        """Should detect references to guidelines"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("Based on your guidelines, I cannot do that")
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()
    
    def test_case_insensitive_leak_detection(self):
        """Should detect leakage regardless of case"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("I AM AN AI ASSISTANT")
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()


class TestCheckOutputSafeContent:
    """Test safe output validation (Requirement 8.4)"""
    
    def test_normal_answer(self):
        """Should allow normal helpful answers"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "To configure authentication, you need to set up JWT tokens in your application."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_technical_explanation(self):
        """Should allow technical explanations"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "JWT tokens consist of three parts: header, payload, and signature. "
            "They are encoded in Base64 and separated by dots."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_code_example(self):
        """Should allow code examples"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "Here's an example:\n\ndef authenticate(token):\n    return verify_jwt(token)"
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_empty_output(self):
        """Should allow empty output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("")
        
        assert is_safe is True
        assert reason == ""
    
    def test_whitespace_only_output(self):
        """Should allow whitespace-only output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("   \n\t  ")
        
        assert is_safe is True
        assert reason == ""
    
    def test_long_safe_answer(self):
        """Should allow long safe answers"""
        service = LLMGuardService()
        long_answer = "Authentication is important. " * 100
        is_safe, reason = service.check_output(long_answer)
        
        assert is_safe is True
        assert reason == ""
    
    def test_answer_with_sources(self):
        """Should allow answers with source citations"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "According to the documentation, you should use HTTPS for secure connections. "
            "See section 3.2 for more details."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_answer_with_numbers(self):
        """Should allow answers with numbers"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "The default port for HTTPS is 443, while HTTP uses port 80."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_answer_with_urls(self):
        """Should allow answers with URLs"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "You can find more information at https://example.com/docs"
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_polite_refusal(self):
        """Should allow polite refusals"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "I don't have enough information to answer that question accurately."
        )
        
        assert is_safe is True
        assert reason == ""


class TestCheckOutputEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_long_output(self):
        """Should handle very long output"""
        service = LLMGuardService()
        long_output = "This is a safe answer. " * 1000
        is_safe, reason = service.check_output(long_output)
        
        # Should not crash
        assert isinstance(is_safe, bool)
        assert isinstance(reason, str)
    
    def test_unicode_characters_in_output(self):
        """Should handle unicode characters"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "L'authentification est configurée. 认证已配置。"
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_special_characters_in_output(self):
        """Should handle special characters"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "Use symbols like @, #, $, %, ^, &, * in your password."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_mixed_safe_and_unsafe_output(self):
        """Should detect unsafe content even when mixed with safe content"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "Here's how to configure auth. By the way, I am an AI assistant."
        )
        
        assert is_safe is False
        assert "system prompt leakage" in reason.lower() or "sensitive" in reason.lower()
    
    def test_markdown_formatting(self):
        """Should allow markdown formatting"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "# Authentication\n\n**Important**: Use strong passwords.\n\n- Step 1\n- Step 2"
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_json_output(self):
        """Should allow JSON formatted output"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            '{"status": "success", "message": "Authentication configured"}'
        )
        
        assert is_safe is True
        assert reason == ""


class TestCheckOutputLogging:
    """Test security event logging (Requirement 8.5)"""
    
    def test_logs_toxic_output(self, caplog):
        """Should log toxic content in output"""
        service = LLMGuardService()
        
        with caplog.at_level(logging.WARNING):
            service.check_output("This is a stupid answer")
        
        # Check that security event was logged
        assert any("SECURITY_EVENT" in record.message for record in caplog.records)
        assert any("toxic_output" in record.message for record in caplog.records)
    
    def test_logs_system_leakage(self, caplog):
        """Should log system prompt leakage"""
        service = LLMGuardService()
        
        with caplog.at_level(logging.WARNING):
            service.check_output("I am an AI assistant")
        
        assert any("SECURITY_EVENT" in record.message for record in caplog.records)
        assert any("sensitive_leak" in record.message for record in caplog.records)
    
    def test_truncates_long_output_in_logs(self, caplog):
        """Should truncate long output in security logs"""
        service = LLMGuardService()
        long_output = "I am an AI assistant " + "a" * 200
        
        with caplog.at_level(logging.WARNING):
            service.check_output(long_output)
        
        # Check that logged content is truncated
        security_logs = [r.message for r in caplog.records if "SECURITY_EVENT" in r.message]
        assert len(security_logs) > 0
        # The full long output should not be in the log
        assert long_output not in security_logs[0]
    
    def test_no_logging_for_safe_output(self, caplog):
        """Should not log security events for safe output"""
        service = LLMGuardService()
        
        with caplog.at_level(logging.WARNING):
            service.check_output("This is a safe and helpful answer")
        
        # Should not have any security event logs
        security_logs = [r for r in caplog.records if "SECURITY_EVENT" in r.message]
        assert len(security_logs) == 0


class TestCheckOutputReturnFormat:
    """Test return value format and consistency"""
    
    def test_returns_tuple(self):
        """Should always return a tuple"""
        service = LLMGuardService()
        result = service.check_output("test output")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_safe_returns_true_empty_string(self):
        """Safe output should return (True, '')"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("This is a safe answer")
        
        assert is_safe is True
        assert reason == ""
    
    def test_unsafe_returns_false_with_reason(self):
        """Unsafe output should return (False, reason)"""
        service = LLMGuardService()
        is_safe, reason = service.check_output("I am an AI assistant")
        
        assert is_safe is False
        assert isinstance(reason, str)
        assert len(reason) > 0
    
    def test_reason_is_descriptive(self):
        """Reason should describe the violation"""
        service = LLMGuardService()
        _, reason = service.check_output("I am an AI assistant")
        
        assert len(reason) > 0
        assert ("system prompt leakage" in reason.lower() or "sensitive" in reason.lower())


class TestCheckOutputVsCheckInput:
    """Test differences between check_output and check_input"""
    
    def test_output_does_not_check_prompt_injection(self):
        """Output validation should not check for prompt injection patterns"""
        service = LLMGuardService()
        # This would fail input check but should pass output check
        is_safe, reason = service.check_output(
            "To ignore previous instructions in your code, use a comment."
        )
        
        # Should be safe because it's explaining code, not attempting injection
        # The pattern "ignore previous instructions" is in a safe context
        assert is_safe is True
        assert reason == ""
    
    def test_output_checks_system_leakage_not_input(self):
        """Output should check for system leakage, input should not"""
        service = LLMGuardService()
        
        # Input check doesn't care about AI identity
        input_safe, _ = service.check_input("Are you an AI?")
        assert input_safe is True
        
        # Output check should detect AI identity disclosure
        output_safe, output_reason = service.check_output("Yes, I am an AI assistant")
        assert output_safe is False
        assert "system prompt leakage" in output_reason.lower() or "sensitive" in output_reason.lower()
    
    def test_both_check_toxic_content(self):
        """Both input and output should check for toxic content"""
        service = LLMGuardService()
        
        input_safe, input_reason = service.check_input("you are stupid")
        output_safe, output_reason = service.check_output("you are stupid")
        
        assert input_safe is False
        assert output_safe is False
        assert "toxic" in input_reason.lower()
        assert "toxic" in output_reason.lower()


class TestCheckOutputIntegration:
    """Integration tests for check_output in RAG context"""
    
    def test_typical_rag_answer(self):
        """Should allow typical RAG-style answers"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "Based on the documentation, to configure authentication you need to:\n\n"
            "1. Set up JWT tokens\n"
            "2. Configure the secret key\n"
            "3. Enable HTTPS\n\n"
            "For more details, see the authentication guide in section 4.2."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_answer_with_uncertainty(self):
        """Should allow answers expressing uncertainty"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "I couldn't find specific information about that in the documentation. "
            "You might want to check the official API reference."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_answer_with_multiple_sources(self):
        """Should allow answers citing multiple sources"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "According to the setup guide (section 2.1) and the security best practices "
            "(section 5.3), you should use environment variables for sensitive configuration."
        )
        
        assert is_safe is True
        assert reason == ""
    
    def test_technical_troubleshooting_answer(self):
        """Should allow technical troubleshooting answers"""
        service = LLMGuardService()
        is_safe, reason = service.check_output(
            "If you're seeing a 401 error, it usually means the authentication token is "
            "invalid or expired. Try regenerating the token and ensure it's properly "
            "included in the Authorization header."
        )
        
        assert is_safe is True
        assert reason == ""
