"""
LLM Guard Service for RAG Application

This module provides safety validation functionality for user inputs and LLM outputs
to detect prompt injection, toxic content, and PII.

Requirements: 7.1, 7.2, 7.3, 8.1, 8.2, 8.3, 8.4, 8.5
"""

from typing import Tuple, List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)


class LLMGuardService:
    """
    LLM Guard service for validating input/output safety.
    
    This class handles:
    - Prompt injection detection in user inputs
    - Toxic content filtering for inputs and outputs
    - PII (Personally Identifiable Information) detection and sanitization
    - System prompt leakage detection in outputs
    - Security event logging
    
    This implementation uses pattern-based detection for safety checks.
    For production use, consider integrating with specialized libraries
    like llm-guard, detoxify, or presidio for more robust detection.
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    # Prompt injection patterns to detect
    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+(instructions|prompts|rules)",
        r"disregard\s+(previous|above|all)\s+(instructions|prompts|rules)",
        r"forget\s+(previous|above|all)\s+(instructions|prompts|rules)",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"###\s*Instruction",
        r"###\s*System",
    ]
    
    # Toxic content patterns (basic detection)
    TOXIC_PATTERNS = [
        r"\b(hate|kill|murder|attack|destroy)\s+(all|every|the)\s+\w+",
        r"\b(stupid|idiot|moron|dumb)\b",
        r"\b(fuck|shit|damn|hell|ass)\b",
    ]
    
    # PII patterns
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }
    
    # System prompt leakage patterns
    SYSTEM_LEAK_PATTERNS = [
        r"you\s+are\s+an?\s+(AI|assistant|chatbot|language\s+model)",
        r"your\s+(instructions|system\s+prompt|guidelines)",
        r"I\s+am\s+an?\s+(AI|assistant|chatbot|language\s+model)",
    ]
    
    def __init__(
        self,
        prompt_injection_threshold: float = 0.5,
        toxicity_threshold: float = 0.7,
        enable_pii_detection: bool = True
    ):
        """
        Initialize the LLMGuardService with safety scanners.
        
        Sets up pattern-based detection for:
        - Prompt injection detection
        - Toxicity filtering
        - PII detection and anonymization
        
        Preconditions:
        - Thresholds are between 0.0 and 1.0
        
        Postconditions:
        - All detection patterns are compiled
        - Service is ready for safety checks
        
        Args:
            prompt_injection_threshold: Threshold for prompt injection detection (0.0-1.0)
            toxicity_threshold: Threshold for toxicity detection (0.0-1.0)
            enable_pii_detection: Whether to enable PII detection and sanitization
            
        Requirements: 7.1, 7.2, 7.3
        """
        logger.info("Initializing LLMGuardService with pattern-based safety scanners")
        
        # Validate thresholds
        if not 0.0 <= prompt_injection_threshold <= 1.0:
            raise ValueError("prompt_injection_threshold must be between 0.0 and 1.0")
        if not 0.0 <= toxicity_threshold <= 1.0:
            raise ValueError("toxicity_threshold must be between 0.0 and 1.0")
        
        self.prompt_injection_threshold = prompt_injection_threshold
        self.toxicity_threshold = toxicity_threshold
        self.enable_pii_detection = enable_pii_detection
        
        # Compile regex patterns for efficiency
        self._prompt_injection_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.PROMPT_INJECTION_PATTERNS
        ]
        self._toxic_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.TOXIC_PATTERNS
        ]
        self._pii_regex = {
            name: re.compile(pattern, re.IGNORECASE) 
            for name, pattern in self.PII_PATTERNS.items()
        }
        self._system_leak_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SYSTEM_LEAK_PATTERNS
        ]
        
        logger.info(f"Initialized with {len(self._prompt_injection_regex)} prompt injection patterns")
        logger.info(f"Initialized with {len(self._toxic_regex)} toxicity patterns")
        logger.info(f"Initialized with {len(self._pii_regex)} PII patterns")
        logger.info(f"PII detection: {'enabled' if enable_pii_detection else 'disabled'}")
        logger.info("LLMGuardService initialization complete")
    
    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate user input for safety concerns.
        
        Checks input text for:
        - Prompt injection attempts
        - Toxic or harmful content
        - Personally identifiable information (PII)
        
        If any check fails, returns False with a reason. If all checks pass,
        returns True with an empty reason string.
        
        Preconditions:
        - text is valid string (may be empty)
        - Safety scanners are initialized
        
        Postconditions:
        - Returns tuple (is_safe: bool, reason: str)
        - If is_safe is True, reason is empty string
        - If is_safe is False, reason contains specific violation
        - No false negatives for known attack patterns
        - Security events are logged
        
        Args:
            text: User input text to validate
            
        Returns:
            Tuple of (is_safe, reason) where:
            - is_safe: True if input passes all checks, False otherwise
            - reason: Empty string if safe, otherwise description of violation
            
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        # Handle empty input
        if not text or not text.strip():
            logger.debug("Empty input received")
            return True, ""  # Empty input is considered safe
        
        try:
            # Step 1: Check for prompt injection
            logger.debug("Checking for prompt injection...")
            for pattern in self._prompt_injection_regex:
                if pattern.search(text):
                    reason = f"Prompt injection detected: pattern matched"
                    logger.warning(f"Input rejected: {reason}")
                    self._log_security_event("prompt_injection", text, reason)
                    return False, reason
            
            # Step 2: Check for toxic content
            logger.debug("Checking for toxic content...")
            toxic_matches = []
            for pattern in self._toxic_regex:
                match = pattern.search(text)
                if match:
                    toxic_matches.append(match.group(0))
            
            if toxic_matches:
                reason = f"Toxic content detected: {len(toxic_matches)} pattern(s) matched"
                logger.warning(f"Input rejected: {reason}")
                self._log_security_event("toxic_content", text, reason)
                return False, reason
            
            # Step 3: Check for PII (if enabled)
            if self.enable_pii_detection:
                logger.debug("Checking for PII...")
                pii_found = []
                for pii_type, pattern in self._pii_regex.items():
                    if pattern.search(text):
                        pii_found.append(pii_type)
                
                if pii_found:
                    reason = f"PII detected: {', '.join(pii_found)}"
                    logger.warning(f"Input rejected: {reason}")
                    self._log_security_event("pii_detected", text, reason)
                    return False, reason
            
            # All checks passed
            logger.debug("Input passed all safety checks")
            return True, ""
            
        except Exception as e:
            # Log error but allow input to proceed (fail open for availability)
            logger.error(f"Error during input safety check: {e}", exc_info=True)
            return True, ""
    
    def check_output(self, text: str) -> Tuple[bool, str]:
        """
        Validate LLM output for safety concerns.
        
        Checks output text for:
        - Toxic or harmful content
        - System prompt leakage or internal information
        - Inappropriate responses
        
        If any check fails, returns False with a reason. If all checks pass,
        returns True with an empty reason string.
        
        Preconditions:
        - text is valid string (may be empty)
        - Safety scanners are initialized
        
        Postconditions:
        - Returns tuple (is_safe: bool, reason: str)
        - If is_safe is True, reason is empty string
        - If is_safe is False, reason contains specific violation
        - Security events are logged
        
        Args:
            text: LLM output text to validate
            
        Returns:
            Tuple of (is_safe, reason) where:
            - is_safe: True if output passes all checks, False otherwise
            - reason: Empty string if safe, otherwise description of violation
            
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Handle empty output
        if not text or not text.strip():
            logger.debug("Empty output received")
            return True, ""  # Empty output is considered safe
        
        try:
            # Step 1: Check for toxic content in output
            logger.debug("Checking output for toxic content...")
            toxic_matches = []
            for pattern in self._toxic_regex:
                match = pattern.search(text)
                if match:
                    toxic_matches.append(match.group(0))
            
            if toxic_matches:
                reason = f"Toxic content in output: {len(toxic_matches)} pattern(s) matched"
                logger.warning(f"Output rejected: {reason}")
                self._log_security_event("toxic_output", text, reason)
                return False, reason
            
            # Step 2: Check for system prompt leakage
            logger.debug("Checking for system prompt leakage...")
            for pattern in self._system_leak_regex:
                if pattern.search(text):
                    reason = "System prompt leakage detected"
                    logger.warning(f"Output rejected: {reason}")
                    self._log_security_event("sensitive_leak", text, reason)
                    return False, reason
            
            # All checks passed
            logger.debug("Output passed all safety checks")
            return True, ""
            
        except Exception as e:
            # Log error but allow output to proceed (fail open for availability)
            logger.error(f"Error during output safety check: {e}", exc_info=True)
            return True, ""
    
    def sanitize_text(self, text: str) -> str:
        """
        Remove or mask sensitive information from text.
        
        Uses PII detection to identify and anonymize sensitive information
        such as emails, phone numbers, SSNs, credit cards, etc.
        
        Preconditions:
        - text is valid string (may be empty)
        - PII patterns are initialized
        
        Postconditions:
        - Returns sanitized text with PII masked or removed
        - Original text structure is preserved
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text with PII masked
            
        Requirements: 7.3
        """
        if not text or not text.strip():
            return text
        
        if not self.enable_pii_detection:
            logger.warning("PII detection is disabled, returning original text")
            return text
        
        try:
            logger.debug("Sanitizing text for PII...")
            sanitized = text
            
            # Replace each type of PII with masked placeholder
            for pii_type, pattern in self._pii_regex.items():
                matches = pattern.findall(sanitized)
                if matches:
                    logger.info(f"Found {len(matches)} {pii_type} instance(s), masking...")
                    # Replace with placeholder
                    placeholder = f"[{pii_type.upper()}_REDACTED]"
                    sanitized = pattern.sub(placeholder, sanitized)
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error during text sanitization: {e}", exc_info=True)
            return text  # Return original text on error
    
    def _log_security_event(self, event_type: str, content: str, reason: str) -> None:
        """
        Log security event for audit purposes.
        
        Logs security events separately from normal application logs for
        security monitoring and audit purposes.
        
        Args:
            event_type: Type of security event (e.g., 'prompt_injection', 'toxic_content')
            content: The content that triggered the event (truncated for logging)
            reason: Reason for the security event
            
        Requirements: 7.5, 8.5, 15.5
        """
        # Truncate content for logging (don't log full potentially malicious content)
        truncated_content = content[:100] + "..." if len(content) > 100 else content
        
        logger.warning(
            f"SECURITY_EVENT: type={event_type}, reason={reason}, "
            f"content_preview={truncated_content}"
        )
