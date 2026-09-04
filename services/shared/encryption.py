"""Field-level encryption utilities for sensitive data.

This module provides encryption and decryption functionality for sensitive fields
in the database, implementing encryption at rest as required by Requirement 15.4.

Uses Fernet (symmetric encryption) from the cryptography library, which provides:
- AES-128 encryption in CBC mode
- HMAC for authentication
- Timestamp for key rotation support
"""

import logging
import base64
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""
    pass


class FieldEncryption:
    """Encrypt and decrypt sensitive fields using Fernet symmetric encryption.
    
    Validates Requirement 15.4: Encrypt sensitive data at rest in the database
    
    Example:
        # Initialize with key from environment
        encryption = FieldEncryption(os.getenv('ENCRYPTION_KEY'))
        
        # Encrypt sensitive data
        encrypted = encryption.encrypt("Sensitive message")
        
        # Decrypt when needed
        decrypted = encryption.decrypt(encrypted)
    """
    
    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize field encryption with key.
        
        Args:
            encryption_key: Base64-encoded Fernet key or passphrase.
                           If None, reads from ENCRYPTION_KEY environment variable.
                           If not found, encryption is disabled (logs warning).
        
        Raises:
            EncryptionError: If key format is invalid
        """
        if encryption_key is None:
            encryption_key = os.getenv('ENCRYPTION_KEY')
        
        if not encryption_key:
            logger.warning(
                "No encryption key provided. Field encryption is DISABLED. "
                "Set ENCRYPTION_KEY environment variable to enable encryption."
            )
            self.fernet = None
            self.enabled = False
            return
        
        try:
            # Check if it's a valid Fernet key (44 characters, base64-encoded)
            if len(encryption_key) == 44 and self._is_base64(encryption_key):
                self.fernet = Fernet(encryption_key.encode())
            else:
                # Derive key from passphrase using PBKDF2
                logger.info("Deriving encryption key from passphrase")
                key = self._derive_key_from_passphrase(encryption_key)
                self.fernet = Fernet(key)
            
            self.enabled = True
            logger.info("Field encryption initialized successfully")
            
        except Exception as e:
            raise EncryptionError(f"Failed to initialize encryption: {e}")
    
    @staticmethod
    def _is_base64(s: str) -> bool:
        """Check if string is valid base64."""
        try:
            base64.urlsafe_b64decode(s)
            return True
        except Exception:
            return False
    
    @staticmethod
    def _derive_key_from_passphrase(passphrase: str, salt: Optional[bytes] = None) -> bytes:
        """Derive a Fernet key from a passphrase using PBKDF2.
        
        Args:
            passphrase: User-provided passphrase
            salt: Salt for key derivation (defaults to fixed salt)
                 In production, use unique salt per deployment
        
        Returns:
            Base64-encoded Fernet key
        """
        if salt is None:
            # Use fixed salt (should be unique per deployment in production)
            salt = b'rag-app-encryption-salt-v1'
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
        return key
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext string.
        
        Args:
            plaintext: String to encrypt
        
        Returns:
            Base64-encoded encrypted string
            If encryption is disabled, returns plaintext unchanged
        
        Raises:
            EncryptionError: If encryption fails
        
        Example:
            encrypted = encryption.encrypt("Sensitive data")
        """
        if not plaintext:
            return plaintext
        
        if not self.enabled:
            logger.debug("Encryption disabled, returning plaintext")
            return plaintext
        
        try:
            encrypted_bytes = self.fernet.encrypt(plaintext.encode('utf-8'))
            # Encode to base64 string for storage
            encrypted_str = base64.urlsafe_b64encode(encrypted_bytes).decode('ascii')
            return encrypted_str
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt data: {e}")
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext string.
        
        Args:
            ciphertext: Base64-encoded encrypted string
        
        Returns:
            Decrypted plaintext string
            If encryption is disabled, returns ciphertext unchanged
        
        Raises:
            EncryptionError: If decryption fails (wrong key, corrupted data)
        
        Example:
            decrypted = encryption.decrypt(encrypted_data)
        """
        if not ciphertext:
            return ciphertext
        
        if not self.enabled:
            logger.debug("Encryption disabled, returning ciphertext as-is")
            return ciphertext
        
        try:
            # Decode from base64 string
            encrypted_bytes = base64.urlsafe_b64decode(ciphertext.encode('ascii'))
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            plaintext = decrypted_bytes.decode('utf-8')
            return plaintext
            
        except InvalidToken:
            logger.error("Decryption failed: Invalid token (wrong key or corrupted data)")
            raise EncryptionError("Failed to decrypt data: Invalid encryption key or corrupted data")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt data: {e}")
    
    def encrypt_if_enabled(self, plaintext: Optional[str]) -> Optional[str]:
        """Encrypt plaintext if encryption is enabled, otherwise return as-is.
        
        Convenience method that handles None values and disabled encryption.
        
        Args:
            plaintext: String to encrypt or None
        
        Returns:
            Encrypted string, plaintext, or None
        """
        if plaintext is None:
            return None
        return self.encrypt(plaintext)
    
    def decrypt_if_enabled(self, ciphertext: Optional[str]) -> Optional[str]:
        """Decrypt ciphertext if encryption is enabled, otherwise return as-is.
        
        Convenience method that handles None values and disabled encryption.
        
        Args:
            ciphertext: Encrypted string or None
        
        Returns:
            Decrypted string, ciphertext, or None
        """
        if ciphertext is None:
            return None
        return self.decrypt(ciphertext)
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key.
        
        Returns:
            Base64-encoded Fernet key (44 characters)
        
        Example:
            key = FieldEncryption.generate_key()
            print(f"ENCRYPTION_KEY={key}")
        """
        return Fernet.generate_key().decode('ascii')
    
    def is_enabled(self) -> bool:
        """Check if encryption is enabled.
        
        Returns:
            True if encryption is enabled, False otherwise
        """
        return self.enabled


# Global encryption instance (initialized lazily)
_encryption_instance: Optional[FieldEncryption] = None


def get_encryption() -> FieldEncryption:
    """Get or create global encryption instance.
    
    Returns:
        FieldEncryption singleton instance
    
    Example:
        from services.shared.encryption import get_encryption
        
        encryption = get_encryption()
        encrypted = encryption.encrypt("Sensitive data")
    """
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = FieldEncryption()
    return _encryption_instance


def encrypt_field(plaintext: Optional[str]) -> Optional[str]:
    """Convenience function to encrypt a field using global encryption instance.
    
    Args:
        plaintext: String to encrypt or None
    
    Returns:
        Encrypted string or None
    
    Example:
        encrypted_content = encrypt_field(message_content)
    """
    return get_encryption().encrypt_if_enabled(plaintext)


def decrypt_field(ciphertext: Optional[str]) -> Optional[str]:
    """Convenience function to decrypt a field using global encryption instance.
    
    Args:
        ciphertext: Encrypted string or None
    
    Returns:
        Decrypted string or None
    
    Example:
        decrypted_content = decrypt_field(encrypted_content)
    """
    return get_encryption().decrypt_if_enabled(ciphertext)


if __name__ == "__main__":
    # Generate a new encryption key
    print("Generating new encryption key...")
    key = FieldEncryption.generate_key()
    print(f"\nAdd this to your .env file:")
    print(f"ENCRYPTION_KEY={key}")
    print("\n⚠️  Keep this key secure and never commit it to version control!")
