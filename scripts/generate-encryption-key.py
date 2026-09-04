#!/usr/bin/env python3
"""Generate encryption key for field-level encryption.

This script generates a Fernet encryption key for use with the RAG application's
field-level encryption feature (Requirement 15.4).

Usage:
    python3 scripts/generate-encryption-key.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cryptography.fernet import Fernet


def main():
    """Generate and display a new encryption key."""
    print("=" * 70)
    print("RAG Application - Encryption Key Generator")
    print("=" * 70)
    print()
    print("Generating new Fernet encryption key...")
    print()
    
    # Generate key
    key = Fernet.generate_key().decode('ascii')
    
    print("✓ Key generated successfully!")
    print()
    print("-" * 70)
    print("Add this line to your .env file:")
    print("-" * 70)
    print()
    print(f"ENCRYPTION_KEY={key}")
    print()
    print("-" * 70)
    print()
    print("⚠️  IMPORTANT SECURITY NOTES:")
    print()
    print("1. Keep this key secure and never commit it to version control")
    print("2. Use different keys for different environments (dev, staging, prod)")
    print("3. Store production keys in a secrets management service")
    print("4. Back up the key securely - data cannot be decrypted without it")
    print("5. Rotate keys periodically (e.g., annually)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
