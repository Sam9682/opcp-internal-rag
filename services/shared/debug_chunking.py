"""
Debug script to investigate chunking coverage issue.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tiktoken
from text_preprocessor import TextPreprocessor


def test_simple_case():
    """Test a simple case to understand the issue."""
    preprocessor = TextPreprocessor()
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Create a simple test case
    content = "The quick brown fox jumps over the lazy dog. " * 20
    
    print("Original content:")
    print(f"  Length: {len(content)} chars")
    print(f"  Tokens: {len(encoding.encode(content))}")
    print(f"  Last 50 chars: ...{content[-50:]}")
    print(f"  Last 5 words: {content.split()[-5:]}")
    print()
    
    # Chunk it
    chunks = preprocessor.chunk_text(content, chunk_size=50, overlap=10)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        tokens = len(encoding.encode(chunk))
        print(f"\nChunk {i}:")
        print(f"  Tokens: {tokens}")
        print(f"  Length: {len(chunk)} chars")
        print(f"  First 50 chars: {chunk[:50]}...")
        print(f"  Last 50 chars: ...{chunk[-50:]}")
        if i == len(chunks) - 1:
            print(f"  Last 5 words: {chunk.split()[-5:]}")
    
    # Check if last words are in last chunk
    content_end_words = content.rstrip().split()[-5:]
    print(f"\nLooking for these words in last chunk: {content_end_words}")
    
    last_chunk = chunks[-1]
    for word in content_end_words:
        if len(word) > 2:
            if word in last_chunk:
                print(f"  ✓ Found '{word}' in last chunk")
            else:
                print(f"  ✗ Missing '{word}' in last chunk")


if __name__ == "__main__":
    test_simple_case()
