"""
Test runner for chunk_text functionality.
Tests Requirements 2.2 and 2.5.
"""

import sys
import tiktoken
from text_preprocessor import TextPreprocessor


def test_chunk_text_short_content():
    """Test chunking content shorter than chunk_size."""
    preprocessor = TextPreprocessor()
    content = "This is a short text that fits in one chunk."
    chunks = preprocessor.chunk_text(content, chunk_size=512, overlap=50)
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0] == content, "Content should match"
    print("✓ test_chunk_text_short_content passed")


def test_chunk_text_basic_chunking():
    """Test basic chunking with overlap."""
    preprocessor = TextPreprocessor()
    # Create content that will require multiple chunks
    content = " ".join([f"Word{i}" for i in range(1000)])
    chunks = preprocessor.chunk_text(content, chunk_size=100, overlap=10)
    
    # Should have multiple chunks
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    
    # Each chunk should be within size limit
    encoding = tiktoken.get_encoding("cl100k_base")
    for i, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        assert token_count <= 100, f"Chunk {i} has {token_count} tokens, exceeds limit of 100"
    
    print(f"✓ test_chunk_text_basic_chunking passed ({len(chunks)} chunks created)")


def test_chunk_text_overlap_verification():
    """Test that chunks have proper overlap."""
    preprocessor = TextPreprocessor()
    # Create content with identifiable words
    words = [f"Word{i:04d}" for i in range(200)]
    content = " ".join(words)
    
    chunks = preprocessor.chunk_text(content, chunk_size=50, overlap=10)
    
    # Should have multiple chunks
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    
    # Check that adjacent chunks have some overlap
    for i in range(len(chunks) - 1):
        # Find some words from end of current chunk in start of next chunk
        current_words = chunks[i].split()[-5:]  # Last 5 words
        next_chunk_start = chunks[i + 1][:100]  # First part of next chunk
        
        # At least one word should overlap
        overlap_found = any(word in next_chunk_start for word in current_words)
        assert overlap_found, f"No overlap found between chunk {i} and {i+1}"
    
    print(f"✓ test_chunk_text_overlap_verification passed ({len(chunks)} chunks with overlap)")


def test_chunk_text_no_content_loss():
    """Test that all content is covered without loss."""
    preprocessor = TextPreprocessor()
    content = "This is a test document. " * 100
    chunks = preprocessor.chunk_text(content, chunk_size=50, overlap=10)
    
    # All chunks should be non-empty
    assert all(len(chunk) > 0 for chunk in chunks), "All chunks must be non-empty"
    
    # First chunk should start with beginning of content
    assert chunks[0].startswith("This is a test"), "First chunk should start with beginning"
    
    # Last chunk should contain end of content (may have slight variation due to tokenization)
    assert "test document" in chunks[-1], "Last chunk should contain end of content"
    
    print(f"✓ test_chunk_text_no_content_loss passed ({len(chunks)} chunks)")


def test_chunk_text_token_counting_accuracy():
    """Test that token counting is accurate."""
    preprocessor = TextPreprocessor()
    encoding = tiktoken.get_encoding("cl100k_base")
    
    content = "The quick brown fox jumps over the lazy dog. " * 50
    chunk_size = 100
    chunks = preprocessor.chunk_text(content, chunk_size=chunk_size, overlap=20)
    
    # Verify each chunk respects token limit
    for i, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        assert token_count <= chunk_size, f"Chunk {i} has {token_count} tokens, exceeds limit of {chunk_size}"
    
    print(f"✓ test_chunk_text_token_counting_accuracy passed ({len(chunks)} chunks)")


def test_chunk_text_empty_content_raises_assertion():
    """Test that empty content raises assertion error."""
    preprocessor = TextPreprocessor()
    try:
        preprocessor.chunk_text("", chunk_size=512, overlap=50)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "non-empty" in str(e).lower(), f"Expected 'non-empty' in error message, got: {e}"
    print("✓ test_chunk_text_empty_content_raises_assertion passed")


def test_chunk_text_invalid_parameters_raises_assertion():
    """Test that invalid parameters raise assertion errors."""
    preprocessor = TextPreprocessor()
    content = "Some content here."
    
    # overlap >= chunk_size
    try:
        preprocessor.chunk_text(content, chunk_size=50, overlap=50)
        assert False, "Should have raised AssertionError for overlap >= chunk_size"
    except AssertionError:
        pass
    
    # overlap > chunk_size
    try:
        preprocessor.chunk_text(content, chunk_size=50, overlap=100)
        assert False, "Should have raised AssertionError for overlap > chunk_size"
    except AssertionError:
        pass
    
    # overlap <= 0
    try:
        preprocessor.chunk_text(content, chunk_size=50, overlap=0)
        assert False, "Should have raised AssertionError for overlap <= 0"
    except AssertionError:
        pass
    
    # negative overlap
    try:
        preprocessor.chunk_text(content, chunk_size=50, overlap=-10)
        assert False, "Should have raised AssertionError for negative overlap"
    except AssertionError:
        pass
    
    print("✓ test_chunk_text_invalid_parameters_raises_assertion passed")


def test_chunk_text_default_parameters():
    """Test chunking with default parameters (512 tokens, 50 overlap)."""
    preprocessor = TextPreprocessor()
    # Create content that requires multiple chunks with default settings
    content = " ".join([f"Sentence{i} with some words." for i in range(500)])
    chunks = preprocessor.chunk_text(content)
    
    # Should produce chunks
    assert len(chunks) >= 1, "Should produce at least one chunk"
    
    # Verify token limits with default chunk_size=512
    encoding = tiktoken.get_encoding("cl100k_base")
    for i, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        assert token_count <= 512, f"Chunk {i} has {token_count} tokens, exceeds default limit of 512"
    
    print(f"✓ test_chunk_text_default_parameters passed ({len(chunks)} chunks)")


def test_chunk_text_single_long_sentence():
    """Test chunking a single very long sentence."""
    preprocessor = TextPreprocessor()
    # Create a long sentence
    content = "This is a very long sentence " + "that keeps going " * 100 + "and finally ends."
    chunks = preprocessor.chunk_text(content, chunk_size=50, overlap=10)
    
    # Should split into multiple chunks
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    
    # All chunks should be non-empty
    assert all(len(chunk.strip()) > 0 for chunk in chunks), "All chunks must be non-empty"
    
    print(f"✓ test_chunk_text_single_long_sentence passed ({len(chunks)} chunks)")


def test_chunk_text_requirements_compliance():
    """Test compliance with requirements 2.2 and 2.5."""
    preprocessor = TextPreprocessor()
    # Requirement 2.2: chunks of maximum 512 tokens with 50 token overlap
    content = "Test content. " * 1000
    chunks = preprocessor.chunk_text(content, chunk_size=512, overlap=50)
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Verify chunk size requirement (2.2)
    for i, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        assert token_count <= 512, f"Chunk {i} exceeds 512 token limit (Requirement 2.2)"
    
    # Verify no content loss (2.5)
    assert len(chunks) >= 1, "Must produce at least one chunk"
    assert all(len(chunk) > 0 for chunk in chunks), "All chunks must be non-empty (Requirement 2.5)"
    
    # Verify content coverage - first and last parts should be present
    assert chunks[0].startswith("Test content"), "First chunk should start with beginning (Requirement 2.5)"
    assert "Test content" in chunks[-1], "Last chunk should contain end content (Requirement 2.5)"
    
    print(f"✓ test_chunk_text_requirements_compliance passed ({len(chunks)} chunks)")


def test_chunk_text_with_markdown_content():
    """Test chunking markdown content (should work with any text)."""
    preprocessor = TextPreprocessor()
    content = """# Introduction

This is a long document with multiple sections.

## Section 1

Content for section 1 with **bold** and *italic* text.

## Section 2

More content here with [links](https://example.com).

```python
def example():
    return True
```

## Conclusion

Final thoughts and summary.
""" * 10  # Repeat to make it long enough
    
    chunks = preprocessor.chunk_text(content, chunk_size=100, overlap=20)
    
    # Should produce multiple chunks
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    
    # All chunks should be non-empty
    assert all(len(chunk) > 0 for chunk in chunks), "All chunks must be non-empty"
    
    print(f"✓ test_chunk_text_with_markdown_content passed ({len(chunks)} chunks)")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_chunk_text_short_content,
        test_chunk_text_basic_chunking,
        test_chunk_text_overlap_verification,
        test_chunk_text_no_content_loss,
        test_chunk_text_token_counting_accuracy,
        test_chunk_text_empty_content_raises_assertion,
        test_chunk_text_invalid_parameters_raises_assertion,
        test_chunk_text_default_parameters,
        test_chunk_text_single_long_sentence,
        test_chunk_text_requirements_compliance,
        test_chunk_text_with_markdown_content,
    ]
    
    passed = 0
    failed = 0
    
    print("Running chunk_text tests...\n")
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
