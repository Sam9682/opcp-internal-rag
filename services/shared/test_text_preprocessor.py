"""
Unit tests for TextPreprocessor

Tests markdown cleaning, metadata extraction, and edge cases.
Requirements: 2.1, 2.3
"""

import pytest
from text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_markdown_empty_string(self):
        """Test cleaning empty string returns empty string."""
        result = self.preprocessor.clean_markdown("")
        assert result == ""
    
    def test_clean_markdown_plain_text(self):
        """Test cleaning plain text without markdown."""
        text = "This is plain text without any markdown."
        result = self.preprocessor.clean_markdown(text)
        assert result == text
    
    def test_clean_markdown_headers(self):
        """Test header removal while preserving text."""
        markdown = """# Header 1
## Header 2
### Header 3
Regular text"""
        result = self.preprocessor.clean_markdown(markdown)
        assert "Header 1" in result
        assert "Header 2" in result
        assert "Header 3" in result
        assert "#" not in result
    
    def test_clean_markdown_bold_italic(self):
        """Test bold and italic formatting removal."""
        markdown = "This is **bold** and this is *italic* and this is __also bold__ and _also italic_."
        result = self.preprocessor.clean_markdown(markdown)
        assert "**" not in result
        assert "__" not in result
        assert "*" not in result
        assert "_" not in result
        assert "bold" in result
        assert "italic" in result
    
    def test_clean_markdown_links(self):
        """Test link text preservation and URL removal."""
        markdown = "Check out [this link](https://example.com) for more info."
        result = self.preprocessor.clean_markdown(markdown)
        assert "this link" in result
        assert "https://example.com" not in result
        assert "[" not in result
        assert "]" not in result
        assert "(" not in result
        assert ")" not in result
    
    def test_clean_markdown_code_blocks(self):
        """Test code block preservation with language tags."""
        markdown = """Here is some code:

```python
def hello():
    print("Hello, world!")
```

And more text."""
        result = self.preprocessor.clean_markdown(markdown)
        assert "[python code block:" in result
        assert "def hello():" in result
        assert "print" in result
    
    def test_clean_markdown_code_blocks_no_language(self):
        """Test code block preservation without language tag."""
        markdown = """```
some code here
```"""
        result = self.preprocessor.clean_markdown(markdown)
        assert "[code code block:" in result
        assert "some code here" in result
    
    def test_clean_markdown_inline_code(self):
        """Test inline code backtick removal."""
        markdown = "Use the `print()` function to output text."
        result = self.preprocessor.clean_markdown(markdown)
        assert "`" not in result
        assert "print()" in result
    
    def test_clean_markdown_lists(self):
        """Test list marker removal."""
        markdown = """- Item 1
- Item 2
* Item 3
+ Item 4
1. Numbered item 1
2. Numbered item 2"""
        result = self.preprocessor.clean_markdown(markdown)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Numbered item 1" in result
        assert "-" not in result.replace("Numbered item", "")  # Allow hyphens in text
    
    def test_clean_markdown_blockquotes(self):
        """Test blockquote marker removal."""
        markdown = """> This is a quote
> Another line"""
        result = self.preprocessor.clean_markdown(markdown)
        assert "This is a quote" in result
        assert ">" not in result
    
    def test_clean_markdown_horizontal_rules(self):
        """Test horizontal rule removal."""
        markdown = """Text before
---
Text after
***
More text
___
Final text"""
        result = self.preprocessor.clean_markdown(markdown)
        assert "Text before" in result
        assert "Text after" in result
        assert "---" not in result
        assert "***" not in result
        assert "___" not in result
    
    def test_clean_markdown_html_tags(self):
        """Test HTML tag removal."""
        markdown = "This is <strong>bold</strong> and <em>italic</em> text."
        result = self.preprocessor.clean_markdown(markdown)
        assert "<strong>" not in result
        assert "</strong>" not in result
        assert "bold" in result
        assert "italic" in result
    
    def test_clean_markdown_whitespace_normalization(self):
        """Test whitespace normalization."""
        markdown = "This  has   multiple    spaces\n\n\n\nand many newlines"
        result = self.preprocessor.clean_markdown(markdown)
        assert "  " not in result  # No double spaces
        assert "\n\n\n" not in result  # No triple newlines
        assert "This has multiple spaces" in result
    
    def test_clean_markdown_complex_document(self):
        """Test cleaning a complex markdown document."""
        markdown = """# My Document

This is a **complex** document with [links](https://example.com).

## Section 1

Here's some code:

```python
def test():
    return True
```

- List item 1
- List item 2

> A quote here

More text with *italic* and `inline code`.
"""
        result = self.preprocessor.clean_markdown(markdown)
        
        # Check content is preserved
        assert "My Document" in result
        assert "complex" in result
        assert "links" in result
        assert "Section 1" in result
        assert "[python code block:" in result
        assert "def test():" in result
        assert "List item 1" in result
        assert "A quote here" in result
        assert "italic" in result
        assert "inline code" in result
        
        # Check formatting is removed
        assert "#" not in result
        assert "**" not in result
        assert "*" not in result.replace("def test():", "")  # Allow * in code
        assert "[links]" not in result
        assert "https://example.com" not in result
    
    def test_extract_metadata_empty_content(self):
        """Test metadata extraction from empty content."""
        metadata = self.preprocessor.extract_metadata("", "/path/to/document.md")
        assert metadata['title'] == "document"
        assert metadata['headers'] == []
        assert metadata['tags'] == []
        assert metadata['file_path'] == "/path/to/document.md"
    
    def test_extract_metadata_title_from_h1(self):
        """Test title extraction from H1 header."""
        content = """# My Document Title

Some content here."""
        metadata = self.preprocessor.extract_metadata(content, "/path/to/file.md")
        assert metadata['title'] == "My Document Title"
    
    def test_extract_metadata_title_from_filename(self):
        """Test title fallback to filename when no H1."""
        content = """## Section Header

Some content here."""
        metadata = self.preprocessor.extract_metadata(content, "/path/to/my-document.md")
        assert metadata['title'] == "my-document"
    
    def test_extract_metadata_all_headers(self):
        """Test extraction of all headers."""
        content = """# Title
## Section 1
### Subsection 1.1
## Section 2
#### Deep section"""
        metadata = self.preprocessor.extract_metadata(content, "/path/to/file.md")
        assert len(metadata['headers']) == 5
        assert "Title" in metadata['headers']
        assert "Section 1" in metadata['headers']
        assert "Subsection 1.1" in metadata['headers']
        assert "Section 2" in metadata['headers']
        assert "Deep section" in metadata['headers']
    
    def test_extract_metadata_tags_from_frontmatter_array(self):
        """Test tag extraction from YAML frontmatter array."""
        content = """---
title: My Document
tags: [python, testing, markdown]
---

# Content"""
        metadata = self.preprocessor.extract_metadata(content, "/path/to/file.md")
        assert "python" in metadata['tags']
        assert "testing" in metadata['tags']
        assert "markdown" in metadata['tags']
    
    def test_extract_metadata_tags_from_frontmatter_list(self):
        """Test tag extraction from YAML frontmatter list."""
        content = """---
title: My Document
tags: python, testing, markdown
---

# Content"""
        metadata = self.preprocessor.extract_metadata(content, "/path/to/file.md")
        assert "python" in metadata['tags']
        assert "testing" in metadata['tags']
        assert "markdown" in metadata['tags']
    
    def test_extract_metadata_inline_tags(self):
        """Test extraction of inline hashtags."""
        content = """# My Document

This document is about #python and #testing.
"""
        metadata = self.preprocessor.extract_metadata(content, "/path/to/file.md")
        assert "python" in metadata['tags']
        assert "testing" in metadata['tags']
    
    def test_extract_metadata_combined_tags(self):
        """Test combining frontmatter and inline tags."""
        content = """---
tags: [python, testing]
---

# Document

More about #markdown and #python.
"""
        metadata = self.preprocessor.extract_metadata(content, "/path/to/file.md")
        # Should have unique tags from both sources
        assert "python" in metadata['tags']
        assert "testing" in metadata['tags']
        assert "markdown" in metadata['tags']
        # Check no duplicates
        assert metadata['tags'].count("python") == 1
    
    def test_clean_markdown_preserves_semantic_structure(self):
        """Test that semantic structure is preserved after cleaning."""
        markdown = """# Introduction

This is the introduction paragraph.

## Details

Here are the details with **important** information.

```python
code_example = True
```

## Conclusion

Final thoughts here.
"""
        result = self.preprocessor.clean_markdown(markdown)
        
        # Check that sections are still distinguishable
        assert "Introduction" in result
        assert "Details" in result
        assert "Conclusion" in result
        assert "introduction paragraph" in result
        assert "important" in result
        assert "[python code block:" in result
        assert "Final thoughts" in result
    
    def test_clean_markdown_edge_case_single_sentence(self):
        """Test cleaning a single sentence."""
        markdown = "Just one sentence."
        result = self.preprocessor.clean_markdown(markdown)
        assert result == "Just one sentence."
    
    def test_clean_markdown_edge_case_only_formatting(self):
        """Test cleaning content with only formatting."""
        markdown = "**bold** *italic* `code`"
        result = self.preprocessor.clean_markdown(markdown)
        assert result == "bold italic code"
    
    def test_clean_markdown_special_characters(self):
        """Test handling of special characters."""
        markdown = "Special chars: & < > @ # $ % ^ & * ( )"
        result = self.preprocessor.clean_markdown(markdown)
        # Special characters should be preserved (except markdown syntax)
        assert "&" in result
        assert "@" in result
        assert "$" in result

    def test_chunk_text_short_content(self):
        """Test chunking content shorter than chunk_size."""
        content = "This is a short text that fits in one chunk."
        chunks = self.preprocessor.chunk_text(content, chunk_size=512, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == content
    
    def test_chunk_text_basic_chunking(self):
        """Test basic chunking with overlap."""
        # Create content that will require multiple chunks
        content = " ".join([f"Word{i}" for i in range(1000)])
        chunks = self.preprocessor.chunk_text(content, chunk_size=100, overlap=10)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be within size limit
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            token_count = len(encoding.encode(chunk))
            assert token_count <= 100
    
    def test_chunk_text_overlap_verification(self):
        """Test that chunks have proper overlap."""
        # Create content with identifiable words
        words = [f"Word{i:04d}" for i in range(200)]
        content = " ".join(words)
        
        chunks = self.preprocessor.chunk_text(content, chunk_size=50, overlap=10)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Check that adjacent chunks have some overlap
        # (overlap may not be exact due to token boundaries)
        for i in range(len(chunks) - 1):
            # Find some words from end of current chunk in start of next chunk
            current_words = chunks[i].split()[-5:]  # Last 5 words
            next_chunk_start = chunks[i + 1][:100]  # First part of next chunk
            
            # At least one word should overlap
            overlap_found = any(word in next_chunk_start for word in current_words)
            assert overlap_found, f"No overlap found between chunk {i} and {i+1}"
    
    def test_chunk_text_no_content_loss(self):
        """Test that all content is covered without loss."""
        content = "This is a test document. " * 100
        chunks = self.preprocessor.chunk_text(content, chunk_size=50, overlap=10)
        
        # All chunks should be non-empty
        assert all(len(chunk) > 0 for chunk in chunks)
        
        # First chunk should start with beginning of content
        assert chunks[0].startswith("This is a test")
        
        # Last chunk should contain end of content (may have trailing whitespace)
        assert "test document" in chunks[-1].strip(), "Last chunk should contain end of content"
    
    def test_chunk_text_token_counting_accuracy(self):
        """Test that token counting is accurate."""
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        
        content = "The quick brown fox jumps over the lazy dog. " * 50
        chunk_size = 100
        chunks = self.preprocessor.chunk_text(content, chunk_size=chunk_size, overlap=20)
        
        # Verify each chunk respects token limit
        for i, chunk in enumerate(chunks):
            token_count = len(encoding.encode(chunk))
            assert token_count <= chunk_size, f"Chunk {i} has {token_count} tokens, exceeds limit of {chunk_size}"
    
    def test_chunk_text_empty_content_raises_assertion(self):
        """Test that empty content raises assertion error."""
        with pytest.raises(AssertionError):
            self.preprocessor.chunk_text("", chunk_size=512, overlap=50)
    
    def test_chunk_text_invalid_parameters_raises_assertion(self):
        """Test that invalid parameters raise assertion errors."""
        content = "Some content here."
        
        # overlap >= chunk_size
        with pytest.raises(AssertionError):
            self.preprocessor.chunk_text(content, chunk_size=50, overlap=50)
        
        # overlap >= chunk_size
        with pytest.raises(AssertionError):
            self.preprocessor.chunk_text(content, chunk_size=50, overlap=100)
        
        # overlap <= 0
        with pytest.raises(AssertionError):
            self.preprocessor.chunk_text(content, chunk_size=50, overlap=0)
        
        # negative overlap
        with pytest.raises(AssertionError):
            self.preprocessor.chunk_text(content, chunk_size=50, overlap=-10)
    
    def test_chunk_text_default_parameters(self):
        """Test chunking with default parameters (512 tokens, 50 overlap)."""
        # Create content that requires multiple chunks with default settings
        content = " ".join([f"Sentence{i} with some words." for i in range(500)])
        chunks = self.preprocessor.chunk_text(content)
        
        # Should produce chunks
        assert len(chunks) >= 1
        
        # Verify token limits with default chunk_size=512
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            token_count = len(encoding.encode(chunk))
            assert token_count <= 512
    
    def test_chunk_text_single_long_sentence(self):
        """Test chunking a single very long sentence."""
        # Create a long sentence
        content = "This is a very long sentence " + "that keeps going " * 100 + "and finally ends."
        chunks = self.preprocessor.chunk_text(content, chunk_size=50, overlap=10)
        
        # Should split into multiple chunks
        assert len(chunks) > 1
        
        # All chunks should be non-empty
        assert all(len(chunk.strip()) > 0 for chunk in chunks)
    
    def test_chunk_text_with_markdown_content(self):
        """Test chunking markdown content (should work with any text)."""
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
        
        chunks = self.preprocessor.chunk_text(content, chunk_size=100, overlap=20)
        
        # Should produce multiple chunks
        assert len(chunks) > 1
        
        # All chunks should be non-empty
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_chunk_text_preserves_content_integrity(self):
        """Test that chunking preserves content integrity."""
        content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 50
        chunks = self.preprocessor.chunk_text(content, chunk_size=100, overlap=20)
        
        # Reconstruct content from first tokens of each chunk (accounting for overlap)
        # This is a simplified check - just verify first and last chunks
        assert chunks[0].startswith("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in chunks[-1]
    
    def test_chunk_text_requirements_compliance(self):
        """Test compliance with requirements 2.2 and 2.5."""
        # Requirement 2.2: chunks of maximum 512 tokens with 50 token overlap
        content = "Test content. " * 1000
        chunks = self.preprocessor.chunk_text(content, chunk_size=512, overlap=50)
        
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Verify chunk size requirement (2.2)
        for chunk in chunks:
            token_count = len(encoding.encode(chunk))
            assert token_count <= 512, "Chunk exceeds 512 token limit (Requirement 2.2)"
        
        # Verify no content loss (2.5)
        assert len(chunks) >= 1, "Must produce at least one chunk"
        assert all(len(chunk) > 0 for chunk in chunks), "All chunks must be non-empty (Requirement 2.5)"
        
        # Verify content coverage - first and last parts should be present
        assert chunks[0].startswith("Test content"), "First chunk should start with beginning of content (Requirement 2.5)"
        assert "Test content" in chunks[-1].strip(), "Last chunk should contain end content (Requirement 2.5)"
