"""
Simple test runner for TextPreprocessor without pytest dependency.
"""

import sys
from text_preprocessor import TextPreprocessor


def test_clean_markdown_empty_string():
    """Test cleaning empty string returns empty string."""
    preprocessor = TextPreprocessor()
    result = preprocessor.clean_markdown("")
    assert result == "", f"Expected empty string, got: {result}"
    print("✓ test_clean_markdown_empty_string passed")


def test_clean_markdown_plain_text():
    """Test cleaning plain text without markdown."""
    preprocessor = TextPreprocessor()
    text = "This is plain text without any markdown."
    result = preprocessor.clean_markdown(text)
    assert result == text, f"Expected: {text}, got: {result}"
    print("✓ test_clean_markdown_plain_text passed")


def test_clean_markdown_headers():
    """Test header removal while preserving text."""
    preprocessor = TextPreprocessor()
    markdown = """# Header 1
## Header 2
### Header 3
Regular text"""
    result = preprocessor.clean_markdown(markdown)
    assert "Header 1" in result, "Header 1 not found"
    assert "Header 2" in result, "Header 2 not found"
    assert "Header 3" in result, "Header 3 not found"
    assert "#" not in result, "# symbols should be removed"
    print("✓ test_clean_markdown_headers passed")


def test_clean_markdown_bold_italic():
    """Test bold and italic formatting removal."""
    preprocessor = TextPreprocessor()
    markdown = "This is **bold** and this is *italic* and this is __also bold__ and _also italic_."
    result = preprocessor.clean_markdown(markdown)
    assert "**" not in result, "** should be removed"
    assert "__" not in result, "__ should be removed"
    assert "bold" in result, "bold text should be preserved"
    assert "italic" in result, "italic text should be preserved"
    print("✓ test_clean_markdown_bold_italic passed")


def test_clean_markdown_links():
    """Test link text preservation and URL removal."""
    preprocessor = TextPreprocessor()
    markdown = "Check out [this link](https://example.com) for more info."
    result = preprocessor.clean_markdown(markdown)
    assert "this link" in result, "Link text should be preserved"
    assert "https://example.com" not in result, "URL should be removed"
    assert "[" not in result, "[ should be removed"
    assert "]" not in result, "] should be removed"
    print("✓ test_clean_markdown_links passed")


def test_clean_markdown_code_blocks():
    """Test code block preservation with language tags."""
    preprocessor = TextPreprocessor()
    markdown = """Here is some code:

```python
def hello():
    print("Hello, world!")
```

And more text."""
    result = preprocessor.clean_markdown(markdown)
    assert "[python code block:" in result, "Code block marker not found"
    assert "def hello():" in result, "Code content should be preserved"
    assert "print" in result, "Code content should be preserved"
    print("✓ test_clean_markdown_code_blocks passed")


def test_clean_markdown_lists():
    """Test list marker removal."""
    preprocessor = TextPreprocessor()
    markdown = """- Item 1
- Item 2
* Item 3
+ Item 4
1. Numbered item 1
2. Numbered item 2"""
    result = preprocessor.clean_markdown(markdown)
    assert "Item 1" in result, "List item should be preserved"
    assert "Item 2" in result, "List item should be preserved"
    assert "Numbered item 1" in result, "Numbered item should be preserved"
    print("✓ test_clean_markdown_lists passed")


def test_clean_markdown_complex_document():
    """Test cleaning a complex markdown document."""
    preprocessor = TextPreprocessor()
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
    result = preprocessor.clean_markdown(markdown)
    
    # Check content is preserved
    assert "My Document" in result, "Title should be preserved"
    assert "complex" in result, "Text should be preserved"
    assert "links" in result, "Link text should be preserved"
    assert "Section 1" in result, "Section header should be preserved"
    assert "[python code block:" in result, "Code block should be preserved"
    assert "def test():" in result, "Code content should be preserved"
    assert "List item 1" in result, "List items should be preserved"
    assert "A quote here" in result, "Quote should be preserved"
    assert "italic" in result, "Text should be preserved"
    assert "inline code" in result, "Inline code should be preserved"
    
    # Check formatting is removed
    assert "#" not in result, "# should be removed"
    assert "**" not in result, "** should be removed"
    assert "https://example.com" not in result, "URL should be removed"
    print("✓ test_clean_markdown_complex_document passed")


def test_extract_metadata_empty_content():
    """Test metadata extraction from empty content."""
    preprocessor = TextPreprocessor()
    metadata = preprocessor.extract_metadata("", "/path/to/document.md")
    assert metadata['title'] == "document", f"Expected 'document', got: {metadata['title']}"
    assert metadata['headers'] == [], "Headers should be empty"
    assert metadata['tags'] == [], "Tags should be empty"
    assert metadata['file_path'] == "/path/to/document.md", "File path should match"
    print("✓ test_extract_metadata_empty_content passed")


def test_extract_metadata_title_from_h1():
    """Test title extraction from H1 header."""
    preprocessor = TextPreprocessor()
    content = """# My Document Title

Some content here."""
    metadata = preprocessor.extract_metadata(content, "/path/to/file.md")
    assert metadata['title'] == "My Document Title", f"Expected 'My Document Title', got: {metadata['title']}"
    print("✓ test_extract_metadata_title_from_h1 passed")


def test_extract_metadata_all_headers():
    """Test extraction of all headers."""
    preprocessor = TextPreprocessor()
    content = """# Title
## Section 1
### Subsection 1.1
## Section 2
#### Deep section"""
    metadata = preprocessor.extract_metadata(content, "/path/to/file.md")
    assert len(metadata['headers']) == 5, f"Expected 5 headers, got: {len(metadata['headers'])}"
    assert "Title" in metadata['headers'], "Title should be in headers"
    assert "Section 1" in metadata['headers'], "Section 1 should be in headers"
    assert "Subsection 1.1" in metadata['headers'], "Subsection 1.1 should be in headers"
    print("✓ test_extract_metadata_all_headers passed")


def test_extract_metadata_tags_from_frontmatter():
    """Test tag extraction from YAML frontmatter."""
    preprocessor = TextPreprocessor()
    content = """---
title: My Document
tags: [python, testing, markdown]
---

# Content"""
    metadata = preprocessor.extract_metadata(content, "/path/to/file.md")
    assert "python" in metadata['tags'], "python tag should be extracted"
    assert "testing" in metadata['tags'], "testing tag should be extracted"
    assert "markdown" in metadata['tags'], "markdown tag should be extracted"
    print("✓ test_extract_metadata_tags_from_frontmatter passed")


def test_extract_metadata_inline_tags():
    """Test extraction of inline hashtags."""
    preprocessor = TextPreprocessor()
    content = """# My Document

This document is about #python and #testing.
"""
    metadata = preprocessor.extract_metadata(content, "/path/to/file.md")
    assert "python" in metadata['tags'], "python tag should be extracted"
    assert "testing" in metadata['tags'], "testing tag should be extracted"
    print("✓ test_extract_metadata_inline_tags passed")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_clean_markdown_empty_string,
        test_clean_markdown_plain_text,
        test_clean_markdown_headers,
        test_clean_markdown_bold_italic,
        test_clean_markdown_links,
        test_clean_markdown_code_blocks,
        test_clean_markdown_lists,
        test_clean_markdown_complex_document,
        test_extract_metadata_empty_content,
        test_extract_metadata_title_from_h1,
        test_extract_metadata_all_headers,
        test_extract_metadata_tags_from_frontmatter,
        test_extract_metadata_inline_tags,
    ]
    
    passed = 0
    failed = 0
    
    print("Running TextPreprocessor tests...\n")
    
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
