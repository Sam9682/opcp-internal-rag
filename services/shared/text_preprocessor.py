"""
Text Preprocessor for RAG Application

This module provides text preprocessing functionality for markdown documents,
including cleaning, normalization, and semantic structure preservation.

Requirements: 2.1, 2.3
"""

import re
from typing import Dict, Any, List
import tiktoken


class TextPreprocessor:
    """
    Text preprocessor for cleaning and normalizing markdown content.
    
    This class handles markdown formatting removal while preserving semantic
    structure, code blocks, and extracting meaningful content for embedding.
    """
    
    def __init__(self):
        """Initialize the TextPreprocessor."""
        pass
    
    def clean_markdown(self, content: str) -> str:
        """
        Remove markdown formatting while preserving semantic structure.
        
        This method processes markdown content to extract clean text suitable
        for embedding generation. It handles:
        - Code blocks (preserved with language tags)
        - Headers (converted to plain text with context)
        - Links (text preserved, URLs removed)
        - Bold/italic formatting (removed)
        - Lists (converted to plain text)
        - Special characters (normalized)
        - Whitespace (normalized)
        
        Preconditions:
        - content is valid string (may be empty)
        - content contains markdown-formatted text
        
        Postconditions:
        - Returns cleaned text string
        - Markdown syntax removed (**, __, [], (), etc.)
        - Code blocks preserved with language tags
        - Headers converted to plain text with context
        - Link text preserved, URLs removed
        - Multiple whitespaces normalized to single space
        - Leading/trailing whitespace removed
        
        Args:
            content: Raw markdown content as string
            
        Returns:
            Cleaned text with markdown formatting removed
            
        Requirements: 2.1, 2.3
        """
        if not content:
            return ""
        
        # Step 1: Preserve code blocks with language tags
        # Extract code blocks and replace with placeholders
        code_blocks = []
        code_block_pattern = r'```(\w+)?\s*\n(.*?)```'
        
        def save_code_block(match):
            lang = match.group(1) or 'code'
            code_content = match.group(2).strip()
            placeholder = f"XOXOCODEBLOCKXOXO{len(code_blocks)}XOXO"
            code_blocks.append(f"[{lang} code block: {code_content}]")
            return placeholder
        
        text = re.sub(code_block_pattern, save_code_block, content, flags=re.DOTALL)
        
        # Step 2: Convert headers to plain text with context
        # H1-H6 headers: remove # symbols but keep text
        text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
        
        # Step 3: Extract link text, remove URLs
        # [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Step 4: Remove bold and italic formatting
        # **bold** or __bold__ -> bold
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        # *italic* or _italic_ -> italic
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Step 5: Remove inline code backticks
        # `code` -> code
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Step 6: Convert lists to plain text
        # Remove list markers (-, *, +, numbers)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Step 7: Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # Step 8: Remove blockquote markers
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        
        # Step 9: Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Step 10: Restore code blocks
        for i, code_block in enumerate(code_blocks):
            placeholder = f"XOXOCODEBLOCKXOXO{i}XOXO"
            text = text.replace(placeholder, code_block)
        
        # Step 11: Normalize whitespace
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline (paragraph separation)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove spaces at start/end of lines
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
        
        # Step 12: Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from markdown content.
        
        Extracts document metadata including:
        - Title (from first H1 or filename)
        - Headers (all H1-H6 headers)
        - Tags (if present in frontmatter or content)
        
        Preconditions:
        - content is valid string (may be empty)
        - file_path is valid path string
        
        Postconditions:
        - Returns dictionary with metadata fields
        - title is non-empty string
        - headers is list of strings
        - tags is list of strings (may be empty)
        
        Args:
            content: Raw markdown content
            file_path: Path to the markdown file
            
        Returns:
            Dictionary containing extracted metadata
            
        Requirements: 2.3
        """
        metadata: Dict[str, Any] = {
            'title': '',
            'headers': [],
            'tags': [],
            'file_path': file_path
        }
        
        if not content:
            # Use filename as title if content is empty
            import os
            metadata['title'] = os.path.splitext(os.path.basename(file_path))[0]
            return metadata
        
        # Extract title from first H1 header
        h1_match = re.search(r'^#\s+(.+)$', content, flags=re.MULTILINE)
        if h1_match:
            metadata['title'] = h1_match.group(1).strip()
        else:
            # Fallback to filename
            import os
            metadata['title'] = os.path.splitext(os.path.basename(file_path))[0]
        
        # Extract all headers (H1-H6)
        header_pattern = r'^#{1,6}\s+(.+)$'
        headers = re.findall(header_pattern, content, flags=re.MULTILINE)
        metadata['headers'] = [h.strip() for h in headers]
        
        # Extract tags from frontmatter (YAML-style)
        # Look for tags: [tag1, tag2] or tags: tag1, tag2
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, flags=re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            tags_match = re.search(r'tags:\s*\[([^\]]+)\]', frontmatter)
            if tags_match:
                tags_str = tags_match.group(1)
                metadata['tags'] = [t.strip().strip('"\'') for t in tags_str.split(',')]
            else:
                tags_match = re.search(r'tags:\s*(.+)$', frontmatter, flags=re.MULTILINE)
                if tags_match:
                    tags_str = tags_match.group(1)
                    metadata['tags'] = [t.strip() for t in tags_str.split(',')]
        
        # Also look for inline tags like #tag
        inline_tags = re.findall(r'#(\w+)', content)
        if inline_tags:
            # Combine with frontmatter tags, remove duplicates
            all_tags = set(metadata['tags'] + inline_tags)
            metadata['tags'] = list(all_tags)
        
        return metadata

    def chunk_text(
        self,
        content: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks for embedding.

        This method splits text into chunks of maximum chunk_size tokens with
        overlap tokens between adjacent chunks. Uses tiktoken for accurate
        token counting with cl100k_base encoding (compatible with OpenAI models
        and suitable for general text processing).

        Preconditions:
        - content is non-empty string
        - chunk_size > overlap > 0
        - overlap < chunk_size

        Postconditions:
        - Returns list of text chunks
        - Each chunk length <= chunk_size (in tokens)
        - Adjacent chunks overlap by ~overlap tokens
        - All content is covered (no text lost)
        - len(chunks) >= 1

        Loop Invariants:
        - All created chunks have valid length
        - Total coverage increases monotonically
        - No duplicate chunks

        Args:
            content: Text content to chunk
            chunk_size: Maximum tokens per chunk (default 512)
            overlap: Number of overlapping tokens between chunks (default 50)

        Returns:
            List of text chunks

        Requirements: 2.2, 2.5
        """
        # Validate preconditions
        assert len(content) > 0, "Content must be non-empty"
        assert chunk_size > overlap > 0, "chunk_size must be greater than overlap, and overlap must be positive"

        # Step 1: Initialize tokenizer (cl100k_base encoding for compatibility)
        encoding = tiktoken.get_encoding("cl100k_base")

        # Step 2: Tokenize content
        tokens = encoding.encode(content)
        total_tokens = len(tokens)

        # Step 3: Handle single chunk case
        if total_tokens <= chunk_size:
            return [content]

        # Step 4: Calculate chunk boundaries and create chunks
        chunks = []
        start_idx = 0

        while start_idx < total_tokens:
            # Loop invariant: start_idx is valid position
            assert 0 <= start_idx < total_tokens

            # Calculate end position
            end_idx = min(start_idx + chunk_size, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Convert back to text
            chunk_text = encoding.decode(chunk_tokens)

            # Validate chunk
            assert len(chunk_tokens) <= chunk_size, f"Chunk has {len(chunk_tokens)} tokens, exceeds limit of {chunk_size}"
            assert len(chunk_text) > 0, "Chunk text must be non-empty"

            chunks.append(chunk_text)

            # Move to next chunk with overlap
            if end_idx >= total_tokens:
                break

            start_idx = end_idx - overlap

        # Postcondition checks
        assert len(chunks) >= 1, "Must produce at least one chunk"
        assert all(len(encoding.encode(chunk)) <= chunk_size for chunk in chunks), "All chunks must be within size limit"

        return chunks

