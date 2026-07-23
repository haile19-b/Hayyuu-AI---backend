import re
from typing import Any, Dict, List, Optional


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 characters per token for English text)."""
    if not text:
        return 0
    # Splits on whitespace and word boundaries for a reasonable approximation
    words = len(re.findall(r"\w+", text))
    char_estimate = len(text) // 4
    return max(words, char_estimate)


class RecursiveTextSplitter:
    """
    Recursive text splitter that splits documents into overlapping chunks.
    Default target chunk size: ~500 tokens (~2000 chars), overlap: ~100 tokens (~400 chars).
    """

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        # Character approximations
        self.chunk_size_chars = chunk_size_tokens * 4
        self.chunk_overlap_chars = chunk_overlap_tokens * 4
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchy of separators."""
        final_chunks: List[str] = []
        if not text:
            return final_chunks

        # Find the first separator present in text
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split using the chosen separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits up to chunk size
        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size_chars:
                good_splits.append(s)
            else:
                # If split item is still too large, split recursively with finer separators
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    other_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    good_splits.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Combine smaller text splits into chunks with target size and overlap."""
        chunks: List[str] = []
        current_doc: List[str] = []
        current_len = 0

        for split in splits:
            len_split = len(split)
            sep_len = len(separator) if current_doc else 0

            if current_len + len_split + sep_len > self.chunk_size_chars:
                if current_doc:
                    doc_str = separator.join(current_doc).strip()
                    if doc_str:
                        chunks.append(doc_str)
                    
                    # Compute overlap: keep trailing items that fit within overlap limit
                    while current_doc and (
                        sum(len(x) for x in current_doc) > self.chunk_overlap_chars
                    ):
                        current_doc.pop(0)
                    
                    current_len = sum(len(x) for x in current_doc)

            current_doc.append(split)
            current_len += len_split + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_str = separator.join(current_doc).strip()
            if doc_str:
                chunks.append(doc_str)

        return chunks

    def split(self, text: str) -> List[Dict[str, Any]]:
        """
        Split raw document text into structured chunk dicts.
        """
        raw_chunks = self._split_text(text, self.separators)
        structured_chunks: List[Dict[str, Any]] = []

        start_char = 0
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_len = len(chunk_text)
            token_count = estimate_tokens(chunk_text)
            structured_chunks.append(
                {
                    "chunk_index": idx,
                    "content": chunk_text,
                    "token_count": token_count,
                    "start_char": start_char,
                    "end_char": start_char + chunk_len,
                }
            )
            # Advance start_char accounting for overlap
            start_char += max(1, chunk_len - self.chunk_overlap_chars)

        return structured_chunks


def chunk_document_text(
    text: str,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """Convenience helper function to chunk text."""
    splitter = RecursiveTextSplitter(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    return splitter.split(text)
