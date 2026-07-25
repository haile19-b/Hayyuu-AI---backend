import re
import logging
from typing import Any, Dict, List, Optional
from io import BytesIO

logger = logging.getLogger("uvicorn.error")

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import DocumentStream
    from docling.chunking import HybridChunker
    doc_chunker_converter = DocumentConverter()
    logger.info("✅ Docling Chunker Converter Initialized Successfully")
except Exception as chunk_err:
    logger.error(f"Failed to initialize Docling chunker imports: {chunk_err}")
    doc_chunker_converter = None


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 characters per token for English text)."""
    if not text:
        return 0
    words = len(re.findall(r"\w+", text))
    char_estimate = len(text) // 4
    return max(words, char_estimate)


class RecursiveTextSplitter:
    """
    Recursive text splitter that splits documents into overlapping chunks.
    Default target chunk size: ~500 tokens (~2000 chars), overlap: ~100 tokens (~400 chars).
    Used as a fallback when layout-aware chunking is not required or fails.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.chunk_size_chars = chunk_size_tokens * 4
        self.chunk_overlap_chars = chunk_overlap_tokens * 4
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        if not text:
            return final_chunks

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

        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size_chars:
                good_splits.append(s)
            else:
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
            start_char += max(1, chunk_len - self.chunk_overlap_chars)

        return structured_chunks


def merge_short_structured_chunks(chunks: List[Dict[str, Any]], min_chars: int = 200) -> List[Dict[str, Any]]:
    """
    Consolidates undersized adjacent structured chunks (e.g. headers, footers, page numbers)
    by merging them forward into the next chunk. Resets indices and merges metadata.
    """
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for c in chunks:
        content = c["content"].strip()
        if not content:
            continue

        if current is None:
            current = c
        else:
            current_content = current["content"].strip()
            # If current chunk is below threshold, merge it forward into the incoming chunk c
            if len(current_content) < min_chars:
                c["content"] = current_content + "\n\n" + content
                c["start_char"] = current["start_char"]
                
                # Merge section paths if present
                curr_path = current.get("metadata", {}).get("section_path", "")
                c_path = c.get("metadata", {}).get("section_path", "")
                if curr_path and curr_path not in c_path:
                    c["metadata"]["section_path"] = curr_path + " > " + c_path if c_path else curr_path
                
                # Carry forward table flags
                if current.get("metadata", {}).get("is_table"):
                    c["metadata"]["is_table"] = True

                current = c
            else:
                merged.append(current)
                current = c

    if current is not None:
        merged.append(current)

    # Re-index chunks sequentially
    for idx, c in enumerate(merged):
        c["chunk_index"] = idx

    return merged


def chunk_document_text(
    text: str,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """
    Ingests raw document text, converts it to a Docling document in-memory,
    and runs a layout-aware HybridChunker to preserve sections, headers, and tables.
    Consolidates short fragments and falls back to RecursiveTextSplitter on error.
    """
    # 1. Attempt layout-aware chunking via IBM Docling
    if doc_chunker_converter:
        logger.info(f"Attempting layout-aware chunking via Docling HybridChunker (max={chunk_size_tokens} tokens)")
        try:
            source = DocumentStream(name="document.txt", stream=BytesIO(text.encode("utf-8")))
            result = doc_chunker_converter.convert(source)
            chunker = HybridChunker(max_tokens=chunk_size_tokens)
            
            raw_chunks = list(chunker.chunk(result.document))
            structured_chunks: List[Dict[str, Any]] = []
            
            start_offset = 0
            for idx, chunk in enumerate(raw_chunks):
                if isinstance(chunk, str):
                    chunk_text = chunk
                    section_path = ""
                    is_table = False
                else:
                    chunk_text = getattr(chunk, "text", str(chunk))
                    
                    # Fetch section headers path (breadcrumbs metadata)
                    meta = getattr(chunk, "meta", None)
                    headings = getattr(meta, "headings", []) if meta else []
                    headers = [h.text if hasattr(h, "text") else str(h) for h in headings] if headings else []
                    section_path = " > ".join(headers)
                    
                    doc_items = getattr(meta, "doc_items", []) if meta else []
                    is_table = any(getattr(item, "label", "") == "Table" for item in doc_items) if doc_items else False
                
                token_count = estimate_tokens(chunk_text)
                chunk_len = len(chunk_text)
                
                structured_chunks.append(
                    {
                        "chunk_index": idx,
                        "content": chunk_text,
                        "token_count": token_count,
                        "start_char": start_offset,
                        "end_char": start_offset + chunk_len,
                        "metadata": {
                            "section_path": section_path,
                            "is_table": is_table
                        }
                    }
                )
                # Overlap approximation
                overlap_chars = chunk_overlap_tokens * 4
                start_offset += max(1, chunk_len - overlap_chars)
                
            if structured_chunks:
                consolidated = merge_short_structured_chunks(structured_chunks, min_chars=200)
                logger.info(f"Layout-aware Docling chunker generated {len(consolidated)} chunks (consolidated from {len(structured_chunks)}).")
                return consolidated
        except Exception as e:
            logger.warning(f"Docling chunker failed: {e}. Falling back to RecursiveTextSplitter.")

    # 2. Legacy fallback
    logger.info("Using RecursiveTextSplitter character chunker fallback")
    splitter = RecursiveTextSplitter(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    return splitter.split(text)
