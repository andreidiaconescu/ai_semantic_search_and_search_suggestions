from itertools import pairwise

import pytest

from chunking import chunk_text


def test_empty_text_returns_no_chunks():
    """An empty string produces no chunks.

    Args:
        None.

    Returns:
        None.
    """
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    """Text shorter than chunk_size comes back as one unmodified chunk.

    Args:
        None.

    Returns:
        None.
    """
    text = "a short sentence"
    assert chunk_text(text, chunk_size=1000, overlap=100) == [text]


def test_prefers_paragraph_boundary_over_mid_word_split():
    """A paragraph break is preferred as the split point over a hard character cutoff.

    Two ~85-90 char paragraphs joined by "\\n\\n" (177 chars total) split
    with chunk_size=100 should produce exactly two chunks, with the first
    chunk ending precisely at the paragraph boundary (not mid-word), and
    the second chunk carrying the requested overlap plus the full second
    paragraph.

    Args:
        None.

    Returns:
        None.
    """
    para1 = "First paragraph. " * 5  # 85 chars
    para2 = "Second paragraph. " * 5  # 90 chars
    text = para1 + "\n\n" + para2

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) == 2
    assert chunks[0] == para1 + "\n\n"
    assert chunks[1].endswith(para2)
    assert len(chunks[1]) - len(para2) == 20


def test_word_boundaries_are_preserved_with_whitespace_fallback():
    """With no paragraph/sentence separators available, splitting still respects whitespace.

    "word " repeated has no "\\n\\n"/". "/"! "/"? " to split on, so the
    splitter falls back to plain whitespace — every resulting chunk should
    still consist only of complete "word" tokens, never a partial word
    fragment like "wor" or "rd".

    Args:
        None.

    Returns:
        None.
    """
    text = "word " * 60
    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert all(token == "word" for token in chunk.split())


def test_consecutive_chunks_share_the_requested_overlap():
    """Each non-first chunk starts with the last `overlap` characters of the previous chunk.

    Args:
        None.

    Returns:
        None.
    """
    text = "word " * 60
    chunks = chunk_text(text, chunk_size=50, overlap=10)

    for previous, current in pairwise(chunks):
        assert current.startswith(previous[-10:])


def test_chunk_size_must_be_positive():
    """chunk_size <= 0 raises ValueError rather than producing a nonsensical split.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0)


def test_overlap_must_be_non_negative():
    """A negative overlap raises ValueError.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, overlap=-1)


def test_overlap_must_be_less_than_chunk_size():
    """overlap >= chunk_size raises ValueError (it would prevent any forward progress).

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, overlap=10)
