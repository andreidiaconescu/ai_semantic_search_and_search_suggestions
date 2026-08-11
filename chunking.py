"""Recursive character-based text chunking.

A dependency-free splitter in the spirit of tools like
`RecursiveCharacterTextSplitter`: it prefers to break text on paragraph
breaks, then sentence-ish boundaries, then whitespace, and only falls back
to a hard character cutoff when none of those are available within
`chunk_size` — so chunk boundaries land on natural text boundaries whenever
possible instead of mid-word/mid-sentence.
"""

# Priority order: paragraph break, sentence enders, line break, whitespace.
# Falls back to a hard character cutoff once these are exhausted.
_SEPARATORS = ["\n\n", ". ", "! ", "? ", "\n", " "]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split `text` into overlapping chunks of at most `chunk_size` characters.

    Recursively splits on `_SEPARATORS` (paragraph > sentence > line >
    whitespace, falling back to a hard character cutoff), then greedily
    merges the resulting pieces back up to `chunk_size`, carrying the last
    `overlap` characters of each chunk into the start of the next one so
    nearby chunks share context.

    Args:
        text: The text to split. An empty string yields no chunks.
        chunk_size: Maximum characters per chunk. Must be greater than
            `overlap`.
        overlap: How many trailing characters of one chunk are repeated at
            the start of the next chunk, to preserve context across chunk
            boundaries. Must be non-negative and less than `chunk_size`.

    Returns:
        The text split into chunks, in reading order. Empty input returns
        an empty list.

    Raises:
        ValueError: If `chunk_size <= 0`, `overlap < 0`, or
            `overlap >= chunk_size`.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    if not text:
        return []

    pieces = _split_recursive(text, chunk_size, _SEPARATORS)
    return _merge_pieces(pieces, chunk_size, overlap)


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split `text` into pieces no longer than `chunk_size`.

    Tries `separators` in order: splits `text` on the first separator that
    actually occurs in it, then recurses on any resulting piece still
    longer than `chunk_size` using the remaining separators. Once
    `separators` is exhausted, falls back to a hard character cutoff.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per returned piece.
        separators: Candidate separators to try, in priority order.

    Returns:
        Pieces of `text`, each at most `chunk_size` characters, in order.
        Empty pieces are dropped. The separator that was split on is kept
        attached to the end of each piece (except the last), so
        concatenating the pieces reconstructs `text` almost exactly.
    """
    if len(text) <= chunk_size:
        return [text] if text else []
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *rest = separators
    parts = text.split(separator)
    if len(parts) == 1:
        # This separator doesn't occur in `text` at all — try the next one.
        return _split_recursive(text, chunk_size, rest)

    pieces = []
    last_index = len(parts) - 1
    for i, part in enumerate(parts):
        piece = part if i == last_index else part + separator
        if not piece:
            continue
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, chunk_size, rest))
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks up to `chunk_size`, with overlap.

    Args:
        pieces: Small text pieces to merge, in order (from `_split_recursive`;
            each piece is already at most `chunk_size` characters).
        chunk_size: Maximum characters per merged chunk.
        overlap: How many trailing characters of a finished chunk are
            carried over to seed the start of the next one.

    Returns:
        The merged chunks, in order.
    """
    if not pieces:
        return []

    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap > 0 else ""
        current += piece
    if current:
        chunks.append(current)
    return chunks
