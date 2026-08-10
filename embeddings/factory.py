import os

from .base import EmbeddingProvider


def get_provider() -> EmbeddingProvider:
    """Construct the EmbeddingProvider selected by the $EMBEDDING_BACKEND env var.

    This is the single point where an embedding backend is chosen — calling
    code (ingestion/query) depends only on the returned EmbeddingProvider
    interface, never on which backend was picked, so switching providers is
    a config change (setting $EMBEDDING_BACKEND) rather than a code change.
    Each branch imports its provider module lazily, so e.g. selecting
    "mock" never requires the `openai` or `sentence-transformers` packages
    to be installed.

    Returns:
        A constructed provider for the selected backend: "local" (default)
        → LocalEmbeddingProvider, "openai" → OpenAIEmbeddingProvider,
        "mock" → MockEmbeddingProvider.

    Raises:
        ValueError: If $EMBEDDING_BACKEND is set to an unrecognized value.
    """
    backend = os.environ.get("EMBEDDING_BACKEND", "local")
    match backend:
        case "openai":
            from .openai_provider import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider()
        case "local":
            from .local_provider import LocalEmbeddingProvider
            return LocalEmbeddingProvider()
        case "mock":
            from .mock_provider import MockEmbeddingProvider
            return MockEmbeddingProvider()
        # "voyage" can be added here later, following the same pattern as
        # "openai" (see embeddings/openai_provider.py) once implemented.
        case _:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
