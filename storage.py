"""Table-name resolution for embedding providers.

See embeddings/base.py's `table_name` attribute for why this is a direct
lookup rather than a slug derived from `model_id`.
"""
from embeddings.base import EmbeddingProvider


def table_name_for(provider: EmbeddingProvider) -> str:
    """Return the embeddings table that `provider`'s vectors belong in.

    Args:
        provider: The EmbeddingProvider instance whose table name is needed.

    Returns:
        The provider's declared `table_name` (e.g. "embeddings_bge_small").
    """
    return provider.table_name
