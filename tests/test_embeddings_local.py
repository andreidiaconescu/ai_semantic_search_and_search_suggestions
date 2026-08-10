import math

import pytest

pytest.importorskip("sentence_transformers")

from embeddings.local_provider import LocalEmbeddingProvider


def _norm(vector: list[float]) -> float:
    """Return the L2 (Euclidean) norm of `vector`.

    Args:
        vector: The vector to measure.

    Returns:
        The L2 norm (`sqrt(sum(v_i^2))`) of `vector`.
    """
    return math.sqrt(sum(v * v for v in vector))


@pytest.fixture(scope="module")
def provider():
    """A real LocalEmbeddingProvider, loaded once per test module (not per test).

    Model loading is the slow part (first-run download + disk cache), so
    it's shared across all three tests in this file rather than reloaded
    for each one.

    Returns:
        A LocalEmbeddingProvider with the real bge-small-en-v1.5 model loaded.
    """
    return LocalEmbeddingProvider()


@pytest.mark.integration
def test_dimension_is_384(provider):
    """The local model's declared dimension is 384, matching embeddings_bge_small.

    Args:
        provider: The module-scoped LocalEmbeddingProvider fixture above.

    Returns:
        None.
    """
    assert provider.dimension == 384


@pytest.mark.integration
def test_embed_documents_shape_and_normalization(provider):
    """embed_documents returns correctly-shaped (384-dim), unit-normalized vectors.

    Args:
        provider: The module-scoped LocalEmbeddingProvider fixture above.

    Returns:
        None.
    """
    result = provider.embed_documents(["hello world", "another sentence"])
    assert len(result.vectors) == 2
    for vector in result.vectors:
        assert len(vector) == 384
        assert math.isclose(_norm(vector), 1.0, rel_tol=1e-3)


@pytest.mark.integration
def test_query_prefix_changes_the_embedding(provider):
    """embed_query differs from embed_documents for the same raw text — proves the BGE instruction prefix is actually applied, not a no-op.

    Args:
        provider: The module-scoped LocalEmbeddingProvider fixture above.

    Returns:
        None.
    """
    text = "how do I reset my password"
    query_vector = provider.embed_query(text)
    document_vector = provider.embed_documents([text]).vectors[0]
    assert query_vector != document_vector
