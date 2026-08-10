import math

from embeddings.mock_provider import MockEmbeddingProvider


def _norm(vector: list[float]) -> float:
    """Return the L2 (Euclidean) norm of `vector`.

    Args:
        vector: The vector to measure.

    Returns:
        The L2 norm (`sqrt(sum(v_i^2))`) of `vector`.
    """
    return math.sqrt(sum(v * v for v in vector))


def test_embed_documents_preserves_order_and_count():
    """embed_documents returns one vector per input text, in order, with the provider's model_id/dimension.

    Args:
        None.

    Returns:
        None.
    """
    provider = MockEmbeddingProvider()
    texts = ["alpha", "beta", "gamma"]
    result = provider.embed_documents(texts)
    assert len(result.vectors) == len(texts)
    assert result.dimension == provider.dimension
    assert result.model_id == provider.model_id


def test_same_text_gives_same_vector():
    """Embedding the same text twice is deterministic — identical output both times.

    Args:
        None.

    Returns:
        None.
    """
    provider = MockEmbeddingProvider()
    assert provider.embed_query("hello") == provider.embed_query("hello")


def test_different_text_gives_different_vector():
    """Different input text produces a different vector.

    Args:
        None.

    Returns:
        None.
    """
    provider = MockEmbeddingProvider()
    assert provider.embed_query("hello") != provider.embed_query("goodbye")


def test_vectors_are_unit_normalized():
    """Generated vectors have L2 norm ≈ 1, matching the convention real providers use.

    Args:
        None.

    Returns:
        None.
    """
    provider = MockEmbeddingProvider()
    vector = provider.embed_query("normalize me")
    assert math.isclose(_norm(vector), 1.0, rel_tol=1e-6)


def test_embed_query_matches_embed_documents_for_same_text():
    """embed_query(t) and embed_documents([t]).vectors[0] are internally consistent for the same text.

    Args:
        None.

    Returns:
        None.
    """
    provider = MockEmbeddingProvider()
    text = "consistency check"
    assert provider.embed_query(text) == provider.embed_documents([text]).vectors[0]


def test_dimension_is_configurable():
    """The `dimension` constructor argument controls the length of generated vectors.

    Args:
        None.

    Returns:
        None.
    """
    provider = MockEmbeddingProvider(dimension=16)
    vector = provider.embed_query("sized")
    assert len(vector) == 16
