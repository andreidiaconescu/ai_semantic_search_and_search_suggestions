import pytest

from embeddings.base import EmbeddingProvider, EmbeddingResult


def test_embedding_provider_cannot_be_instantiated_directly():
    """EmbeddingProvider is an ABC — only subclasses implementing both abstract methods can be constructed.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_embedding_result_holds_its_fields():
    """EmbeddingResult stores exactly the vectors/model_id/dimension it's constructed with.

    Args:
        None.

    Returns:
        None.
    """
    result = EmbeddingResult(vectors=[[0.1, 0.2]], model_id="test-model", dimension=2)
    assert result.vectors == [[0.1, 0.2]]
    assert result.model_id == "test-model"
    assert result.dimension == 2
