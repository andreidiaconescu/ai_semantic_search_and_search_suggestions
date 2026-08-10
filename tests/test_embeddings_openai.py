import os
from unittest.mock import Mock

import openai
import pytest

from embeddings.openai_provider import OpenAIEmbeddingProvider


class _FakeItem:
    """Stand-in for one item of OpenAI's `embeddings.create()` response `data` list.

    Attributes:
        index: Position this embedding corresponds to in the request's
            input list, per the real API's response shape.
        embedding: The (fake) embedding vector for that input.
    """

    def __init__(self, index, embedding):
        """Store the given index/embedding as-is.

        Args:
            index: Position in the request's input list this item
                corresponds to.
            embedding: The (fake) embedding vector.

        Returns:
            None.
        """
        self.index = index
        self.embedding = embedding


class _FakeResponse:
    """Stand-in for OpenAI's `embeddings.create()` response object.

    Attributes:
        data: List of `_FakeItem`s, mirroring the real response's `.data`.
    """

    def __init__(self, data):
        """Store the given item list as `.data`.

        Args:
            data: A list of `_FakeItem`s to expose as `.data`.

        Returns:
            None.
        """
        self.data = data


@pytest.fixture
def provider(monkeypatch):
    """An OpenAIEmbeddingProvider built with a dummy API key — no real network calls.

    Individual tests replace `provider.client.embeddings.create` with a
    Mock to control what the "API" returns/raises, so nothing here ever
    touches the real OpenAI service.

    Args:
        monkeypatch: pytest's built-in fixture for setting/deleting env
            vars for the duration of one test.

    Returns:
        A freshly constructed OpenAIEmbeddingProvider.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    return OpenAIEmbeddingProvider()


def test_embed_documents_sorts_by_index_defensively(provider):
    """Vectors come back in input order even if the API response's `data` items are out of order.

    Args:
        provider: The `provider` fixture above (dummy-keyed, no real network).

    Returns:
        None.
    """
    out_of_order = _FakeResponse(
        [
            _FakeItem(index=1, embedding=[0.2]),
            _FakeItem(index=0, embedding=[0.1]),
        ]
    )
    provider.client.embeddings.create = Mock(return_value=out_of_order)
    result = provider.embed_documents(["first", "second"])
    assert result.vectors == [[0.1], [0.2]]


def test_retries_on_retryable_error_then_succeeds(provider):
    """Two RateLimitErrors followed by success still return the correct result — proves the tenacity retry wiring works.

    Args:
        provider: The `provider` fixture above (dummy-keyed, no real network).

    Returns:
        None.
    """
    retryable = openai.RateLimitError.__new__(openai.RateLimitError)
    success = _FakeResponse([_FakeItem(index=0, embedding=[0.5])])
    mock_create = Mock(side_effect=[retryable, retryable, success])
    provider.client.embeddings.create = mock_create

    result = provider.embed_documents(["retry me"])

    assert result.vectors == [[0.5]]
    assert mock_create.call_count == 3


def test_non_retryable_error_is_not_retried(provider):
    """An AuthenticationError propagates immediately (no retries) — retrying it could never succeed.

    Args:
        provider: The `provider` fixture above (dummy-keyed, no real network).

    Returns:
        None.
    """
    auth_error = openai.AuthenticationError.__new__(openai.AuthenticationError)
    mock_create = Mock(side_effect=auth_error)
    provider.client.embeddings.create = mock_create

    with pytest.raises(openai.AuthenticationError):
        provider.embed_documents(["fail fast"])

    assert mock_create.call_count == 1


def test_embed_query_delegates_to_embed_documents(provider):
    """embed_query returns the same vector embed_documents([text]) would for that one text.

    Args:
        provider: The `provider` fixture above (dummy-keyed, no real network).

    Returns:
        None.
    """
    resp = _FakeResponse([_FakeItem(index=0, embedding=[0.7, 0.8])])
    provider.client.embeddings.create = Mock(return_value=resp)
    assert provider.embed_query("q") == [0.7, 0.8]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)
def test_real_openai_embedding_dimension_matches():
    """A real API call with a fixed short input returns a vector of length provider.dimension (1536).

    Skipped unless OPENAI_API_KEY is set to a real key — this makes an
    actual, billed call to OpenAI, unlike every other test in this file.

    Args:
        None — constructs its own OpenAIEmbeddingProvider from the real
        $OPENAI_API_KEY rather than using the `provider` fixture, since it
        needs the real key, not the fixture's dummy one.

    Returns:
        None.
    """
    provider = OpenAIEmbeddingProvider()
    result = provider.embed_documents(["hello world"])
    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == provider.dimension == 1536
