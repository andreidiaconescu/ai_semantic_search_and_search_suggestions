import pytest

from embeddings.factory import get_provider
from embeddings.mock_provider import MockEmbeddingProvider
from embeddings.openai_provider import OpenAIEmbeddingProvider


def test_mock_backend(monkeypatch):
    """EMBEDDING_BACKEND=mock resolves to a MockEmbeddingProvider.

    Args:
        monkeypatch: pytest's built-in fixture for setting/deleting env
            vars and patching attributes for the duration of one test.

    Returns:
        None.
    """
    monkeypatch.setenv("EMBEDDING_BACKEND", "mock")
    provider = get_provider()
    assert isinstance(provider, MockEmbeddingProvider)


def test_openai_backend(monkeypatch):
    """EMBEDDING_BACKEND=openai resolves to an OpenAIEmbeddingProvider with the expected attributes.

    Uses a dummy OPENAI_API_KEY — constructing the client doesn't make a
    network call, so this stays a fast unit test.

    Args:
        monkeypatch: pytest's built-in fixture for setting/deleting env
            vars and patching attributes for the duration of one test.

    Returns:
        None.
    """
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    provider = get_provider()
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model_id == "text-embedding-3-small"
    assert provider.dimension == 1536
    assert provider.table_name == "embeddings_openai_small"


def test_local_backend(monkeypatch):
    """EMBEDDING_BACKEND=local resolves to a LocalEmbeddingProvider with the expected attributes.

    SentenceTransformer is monkeypatched so construction doesn't actually
    download/load a model. Skipped if sentence-transformers isn't
    installed (e.g. in CI, which only installs requirements-dev.txt).

    Args:
        monkeypatch: pytest's built-in fixture for setting/deleting env
            vars and patching attributes for the duration of one test.

    Returns:
        None.
    """
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(
        "embeddings.local_provider.SentenceTransformer",
        lambda model_id, device="cpu": object(),
    )
    from embeddings.local_provider import LocalEmbeddingProvider

    provider = get_provider()
    assert isinstance(provider, LocalEmbeddingProvider)
    assert provider.dimension == 384
    assert provider.table_name == "embeddings_bge_small"


def test_default_backend_is_local(monkeypatch):
    """With no EMBEDDING_BACKEND set at all, get_provider() defaults to local (matches .env.example).

    Args:
        monkeypatch: pytest's built-in fixture for setting/deleting env
            vars and patching attributes for the duration of one test.

    Returns:
        None.
    """
    pytest.importorskip("sentence_transformers")
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.setattr(
        "embeddings.local_provider.SentenceTransformer",
        lambda model_id, device="cpu": object(),
    )
    from embeddings.local_provider import LocalEmbeddingProvider

    provider = get_provider()
    assert isinstance(provider, LocalEmbeddingProvider)


def test_unknown_backend_raises(monkeypatch):
    """An unrecognized EMBEDDING_BACKEND value raises ValueError rather than silently picking one.

    Args:
        monkeypatch: pytest's built-in fixture for setting/deleting env
            vars and patching attributes for the duration of one test.

    Returns:
        None.
    """
    monkeypatch.setenv("EMBEDDING_BACKEND", "not-a-real-backend")
    with pytest.raises(ValueError):
        get_provider()
