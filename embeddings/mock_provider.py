import hashlib

from .base import EmbeddingProvider, EmbeddingResult


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic fixed-size vectors for tests — no model, no network call.

    Each vector is derived from a SHA-256 hash of the input text, so the
    same text always produces the same vector and different texts produce
    different vectors, with no model download or API call. Selected via
    EMBEDDING_BACKEND=mock, or constructed directly by tests that need a
    stand-in provider.

    Attributes:
        model_id: Fixed identifier ("mock-embedding-v1") — there's no real
            model behind this provider, but the interface still requires one.
        table_name: Fixed placeholder ("embeddings_mock"); no such table is
            ever created — this provider is for tests only.
        dimension: Length of the generated vectors, set per instance via
            the constructor (default 8).
    """

    model_id = "mock-embedding-v1"
    table_name = "embeddings_mock"

    def __init__(self, dimension: int = 8):
        """Create a mock provider that generates `dimension`-length vectors.

        Args:
            dimension: Length of the vectors this instance will generate.
                Defaults to 8 — small and fast for unit tests; pass a
                specific value (e.g. 384) if a test needs vectors shaped
                like a real provider's output.

        Returns:
            None.
        """
        self.dimension = dimension

    def _vector_for(self, text: str) -> list[float]:
        """Deterministically derive a unit-normalized vector from `text`.

        Args:
            text: The text to derive a vector from.

        Returns:
            A `self.dimension`-length unit vector, identical for identical
            `text` and (with overwhelming probability) different across
            different `text`.
        """
        digest = hashlib.sha256(text.encode()).digest()
        raw = (digest * ((self.dimension // len(digest)) + 1))[: self.dimension]
        floats = [b / 255.0 for b in raw]
        norm = sum(f * f for f in floats) ** 0.5 or 1.0
        return [f / norm for f in floats]

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts by deriving a deterministic vector for each.

        Args:
            texts: The chunk texts to embed, in order.

        Returns:
            An EmbeddingResult with one vector per input text, in the same
            order as `texts`.
        """
        vectors = [self._vector_for(t) for t in texts]
        return EmbeddingResult(vectors, self.model_id, self.dimension)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string — identical behavior to embed_documents for one text.

        Args:
            text: The query text to embed.

        Returns:
            The deterministic vector for `text`.
        """
        return self._vector_for(text)
