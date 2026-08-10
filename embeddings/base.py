from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """The output of an embedding call: one vector per input text, plus provenance.

    Attributes:
        vectors: One embedding vector per input text, in the same order the
            texts were given to embed_documents.
        model_id: Identifier of the model that produced these vectors (e.g.
            "BAAI/bge-small-en-v1.5"), carried alongside the vectors so
            callers can record/verify which model produced a given result.
        dimension: Length of each vector in `vectors`; matches the owning
            provider's `dimension` attribute.
    """

    vectors: list[list[float]]
    model_id: str
    dimension: int


class EmbeddingProvider(ABC):
    """Vendor-agnostic embedding interface. All providers implement this.

    Concrete subclasses (LocalEmbeddingProvider, OpenAIEmbeddingProvider,
    MockEmbeddingProvider) declare the three attributes below as class-level
    constants and implement the two abstract methods. Calling code depends
    only on this interface, never on a specific provider — that's what lets
    EMBEDDING_BACKEND be a config change instead of a code change.

    Attributes:
        model_id: Identifier of the underlying model (e.g.
            "text-embedding-3-small"), also stored on every EmbeddingResult
            and written to the `model_id` column of the embeddings table.
        dimension: The fixed vector length this provider produces. Must
            match the `vector(N)` dimension of the provider's embeddings
            table — pgvector requires a fixed dimension per column.
        table_name: The embeddings table this provider's vectors belong in
            (e.g. "embeddings_bge_small"). Explicit rather than derived from
            model_id — see the comment below for why.
    """

    model_id: str
    dimension: int
    # Explicit, not derived from model_id — table_name_for() (Phase 4) just
    # returns this. See migrations/0003_create_embeddings_bge_small_table.sql's
    # comment for why deriving it from model_id was dropped.
    table_name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of documents/chunks for storage.

        Args:
            texts: The chunk texts to embed, in the order they should be
                returned in — implementations must not reorder or drop
                items (see bulk-ingest's reliance on this in Phase 4).

        Returns:
            An EmbeddingResult with one vector per input text, in the same
            order as `texts`.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for search.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for `text`, of length `self.dimension`.
        """
        ...
