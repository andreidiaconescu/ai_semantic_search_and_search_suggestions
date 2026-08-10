from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider, EmbeddingResult


class LocalEmbeddingProvider(EmbeddingProvider):
    """Self-hosted embedding provider: BAAI/bge-small-en-v1.5 via sentence-transformers.

    Runs fully offline after the model is downloaded and cached on first use
    (~130MB). No API key and no per-call cost. Selected via
    EMBEDDING_BACKEND=local (the default).

    Attributes:
        model_id: Hugging Face model identifier
            ("BAAI/bge-small-en-v1.5"), used to download/cache the model.
        dimension: Fixed output size of this model (384), matching the
            `embedding vector(384)` column in `embeddings_bge_small`.
        table_name: The embeddings table this provider's vectors belong in
            ("embeddings_bge_small").
        model: The loaded SentenceTransformer instance, created in
            __init__ and reused for every embed_documents/embed_query call.
    """

    model_id = "BAAI/bge-small-en-v1.5"
    dimension = 384
    table_name = "embeddings_bge_small"

    def __init__(self, device: str = "cpu"):
        """Load the model onto `device`, downloading it to the local cache if needed.

        Args:
            device: The torch device to load the model onto — "cpu"
                (default) or "cuda" if a GPU is available.

        Returns:
            None.
        """
        self.model = SentenceTransformer(self.model_id, device=device)

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts with the local model.

        Args:
            texts: The chunk texts to embed, in order.

        Returns:
            An EmbeddingResult with one 384-dim, unit-normalized vector per
            input text, in the same order as `texts`.
        """
        vectors = self.model.encode(texts, normalize_embeddings=True).tolist()
        return EmbeddingResult(vectors, self.model_id, self.dimension)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string, applying BGE's query instruction prefix.

        Args:
            text: The raw query text (the instruction prefix is added
                internally — callers should not add it themselves).

        Returns:
            The 384-dim, unit-normalized embedding vector for the
            prefixed query text.
        """
        # BGE models want an instruction prefix on queries for best retrieval quality
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return self.model.encode([prefixed], normalize_embeddings=True).tolist()[0]
