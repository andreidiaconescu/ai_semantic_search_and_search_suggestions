import os

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import EmbeddingProvider, EmbeddingResult

# Only retry errors that are plausibly transient. Auth/bad-request errors
# should fail fast rather than retry — retrying them just delays a failure
# that retrying can never fix.
_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Hosted embedding provider using OpenAI's text-embedding-3-small API.

    embed_documents is retried with exponential backoff (via tenacity) on
    transient failures (rate limits, timeouts, connection errors);
    non-transient errors (e.g. authentication) propagate immediately rather
    than being retried. Selected via EMBEDDING_BACKEND=openai; requires
    OPENAI_API_KEY.

    Attributes:
        model_id: OpenAI model identifier ("text-embedding-3-small").
        dimension: Fixed output size of this model (1536), matching the
            `embedding vector(1536)` column an `embeddings_openai_small`
            table would use.
        table_name: The embeddings table this provider's vectors belong in
            ("embeddings_openai_small").
        client: The OpenAI SDK client used for all API calls, created in
            __init__ from the given or environment API key.
    """

    model_id = "text-embedding-3-small"
    dimension = 1536
    table_name = "embeddings_openai_small"

    def __init__(self, api_key: str | None = None):
        """Create the OpenAI client, using `api_key` or $OPENAI_API_KEY.

        Args:
            api_key: Explicit API key to use. If omitted (the normal case),
                falls back to the $OPENAI_API_KEY environment variable.

        Returns:
            None.

        Raises:
            KeyError: If `api_key` is omitted and $OPENAI_API_KEY isn't set.
        """
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts via the OpenAI API, retrying transient failures.

        Args:
            texts: The chunk texts to embed, in order.

        Returns:
            An EmbeddingResult with one 1536-dim vector per input text, in
            the same order as `texts` (restored via the response's `index`
            field regardless of the order the API returned items in).

        Raises:
            openai.RateLimitError, openai.APIConnectionError,
            openai.APITimeoutError: If still failing after 5 retries with
                exponential backoff.
            openai.APIStatusError subclasses (e.g. AuthenticationError,
            BadRequestError): Raised immediately, without retrying, since
                these can never succeed on retry.
        """
        resp = self.client.embeddings.create(model=self.model_id, input=texts)
        # Defensive: OpenAI's response includes an explicit `index` per item.
        # Sort by it rather than trusting array order, since the bulk-ingest
        # pipeline (Phase 4) depends on strict input/output order alignment.
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors = [d.embedding for d in ordered]
        return EmbeddingResult(vectors, self.model_id, self.dimension)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string — delegates to embed_documents for the one text.

        Args:
            text: The query text to embed.

        Returns:
            The 1536-dim embedding vector for `text`.
        """
        return self.embed_documents([text]).vectors[0]
