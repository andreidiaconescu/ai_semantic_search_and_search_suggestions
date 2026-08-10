"""Vendor-agnostic embedding providers.

See embeddings/base.py for the EmbeddingProvider interface every provider
implements, and embeddings/factory.py's get_provider() for how a provider
is selected at runtime via the EMBEDDING_BACKEND env var.
"""
