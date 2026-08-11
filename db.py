"""Database connection helper.

See storage.py for table-name resolution and ingest.py/bulk_ingest.py for
what actually uses connections from here.
"""
import os

import psycopg
from pgvector.psycopg import register_vector


def get_connection() -> psycopg.Connection:
    """Open a new connection to $DATABASE_URL with pgvector adaptation enabled.

    Each call opens a fresh connection — no pooling yet (deferred to
    Phase 7, per the master plan). `register_vector` teaches psycopg to
    adapt Python lists/numpy arrays to/from the `vector` type automatically,
    so callers never need to manually serialize embeddings.

    Args:
        None.

    Returns:
        A new, vector-aware psycopg connection.

    Raises:
        KeyError: If $DATABASE_URL is not set.
    """
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn
