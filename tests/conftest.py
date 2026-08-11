import os

import pytest

from db import get_connection


@pytest.fixture
def db_conn(monkeypatch):
    """Provide a psycopg connection to the test database for one test.

    Delegates to db.get_connection() — the same connect + register_vector
    logic the application code uses, not a second copy of it — after
    pointing $DATABASE_URL at $TEST_DATABASE_URL for the duration of the
    test (never the dev database). Closes the connection after the test —
    a fresh connection per test, not shared/pooled across tests.

    Args:
        monkeypatch: pytest's built-in fixture for setting env vars for the
            duration of one test.

    Yields:
        An open, vector-aware psycopg connection.
    """
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    conn = get_connection()
    yield conn
    conn.close()


@pytest.fixture
def local_ingest_env(monkeypatch):
    """Point db.get_connection()/embeddings.factory.get_provider() at the test DB and local model.

    Used by ingestion integration tests (tests/test_ingest.py,
    tests/test_bulk_ingest.py) so ingest_document/bulk_ingest_documents —
    which read $DATABASE_URL and $EMBEDDING_BACKEND internally rather than
    taking them as arguments — write to the test database via the local
    provider instead of the dev database or whatever backend is configured
    in the environment.

    Args:
        monkeypatch: pytest's built-in fixture for setting env vars for the
            duration of one test.

    Returns:
        None.
    """
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
