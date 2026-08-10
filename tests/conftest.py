import os

import psycopg
import pytest
from pgvector.psycopg import register_vector


@pytest.fixture
def db_conn():
    """Provide a psycopg connection to the test database for one test.

    Connects to $TEST_DATABASE_URL (never the dev database), registers
    pgvector's adapter so `vector` columns convert to/from Python lists
    automatically, and closes the connection after the test — a fresh
    connection per test, not shared/pooled across tests.

    Yields:
        An open, vector-aware psycopg connection.
    """
    url = os.environ["TEST_DATABASE_URL"]
    conn = psycopg.connect(url)
    register_vector(conn)
    yield conn
    conn.close()
