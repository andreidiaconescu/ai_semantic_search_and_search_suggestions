import os

import psycopg
import pytest
from pgvector.psycopg import register_vector


@pytest.fixture
def db_conn():
    url = os.environ["TEST_DATABASE_URL"]
    conn = psycopg.connect(url)
    register_vector(conn)
    yield conn
    conn.close()
