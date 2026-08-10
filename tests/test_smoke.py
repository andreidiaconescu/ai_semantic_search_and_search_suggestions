def test_vector_extension_enabled(db_conn):
    """The `vector` extension (migration 0001) is enabled on the test database.

    A basic sanity check that migrations actually ran before anything
    schema-specific (tests/test_schema.py) is tested.

    Args:
        db_conn: pytest fixture (tests/conftest.py) providing a connection
            to the test database.

    Returns:
        None.
    """
    with db_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        assert cur.fetchone() is not None
