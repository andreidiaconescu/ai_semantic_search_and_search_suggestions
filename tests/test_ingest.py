import pytest

from chunking import chunk_text
from ingest import _rows_for, ingest_document


def test_rows_for_reattaches_vectors_to_correct_keys():
    """_rows_for zips (document_id, chunk_id) keys back onto their texts/vectors, in order.

    Args:
        None.

    Returns:
        None.
    """
    keys = [(1, 0), (1, 1), (2, 0)]
    texts = ["a", "b", "c"]
    vectors = [[0.1], [0.2], [0.3]]

    rows = _rows_for(keys, texts, vectors, "test-model")

    assert rows == [
        (1, 0, "a", [0.1], "test-model"),
        (1, 1, "b", [0.2], "test-model"),
        (2, 0, "c", [0.3], "test-model"),
    ]


@pytest.mark.integration
def test_ingest_document_creates_expected_chunks(local_ingest_env, db_conn):
    """A realistic sample document ingests into embeddings_bge_small with the right chunk count and document_id.

    Args:
        local_ingest_env: Fixture pointing DATABASE_URL/EMBEDDING_BACKEND at
            the test database and local provider.
        db_conn: pytest fixture (tests/conftest.py) providing a connection
            to the test database, used here to inspect the result directly.

    Returns:
        None.
    """
    pytest.importorskip("sentence_transformers")

    content = (
        "Vector search lets you find documents by meaning instead of exact "
        "keyword matches. It works by embedding text into a numeric vector "
        "and finding nearby vectors with an approximate nearest-neighbor "
        "index such as HNSW.\n\n"
        "This is useful for semantic search, recommendation, and "
        "deduplication, and it works the same way regardless of which "
        "embedding model produced the vectors."
    )
    source_uri = "test://ingest/doc-1"

    try:
        doc_id = ingest_document(source_uri, content)

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM embeddings_bge_small "
                "WHERE document_id = %s ORDER BY chunk_id",
                (doc_id,),
            )
            chunk_ids = [row[0] for row in cur.fetchall()]

        expected_chunk_count = len(chunk_text(content))
        assert chunk_ids == list(range(expected_chunk_count))
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE source_uri = %s", (source_uri,))
        db_conn.commit()


@pytest.mark.integration
def test_reingesting_same_source_uri_replaces_chunks(local_ingest_env, db_conn):
    """Re-ingesting an existing source_uri with different content replaces its chunks, doesn't accumulate them.

    Args:
        local_ingest_env: Fixture pointing DATABASE_URL/EMBEDDING_BACKEND at
            the test database and local provider.
        db_conn: pytest fixture (tests/conftest.py) providing a connection
            to the test database, used here to inspect the result directly.

    Returns:
        None.
    """
    pytest.importorskip("sentence_transformers")

    source_uri = "test://ingest/doc-2"
    try:
        first_id = ingest_document(source_uri, "short content")
        second_id = ingest_document(
            source_uri,
            "different, longer content than before, spanning quite a few more words",
        )

        assert second_id == first_id

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_text FROM embeddings_bge_small WHERE document_id = %s",
                (second_id,),
            )
            chunk_texts = [row[0] for row in cur.fetchall()]

        assert chunk_texts  # the new content still produced at least one chunk
        assert all("short content" not in text for text in chunk_texts)
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE source_uri = %s", (source_uri,))
        db_conn.commit()
