import time


def _columns(db_conn, table_name: str) -> set[str]:
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table_name,),
        )
        return {row[0] for row in cur.fetchall()}


def test_documents_table_columns(db_conn):
    assert _columns(db_conn, "documents") == {
        "id",
        "source_uri",
        "content",
        "metadata",
        "created_at",
        "updated_at",
    }


def test_embeddings_bge_small_table_columns(db_conn):
    assert _columns(db_conn, "embeddings_bge_small") == {
        "document_id",
        "chunk_id",
        "chunk_text",
        "embedding",
        "model_id",
        "created_at",
        "updated_at",
    }


def test_embedding_column_dimension_is_384(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'embeddings_bge_small'::regclass AND attname = 'embedding'"
        )
        (dimension,) = cur.fetchone()
    assert dimension == 384


def test_hnsw_index_exists_with_inner_product_ops(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'embeddings_bge_small'"
        )
        indexdefs = [row[0] for row in cur.fetchall()]
    assert any(
        "USING hnsw" in d and "vector_ip_ops" in d for d in indexdefs
    ), f"no HNSW/vector_ip_ops index found among: {indexdefs}"


def test_updated_at_auto_updates_on_change(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (source_uri, content) VALUES (%s, %s) "
            "RETURNING id, created_at, updated_at",
            ("test://trigger-check", "original content"),
        )
        doc_id, created_at, updated_at_before = cur.fetchone()
    db_conn.commit()

    try:
        time.sleep(0.01)  # ensure now() in the trigger differs from the insert timestamp
        with db_conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET content = %s WHERE id = %s "
                "RETURNING created_at, updated_at",
                ("changed content", doc_id),
            )
            created_at_after, updated_at_after = cur.fetchone()
        db_conn.commit()

        assert created_at_after == created_at
        assert updated_at_after > updated_at_before
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        db_conn.commit()
