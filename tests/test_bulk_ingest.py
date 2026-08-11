import pytest

from bulk_ingest import bulk_ingest_documents
from chunking import chunk_text
from embeddings.mock_provider import MockEmbeddingProvider


class _FakeCursor:
    """A minimal stand-in for a psycopg cursor, backed by in-memory lists.

    Simulates just enough behavior for ingest.py/bulk_ingest.py's queries:
    `documents` lookups always report "not found" (so every upsert takes
    the insert path), inserts return incrementing fake ids, and
    `executemany`/`execute` calls are simply recorded for later assertion
    — nothing touches a real database.

    Attributes:
        executemany_calls: Every (query, rows) passed to `executemany`, in
            call order.
        execute_calls: Every (query, params) passed to `execute`, in call
            order.
    """

    def __init__(self, first_id: int = 1):
        """Set up empty call logs and the first id to hand out on insert.

        Args:
            first_id: The id returned for the first `INSERT INTO documents`
                call; each subsequent insert increments by 1.

        Returns:
            None.
        """
        self.executemany_calls: list[tuple[str, list[tuple]]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self._next_id = first_id
        self._last_result = None

    def execute(self, query: str, params: tuple | None = None) -> None:
        """Record the call and simulate a result for INSERT INTO documents.

        Args:
            query: The SQL text (only its shape is inspected — never
                actually executed).
            params: The query parameters, if any.

        Returns:
            None.
        """
        self.execute_calls.append((query, params))
        self._last_result = None
        if query.strip().startswith("INSERT INTO documents"):
            self._last_result = (self._next_id,)
            self._next_id += 1

    def fetchone(self) -> tuple | None:
        """Return whatever result `execute` most recently simulated.

        Args:
            None.

        Returns:
            A one-element tuple `(id,)` right after a simulated
            `INSERT INTO documents`, otherwise None (simulating "not found"
            for the `SELECT id FROM documents WHERE source_uri = ...`
            lookup, so every upsert in these tests takes the insert path).
        """
        return self._last_result

    def executemany(self, query: str, rows) -> None:
        """Record the query and rows instead of executing them.

        Args:
            query: The SQL text.
            rows: The rows that would have been inserted.

        Returns:
            None.
        """
        self.executemany_calls.append((query, list(rows)))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _FakeConnection:
    """A minimal stand-in for a psycopg connection, backed by a `_FakeCursor`.

    Attributes:
        cursor_obj: The single `_FakeCursor` this connection always
            returns — bulk_ingest.py only ever opens one cursor per call,
            so a single shared instance is enough to observe everything
            that happened.
        committed: Whether `commit()` has been called.
    """

    def __init__(self):
        """Create the fake connection with a fresh, uncommitted cursor.

        Args:
            None.

        Returns:
            None.
        """
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def cursor(self) -> _FakeCursor:
        """Return the (single, shared) fake cursor for this connection.

        Args:
            None.

        Returns:
            The `_FakeCursor` instance backing this connection.
        """
        return self.cursor_obj

    def commit(self) -> None:
        """Record that commit() was called.

        Args:
            None.

        Returns:
            None.
        """
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _FailingProvider(MockEmbeddingProvider):
    """A MockEmbeddingProvider that raises on its second embed_documents call.

    Used to simulate an embedding failure partway through a multi-batch
    bulk_ingest_documents call, to verify the fail-whole-batch policy.

    Attributes:
        calls: How many times embed_documents has been called so far.
    """

    def __init__(self):
        """Initialize the underlying mock provider and the call counter.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()
        self.calls = 0

    def embed_documents(self, texts: list[str]):
        """Raise on the 2nd call; otherwise behave like MockEmbeddingProvider.

        Args:
            texts: The chunk texts to embed.

        Returns:
            An EmbeddingResult, as MockEmbeddingProvider.embed_documents
            would produce — except on the 2nd call, which raises instead.

        Raises:
            RuntimeError: On the 2nd call, simulating a failed embedding batch.
        """
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated embedding failure")
        return super().embed_documents(texts)


def test_bulk_ingest_reattaches_vectors_across_documents_and_batches(monkeypatch):
    """Chunks from multiple documents, embedded across multiple batches, land on the correct document_id/chunk_id.

    Uses MockEmbeddingProvider (no real model) and a fake DB connection
    (no real Postgres), so this stays a fast, network/DB-free unit test of
    just the row-building/reattachment logic in bulk_ingest_documents.
    `batch_size=1` forces one embed_documents call per chunk, exercising
    the multi-batch path.

    Args:
        monkeypatch: pytest's built-in fixture for patching module-level
            names for the duration of one test.

    Returns:
        None.
    """
    fake_conn = _FakeConnection()
    monkeypatch.setattr("bulk_ingest.get_connection", lambda: fake_conn)
    monkeypatch.setattr("bulk_ingest.get_provider", lambda: MockEmbeddingProvider())

    docs = [
        {"source_uri": "doc-a", "content": "alpha beta"},
        {"source_uri": "doc-b", "content": "gamma delta epsilon"},
    ]

    doc_ids = bulk_ingest_documents(docs, batch_size=1)

    assert doc_ids == [1, 2]
    assert len(fake_conn.cursor_obj.executemany_calls) == 1
    (_, rows) = fake_conn.cursor_obj.executemany_calls[0]

    by_doc: dict[int, list[tuple[int, str]]] = {}
    for doc_id, chunk_id, text, _vector, _model_id in rows:
        by_doc.setdefault(doc_id, []).append((chunk_id, text))

    assert by_doc[1] == [(0, "alpha beta")]
    assert by_doc[2] == [(0, "gamma delta epsilon")]
    assert fake_conn.committed


def test_bulk_ingest_writes_nothing_if_any_batch_fails(monkeypatch):
    """If one embedding batch raises, nothing is written — not even the earlier document upserts.

    Args:
        monkeypatch: pytest's built-in fixture for patching module-level
            names for the duration of one test.

    Returns:
        None.
    """
    fake_conn = _FakeConnection()
    failing_provider = _FailingProvider()
    monkeypatch.setattr("bulk_ingest.get_connection", lambda: fake_conn)
    monkeypatch.setattr("bulk_ingest.get_provider", lambda: failing_provider)

    docs = [
        {"source_uri": "doc-a", "content": "alpha beta"},
        {"source_uri": "doc-b", "content": "gamma delta epsilon"},
    ]

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        bulk_ingest_documents(docs, batch_size=1)

    assert fake_conn.cursor_obj.executemany_calls == []
    assert not fake_conn.committed


@pytest.mark.integration
def test_bulk_ingest_document_ids_and_chunk_counts(local_ingest_env, db_conn):
    """Bulk-ingesting a few real documents against local + real Postgres lands the right document_ids and chunk counts.

    Args:
        local_ingest_env: Fixture pointing DATABASE_URL/EMBEDDING_BACKEND at
            the test database and local provider.
        db_conn: pytest fixture (tests/conftest.py) providing a connection
            to the test database, used here to inspect the result directly.

    Returns:
        None.
    """
    pytest.importorskip("sentence_transformers")

    docs = [
        {"source_uri": "test://bulk/doc-1", "content": "Alpha document about search."},
        {
            "source_uri": "test://bulk/doc-2",
            "content": "Beta document about vectors and retrieval.",
        },
    ]
    source_uris = [d["source_uri"] for d in docs]

    try:
        doc_ids = bulk_ingest_documents(docs, batch_size=10)
        assert len(doc_ids) == 2

        with db_conn.cursor() as cur:
            for doc_id, doc in zip(doc_ids, docs):
                cur.execute(
                    "SELECT count(*) FROM embeddings_bge_small WHERE document_id = %s",
                    (doc_id,),
                )
                (chunk_count,) = cur.fetchone()
                assert chunk_count == len(chunk_text(doc["content"]))
    finally:
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE source_uri = ANY(%s)", (source_uris,)
            )
        db_conn.commit()
