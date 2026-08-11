"""Single-document ingestion: text -> chunks -> embeddings -> storage.

Idempotency policy: upsert by source_uri. Re-ingesting an existing
source_uri updates the documents row and replaces its chunks (deletes the
old ones, re-embeds from the new content) instead of accumulating
duplicates — see README's "Ingestion" section for the reasoning.
"""
import logging

from psycopg.types.json import Jsonb

from chunking import chunk_text
from db import get_connection
from embeddings.factory import get_provider
from storage import table_name_for

logger = logging.getLogger(__name__)


def _rows_for(
    keys: list[tuple[int, int]],
    texts: list[str],
    vectors: list[list[float]],
    model_id: str,
) -> list[tuple[int, int, str, list[float], str]]:
    """Reattach embedding vectors to their (document_id, chunk_id).

    This is the "rule that makes batching safe" from the master plan's
    §6.1: `embed_documents` never reorders or drops items, so `keys[i]`,
    `texts[i]`, and `vectors[i]` all refer to the same chunk for every
    `i` — zipping them back together is correct regardless of how many
    documents or embedding-call batches were involved.

    Args:
        keys: (document_id, chunk_id) pairs, one per text, in the same
            order the texts were embedded in.
        texts: The chunk texts that were embedded, same order as `keys`.
        vectors: The embedding vectors returned for `texts`, same order.
        model_id: The model identifier to stamp on every row.

    Returns:
        One (document_id, chunk_id, chunk_text, embedding, model_id) tuple
        per input chunk, ready for an `INSERT INTO embeddings_<model>`.
    """
    return [
        (doc_id, chunk_id, text, vector, model_id)
        for (doc_id, chunk_id), text, vector in zip(keys, texts, vectors)
    ]


def _find_document_id(cur, source_uri: str) -> int | None:
    """Look up an existing documents.id by source_uri.

    Args:
        cur: An open psycopg cursor.
        source_uri: The source_uri to look up.

    Returns:
        The matching documents.id, or None if no row has that source_uri.
    """
    cur.execute("SELECT id FROM documents WHERE source_uri = %s", (source_uri,))
    row = cur.fetchone()
    return row[0] if row else None


def _upsert_document(cur, source_uri: str, content: str, metadata: dict) -> int:
    """Insert a new documents row, or update the existing one for source_uri.

    Implements the upsert-by-source_uri idempotency policy: a document is
    identified by its source_uri, not a caller-supplied id. Re-ingesting an
    existing source_uri overwrites its content/metadata in place rather
    than creating a second row.

    Args:
        cur: An open psycopg cursor.
        source_uri: Identifies the document; existing rows are matched on
            this.
        content: Full document text to store.
        metadata: Arbitrary JSON-able metadata to store in the `metadata`
            JSONB column.

    Returns:
        The document's id — newly inserted, or the existing row's id.
    """
    doc_id = _find_document_id(cur, source_uri)
    if doc_id is None:
        cur.execute(
            "INSERT INTO documents (source_uri, content, metadata) "
            "VALUES (%s, %s, %s) RETURNING id",
            (source_uri, content, Jsonb(metadata)),
        )
        doc_id = cur.fetchone()[0]
        logger.info("inserted new document id=%s source_uri=%s", doc_id, source_uri)
    else:
        cur.execute(
            "UPDATE documents SET content = %s, metadata = %s WHERE id = %s",
            (content, Jsonb(metadata), doc_id),
        )
        logger.info("updated existing document id=%s source_uri=%s", doc_id, source_uri)
    return doc_id


def _delete_existing_chunks(cur, table: str, doc_id: int) -> None:
    """Delete all embedding rows for `doc_id` from `table`, ahead of re-embedding.

    A no-op if `doc_id` has no existing chunks (e.g. it was just inserted),
    so callers can always call this unconditionally before inserting fresh
    chunks — that's what makes re-ingestion safe against accumulating
    stale chunks.

    Args:
        cur: An open psycopg cursor.
        table: The embeddings table to delete from (the active provider's
            `table_name`, via `storage.table_name_for`).
        doc_id: The documents.id whose chunks should be removed.

    Returns:
        None.
    """
    cur.execute(f"DELETE FROM {table} WHERE document_id = %s", (doc_id,))


def ingest_document(
    source_uri: str, content: str, metadata: dict | None = None
) -> int:
    """Ingest a single document: chunk, embed, and store it.

    Upsert-by-source_uri: if a document with this source_uri already
    exists, its content/metadata are updated and its old chunks are
    deleted before the new content is re-chunked and re-embedded — so
    re-ingesting the same source never accumulates duplicate chunks.
    Runs as one transaction: if embedding or any DB step fails, nothing
    (including the document upsert) is committed.

    Args:
        source_uri: Identifies the document (e.g. a file path or URL).
            Existing documents are matched on this.
        content: The full document text.
        metadata: Optional JSON-able metadata to store alongside the
            document. Defaults to `{}`.

    Returns:
        The document's id (new or existing).
    """
    provider = get_provider()
    table = table_name_for(provider)

    with get_connection() as conn, conn.cursor() as cur:
        doc_id = _upsert_document(cur, source_uri, content, metadata or {})
        _delete_existing_chunks(cur, table, doc_id)

        chunks = chunk_text(content)
        result = provider.embed_documents(chunks)
        keys = [(doc_id, i) for i in range(len(chunks))]
        rows = _rows_for(keys, chunks, result.vectors, provider.model_id)

        cur.executemany(
            f"INSERT INTO {table} (document_id, chunk_id, chunk_text, embedding, model_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            rows,
        )
        conn.commit()

    logger.info(
        "ingested document id=%s source_uri=%s chunks=%d", doc_id, source_uri, len(chunks)
    )
    return doc_id
