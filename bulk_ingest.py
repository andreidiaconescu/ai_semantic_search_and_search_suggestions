"""Batched multi-document ingestion.

Batches embedding calls across documents (master plan §6.1) to cut
round-trips and respect commercial API rate limits, while keeping the same
upsert-by-source_uri idempotency policy as ingest.py. Partial-batch-failure
policy: fail the whole call — the entire operation runs as a single
transaction, so if any embedding batch raises, nothing is committed at
all (not even the document upserts) — see README's "Ingestion" section.
"""
import logging

from chunking import chunk_text
from db import get_connection
from embeddings.factory import get_provider
from ingest import _delete_existing_chunks, _rows_for, _upsert_document
from storage import table_name_for

logger = logging.getLogger(__name__)


def bulk_ingest_documents(docs: list[dict], batch_size: int = 200) -> list[int]:
    """Ingest many documents, batching embedding calls across all of them.

    Upserts every document first (by source_uri, same policy as
    `ingest_document`), deletes their old chunks, flattens all chunks
    across all documents into one list, and embeds it in batches of
    `batch_size` chunks (which may span multiple documents). Everything
    — document upserts, chunk deletes, and chunk inserts — happens in one
    transaction: if any embedding call raises partway through, the whole
    call rolls back and nothing is written, including the document
    upserts from step 1 (fail-whole-batch, taken to the level of the
    whole bulk call rather than just the one failing batch, for a
    simpler, stronger guarantee).

    Args:
        docs: Each dict must have "source_uri" and "content" keys, and may
            have an optional "metadata" dict.
        batch_size: How many chunks to send to `embed_documents` per call,
            regardless of how many documents they came from.

    Returns:
        The ingested documents' ids, in the same order as `docs`.
    """
    provider = get_provider()
    table = table_name_for(provider)

    with get_connection() as conn, conn.cursor() as cur:
        doc_ids = [
            _upsert_document(
                cur, doc["source_uri"], doc["content"], doc.get("metadata") or {}
            )
            for doc in docs
        ]
        for doc_id in doc_ids:
            _delete_existing_chunks(cur, table, doc_id)

        flat_texts: list[str] = []
        flat_keys: list[tuple[int, int]] = []
        for doc_id, doc in zip(doc_ids, docs):
            chunks = chunk_text(doc["content"])
            for chunk_id, chunk in enumerate(chunks):
                flat_texts.append(chunk)
                flat_keys.append((doc_id, chunk_id))

        all_rows: list[tuple[int, int, str, list[float], str]] = []
        for start in range(0, len(flat_texts), batch_size):
            batch_texts = flat_texts[start : start + batch_size]
            batch_keys = flat_keys[start : start + batch_size]
            # Fail-whole-batch: any exception here propagates out of this
            # function entirely; the `with get_connection()` block rolls
            # back on exception, so nothing committed above survives either.
            result = provider.embed_documents(batch_texts)
            all_rows.extend(
                _rows_for(batch_keys, batch_texts, result.vectors, provider.model_id)
            )

        cur.executemany(
            f"INSERT INTO {table} (document_id, chunk_id, chunk_text, embedding, model_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            all_rows,
        )
        conn.commit()

    logger.info(
        "bulk-ingested %d documents, %d chunks", len(doc_ids), len(all_rows)
    )
    return doc_ids
