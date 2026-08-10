#!/usr/bin/env python3
"""Throwaway load-test: validate HNSW query latency on embeddings_bge_small
at a synthetic ~100k-row scale close to expected production volume.

Not part of the application — run manually against a scratch database,
never imported by app code. Re-running is safe: it skips loading if
synthetic rows (source_uri LIKE 'synthetic://%') are already present.

Usage:
    python migrations/loadtest_synthetic_bge_small.py --url postgresql://postgres:postgres@localhost:5432/postgres
"""
import argparse
import os
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DIM = 384
N_DOCUMENTS = 2_000
N_CHUNKS = 100_000
N_QUERIES = 50
LATENCY_TARGET_MS = 100
INSERT_BATCH_SIZE = 1_000


def random_unit_vectors(n: int, dim: int) -> np.ndarray:
    """Generate `n` random `dim`-dimensional vectors, each L2-normalized to unit length.

    Unit-normalized to realistically match what LocalEmbeddingProvider
    produces (see embeddings/local_provider.py), since the HNSW index uses
    inner product (§3.4 of the master plan), which only ranks like cosine
    similarity for normalized vectors.

    Args:
        n: Number of vectors to generate.
        dim: Length of each vector.

    Returns:
        An (n, dim) float32 array, each row a unit vector.
    """
    vecs = np.random.default_rng().normal(size=(n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def insert_synthetic_documents(conn, n_documents: int) -> list[int]:
    """Insert `n_documents` synthetic `documents` rows, batched for speed.

    Each row gets a distinguishable `source_uri` ("synthetic://doc/<i>") so
    `load_synthetic_data` can detect and skip re-loading on a later run.

    Args:
        conn: An open psycopg connection with `register_vector` applied.
        n_documents: How many synthetic document rows to insert.

    Returns:
        The inserted rows' `id` values, in insertion order.
    """
    doc_ids: list[int] = []
    with conn.cursor() as cur:
        for start in range(0, n_documents, INSERT_BATCH_SIZE):
            batch = range(start, min(start + INSERT_BATCH_SIZE, n_documents))
            placeholders = ",".join(["(%s, %s)"] * len(batch))
            args = [
                arg
                for i in batch
                for arg in (f"synthetic://doc/{i}", "synthetic load-test document")
            ]
            cur.execute(
                f"INSERT INTO documents (source_uri, content) VALUES {placeholders} RETURNING id",
                args,
            )
            doc_ids.extend(row[0] for row in cur.fetchall())
    conn.commit()
    return doc_ids


def insert_synthetic_embeddings(conn, doc_ids: list[int], n_chunks: int, dim: int) -> None:
    """Generate and bulk-insert `n_chunks` synthetic embedding rows.

    Spreads `n_chunks` evenly across `doc_ids` (`n_chunks // len(doc_ids)`
    chunks per document, sequential `chunk_id`s starting at 0 per document)
    and inserts them into `embeddings_bge_small` in batches of
    INSERT_BATCH_SIZE rows per statement, to avoid one network round-trip
    per row.

    Args:
        conn: An open psycopg connection with `register_vector` applied.
        doc_ids: The `documents.id` values to attach chunks to (from
            `insert_synthetic_documents`).
        n_chunks: Total number of embedding rows to generate, across all
            documents combined.
        dim: Vector dimension to generate (must match the `embedding`
            column's declared dimension).

    Returns:
        None. Prints how many rows were loaded when done.
    """
    chunks_per_doc = n_chunks // len(doc_ids)
    vectors = random_unit_vectors(len(doc_ids) * chunks_per_doc, dim)

    rows = []
    vec_idx = 0
    for doc_id in doc_ids:
        for chunk_id in range(chunks_per_doc):
            rows.append((doc_id, chunk_id, f"synthetic chunk {vec_idx}", vectors[vec_idx]))
            vec_idx += 1

    with conn.cursor() as cur:
        for start in range(0, len(rows), INSERT_BATCH_SIZE):
            batch = rows[start:start + INSERT_BATCH_SIZE]
            placeholders = ",".join(["(%s, %s, %s, %s)"] * len(batch))
            args = [value for row in batch for value in row]
            cur.execute(
                "INSERT INTO embeddings_bge_small (document_id, chunk_id, chunk_text, embedding) "
                f"VALUES {placeholders}",
                args,
            )
    conn.commit()
    print(f"loaded {len(rows)} synthetic embeddings across {len(doc_ids)} documents")


def load_synthetic_data(conn, n_documents: int, n_chunks: int, dim: int) -> None:
    """Load synthetic documents/embeddings, unless they're already present.

    Idempotency check: looks for any `documents` row whose `source_uri`
    starts with "synthetic://" and skips loading entirely if one is found,
    so re-running this script against the same database doesn't duplicate
    data or waste time re-generating vectors.

    Args:
        conn: An open psycopg connection with `register_vector` applied.
        n_documents: Number of synthetic documents to insert if not already loaded.
        n_chunks: Number of synthetic embedding rows to insert if not already loaded.
        dim: Vector dimension to generate.

    Returns:
        None.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE source_uri LIKE 'synthetic://%'")
        if cur.fetchone()[0] > 0:
            print("synthetic documents already present, skipping load")
            return

    doc_ids = insert_synthetic_documents(conn, n_documents)
    insert_synthetic_embeddings(conn, doc_ids, n_chunks, dim)


def show_query_plan(conn, dim: int) -> None:
    """Run EXPLAIN ANALYZE on one sample query and print the plan.

    Lets a human visually confirm Postgres is using the HNSW index (an
    "Index Scan using embeddings_bge_small_embedding_idx" line) rather than
    silently falling back to a sequential scan.

    Args:
        conn: An open psycopg connection with `register_vector` applied.
        dim: Vector dimension of the random query vector to search with.

    Returns:
        None. Prints the EXPLAIN ANALYZE plan lines to stdout.
    """
    query_vec = random_unit_vectors(1, dim)[0]
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100;")
        cur.execute(
            "EXPLAIN ANALYZE SELECT document_id FROM embeddings_bge_small "
            "ORDER BY embedding <#> %s LIMIT 10",
            (query_vec,),
        )
        print("\n--- EXPLAIN ANALYZE (confirms HNSW index usage) ---")
        for (line,) in cur.fetchall():
            print(line)
        print()


def measure_latency(conn, n_queries: int, dim: int) -> list[float]:
    """Time `n_queries` nearest-neighbor searches against a fresh random query vector each.

    Args:
        conn: An open psycopg connection with `register_vector` applied.
        n_queries: How many timed queries to run.
        dim: Vector dimension of each random query vector.

    Returns:
        One latency per query, in milliseconds, in the order the queries ran.
    """
    latencies = []
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100;")
        for _ in range(n_queries):
            query_vec = random_unit_vectors(1, dim)[0]
            start = time.perf_counter()
            cur.execute(
                "SELECT document_id FROM embeddings_bge_small "
                "ORDER BY embedding <#> %s LIMIT 10",
                (query_vec,),
            )
            cur.fetchall()
            latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def main() -> None:
    """CLI entry point: load synthetic data (if needed), then measure and report query latency.

    Parses `--url`/`--n-documents`/`--n-chunks`/`--n-queries`, loads
    synthetic data via `load_synthetic_data`, prints an `EXPLAIN ANALYZE`
    plan via `show_query_plan`, then times `--n-queries` queries via
    `measure_latency` and prints p50/p95/max latency plus a PASS/FAIL line
    against LATENCY_TARGET_MS.

    Args:
        None directly — reads `--url`/`--n-documents`/`--n-chunks`/
        `--n-queries` from sys.argv via argparse.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--n-documents", type=int, default=N_DOCUMENTS)
    parser.add_argument("--n-chunks", type=int, default=N_CHUNKS)
    parser.add_argument("--n-queries", type=int, default=N_QUERIES)
    args = parser.parse_args()
    if not args.url:
        parser.error("no --url given and DATABASE_URL is not set")

    conn = psycopg.connect(args.url)
    register_vector(conn)

    load_synthetic_data(conn, args.n_documents, args.n_chunks, DIM)
    show_query_plan(conn, DIM)

    latencies = sorted(measure_latency(conn, args.n_queries, DIM))
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    print(
        f"queries: {len(latencies)}  p50: {p50:.2f}ms  p95: {p95:.2f}ms  "
        f"max: {latencies[-1]:.2f}ms"
    )

    if p95 < LATENCY_TARGET_MS:
        print(f"PASS: p95 {p95:.2f}ms < {LATENCY_TARGET_MS}ms target")
    else:
        print(f"FAIL: p95 {p95:.2f}ms >= {LATENCY_TARGET_MS}ms target")

    conn.close()


if __name__ == "__main__":
    main()
