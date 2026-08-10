-- Inner product, not cosine: bge-small-en-v1.5 output is unit-normalized
-- (see LocalEmbeddingProvider, normalize_embeddings=True), so vector_ip_ops
-- ranks identically to vector_cosine_ops here but skips the norm division,
-- which is cheaper per comparison at both build and query time.
--
-- Plain (non-CONCURRENTLY) index build: the table is empty at this point in
-- the migration sequence, and the runner wraps each migration in a
-- transaction, which CREATE INDEX CONCURRENTLY cannot run inside anyway.
-- A migration against an already-populated table (e.g. a later environment
-- bootstrapped from a snapshot) would need a different, non-transactional
-- approach — revisit if/when that comes up.
CREATE INDEX ON embeddings_bge_small
  USING hnsw (embedding vector_ip_ops)
  WITH (m = 16, ef_construction = 64);
