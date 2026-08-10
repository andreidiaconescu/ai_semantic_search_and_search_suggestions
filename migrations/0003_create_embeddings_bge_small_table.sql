-- Embeddings table for the local model (BAAI/bge-small-en-v1.5, 384-dim).
-- Table name is intentionally the literal "embeddings_bge_small" rather than
-- derived from the model_id string — see storage.table_name_for's docstring
-- (added in Phase 3) for why the two must be kept in sync explicitly.
CREATE TABLE embeddings_bge_small (
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id    INT NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    model_id    TEXT NOT NULL DEFAULT 'bge-small-en-v1.5',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (document_id, chunk_id)
);

-- Reuses the set_updated_at() function created in 0002.
CREATE TRIGGER embeddings_bge_small_set_updated_at
    BEFORE UPDATE ON embeddings_bge_small
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
