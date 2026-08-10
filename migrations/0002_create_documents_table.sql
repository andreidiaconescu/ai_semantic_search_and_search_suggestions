-- The single source of truth for original documents. One row per ingested
-- document, independent of any embedding model — this is what gets re-chunked
-- and re-embedded whenever a new/different model is adopted, so its `content`
-- must always be enough to fully reconstruct any embeddings table.
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    source_uri  TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Postgres has no built-in "ON UPDATE CURRENT_TIMESTAMP" (unlike MySQL), so
-- automatic updated_at maintenance requires a trigger function. Defined once
-- here (documents is the first table that needs it) and reused by every
-- other table's own BEFORE UPDATE trigger.
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_set_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
