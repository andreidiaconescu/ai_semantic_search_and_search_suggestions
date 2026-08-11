# Running migrations

`migrations/runner.py` applies every `.sql` file in `migrations/`, in filename
order, to the target Postgres database. Applied migrations are tracked in a
`schema_migrations` table, so re-running the script is safe — it only applies
files it hasn't seen before.

## 1. Start Postgres

```bash
docker compose up -d db
```

This brings up the `pgvector/pgvector:pg16` dev database on `localhost:5432`
(see `docker-compose.yml`). A separate `test_db` service is available on
`localhost:5433` for running tests against.

## 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Set the database URL

Copy `.env.example` to `.env` and adjust if needed, then export
`DATABASE_URL` (the runner does not load `.env` files itself):

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

## 4. Run the migrations

```bash
python migrations/runner.py
```

This connects using `$DATABASE_URL`, creates `schema_migrations` if it
doesn't exist, and applies any pending `migrations/*.sql` file inside its own
transaction, printing `applied <filename>` for each one it runs.

To target a different database without exporting the env var, pass `--url`
directly:

```bash
python migrations/runner.py --url postgresql://postgres:postgres@localhost:5433/postgres
```

(Useful for applying migrations to the `test_db` instance.)

## 5. Keep the dev and test databases in sync

Locally, `db` (`:5432`) and `test_db` (`:5433`) are separate Postgres
instances (see `docker-compose.yml`), so new migrations must be applied to
**both** before running `pytest`:

```bash
python migrations/runner.py --url "$DATABASE_URL"
python migrations/runner.py --url "$TEST_DATABASE_URL"
```

CI only has one Postgres service and points both `DATABASE_URL` and
`TEST_DATABASE_URL` at it, applying migrations once (see
`.github/workflows/ci.yml` / `.gitlab-ci.yml`) — the two-command form above
is only needed for local dev.

## Running tests

> **Keep this section in sync**: whenever a test is added, removed, or its
> behavior changes, update the corresponding entry below in the same change.

Two kinds of tests live side by side here:

- **`tests/test_smoke.py`, `tests/test_schema.py`** — hit a real Postgres.
  They use the `db_conn` fixture in `tests/conftest.py`, which connects to
  `TEST_DATABASE_URL` (via `pgvector.psycopg.register_vector`, so `vector`
  columns adapt to/from Python lists automatically) — every test in these
  files runs against `test_db` (`:5433`), never the dev database.
- **`tests/test_embeddings_*.py`** — pure Python, no database at all. Most
  are plain unit tests (no network either — real API/model calls are
  mocked); a few are marked `@pytest.mark.integration` because they need a
  real model download or a real API key (see "Running the integration
  tests" below).

### Prerequisites

```bash
docker compose up -d test_db
export TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/postgres
python migrations/runner.py --url "$TEST_DATABASE_URL"   # apply all migrations to test_db
pip install -r requirements-dev.txt -r requirements-openai.txt   # gets pytest + the openai SDK
```

The Postgres steps are only needed for `test_schema.py`/`test_smoke.py`;
`test_embeddings_*.py` doesn't touch the database and will pass without
`test_db` running. `requirements-openai.txt` is needed because
`embeddings/openai_provider.py` imports the `openai` SDK at module level —
even the tests that mock the network call still need the package installed
to construct the class.

### Run everything (excluding integration tests — this is what CI runs)

```bash
pytest -m "not integration"
```

Plain `pytest` (no `-m` filter) also works locally and is usually safe even
without a real API key or `sentence-transformers` installed — the
integration tests skip themselves (via `skipif`/`importorskip`) rather than
failing when their prerequisites aren't met. `-m "not integration"` is the
explicit, intentional version of the same result, and what CI relies on.

### Run a single file, or a single test

```bash
pytest tests/test_schema.py            # one file
pytest tests/test_schema.py -v         # ... with per-test output
pytest -k test_hnsw_index_exists       # one test, by name, from any file
```

### `tests/test_smoke.py`

| Test | What it checks |
|---|---|
| `test_vector_extension_enabled` | The `vector` extension (from migration `0001`) is actually enabled on the connected database — a sanity check that migrations ran at all before anything schema-specific is tested. |

### `tests/test_schema.py`

Verifies the Phase 2 schema (migrations `0002`–`0004`) was created correctly — a correctness check on the migrations themselves, not on application code.

| Test | What it checks |
|---|---|
| `test_documents_table_columns` | `documents` has exactly the expected columns (`id`, `source_uri`, `content`, `metadata`, `created_at`, `updated_at`), via `information_schema.columns`. |
| `test_embeddings_bge_small_table_columns` | `embeddings_bge_small` has exactly the expected columns (`document_id`, `chunk_id`, `chunk_text`, `embedding`, `model_id`, `created_at`, `updated_at`). |
| `test_embedding_column_dimension_is_384` | The `embedding` column is `vector(384)`, checked via `pg_attribute.atttypmod` — catches an accidental dimension mismatch with the local model (`bge-small-en-v1.5`). |
| `test_hnsw_index_exists_with_inner_product_ops` | An HNSW index using `vector_ip_ops` (inner product, not cosine — see §3.4 of the master plan) exists on `embeddings_bge_small`, via `pg_indexes.indexdef`. |
| `test_updated_at_auto_updates_on_change` | Inserts a `documents` row, updates it, and asserts the `set_updated_at()` trigger bumped `updated_at` while leaving `created_at` unchanged; deletes the row it inserted afterward so it doesn't leave test data behind. |

### `tests/test_embeddings_base.py`

Checks the `EmbeddingProvider` ABC/`EmbeddingResult` contract itself.

| Test | What it checks |
|---|---|
| `test_embedding_provider_cannot_be_instantiated_directly` | `EmbeddingProvider()` raises `TypeError` — it's an ABC with abstract methods, so only subclasses that implement `embed_documents`/`embed_query` can be constructed. |
| `test_embedding_result_holds_its_fields` | `EmbeddingResult(vectors=..., model_id=..., dimension=...)` stores exactly what it's given. |

### `tests/test_embeddings_mock.py`

Checks `MockEmbeddingProvider` — the deterministic, network-free provider used to test downstream code without a real model.

| Test | What it checks |
|---|---|
| `test_embed_documents_preserves_order_and_count` | `embed_documents(texts)` returns one vector per input text, in order, and `EmbeddingResult.model_id`/`dimension` match the provider's. |
| `test_same_text_gives_same_vector` | Calling `embed_query` twice with the same text returns identical vectors (determinism). |
| `test_different_text_gives_different_vector` | Different input text produces a different vector. |
| `test_vectors_are_unit_normalized` | Every vector has L2 norm ≈ 1 — matches the normalization convention the real providers use (§3.4 of the master plan). |
| `test_embed_query_matches_embed_documents_for_same_text` | `embed_query(t) == embed_documents([t]).vectors[0]` — internal consistency between the two entry points. |
| `test_dimension_is_configurable` | `MockEmbeddingProvider(dimension=16)` produces 16-length vectors. |

### `tests/test_embeddings_factory.py`

Checks `get_provider()` — that `EMBEDDING_BACKEND` alone selects the provider, with no other code changes (the master plan's "no vendor lock-in" exit criterion).

| Test | What it checks |
|---|---|
| `test_mock_backend` | `EMBEDDING_BACKEND=mock` → a `MockEmbeddingProvider` instance. |
| `test_openai_backend` | `EMBEDDING_BACKEND=openai` (with a dummy `OPENAI_API_KEY`, no real network) → an `OpenAIEmbeddingProvider` with the expected `model_id`/`dimension`/`table_name`. |
| `test_local_backend` | `EMBEDDING_BACKEND=local` → a `LocalEmbeddingProvider`, with `SentenceTransformer` monkeypatched so it doesn't actually download/load a model. Skipped if `sentence-transformers` isn't installed. |
| `test_default_backend_is_local` | No `EMBEDDING_BACKEND` set at all → defaults to local (matches `.env.example`). Same monkeypatch/skip treatment as above. |
| `test_unknown_backend_raises` | An unrecognized `EMBEDDING_BACKEND` value raises `ValueError`. |

### `tests/test_embeddings_openai.py`

Checks `OpenAIEmbeddingProvider`. The first four tests mock `client.embeddings.create` and use a dummy API key — no real network call. The last one is a real, opt-in integration test.

| Test | What it checks |
|---|---|
| `test_embed_documents_sorts_by_index_defensively` | If the API response's `data` items come back out of order, the provider still returns vectors matching the *input* order (sorts by the response's `index` field first). |
| `test_retries_on_retryable_error_then_succeeds` | Two simulated `RateLimitError`s followed by success still returns the correct result — proves the `tenacity` retry wiring works. |
| `test_non_retryable_error_is_not_retried` | A simulated `AuthenticationError` propagates immediately (call count stays at 1) — retrying an unfixable error would just waste time. |
| `test_embed_query_delegates_to_embed_documents` | `embed_query` returns the same vector `embed_documents([text])` would for that one text. |
| `test_ingest_row_building_works_for_the_openai_path` | Proves `ingest.py`'s `_rows_for` helper reattaches OpenAI-shaped vectors to `(document_id, chunk_id)` correctly — no real network, and no write to `embeddings_openai_small` (that table doesn't exist yet; see "Ingestion" below). |
| `test_real_openai_embedding_dimension_matches` *(integration)* | A real API call with a fixed short input returns a vector of length 1536 (`provider.dimension`). Skipped automatically unless `OPENAI_API_KEY` is set to a real key. |

### `tests/test_embeddings_local.py` *(integration)*

Checks `LocalEmbeddingProvider` against the real `sentence-transformers` model. The whole file is skipped if `sentence-transformers` isn't installed (`pytest.importorskip` at module level) — see "Running the integration tests" below.

| Test | What it checks |
|---|---|
| `test_dimension_is_384` | `provider.dimension == 384`. |
| `test_embed_documents_shape_and_normalization` | Each returned vector has length 384 and L2 norm ≈ 1. |
| `test_query_prefix_changes_the_embedding` | `embed_query(text)` differs from `embed_documents([text]).vectors[0]` for the same raw text — proves the BGE instruction prefix is actually applied at query time, not a no-op. |

### `tests/test_chunking.py`

Pure unit tests of `chunking.chunk_text` — no DB, no provider, no network.

| Test | What it checks |
|---|---|
| `test_empty_text_returns_no_chunks` | `chunk_text("")` returns `[]`. |
| `test_short_text_returns_single_chunk` | Text shorter than `chunk_size` comes back as one unmodified chunk. |
| `test_prefers_paragraph_boundary_over_mid_word_split` | Two paragraphs joined by `\n\n` split cleanly at the paragraph boundary, not mid-word — proves the recursive splitter's separator priority. |
| `test_word_boundaries_are_preserved_with_whitespace_fallback` | With no paragraph/sentence separators available, the whitespace fallback still never splits a word in half. |
| `test_consecutive_chunks_share_the_requested_overlap` | Each non-first chunk starts with the last `overlap` characters of the previous one. |
| `test_chunk_size_must_be_positive`, `test_overlap_must_be_non_negative`, `test_overlap_must_be_less_than_chunk_size` | Invalid parameters raise `ValueError` instead of producing a nonsensical split. |

### `tests/test_ingest.py`

Checks `ingest.py`'s single-document path — upsert-by-`source_uri` (§ below).

| Test | What it checks |
|---|---|
| `test_rows_for_reattaches_vectors_to_correct_keys` | The `_rows_for` helper (no DB) zips `(document_id, chunk_id)` keys back onto texts/vectors correctly. |
| `test_ingest_document_creates_expected_chunks` *(integration)* | Ingesting a realistic sample document produces exactly the expected chunk rows in `embeddings_bge_small`, with correct `document_id`s. |
| `test_reingesting_same_source_uri_replaces_chunks` *(integration)* | Calling `ingest_document` twice with the same `source_uri` but different content replaces the old chunks rather than accumulating both sets. |

### `tests/test_bulk_ingest.py`

Checks `bulk_ingest.py`'s batched multi-document path — fail-whole-batch policy (§ below). The pure-logic tests use a fake in-memory DB connection/cursor (`_FakeConnection`/`_FakeCursor`) so they never touch Postgres.

| Test | What it checks |
|---|---|
| `test_bulk_ingest_reattaches_vectors_across_documents_and_batches` | With `batch_size=1` (forcing one `embed_documents` call per chunk), every vector still lands on the correct document/chunk — no DB, `MockEmbeddingProvider` only. |
| `test_bulk_ingest_writes_nothing_if_any_batch_fails` | A provider that raises on its 2nd `embed_documents` call results in *nothing* being written — not even the document upserts from earlier in the same call — proving the whole operation is one all-or-nothing transaction. |
| `test_bulk_ingest_document_ids_and_chunk_counts` *(integration)* | Bulk-ingesting a few real documents against `local` + real Postgres produces the expected row counts per `document_id`. |

### Running the integration tests

Integration tests are skipped by default (`pytest -m "not integration"`, what CI runs) because they need a real model download or a real paid API call. To run them deliberately:

```bash
# Local model — needs sentence-transformers/torch installed (~130MB model
# download on first run, cached afterward):
pip install -r requirements-local.txt
pytest tests/test_embeddings_local.py -m integration

# Ingestion against the local model + real Postgres — same prerequisite,
# plus test_db migrated (§1-4) and TEST_DATABASE_URL exported:
pytest tests/test_ingest.py tests/test_bulk_ingest.py -m integration

# OpenAI — needs a real key (this makes a real, billed API call):
export OPENAI_API_KEY=sk-...
pytest tests/test_embeddings_openai.py -m integration

# Both, plus everything else:
pytest -m integration
```

### When these run automatically

Both `.github/workflows/ci.yml` and `.gitlab-ci.yml` install
`requirements-dev.txt` + `requirements-openai.txt`, run `python
migrations/runner.py` against a fresh Postgres service, then `pytest -m "not
integration"` on every push/PR — so every non-integration test above always
runs in CI. The integration tests (`test_embeddings_local.py`,
`test_ingest_document_creates_expected_chunks`/
`test_reingesting_same_source_uri_replaces_chunks` in `test_ingest.py`,
`test_bulk_ingest_document_ids_and_chunk_counts` in `test_bulk_ingest.py`,
and `test_real_openai_embedding_dimension_matches` in
`test_embeddings_openai.py`) never run in CI (no `sentence-transformers`/
`torch` installed there, no `OPENAI_API_KEY` configured) — they're for
local, deliberate runs only (see above).

## Embedding providers

`embeddings/` (added in Phase 3) implements the vendor-agnostic
`EmbeddingProvider` interface — `embeddings/base.py` — with three
implementations, selected at runtime by the `EMBEDDING_BACKEND` env var via
`embeddings/factory.py`'s `get_provider()`:

| `EMBEDDING_BACKEND` | Provider | Requires | Distance metric |
|---|---|---|---|
| `local` (default) | `LocalEmbeddingProvider` (`embeddings/local_provider.py`) — `BAAI/bge-small-en-v1.5` via `sentence-transformers`, fully offline after first download | `pip install -r requirements-local.txt` | Inner product (`vector_ip_ops`/`<#>`) — output is unit-normalized (`normalize_embeddings=True`), so this ranks identically to cosine but is cheaper (§3.4 of the master plan). Matches the `embeddings_bge_small` HNSW index from Phase 2. |
| `openai` | `OpenAIEmbeddingProvider` (`embeddings/openai_provider.py`) — `text-embedding-3-small`, retried with `tenacity` on rate-limit/timeout/connection errors | `pip install -r requirements-openai.txt` and a real `OPENAI_API_KEY` | Inner product — OpenAI's embeddings API returns unit-normalized vectors, so the same reasoning as above applies. |
| `mock` | `MockEmbeddingProvider` (`embeddings/mock_provider.py`) — deterministic, hash-derived vectors, no model/network | nothing extra | N/A — for tests only, never written to a real embeddings table. |

Voyage (`voyage-3-lite`, the master plan's other commercial option) is not
implemented yet; `requirements-voyage.txt` exists but is currently unused.

Every provider's `model_id`/`dimension`/`table_name` class attributes are
what `storage.table_name_for()` (Phase 4) uses to pick the right
`embeddings_<model>` table without hardcoding it — see the code comment in
`embeddings/base.py` for why `table_name` is explicit rather than derived
from `model_id`.

**Whichever backend is used at ingestion time must be the same one used at
query time** — mixing vectors from two different models in one similarity
search silently returns meaningless results, not an error (this is called
out as a pitfall in the master plan's Phase 3 section).

## Ingestion

`ingest.py` and `bulk_ingest.py` (Phase 4) turn text into stored,
searchable chunks: `chunk_text` (`chunking.py`) → `provider.embed_documents`
→ rows in the active provider's embeddings table (`embeddings_bge_small`
for `local`, the only one that actually exists right now — see "Embedding
providers" above).

```python
from ingest import ingest_document
from bulk_ingest import bulk_ingest_documents

doc_id = ingest_document("docs://readme", "... document text ...")

doc_ids = bulk_ingest_documents([
    {"source_uri": "docs://a", "content": "..."},
    {"source_uri": "docs://b", "content": "...", "metadata": {"tag": "x"}},
])
```

Both read `$DATABASE_URL` and `$EMBEDDING_BACKEND` from the environment
(via `db.get_connection()`/`embeddings.factory.get_provider()`) rather than
taking them as arguments — set them (or use the `local_ingest_env` pytest
fixture in tests) before calling either function.

**Idempotency: upsert by `source_uri`.** A document is identified by its
`source_uri`, not a caller-supplied id. Ingesting the same `source_uri`
again updates the existing `documents` row's `content`/`metadata` and
**replaces** its chunks (old ones are deleted before the new content is
re-embedded) — so re-ingesting a changed document never accumulates stale
or duplicate chunks. If you want two logically-different documents stored
side by side, give them different `source_uri`s.

**Partial-batch failure: fail the whole call.** `bulk_ingest_documents`
runs as a single transaction covering every document upsert, chunk
delete, and chunk insert. If any `embed_documents` call fails partway
through a large batch, the exception propagates and **nothing** from that
call is written — not even the document upserts that happened earlier in
the same call. This trades off "can't lose partial progress on a huge
bulk job" for "never has to reason about a half-ingested batch" — simplest
and safest, per the master plan's own warning that silent partial writes
can quietly degrade search quality over time. Rerunning
`bulk_ingest_documents` after a failure is safe (it's exactly the upsert
path again) once the underlying issue (e.g. a rate limit) is resolved.

**Why `embeddings_openai_small` isn't tested end-to-end yet:** Phase 2
only created `embeddings_bge_small` (local-model-only, deliberately —
see Phase 2's plan notes), and there's still no `OPENAI_API_KEY`. The
OpenAI provider's row-building is unit-tested with a mocked API client
(`test_ingest_row_building_works_for_the_openai_path` in
`tests/test_embeddings_openai.py`), but ingesting for real with
`EMBEDDING_BACKEND=openai` will fail with "relation
embeddings_openai_small does not exist" until that table is added in a
future migration.

## Load-testing `embeddings_bge_small` with synthetic data

`migrations/loadtest_synthetic_bge_small.py` is a throwaway validation
script — **not** part of the application, never imported by app code. It
loads ~100k synthetic 384-dim unit-normalized vectors into
`embeddings_bge_small`, then measures HNSW query latency against them, to
confirm the index meets the p95 < 100ms target from Phase 2 before real data
is loaded.

### Prerequisites

- Migrations `0001`–`0004` applied to the target database (§1–4 above) —
  the script assumes `documents` and `embeddings_bge_small` (with its HNSW
  index) already exist.
- Dev dependencies installed (the script needs `numpy` on top of the base
  requirements):

  ```bash
  pip install -r requirements-dev.txt
  ```

### Run it

Point it at a **scratch/dev** database — it writes ~100k rows, so don't run
it against `test_db` if you rely on that instance staying small/fast for
`pytest`:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
python migrations/loadtest_synthetic_bge_small.py
```

Or pass `--url` explicitly instead of exporting `DATABASE_URL`:

```bash
python migrations/loadtest_synthetic_bge_small.py --url postgresql://postgres:postgres@localhost:5432/postgres
```

To use a different data volume than the 2,000 documents / 100,000 chunks
default (e.g. a quicker smoke run):

```bash
python migrations/loadtest_synthetic_bge_small.py --n-documents 100 --n-chunks 2000 --n-queries 20
```

Flags: `--n-documents` (synthetic `documents` rows), `--n-chunks` (total
embedding rows, spread evenly across the documents), `--n-queries` (how many
timed queries to run for the latency stats). All default to the values in
the script (`N_DOCUMENTS = 2_000`, `N_CHUNKS = 100_000`, `N_QUERIES = 50`).

### What it does, step by step

1. Checks whether synthetic rows are already loaded (`documents.source_uri
   LIKE 'synthetic://%'`) — if so, skips straight to the query benchmark, so
   re-running the script is safe and fast.
2. Otherwise, inserts the synthetic `documents` rows, then generates and
   bulk-inserts the embedding rows (batched 1,000 rows per `INSERT`
   statement). Loading the full 100k-row default takes a few minutes —
   dominated by the inserts, not the vector generation.
3. Runs `EXPLAIN ANALYZE` on a sample query and prints the plan, so you can
   confirm Postgres is using the HNSW index (`Index Scan using
   embeddings_bge_small_embedding_idx`) rather than a sequential scan.
4. Runs `--n-queries` timed queries and prints p50/p95/max latency in
   milliseconds, followed by a `PASS`/`FAIL` line against the 100ms p95
   target (`LATENCY_TARGET_MS` in the script).

Expected output looks like:

```
loaded 100000 synthetic embeddings across 2000 documents

--- EXPLAIN ANALYZE (confirms HNSW index usage) ---
Limit  (cost=...) (actual time=4.306..4.334 rows=10 loops=1)
  ->  Index Scan using embeddings_bge_small_embedding_idx on embeddings_bge_small ...
Planning Time: 0.979 ms
Execution Time: 4.353 ms

queries: 50  p50: 4.15ms  p95: 5.17ms  max: 5.43ms
PASS: p95 5.17ms < 100ms target
```

### Cleaning up afterward

The script has no built-in cleanup. To remove the synthetic rows (cascades
from `documents` to `embeddings_bge_small` via `ON DELETE CASCADE`):

```sql
DELETE FROM documents WHERE source_uri LIKE 'synthetic://%';
```

or just tear down and recreate the container if it's a disposable dev
instance:

```bash
docker compose down -v db
docker compose up -d db
python migrations/runner.py
```
