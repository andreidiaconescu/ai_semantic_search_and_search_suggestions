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

All tests use the `db_conn` fixture in `tests/conftest.py`, which connects
to `TEST_DATABASE_URL` (via `pgvector.psycopg.register_vector`, so `vector`
columns adapt to/from Python lists automatically) — every test runs against
`test_db` (`:5433`), never the dev database.

### Prerequisites

```bash
docker compose up -d test_db
export TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/postgres
python migrations/runner.py --url "$TEST_DATABASE_URL"   # apply all migrations to test_db
pip install -r requirements-dev.txt                       # gets pytest
```

### Run everything

```bash
pytest
```

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

### When these run automatically

Both `.github/workflows/ci.yml` and `.gitlab-ci.yml` run `python
migrations/runner.py` against a fresh Postgres service and then `pytest` on
every push/PR — so the full suite above always runs in CI, in addition to
whenever you run it locally.

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

## Loading local model dependencies

To install the local embedding provider's dependencies (`sentence-transformers`,
`torch`) on top of the base requirements:

```bash
pip install -r requirements-local.txt
```

For running tests and lint: `pip install -r requirements-dev.txt`.
