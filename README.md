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
