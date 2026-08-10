#!/usr/bin/env python3
"""Apply pending .sql migrations in this directory, in filename order."""
import argparse
import os
import pathlib

import psycopg

MIGRATIONS_DIR = pathlib.Path(__file__).parent


def applied_versions(cur) -> set[str]:
    """Return the filenames of migrations already recorded as applied.

    Creates the `schema_migrations` bookkeeping table on first run if it
    doesn't exist yet, then reads back every version it already knows about.

    Args:
        cur: An open psycopg cursor on the target database.

    Returns:
        The set of migration filenames (e.g. "0001_enable_vector_extension.sql")
        already applied to this database.
    """
    cur.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
    )
    cur.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def run(url: str) -> None:
    """Apply every pending `.sql` file in MIGRATIONS_DIR, in filename order.

    Each pending migration's SQL is executed and its filename recorded in
    `schema_migrations` inside a single transaction, so a migration that
    fails partway never gets marked as applied. Already-applied migrations
    (per `applied_versions`) are skipped, so re-running this is safe.

    Args:
        url: Postgres connection URL to apply migrations against.

    Returns:
        None. Prints "applied <filename>" to stdout for each migration run.
    """
    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            done = applied_versions(cur)

        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.name
            if version in done:
                continue
            sql = path.read_text()
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
                )
            print(f"applied {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=os.environ.get("DATABASE_URL"),
        help="Postgres connection URL (defaults to $DATABASE_URL)",
    )
    args = parser.parse_args()
    if not args.url:
        parser.error("no --url given and DATABASE_URL is not set")
    run(args.url)
