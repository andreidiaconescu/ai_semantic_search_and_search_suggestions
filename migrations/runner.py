#!/usr/bin/env python3
"""Apply pending .sql migrations in this directory, in filename order."""
import argparse
import os
import pathlib

import psycopg

MIGRATIONS_DIR = pathlib.Path(__file__).parent


def applied_versions(cur) -> set[str]:
    cur.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
    )
    cur.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def run(url: str) -> None:
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
