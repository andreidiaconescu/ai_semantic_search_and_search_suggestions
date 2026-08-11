# Vector Search Implementation Plan
### pgvector + Python + Swappable Embedding Providers

---

## 0. Why This Project Makes Sense (Business Case)

This section explains the value of the project in plain terms, with the technical facts that back each claim.

### 0.1 What this actually gives the business

At its core, this system turns any collection of text — support tickets, product catalogs, documentation, contracts, knowledge-base articles, past customer conversations — into something that can be **searched and matched by meaning**, not just by matching words. Two capabilities fall directly out of that:

**1. Semantic search.** A user (or an internal tool) can type a question or phrase in natural language, and the system finds the most relevant documents even if none of the exact words match. Example: a search for "how do I get my money back" correctly surfaces a document titled "Refund Policy" even though it shares almost no words with the query. Keyword search (the traditional `LIKE '%refund%'` or full-text-search approach) fundamentally cannot do this, because it matches strings, not meaning. Semantic search matches on the underlying concept, encoded as a vector — which is why this is often the single biggest jump in "can our users actually find what they're looking for" that a search feature can deliver.

**2. Suggestion / recommendation from free text.** The same underlying mechanism — embed some text, find the nearest vectors in a document collection — powers "find documents similar to this one," "recommend related articles based on what the customer just typed," or "surface similar past support tickets given this new one." This is not a separate system; it's the same `search()` function from §7 applied to a different kind of input (a whole document or ticket instead of a short query). One piece of infrastructure serves both use cases, which matters for cost and maintenance: there's no separate recommendation engine to build, license, or operate.

Concretely, both capabilities reduce to the same measurable thing: given some input text, return the top-K most relevant items from a collection, ranked by similarity, in well under a second. Section 7's `search()` function and the HNSW index in §3.4 are what make that fast — pgvector's HNSW index is designed specifically so this kind of nearest-neighbor lookup stays fast (sub-100ms is a realistic target, see the exit criteria in Phase 2) even as the collection grows into the hundreds of thousands or millions of documents.

### 0.2 Business impact, by use case

| Capability | Business outcome | Where it shows up |
|---|---|---|
| Semantic search | Users find answers themselves instead of contacting support; fewer abandoned searches; better conversion on product/content search | Self-service help centers, e-commerce search, internal knowledge bases |
| Similar-document suggestions | Faster resolution (agents see similar past tickets instantly); better content discovery/engagement; automated routing | Support ticket triage, "related articles," content recommendation |
| Combined | New content becomes searchable/useful the moment it's ingested — no manual tagging, no keyword curation required | Any growing document collection |

Both capabilities are **immediately useful with an existing document collection and no model training** — this is the key point that makes the next section matter.

### 0.3 Why this approach instead of training/fine-tuning a custom AI model

A natural alternative some teams consider is: "why not just train or fine-tune a model on our own data instead?" It's worth being explicit about why vector search on pre-trained embeddings is the better starting point for search/recommendation use cases specifically (fine-tuning still has its place for other problems — this is not a blanket argument against it).

**Cost.** Fine-tuning a language model requires curated training data, GPU compute for training runs, and specialized ML expertise to do well — realistically a multi-week effort with recurring cost every time the underlying knowledge changes. The approach in this plan uses embedding models exactly as published, with no training step at all. The only ongoing cost is the (small) per-token cost of embedding calls — for the commercial option in this plan, on the order of **$0.02 per million tokens** (OpenAI `text-embedding-3-small`), or **zero marginal cost** using the self-hosted open-source option (§4.3), which runs on commodity CPU hardware.

**Speed to value.** Standing up the pipeline in this plan (ingest documents, embed, index, search) is a project measured in days to a few weeks (see the phased plan in §12), not the months typically required to collect training data, fine-tune, evaluate, and safely deploy a custom model.

**Freshness — this is the big one.** A fine-tuned model's knowledge is frozen at training time; incorporating new or changed documents means retraining or fine-tuning again. In this architecture, adding new content is just running the ingestion pipeline (§6) on the new document — it's searchable within seconds, with no retraining, no evaluation cycle, no redeployment. For any business where content changes regularly (new products, updated policies, new support tickets every day), this difference alone can be decisive.

**Transparency and control.** A fine-tuned model's answers come from inside a black box — it's difficult to say precisely *why* it produced a given answer, and difficult to remove or correct a specific piece of bad information after the fact. In this architecture, every result traces directly back to a specific row in the `documents` table (via the `document_id` design in §3.3/§6) — you always know exactly which source document produced a given result, and removing or correcting a document takes effect immediately (delete the row, or re-ingest it — no retraining).

**No lock-in, lower risk.** Because this plan is built around the vendor-agnostic `EmbeddingProvider` interface (§4), the business isn't betting on one company's model or pricing. If a cheaper or better embedding model becomes available, or a vendor changes pricing or deprecates a model, switching is a configuration change and a re-embedding job (§8), not a re-architecture. Training a custom model, by contrast, ties the business to the specific tooling and infrastructure used to produce it.

**Where fine-tuning still makes sense (for completeness).** Fine-tuning is the right tool when the goal is to change how a model *generates* language or performs a specific reasoning/generation task in a specialized way — not when the goal is finding the right existing document quickly. This project is squarely the latter, so a pre-trained embedding model plus a proper index (this plan) is both cheaper and better suited to the actual problem than training would be.

### 0.4 Summary for a business stakeholder

- Turns an existing document collection into a system that answers "what's relevant to this" instantly and accurately, in the user's own words.
- Powers both search and recommendation from the same infrastructure — no separate systems to buy or build.
- Costs cents per million tokens processed (or nothing at all with the free self-hosted option), with no training pipeline, no GPU training cluster, and no ML research team required to get started.
- Stays current automatically — new or changed content is usable within seconds of being added, unlike a trained model which goes stale until retrained.
- Every result is traceable to a specific source document, which matters for trust, correction, and any compliance/audit requirement.
- Not locked into a single AI vendor — the architecture in this plan (§4) is explicitly designed so the underlying model can be swapped without touching the rest of the system.

---

## 1. Design Goals

| Goal | Approach |
|---|---|
| No vendor lock-in | Abstract `EmbeddingProvider` interface; providers are plug-ins |
| Storage | PostgreSQL + `pgvector` extension |
| Commercial option | Cheap hosted API (OpenAI `text-embedding-3-small` primary, Voyage `voyage-3-lite` as drop-in alt) |
| Free option | Self-hosted open-source model via `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Language | Pure Python, `psycopg` + `pgvector-python` for DB access |

The core idea: **the embedding provider is a config value, not a code dependency.** Nothing downstream of the `EmbeddingProvider.embed()` call knows or cares which model produced the vector — it only needs the dimension, which is stored alongside the vector.

---

## 2. Architecture Overview

```
┌─────────────┐      ┌───────────────────┐      ┌────────────────┐
│  Documents  │─────▶│  Chunker/Loader    │─────▶│ EmbeddingProvider│
└─────────────┘      └───────────────────┘      │  (interface)     │
                                                  └─────┬───────────┘
                                    ┌────────────────────┼────────────────────┐
                                    ▼                                         ▼
                        ┌───────────────────────┐                ┌───────────────────────┐
                        │ CommercialProvider     │                │ LocalProvider          │
                        │ (OpenAI / Voyage API)  │                │ (sentence-transformers)│
                        └───────────────────────┘                └───────────────────────┘
                                    │                                         │
                                    └────────────────────┬────────────────────┘
                                                          ▼
                                              ┌────────────────────────┐
                                              │ PostgreSQL + pgvector  │
                                              │ (HNSW, inner product)  │
                                              └────────────────────────┘
                                                          ▲
                                                          │
                                              ┌────────────────────────┐
                                              │ Query embed + kNN search│
                                              └────────────────────────┘
```

Key principle: **one Postgres table per embedding model/dimension.** Different models produce different vector spaces — you cannot mix vectors from two models in the same similarity search. The schema below handles this cleanly via a `model_id` column plus per-model tables if you need to run multiple models side by side (e.g. during migration/evaluation).

---

## 3. Database Layer (pgvector)

### 3.1 Install

```bash
# Postgres 14+ recommended. Install pgvector extension:
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install   # requires postgres dev headers (postgresql-server-dev-XX)
```

Or use a prebuilt image: `pgvector/pgvector:pg16` (Docker) — simplest path for local dev/CI.

```bash
docker run -d --name pgvec \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 3.2 Enable extension

```sql
-- Registers the `vector` type and its distance operators (<=>, <->, <#>)
-- with this database — a one-time, per-database prerequisite for every
-- table/index/query in the rest of this document.
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3.3 Schema

Design so the vector dimension is fixed per table (pgvector requires this). Keep provider/model metadata so you can safely re-embed and compare.

```sql
-- The single source of truth for original documents. One row per ingested
-- document, independent of any embedding model — this is what gets re-chunked
-- and re-embedded whenever a new/different model is adopted (see §8), so its
-- `content` must always be enough to fully reconstruct the embeddings tables.
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,      -- stable internal identifier; referenced by every embeddings table as document_id, and returned on every search result (§7) so the app can trace a match back to its source
    source_uri  TEXT NOT NULL,              -- where this document came from (file path, URL, ticket ID, etc.) — used for dedupe-on-ingest decisions (§Phase 4) and shown/returned alongside search results
    content     TEXT NOT NULL,              -- the full original text, kept independently of any chunking/embedding — required because embeddings can't be converted between models; re-embedding always re-derives chunks from this column (§8)
    metadata    JSONB DEFAULT '{}',         -- free-form application fields (tags, tenant_id, access-control flags, etc.); add real columns instead of relying on this for anything that needs to be indexed/filtered efficiently at query time (§Phase 5)
    created_at  TIMESTAMPTZ DEFAULT now(),  -- ingestion timestamp, useful for auditing and for date-range filtering at query time; never changes after the row is inserted
    updated_at  TIMESTAMPTZ DEFAULT now()   -- last-modified timestamp, auto-refreshed on every UPDATE by the set_updated_at trigger below (e.g. content/metadata edits on re-ingestion, §Phase 4) — do not set this manually from application code
);

-- One embeddings table per model/dimension in use (see §2 "one Postgres table
-- per embedding model/dimension" and the note below on why vectors from
-- different models must never share a table).
-- Example: 384-dim local model (bge-small) and 1536-dim commercial model.
CREATE TABLE embeddings_bge_small (
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE, -- FK back to the source document; ON DELETE CASCADE means deleting a document automatically removes all of its chunks/vectors here — no orphaned embeddings to clean up manually
    chunk_id    INT NOT NULL,               -- position of this chunk within its document (0, 1, 2, ...), assigned by the chunker (§6); combined with document_id this uniquely identifies a chunk and lets re-ingestion delete-and-replace a document's old chunks cleanly (§Phase 4)
    chunk_text  TEXT NOT NULL,              -- the literal chunk of text that was embedded — stored so search results (§7) can return matched text directly without a second lookup/re-chunk of `documents.content`
    embedding   vector(384) NOT NULL,       -- the actual embedding vector for chunk_text, produced by this table's model; dimension (384) is fixed per table because pgvector requires a fixed vector size per column, and it must exactly match the owning model's output dimension (bge-small-en-v1.5 here)
    model_id    TEXT NOT NULL DEFAULT 'bge-small-en-v1.5', -- records which exact model/version produced this vector; the table name already implies the model, but this column keeps that fact queryable/auditable and catches any accidental cross-model writes into the wrong table
    created_at  TIMESTAMPTZ DEFAULT now(),  -- when this chunk/vector was first written — useful for auditing ingestion runs and index-growth tracking (§Phase 7)
    updated_at  TIMESTAMPTZ DEFAULT now(),  -- last-modified timestamp, auto-refreshed on every UPDATE by the set_updated_at trigger below (e.g. re-embedding a chunk in place) — do not set this manually from application code
    PRIMARY KEY (document_id, chunk_id)     -- a document's chunks are numbered from 0, so (document_id, chunk_id) is naturally unique and doubles as the natural key for upserts during re-ingestion
);

CREATE TABLE embeddings_openai_small (
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE, -- same role as in embeddings_bge_small: link back to the source document, cascading deletes with it
    chunk_id    INT NOT NULL,               -- same role as in embeddings_bge_small: this chunk's position within its document
    chunk_text  TEXT NOT NULL,              -- same role as in embeddings_bge_small: the literal text that was embedded, returned directly in search results
    embedding   vector(1536) NOT NULL,      -- dimension (1536) matches OpenAI text-embedding-3-small's output — different from embeddings_bge_small's 384, which is exactly why this must be a separate table rather than a shared one (mixing dimensions/vector spaces in one similarity search is meaningless, see the warning below)
    model_id    TEXT NOT NULL DEFAULT 'text-embedding-3-small', -- same role as in embeddings_bge_small: records the exact model/version, for auditability and safe multi-model migration/comparison (§8)
    created_at  TIMESTAMPTZ DEFAULT now(),  -- same role as in embeddings_bge_small: when this chunk/vector was first written
    updated_at  TIMESTAMPTZ DEFAULT now(),  -- same role as in embeddings_bge_small: auto-refreshed on every UPDATE by the set_updated_at trigger below
    PRIMARY KEY (document_id, chunk_id)     -- same role as in embeddings_bge_small: natural key over a document's ordered chunks
);

-- Postgres has no built-in "ON UPDATE CURRENT_TIMESTAMP" (unlike MySQL), so
-- automatic updated_at maintenance requires a trigger function attached to
-- every table that has the column. One shared function, reused via one
-- trigger per table, keeps this from being duplicated per table.
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now(); -- stamp the row with the current time on every UPDATE, overriding whatever the caller sent (or didn't send) for updated_at
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_set_updated_at
    BEFORE UPDATE ON documents          -- fires before any UPDATE to documents (e.g. content/metadata changes on re-ingestion, §Phase 4) so updated_at reflects the write that's about to commit
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER embeddings_bge_small_set_updated_at
    BEFORE UPDATE ON embeddings_bge_small -- fires before any UPDATE to a chunk/vector row in this table (e.g. in-place re-embedding rather than delete+reinsert)
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER embeddings_openai_small_set_updated_at
    BEFORE UPDATE ON embeddings_openai_small -- same purpose as embeddings_bge_small_set_updated_at, scoped to this table
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
```

> **`updated_at` is maintained automatically, not by application code.** The `set_updated_at()` trigger function overwrites `updated_at` with `now()` on every `UPDATE`, regardless of what value (if any) the application supplied — this guarantees the column is trustworthy even if a caller forgets to set it. Any new table added later that needs the same behavior just needs its own `CREATE TRIGGER ... BEFORE UPDATE ... EXECUTE FUNCTION set_updated_at();` — the function itself doesn't need to change.

> Alternative if you want a single table: store dimension as the max needed and zero-pad smaller vectors — **don't do this**. It corrupts cosine similarity. Separate tables (or separate Postgres schemas) per model is the correct approach.

> **`document_id` is the link back to the source record.** Every row in an embeddings table carries the `document_id` of the original document it was chunked from (via the `documents(id)` foreign key). This is what makes the vectors useful to the application: a similarity search doesn't just return "a piece of matching text," it returns the ID your app can use to fetch the full document, look up its metadata, apply access-control checks, or build a link back into your own system. This ID is set at ingestion time and returned on every search result — see Sections 6 and 7.

### 3.4 Indexing

Two index types in pgvector: `ivfflat` (faster build, needs pre-existing data to train) and `hnsw` (slower build, better recall/query speed, no training step). For most workloads, **use HNSW**.

**Operator choice: inner product, not cosine.** Both providers in §4 emit unit-normalized vectors (OpenAI's API pre-normalizes its embeddings; the local `LocalEmbeddingProvider` calls `.encode(..., normalize_embeddings=True)`, §4.3). For unit-length vectors, cosine similarity and inner product produce **identical rankings** — cosine distance's extra division by `‖a‖‖b‖` is a division by `1`. Inner product (`vector_ip_ops` / `<#>`) skips that normalization step entirely, so it's cheaper per comparison at both index-build and query time, with no precision loss versus cosine. This only holds because the vectors are normalized — if you ever add a provider that doesn't normalize its output, either normalize it yourself before storing, or fall back to `vector_cosine_ops` for that table.

```sql
-- Index built on the `embedding` column of embeddings_bge_small — this is what
-- makes `ORDER BY embedding <#> $1 LIMIT k` (§3.5) fast instead of scanning
-- every row; `document_id`/`chunk_id`/`chunk_text`/`model_id` don't need an
-- index here since they're not what similarity search orders by.
CREATE INDEX ON embeddings_bge_small
  USING hnsw (embedding vector_ip_ops)     -- vector_ip_ops: build the graph for inner product — equivalent ranking to cosine here because bge-small-en-v1.5's output is unit-normalized (§4.3), but cheaper per comparison (§3.5)
  WITH (m = 16, ef_construction = 64);     -- m: graph connectivity per node; ef_construction: build-time search width — both tunable per §3.4's tuning notes

-- Same purpose as above, applied to embeddings_openai_small's `embedding`
-- column — each per-model table needs its own HNSW index, since indexes
-- aren't shared across tables.
CREATE INDEX ON embeddings_openai_small
  USING hnsw (embedding vector_ip_ops)     -- vector_ip_ops: inner product — equivalent ranking to cosine here because OpenAI text-embedding-3-small's output is already unit-normalized by the API
  WITH (m = 16, ef_construction = 64);
```

Tuning notes:
- `m`: graph connectivity (default 16). Higher = better recall, more memory/build time.
- `ef_construction`: build-time search width (default 64). Higher = better index quality, slower build.
- At query time, set `SET hnsw.ef_search = 100;` (higher = better recall, slower query). Tune per latency budget.
- If dataset < ~50k rows, a plain sequential scan with inner product distance is often fast enough — don't over-engineer the index for small datasets.
- If you ever store non-normalized vectors in a future table, use `vector_cosine_ops`/`<=>` for that table instead — inner product on non-normalized vectors is biased toward larger-magnitude vectors and is **not** equivalent to cosine similarity in that case.

### 3.5 Query

```sql
-- $1 is the query embedding, produced by the SAME model that owns this table
-- (embed_query, not embed_documents — see §4.1, §7); mixing a query vector
-- from a different model into this table's search would silently return
-- meaningless results (§Phase 3 pitfalls).
SELECT document_id,              -- returned so the app can fetch the full source document, its metadata, or apply access control (§3.3 note)
       chunk_text,                -- the matched text itself, returned directly so the app doesn't need a second lookup
       -(embedding <#> $1) AS similarity -- <#> is pgvector's NEGATIVE inner product operator (it negates so that ORDER BY ASC still means "closest first", matching <=> and <->); negating it back gives the raw dot product, which equals cosine similarity here because both vectors are unit-normalized (§3.4) — no norm division needed
FROM embeddings_bge_small
ORDER BY embedding <#> $1        -- ordering by negative inner product (ascending = most negative = largest actual dot product = closest first) is what lets the HNSW index (§3.4) satisfy this query without a full scan
LIMIT 10;                        -- top_k — how many nearest chunks to return; configurable per §Phase 5
```

`<#>` is the negative inner product operator in pgvector (chosen here because both providers store unit-normalized vectors — see §3.4's "Operator choice" note); `<=>` is cosine distance, `<->` is L2. If you introduce a table of non-normalized vectors, use `<=>` for that table instead, since `<#>`'s ranking is only equivalent to cosine similarity when the vectors are unit length.

---

## 4. Python Abstraction Layer

### 4.1 Interface

```python
# embeddings/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model_id: str
    dimension: int

class EmbeddingProvider(ABC):
    """Vendor-agnostic embedding interface. All providers implement this."""

    model_id: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of documents/chunks for storage."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for search."""
        ...
```

Some models distinguish "document" vs "query" embeddings (asymmetric embedding models like BGE need an instruction prefix on queries). Keeping two methods in the interface avoids leaking that detail into calling code.

### 4.2 Commercial provider (cheap, hosted)

```python
# embeddings/openai_provider.py
import os
from openai import OpenAI
from .base import EmbeddingProvider, EmbeddingResult

class OpenAIEmbeddingProvider(EmbeddingProvider):
    model_id = "text-embedding-3-small"
    dimension = 1536

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        resp = self.client.embeddings.create(model=self.model_id, input=texts)
        # Defensive: OpenAI's response includes an explicit `index` per item.
        # Sort by it rather than trusting array order, since the bulk-ingest
        # pipeline (§6.1) depends on strict input/output order alignment.
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors = [d.embedding for d in ordered]
        return EmbeddingResult(vectors, self.model_id, self.dimension)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text]).vectors[0]
```

Swap-in alternative (Voyage, similarly cheap, Anthropic's recommended partner):

```python
# embeddings/voyage_provider.py
import os, voyageai
from .base import EmbeddingProvider, EmbeddingResult

class VoyageEmbeddingProvider(EmbeddingProvider):
    model_id = "voyage-3-lite"
    dimension = 512

    def __init__(self, api_key: str | None = None):
        self.client = voyageai.Client(api_key=api_key or os.environ["VOYAGE_API_KEY"])

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        resp = self.client.embed(texts, model=self.model_id, input_type="document")
        return EmbeddingResult(resp.embeddings, self.model_id, self.dimension)

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embed([text], model=self.model_id, input_type="query")
        return resp.embeddings[0]
```

Cost ballpark (check current pricing pages before committing):
- OpenAI `text-embedding-3-small`: ~$0.02 / 1M tokens
- Voyage `voyage-3-lite`: similarly low, optimized for cost/latency tradeoff

### 4.3 Free / self-hosted provider

```python
# embeddings/local_provider.py
from sentence_transformers import SentenceTransformer
from .base import EmbeddingProvider, EmbeddingResult

class LocalEmbeddingProvider(EmbeddingProvider):
    model_id = "BAAI/bge-small-en-v1.5"
    dimension = 384

    def __init__(self, device: str = "cpu"):
        self.model = SentenceTransformer(self.model_id, device=device)

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        vectors = self.model.encode(texts, normalize_embeddings=True).tolist()
        return EmbeddingResult(vectors, self.model_id, self.dimension)

    def embed_query(self, text: str) -> list[float]:
        # BGE models want an instruction prefix on queries for best retrieval quality
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return self.model.encode([prefixed], normalize_embeddings=True).tolist()[0]
```

Install: `pip install sentence-transformers` (pulls in `torch`; use CPU build if no GPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`). Runs fully offline after first model download (~130MB for bge-small).

Alternative local runtime if you'd rather not manage Python-side model weights directly: run **Ollama** with `nomic-embed-text` and call its HTTP API (`http://localhost:11434/api/embeddings`) — same `EmbeddingProvider` interface, different implementation, zero change to calling code.

### 4.4 Provider factory (config-driven selection)

```python
# embeddings/factory.py
import os
from .base import EmbeddingProvider

def get_provider() -> EmbeddingProvider:
    backend = os.environ.get("EMBEDDING_BACKEND", "local")
    match backend:
        case "openai":
            from .openai_provider import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider()
        case "voyage":
            from .voyage_provider import VoyageEmbeddingProvider
            return VoyageEmbeddingProvider()
        case "local":
            from .local_provider import LocalEmbeddingProvider
            return LocalEmbeddingProvider()
        case _:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
```

Switching providers is now a single environment variable — no code changes anywhere in the ingestion or query pipeline.

---

## 5. Database Access Layer

```python
# db.py
import os
import psycopg
from pgvector.psycopg import register_vector

def get_connection():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn
```

`pgvector-python`'s `register_vector` teaches psycopg to adapt Python lists/numpy arrays to the `vector` type automatically — no manual serialization.

### 5.1 Table naming by provider

Since dimension varies per model, derive the table name from the active provider so ingestion/query code never hardcodes it:

```python
# storage.py
def table_name_for(provider) -> str:
    slug = provider.model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"embeddings_{slug}"
```

---

## 6. Ingestion Pipeline

**How `document_id` survives the embedding call.** `embed_documents(texts)` is a pure function: text list in, vector list out, same order, no knowledge of documents or IDs. That's intentional — it keeps the provider interface generic. The *caller* is responsible for remembering which text came from which document, by keeping a parallel list of IDs in lockstep with the texts list, then zipping them back together once the vectors come back.

For a single document this is trivial (every chunk shares the same `doc_id`):

```python
# ingest.py
from embeddings.factory import get_provider
from db import get_connection
from storage import table_name_for

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def ingest_document(source_uri: str, content: str, metadata: dict | None = None):
    provider = get_provider()
    table = table_name_for(provider)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (source_uri, content, metadata) VALUES (%s, %s, %s) RETURNING id",
            (source_uri, content, metadata or {}),
        )
        doc_id = cur.fetchone()[0]

        chunks = chunk_text(content)
        result = provider.embed_documents(chunks)
        # chunks[i] and result.vectors[i] refer to the same text, in order —
        # doc_id is constant here because every chunk came from this one document.
        rows = [
            (doc_id, i, chunk, vec, provider.model_id)
            for i, (chunk, vec) in enumerate(zip(chunks, result.vectors))
        ]
        cur.executemany(
            f"INSERT INTO {table} (document_id, chunk_id, chunk_text, embedding, model_id) "
            f"VALUES (%s, %s, %s, %s, %s)",
            rows,
        )
        conn.commit()
    return doc_id
```

### 6.1 Batching across multiple documents

If you're ingesting many documents and want to batch the embedding calls (e.g. 500 chunks per API call, spanning several documents, to cut round-trips and respect commercial API rate limits), the correspondence is no longer "one document per call" — so you must carry an explicit parallel list of `(document_id, chunk_id)` alongside the flattened text list, and zip everything back together by index once the vectors return:

```python
# bulk_ingest.py
from embeddings.factory import get_provider
from db import get_connection
from storage import table_name_for
from ingest import chunk_text

def bulk_ingest_documents(docs: list[dict], batch_size: int = 200):
    """docs: list of {"source_uri": str, "content": str, "metadata": dict}"""
    provider = get_provider()
    table = table_name_for(provider)

    with get_connection() as conn, conn.cursor() as cur:
        # 1. Insert all documents first, capturing their real doc_ids.
        doc_ids = []
        for doc in docs:
            cur.execute(
                "INSERT INTO documents (source_uri, content, metadata) VALUES (%s, %s, %s) RETURNING id",
                (doc["source_uri"], doc["content"], doc.get("metadata", {})),
            )
            doc_ids.append(cur.fetchone()[0])
        conn.commit()

        # 2. Flatten chunks across all documents, keeping a PARALLEL list of
        #    (document_id, chunk_id) so we know exactly which chunk each text is,
        #    even after the texts have been mixed together into one flat list.
        flat_texts: list[str] = []
        flat_keys: list[tuple[int, int]] = []   # (document_id, chunk_id) per text, same order
        for doc_id, doc in zip(doc_ids, docs):
            chunks = chunk_text(doc["content"])
            for chunk_id, chunk in enumerate(chunks):
                flat_texts.append(chunk)
                flat_keys.append((doc_id, chunk_id))

        # 3. Embed in batches. The provider still just sees "texts in, vectors out" —
        #    it never sees document_id — but flat_keys[i] still lines up with flat_texts[i],
        #    so vectors[i] can be reattached to the right (document_id, chunk_id) afterward.
        all_rows = []
        for start in range(0, len(flat_texts), batch_size):
            batch_texts = flat_texts[start:start + batch_size]
            batch_keys = flat_keys[start:start + batch_size]

            result = provider.embed_documents(batch_texts)

            for (doc_id, chunk_id), text, vec in zip(batch_keys, batch_texts, result.vectors):
                all_rows.append((doc_id, chunk_id, text, vec, provider.model_id))

        cur.executemany(
            f"INSERT INTO {table} (document_id, chunk_id, chunk_text, embedding, model_id) "
            f"VALUES (%s, %s, %s, %s, %s)",
            all_rows,
        )
        conn.commit()

    return doc_ids
```

The rule that makes this safe: **never let a provider call reorder or drop items.** `embed_documents` must return vectors in the exact same order and count as the texts it was given (all implementations in §4 satisfy this — `SentenceTransformer.encode` and both hosted APIs preserve input order). As long as that holds, `zip(batch_keys, batch_texts, result.vectors)` is a correct and sufficient way to reattach every vector to its `(document_id, chunk_id)`, no matter how many documents were mixed into one batch.

## 7. Query Pipeline

Every result includes `document_id` — the primary key of the original row in `documents` — so the calling application can immediately fetch the full document, its metadata, or enforce access control, without needing a second lookup by URI or text match.

```python
# search.py
from embeddings.factory import get_provider
from db import get_connection
from storage import table_name_for

def search(query: str, top_k: int = 10) -> list[dict]:
    provider = get_provider()
    table = table_name_for(provider)
    query_vec = provider.embed_query(query)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100;")
        cur.execute(
            f"""
            SELECT e.document_id, d.source_uri, e.chunk_id, e.chunk_text,
                   -(e.embedding <#> %s) AS similarity
            FROM {table} e
            JOIN documents d ON d.id = e.document_id
            ORDER BY e.embedding <#> %s
            LIMIT %s
            """,
            (query_vec, query_vec, top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "document_id": r[0],   # original doc's primary key — use this in the app
            "source": r[1],
            "chunk_id": r[2],
            "text": r[3],
            "similarity": r[4],
        }
        for r in rows
    ]
```

---

## 8. Migrating / Comparing Models

Because each model has its own table, you can run both providers side by side, compare retrieval quality (e.g. via a labeled eval set and recall@k), and cut over by changing `EMBEDDING_BACKEND` — no data loss, no downtime, easy rollback. To fully migrate, re-run ingestion against all source documents with the new provider (embeddings can't be converted between models — they must be recomputed from source text, so always retain original `content` in the `documents` table).

---

## 9. Evaluation

Minimal offline eval harness to sanity-check any provider before committing to it:

```python
# eval.py
def recall_at_k(queries: list[tuple[str, str]], search_fn, k: int = 10) -> float:
    """queries: list of (query_text, expected_source_uri)"""
    hits = 0
    for query, expected in queries:
        results = search_fn(query, top_k=k)
        if any(r["source"] == expected for r in results):
            hits += 1
    return hits / len(queries)
```

Run this against both the commercial and local provider on a small labeled set (20–50 query/expected-doc pairs) before deciding which to use in production, or whether to keep both (e.g. local for bulk/internal search, commercial for customer-facing search where quality matters more than cost).

---

## 10. Dependencies

```
# requirements.txt
psycopg[binary]>=3.1
pgvector>=0.3.0
sentence-transformers>=3.0        # local provider
torch>=2.0                        # local provider (CPU wheel if no GPU)
openai>=1.30                      # commercial provider (optional)
voyageai>=0.2                     # commercial provider (optional, alt)
```

Keep provider-specific SDKs (`openai`, `voyageai`) as **optional extras**, not core dependencies, so a deployment using only the local model doesn't need to install or configure any hosted-API SDK:

```toml
# pyproject.toml
[project.optional-dependencies]
openai = ["openai>=1.30"]
voyage = ["voyageai>=0.2"]
local = ["sentence-transformers>=3.0", "torch>=2.0"]
```

---

## 11. Deployment Checklist

- [ ] Postgres with pgvector extension installed and `CREATE EXTENSION vector;` run
- [ ] One embeddings table + HNSW index per model dimension in use
- [ ] `EMBEDDING_BACKEND` env var controls provider selection; no hardcoded model in app code
- [ ] API keys (if using commercial provider) in secrets manager, not source control
- [ ] Local model weights cached in a persistent volume (avoid re-downloading on every container start)
- [ ] Batch size tuned for commercial API rate limits
- [ ] `hnsw.ef_search` tuned against a latency/recall target
- [ ] Eval set in place before switching providers in production
- [ ] Connection pooling (e.g. `psycopg_pool`) for concurrent query load

---

## 12. Implementation Phases

Each phase has a clear exit criterion — don't move to the next phase until the current one's criterion is met. Phases 1–4 can mostly be built and tested locally with Docker before touching any commercial API, which keeps early iteration free and fast.

### Phase 1 — Environment & Infrastructure Setup ✅ DONE

**Goal:** a running Postgres instance with pgvector installed, and a Python project skeleton that everything else plugs into.

Tasks:
- Stand up Postgres via the `pgvector/pgvector:pg16` Docker image for local dev; decide on the equivalent for staging/prod (managed Postgres with pgvector support — e.g. AWS RDS ≥ 15 with the extension allow-listed, Supabase, Neon, or self-managed with the extension compiled from source per §3.1).
- Run `CREATE EXTENSION IF NOT EXISTS vector;` and confirm with `SELECT * FROM pg_extension WHERE extname = 'vector';`.
- Initialize the Python project: `pyproject.toml` with the optional-dependency groups from §10, a virtual environment, and a `.env.example` documenting `DATABASE_URL`, `EMBEDDING_BACKEND`, `OPENAI_API_KEY` / `VOYAGE_API_KEY`.
- Set up a lightweight migration tool for schema changes (`alembic` or plain versioned `.sql` files in a `migrations/` folder) — even a small project benefits from not hand-running DDL against prod later.
- Set up a test database (either a second Docker container or a schema-per-test-run convention) so ingestion/query tests don't pollute dev data.
- Wire up basic CI (lint + unit tests) even before there's much to test — cheaper to add now than retrofit later.

Deliverables: `docker-compose.yml` (Postgres + app), `pyproject.toml`, `.env.example`, empty `migrations/` with the extension-creation migration checked in, CI config running on every push.

**Status: implemented** (commit `be46132`, 2026-08-07). Delivered: `docker-compose.yml` with dev + test Postgres instances on `pgvector/pgvector:pg16`, `.env.example`, a plain-`.sql` migration runner (`migrations/runner.py`) with `migrations/0001_enable_vector_extension.sql` checked in, a `requirements*.txt`-based Python skeleton (base + optional `local`/`openai`/`voyage`/`dev` extras, per §10) in place of `pyproject.toml` optional-dependency groups, a `tests/` scaffold with a passing smoke test, and both GitHub Actions and GitLab CI configs kept in sync.

Exit criteria: `docker compose up` gives a Postgres instance with `vector` extension enabled, reachable from a local Python `psycopg` connection; CI passes on an empty test suite. — **Met.**

Pitfalls: managed Postgres providers vary in whether pgvector is pre-approved — check this before committing to a cloud provider, since some require opening a support ticket to allow-list extensions. Pin the pgvector version (`v0.8.0` or later) — index syntax and available operators have changed across versions.

---

### Phase 2 — Database Schema & Indexing ✅ DONE

**Goal:** finalized schema for `documents` and per-model embedding tables, with indexes tuned for the expected data volume.

Tasks:
- Create the `documents` table (§3.3) with whatever metadata columns your application actually needs (tags, access-control fields, timestamps) — don't over-design this speculatively, but don't leave out fields you already know you'll need for filtering (e.g. `tenant_id` for multi-tenant apps, since that will matter for query-time filtering later).
- Create the first embeddings table for whichever model you'll prototype with first (local model recommended for phase 2, since it avoids API costs during schema iteration).
- Decide the distance operator up front (`vector_ip_ops`/`<#>` for unit-normalized text embeddings, per §3.4 — inner product is faster than `vector_cosine_ops` and ranks identically as long as the model's output is normalized; verify per model, don't assume, and use `vector_cosine_ops` instead for any model that isn't normalized) and encode that decision in the migration, not just in application code.
- Add the HNSW index (§3.4). For initial development, default `m = 16, ef_construction = 64` is fine; defer serious tuning to Phase 6 once you have real data volume and can measure recall/latency tradeoffs.
- Write a small script that inserts synthetic rows (random vectors) at a scale close to your expected production volume (e.g. 100k rows) purely to validate that index creation completes in reasonable time and query latency is acceptable — catching this now is much cheaper than discovering it after real data is loaded.
- Decide and document the multi-model table-naming convention (§5.1) before writing any ingestion code that depends on it.

Deliverables: SQL migrations for `documents` and the first `embeddings_<model>` table with its HNSW index; a throwaway load-test script (not part of the app, just a validation tool).

**Status: implemented** (2026-08-10, uncommitted — pending review). Delivered: `migrations/0002_create_documents_table.sql` (`documents` table + a shared `set_updated_at()` trigger function), `migrations/0003_create_embeddings_bge_small_table.sql` (the local-model-only `embeddings_bge_small` table, 384-dim, scoped to `bge-small-en-v1.5` per the phase's own recommendation — no `embeddings_openai_small` table yet, deferred per this phase's pitfalls note), `migrations/0004_index_embeddings_bge_small_hnsw.sql` (HNSW index using `vector_ip_ops`, not `vector_cosine_ops`, matching §3.4's inner-product decision), and `migrations/loadtest_synthetic_bge_small.py` (throwaway synthetic-data + latency validation script). No `tenant_id`/extra `documents` columns were added — no known filtering requirement yet, so the schema matches §3.3 as-is (plus `updated_at`). New tests in `tests/test_schema.py` verify table columns, embedding dimension, HNSW index presence, and `updated_at` trigger behavior; all pass, and `ruff check .` is clean.

One deviation flagged for Phase 3: `table_name_for()` (§5.1) derives the table name from `provider.model_id` (`"BAAI/bge-small-en-v1.5"`), which slugifies to `embeddings_BAAI_bge_small_en_v1_5` — not the `embeddings_bge_small` name actually created here (matching §3.3's hardcoded example). Phase 3 should add an explicit `table_name` attribute per provider instead of deriving it, or ingestion/query code will silently look at the wrong table.

Exit criteria: a query against the synthetic-data table using `ORDER BY embedding <#> $1 LIMIT 10` returns in acceptable time (define "acceptable" now — e.g. p95 < 100ms — so Phase 6 has a concrete target to test against). — **Met.** The load-test script loaded 100,000 synthetic 384-dim unit-normalized vectors across 2,000 documents against the dev DB; `EXPLAIN ANALYZE` confirmed the HNSW index (not a sequential scan) is used; measured **p95 = 5.17ms, max = 5.43ms** — well under the 100ms target.

Pitfalls: building an HNSW index on a large table blocks other writes for a while (`CREATE INDEX CONCURRENTLY` avoids the write-lock but has its own caveats — check pgvector's docs for concurrent-build support in your installed version). Don't add multiple embedding tables until you actually need to compare/migrate models — one table is enough to start.

---

### Phase 3 — Embedding Provider Abstraction Layer ✅ DONE

**Goal:** the `EmbeddingProvider` interface (§4) implemented for both a local and a commercial backend, swappable via config, with no calling code aware of which is active.

Tasks:
- Implement `EmbeddingProvider` ABC (§4.1) first, with nothing but the interface — no implementation yet. Write a fake/mock provider for unit tests that returns deterministic fixed-size vectors, so downstream code (ingestion, query) can be tested without any real model or network call.
- Implement `LocalEmbeddingProvider` (§4.3) first, since it requires no API key and no cost — good for making the rest of the pipeline work end-to-end quickly. Confirm the model downloads and caches correctly, and that `embed_query` applies the correct instruction prefix if the model needs one (BGE-family models do; not all models do — check the model card).
- Implement the commercial provider (§4.2) second. Handle the practical realities hosted APIs bring that a local model doesn't: network timeouts, rate-limit (`429`) responses with retry/backoff, and partial-batch failures. Wrap calls with a retry decorator (e.g. `tenacity`) rather than hand-rolling retry loops.
- Implement the provider factory (§4.4) and confirm switching `EMBEDDING_BACKEND` between `local` and `openai` changes behavior with zero other code changes — this is the actual test of "no vendor lock-in," so don't skip verifying it.
- Add a small integration test that calls each real provider (local and commercial) with a fixed short input and asserts the returned dimension matches `provider.dimension` — cheap insurance against a provider silently changing its output shape.
- Document, in code comments or a README, which distance metric each provider's embeddings expect (cosine vs dot product) — this is a common source of silent quality bugs since Postgres won't error if you use the wrong operator, it'll just return worse rankings.

Deliverables: `embeddings/base.py`, `embeddings/local_provider.py`, `embeddings/openai_provider.py` (or `voyage_provider.py`), `embeddings/factory.py`, `embeddings/mock_provider.py` for tests, unit + integration tests for all.

**Status: implemented** (2026-08-10). Delivered exactly the deliverables above: `embeddings/base.py` (`EmbeddingProvider` ABC + `EmbeddingResult`), `embeddings/local_provider.py` (`BAAI/bge-small-en-v1.5`, matching §4.3), `embeddings/openai_provider.py` (`text-embedding-3-small`, matching §4.2's defensive index-sort, plus `tenacity` retry/backoff scoped to `RateLimitError`/`APIConnectionError`/`APITimeoutError` only — not auth/bad-request errors, which fail fast), `embeddings/factory.py` (`local`/`openai`/`mock` via `EMBEDDING_BACKEND`, matching §4.4), and `embeddings/mock_provider.py` (deterministic, hash-derived, unit-normalized vectors — zero network/model, used by downstream tests). Voyage was **not** implemented this phase — deliberately scoped to OpenAI-only, per an explicit user decision (`requirements-voyage.txt` stays unused).

One interface addition beyond §4.1: `EmbeddingProvider` also requires a **`table_name`** class attribute (alongside `model_id`/`dimension`), fixing the table-naming bug flagged at the end of Phase 2 — `table_name_for()` (§5.1) deriving a name from `model_id` broke for `"BAAI/bge-small-en-v1.5"`. Each provider now declares its table explicitly (`LocalEmbeddingProvider.table_name = "embeddings_bge_small"`, `OpenAIEmbeddingProvider.table_name = "embeddings_openai_small"`), so Phase 4's `table_name_for()` becomes a trivial `return provider.table_name` instead of a slugifier.

Tests: 5 new files (`tests/test_embeddings_base.py`, `_mock.py`, `_factory.py`, `_openai.py`, `_local.py`) — 26 tests pass, 1 correctly skipped (the real-API OpenAI test, no key available). The local-model integration test was **actually run** (not just written): it downloaded and cached `bge-small-en-v1.5` and confirmed 384-dim, unit-normalized output with the BGE query instruction prefix actually changing the embedding. The OpenAI provider's retry/backoff and defensive-sort behavior are unit-tested with a mocked client (no real network needed). A new `pytest.ini` registers an `integration` marker so real-network/real-model tests are excluded by default (`pytest -m "not integration"`, what CI runs) and opt-in otherwise. `ruff check .` is clean.

Distance metric documented per this phase's own task list: both implemented providers are inner-product (`vector_ip_ops`/`<#>`), not cosine — both emit unit-normalized vectors, so this matches §3.4's Phase 2 decision. Documented in a new README "Embedding providers" section (table of `EMBEDDING_BACKEND` values, requirements, and distance metric per provider), not just code comments.

Every class, function, and method across all of this phase's `.py` files (and the migrations/tests scripts from earlier phases) was given full docstrings — one-line summary plus explicit `Args:`/`Returns:`/`Raises:` sections enumerating every parameter and the return value, including `None`/no-args cases — per an explicit user request to treat this as standing practice going forward, not a one-off pass.

Exit criteria: the same ingestion/query code (even if just a throwaway script at this stage) works unmodified against both providers by only changing `EMBEDDING_BACKEND`. — **Met, adapted.** Ingestion/query code doesn't exist until Phase 4, so this was verified at the provider layer instead: `tests/test_embeddings_factory.py` proves `EMBEDDING_BACKEND` alone (`local`/`openai`/`mock`) determines which provider class is constructed and what `model_id`/`dimension`/`table_name` it reports, with zero other code changes.

Pitfalls: forgetting that query-time embedding must use the same provider (and ideally the same `embed_query`, not `embed_documents`) as was used at ingestion time — mixing vectors from different models in one similarity search silently produces meaningless results, not an error.

---

### Phase 4 — Ingestion Pipeline ✅ DONE

**Goal:** reliable text → chunk → embed → store pipeline, correct for both single-document and batched multi-document cases (§6, §6.1).

Tasks:
- Implement chunking (§6) with a real strategy, not just word-count splitting — evaluate `RecursiveCharacterTextSplitter` (token- or character-aware) or a sentence-boundary splitter (`spaCy`, `nltk`) depending on your content type. Chunking quality has a bigger effect on retrieval quality than most people expect; don't treat it as a placeholder detail.
- Implement `ingest_document` (§6) for the single-document path and test it against both providers via the factory.
- Implement `bulk_ingest_documents` (§6.1) for the batched path, and specifically test the failure mode where a batch partially fails (e.g. one malformed chunk causes the whole API call to error) — decide whether to fail the whole batch, retry only the offending item, or skip-and-log. This decision matters more than it seems, since silent skips can quietly degrade search quality over time.
- Add idempotency: decide what happens if the same document is ingested twice (dedupe by `source_uri` with an upsert, or allow duplicates and dedupe at query time — pick one deliberately).
- Add a way to re-ingest an existing document when its content changes (delete old chunks for that `document_id` and re-embed, rather than accumulating stale chunks).
- Instrument ingestion with basic logging/metrics: documents processed, chunks created, embedding API latency, failures — this pays off heavily once you're troubleshooting a production ingestion job.

Deliverables: `ingest.py`, `bulk_ingest.py`, chunking module, tests covering single-doc ingest, batch ingest across multiple documents, re-ingestion of updated content, and partial-batch-failure handling.

**Status: implemented** (2026-08-11). Delivered exactly the deliverables above, plus the two files §5 already specified but had never been built: `db.py` (`get_connection()` per §5) and `storage.py` (`table_name_for()` per §5.1, now a trivial `return provider.table_name` thanks to the Phase 3 fix). `chunking.py` implements a dependency-free recursive splitter (paragraph → sentence → whitespace → hard cutoff, character-based `chunk_size`/overlap) — no `nltk`/`spaCy`/`langchain`, a deliberate user decision to avoid new dependencies. `ingest.py`'s `ingest_document` and `bulk_ingest.py`'s `bulk_ingest_documents` both implement upsert-by-`source_uri` idempotency (re-ingesting a `source_uri` updates the row and replaces its chunks rather than accumulating duplicates) and fail-whole-call partial-failure handling — `bulk_ingest_documents` runs as a single transaction, so any embedding error anywhere rolls back everything from that call, including earlier document upserts, not just the failing batch. The `(document_id, chunk_id, chunk_text, embedding, model_id)` reattachment logic from §6.1 was factored into a small pure helper (`ingest._rows_for`), independently unit-tested with synthetic data, no DB or model required.

A known, deliberate gap: `embeddings_openai_small` still doesn't exist as a real table (no `OPENAI_API_KEY` yet, consistent with Phase 3's scoping), so OpenAI's ingestion path is only unit-tested via a mocked API client proving correct row-building — not exercised end-to-end against Postgres. All real DB-write tests run against `local` + real Postgres instead.

Tests: 3 new files (`tests/test_chunking.py`, `tests/test_ingest.py`, `tests/test_bulk_ingest.py`) plus one new test in `tests/test_embeddings_openai.py` — 41 tests pass, 1 correctly skipped (real OpenAI API, no key). The integration tests were **actually run** against real Postgres and the real local model: ingesting a realistic sample document produced the expected chunks with correct `document_id`s, re-ingesting the same `source_uri` with different content replaced the old chunks, and a fake-DB unit test confirmed the fail-whole-batch policy writes nothing — not even prior document upserts — when one embedding call in a multi-batch call raises. `ruff check .` is clean; every new function/method has full `Args:`/`Returns:`/`Raises:` docstrings.

One follow-up refactor after this phase landed: `tests/conftest.py`'s `db_conn` fixture was updated to delegate to `db.get_connection()` (pointing `$DATABASE_URL` at `$TEST_DATABASE_URL` via `monkeypatch`) instead of duplicating the connect + `register_vector` logic — one canonical connection path for both application code and tests.

Exit criteria: ingesting a representative sample of your real documents (not synthetic data) completes without errors, and spot-checking the DB shows the expected number of chunks per document with correctly attached `document_id`s (this is where the ID-tracking logic from §6.1 gets its real test). — **Met.** Automated as `@pytest.mark.integration` tests rather than manual spot-checking: `test_ingest_document_creates_expected_chunks` and `test_bulk_ingest_document_ids_and_chunk_counts` both query `embeddings_bge_small` directly after ingestion and assert exact chunk counts/`document_id`s.

Pitfalls: chunk size/overlap tuned for one content type (e.g. long-form articles) can perform poorly on another (e.g. short structured records) — don't assume one chunking config fits all document types in your corpus.

---

### Phase 5 — Query / Search Pipeline

**Goal:** a `search()` function (§7) that takes a query string and returns ranked results with `document_id` attached, ready for the application layer to consume.

Tasks:
- Implement `search()` and confirm it uses `embed_query`, not `embed_documents`, for the query text.
- Add query-time filtering support if your application needs it (e.g. filter by `metadata->>'tenant_id'` or a date range) — this typically means adding a `WHERE` clause alongside the `ORDER BY ... <#>` and may need a composite index if filters are highly selective; test that the HNSW index is still used (`EXPLAIN ANALYZE`) once a `WHERE` clause is added, since some filter patterns can cause Postgres to fall back to a sequential scan.
- Tune `hnsw.ef_search` empirically against the latency/recall target set in Phase 2, not by guessing.
- Add pagination/top_k configurability, and decide on a similarity-score floor if your application shouldn't show very weak matches (e.g. discard results below a cosine similarity threshold rather than always returning exactly `top_k`).
- Add a fallback path for when the search returns zero results (common for narrow or sparse corpora) — decide whether the application should broaden the query, fall back to keyword search, or simply message "no results."

Deliverables: `search.py` with filtering and threshold support, `EXPLAIN ANALYZE` output captured for a representative query showing index usage, tests covering filtered and unfiltered search.

Exit criteria: search latency meets the target from Phase 2 under realistic data volume and concurrency (test with a simple load-testing script — even a handful of concurrent requests via `asyncio` or `locust` reveals connection-pooling issues early).

Pitfalls: adding filters without the right supporting index can silently make queries slow (sequential scan instead of HNSW) even though the query still "works" — always check `EXPLAIN ANALYZE` after adding a new filter pattern.

---

### Phase 6 — Evaluation & Model Selection

**Goal:** an evidence-based decision on which embedding provider(s) to actually use in production, backed by measurement rather than assumption.

Tasks:
- Build a labeled evaluation set (§9): 30–100 realistic queries paired with the document(s) that should be retrieved for each. This is the single highest-leverage artifact in the whole project — invest real time here, ideally with input from whoever understands the domain best (not just synthetic/guessed queries).
- Run `recall_at_k` (§9) for both the local and commercial provider against the same eval set and same corpus.
- Measure cost and latency per provider under realistic load: local model inference time per chunk on your actual hardware (CPU vs GPU matters a lot here), versus commercial API latency and per-token cost at your expected ingestion/query volume.
- Make an explicit, documented decision: one provider for everything, or a split (e.g. local for internal/low-stakes search, commercial for customer-facing search where quality matters more than infrastructure cost) — and record the reasoning, since this is exactly the kind of decision that gets silently second-guessed later without a written record of why it was made.
- If recall is unexpectedly low for either provider, revisit chunking (Phase 4) before concluding the model itself is the problem — poor chunking is a more common root cause than model quality.

Deliverables: eval dataset (version-controlled), a short written comparison (recall@k, latency, cost per 1k documents) for both providers, a documented decision.

Exit criteria: a written recommendation with numbers behind it, not a guess — this becomes the basis for the config used in Phase 7 onward.

Pitfalls: evaluating on a tiny or non-representative query set (e.g. 5 queries you wrote in ten minutes) will produce a confident-looking number that doesn't generalize — treat eval-set quality with the same seriousness as the retrieval system itself.

---

### Phase 7 — Production Hardening & Observability

**Goal:** the system is safe to run unattended: it handles failures gracefully, is observable when something goes wrong, and doesn't leak secrets or degrade silently.

Tasks:
- Add connection pooling (`psycopg_pool`) sized appropriately for expected concurrent query load; verify behavior under connection exhaustion rather than assuming the defaults are fine.
- Add structured logging around every external call (embedding API requests, DB queries) including latency and outcome, so a production incident can be diagnosed from logs alone.
- Add metrics/alerting on: embedding API error rate, ingestion queue depth/backlog (if ingestion is async), search latency percentiles, and HNSW index bloat/size growth over time.
- Move all secrets (API keys, `DATABASE_URL`) into a proper secrets manager (not `.env` files) for any non-local environment.
- Add a runbook entry for "commercial embedding API is down" — since the whole point of the abstraction layer is that you can fail over to the local provider; make sure that failover path is actually tested, not just theoretically possible.
- Load-test the full ingestion pipeline at expected production volume (not just the query path) — ingestion is often the actual bottleneck, especially against commercial API rate limits.
- Set up index maintenance: monitor for the need to `REINDEX` as data grows and changes significantly, since HNSW index quality can degrade with heavy update/delete churn over time.

Deliverables: pooled DB access layer, logging/metrics wired into an observability stack (whatever your org already uses — Prometheus/Grafana, Datadog, CloudWatch, etc.), a written runbook for provider failover, load-test results for both ingestion and query paths.

Exit criteria: a simulated commercial-API outage (e.g. point `OPENAI_API_KEY` at an invalid value in staging) is caught by monitoring and the system either fails over to the local provider or fails loudly and safely — not silently and slowly.

Pitfalls: treating observability as a "nice to have" added at the end rather than validated under actual failure conditions — an alert that's never been tested against a real failure is not a reliable alert.

---

### Phase 8 — Migration & Scaling Strategy

**Goal:** the ability to change embedding models, scale data volume, or adjust index tuning in the future without downtime or a risky one-shot cutover.

Tasks:
- Confirm the side-by-side migration pattern (§8) works end-to-end: stand up a second embeddings table for a candidate new model, backfill it from `documents.content` (not from existing vectors — embeddings can't be converted between models), and run the Phase 6 evaluation against both tables before cutting `EMBEDDING_BACKEND` over.
- Plan for horizontal scaling if data volume will exceed what a single Postgres instance comfortably handles: read replicas for query load, partitioning the embeddings table (e.g. by tenant or time range) if it grows very large, and revisiting whether a dedicated vector database becomes justified at that scale (a decision to defer, not to pre-optimize for, per the goals in §1 — pgvector comfortably handles millions of rows with proper indexing before this becomes necessary).
- Document the re-embedding cost/time estimate for the full corpus under the current provider, so a future model migration has a known cost instead of being an open-ended unknown.
- Set a policy for embedding table cleanup: when is it safe to drop an old model's table after a migration (e.g. only after a defined grace period with the new table validated in production).

Deliverables: a tested migration runbook, a documented scaling plan with concrete trigger conditions (e.g. "reconsider architecture above N million rows or M ms p95 latency"), a re-embedding cost estimate.

Exit criteria: a full dry-run migration to a second model in staging, including evaluation comparison and cutover, completed without needing to touch the application layer beyond the `EMBEDDING_BACKEND` config value.

Pitfalls: waiting until a migration is urgently needed (e.g. a provider deprecates a model) to discover the process was never actually tested end-to-end.

---

## Summary

- **Storage**: pgvector, HNSW index, one table per model dimension, inner product distance (equivalent to cosine on the unit-normalized vectors both providers emit, but faster — §3.4).
- **No lock-in**: a single `EmbeddingProvider` ABC; commercial and local implementations are interchangeable via one env var.
- **Commercial cheap option**: OpenAI `text-embedding-3-small` (or Voyage `voyage-3-lite` as a drop-in alternative).
- **Free option**: `sentence-transformers` running `BAAI/bge-small-en-v1.5` fully offline, or Ollama + `nomic-embed-text` if you prefer an HTTP-served local model.
- **Migration path**: side-by-side tables let you A/B evaluate before cutting over, and re-embedding from stored source text is always possible since raw content is retained independently of any vector.
