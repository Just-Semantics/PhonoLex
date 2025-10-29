# PhonoLex v2.0 Deployment Guide

This guide covers deploying the PhonoLex v2.0 web application with a database-centric architecture.

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   React Frontend (Netlify/Cloudflare)  │
│   - Thin client: rendering only         │
│   - Minimal state: UI + recent results  │
└───────────────┬─────────────────────────┘
                │ API calls
                ▼
┌─────────────────────────────────────────┐
│   FastAPI Backend (Serverless)          │
│   - Thin orchestration layer            │
│   - Query builder                       │
└───────────────┬─────────────────────────┘
                │ SQL queries
                ▼
┌─────────────────────────────────────────┐
│   PostgreSQL + pgvector (Managed)       │
│   - All data + embeddings               │
│   - GIN/HNSW/B-tree indexes             │
│   - Recursive CTEs for graph traversal  │
└─────────────────────────────────────────┘
```

See [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) for detailed architecture documentation.

---

## Deployment Options

### Option 1: Netlify + Neon (RECOMMENDED)

**Best for**: Quick setup, generous free tier, native integration

#### Why This Wins
- ✅ **Native integration**: Neon is Netlify's default database
- ✅ **One-click setup**: No separate account needed
- ✅ **Free tier**: 10 GB storage, 1 GB RAM, 100 hours compute/month
- ✅ **Serverless Postgres**: Auto-scaling, instant cold starts (<100ms)
- ✅ **pgvector support**: Pre-installed, ready to use
- ✅ **Branching**: Database branches for preview deployments

#### Setup Steps

**1. Deploy Frontend to Netlify**
```bash
cd webapp/frontend
npm run build

# Deploy to Netlify (via CLI or Git integration)
netlify deploy --prod
```

**2. Add Neon Database**
```bash
# From Netlify dashboard:
# Integrations → Browse Integrations → Neon
# Click "Enable" → Creates database automatically

# Get connection string from Netlify environment variables
# NEON_DATABASE_URL will be automatically set
```

**3. Deploy Backend as Netlify Functions**
```bash
cd webapp/backend

# Create netlify/functions/api.py
# This wraps your FastAPI app for Netlify Functions
netlify functions:create api

# Deploy
netlify deploy --prod
```

**4. Initialize Database**
```bash
# Run migration script
python scripts/migrate_to_postgres.py --db-url $NEON_DATABASE_URL
```

#### Netlify Functions Setup

**File**: `netlify/functions/api.py`
```python
from fastapi import FastAPI
from mangum import Mangum
from webapp.backend.main import app  # Your FastAPI app

# Wrap FastAPI with Mangum for serverless
handler = Mangum(app)
```

**Configuration**: `netlify.toml`
```toml
[build]
  command = "npm run build"
  publish = "webapp/frontend/dist"

[functions]
  directory = "netlify/functions"
  node_bundler = "esbuild"

[[redirects]]
  from = "/api/*"
  to = "/.netlify/functions/api/:splat"
  status = 200

[dev]
  command = "npm run dev"
  port = 3000
```

#### Cost (as of 2025)
- **Free tier**:
  - Frontend: Unlimited
  - Functions: 125K requests/month
  - Database: 10 GB storage, 1 GB RAM, 100 hours compute
- **Paid**: Scales as needed

---

### Option 2: Cloudflare Pages + D1

**Best for**: Global edge deployment, ultra-low latency

#### Why Consider This
- ✅ **Edge-first**: Deploy to 300+ locations worldwide
- ✅ **D1 database**: SQLite at the edge (in beta, free during beta)
- ✅ **Workers**: Serverless functions with <50ms cold starts
- ✅ **Free tier**: 100K requests/day, 10 GB storage
- ⚠️ **pgvector**: Not native (would need custom HNSW implementation)

#### Setup Steps

**1. Create Cloudflare Pages Project**
```bash
cd webapp/frontend
npm run build

# Deploy via Wrangler
npx wrangler pages deploy dist --project-name=phonolex
```

**2. Create D1 Database**
```bash
npx wrangler d1 create phonolex-db

# Output: database_id = "xxxx-xxxx-xxxx"
```

**3. Deploy Backend as Cloudflare Worker**
```bash
cd webapp/backend

# Create wrangler.toml
npx wrangler init

# Deploy
npx wrangler deploy
```

**4. Bind Database to Worker**
```toml
# wrangler.toml
name = "phonolex-api"
main = "src/index.ts"

[[d1_databases]]
binding = "DB"
database_name = "phonolex-db"
database_id = "xxxx-xxxx-xxxx"
```

#### Vector Search on D1

Since D1 doesn't have pgvector, you'd need to:

**Option A**: Implement ANN with SQLite extensions
```sql
-- Use SQLite vec extension (experimental)
-- https://github.com/asg017/sqlite-vec
```

**Option B**: Hybrid approach
```
D1 (structured data) + Cloudflare Vectorize (vector search)
```

**Option C**: Store vectors, compute similarity in application
```python
# Fetch candidate vectors from D1
# Compute cosine similarity in Python/JS
# This works for small result sets (<1000 vectors)
```

#### Cost (as of 2025)
- **Free tier**:
  - Pages: Unlimited bandwidth
  - Workers: 100K requests/day
  - D1: 10 GB storage (beta, free)
- **Paid**: $5/month for Workers, D1 pricing TBD

---

### Option 3: Vercel + Neon

**Best for**: Next.js projects, seamless DX

#### Why Consider This
- ✅ **Next.js optimized**: If you want to switch from React to Next.js
- ✅ **Neon integration**: Same database as Option 1
- ✅ **Edge functions**: Similar to Netlify
- ⚠️ **Python backend**: Not as well-supported (Node.js preferred)

#### Setup Steps

Similar to Netlify, but:
- Frontend: Deploy Next.js app to Vercel
- Backend: Rewrite FastAPI → Next.js API routes (TypeScript)
- Database: Add Neon integration from Vercel dashboard

---

### Option 4: Self-Hosted (Docker Compose)

**Best for**: Full control, on-premise deployment, development

#### Setup Steps

**File**: `docker-compose.yml`
```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: phonolex
      POSTGRES_DB: phonolex
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: ./webapp/backend
    environment:
      DATABASE_URL: postgresql://postgres:phonolex@postgres:5432/phonolex
    ports:
      - "8000:8000"
    depends_on:
      - postgres

  frontend:
    build: ./webapp/frontend
    environment:
      VITE_API_URL: http://localhost:8000
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  pgdata:
```

**Run**:
```bash
docker-compose up -d
```

---

## Database Setup

### Schema Migration

**File**: `scripts/migrate_to_postgres.py`

```python
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import pickle
import numpy as np

def migrate(db_url):
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Enable pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS words (
        word_id SERIAL PRIMARY KEY,
        word VARCHAR(100) UNIQUE NOT NULL,
        ipa VARCHAR(200),
        phonemes_json JSONB NOT NULL,
        syllable_count INT,
        syllable_embedding vector(384),
        frequency FLOAT,
        aoa FLOAT,
        imageability FLOAT,
        familiarity FLOAT,
        concreteness FLOAT,
        valence FLOAT,
        arousal FLOAT,
        dominance FLOAT,
        wcm_score INT,
        msh_stage INT
    );
    """)

    # Create indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_words_phonemes_gin ON words USING gin (phonemes_json jsonb_path_ops);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_words_syllable_embedding ON words USING hnsw (syllable_embedding vector_cosine_ops);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_words_frequency ON words (frequency DESC);")

    # Load graph
    with open('data/phonological_graph.pkl', 'rb') as f:
        data = pickle.load(f)

    graph = data['graph']

    # Insert words
    for node, attrs in graph.nodes(data=True):
        cur.execute("""
        INSERT INTO words (word, ipa, phonemes_json, syllable_count, frequency,
                          aoa, imageability, familiarity, concreteness,
                          valence, arousal, dominance, wcm_score, msh_stage)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (word) DO NOTHING;
        """, (
            node,
            attrs.get('ipa'),
            json.dumps(attrs.get('phonemes', [])),
            len(attrs.get('syllables', [])),
            attrs.get('frequency'),
            attrs.get('aoa'),
            attrs.get('imageability'),
            attrs.get('familiarity'),
            attrs.get('concreteness'),
            attrs.get('valence'),
            attrs.get('arousal'),
            attrs.get('dominance'),
            attrs.get('wcm_score'),
            attrs.get('msh_stage')
        ))

    # Create edges table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS word_edges (
        edge_id SERIAL PRIMARY KEY,
        word1_id INT REFERENCES words(word_id),
        word2_id INT REFERENCES words(word_id),
        relation_type VARCHAR(50) NOT NULL,
        metadata JSONB,
        UNIQUE(word1_id, word2_id, relation_type)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON word_edges (relation_type);")

    # Insert edges
    for u, v, data in graph.edges(data=True):
        cur.execute("""
        INSERT INTO word_edges (word1_id, word2_id, relation_type, metadata)
        SELECT w1.word_id, w2.word_id, %s, %s
        FROM words w1, words w2
        WHERE w1.word = %s AND w2.word = %s
        ON CONFLICT DO NOTHING;
        """, (
            data.get('type', 'UNKNOWN'),
            json.dumps(data),
            u,
            v
        ))

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Migrated {graph.number_of_nodes()} words and {graph.number_of_edges()} edges")

if __name__ == "__main__":
    import sys
    db_url = sys.argv[1]
    migrate(db_url)
```

**Run**:
```bash
python scripts/migrate_to_postgres.py $DATABASE_URL
```

---

## Performance Optimization

### Index Tuning

```sql
-- Pattern matching (JSONB)
CREATE INDEX idx_words_phonemes_gin ON words USING gin (phonemes_json jsonb_path_ops);

-- Vector similarity (HNSW for ANN)
CREATE INDEX idx_words_syllable_embedding ON words
USING hnsw (syllable_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Property filtering
CREATE INDEX idx_words_frequency ON words (frequency DESC);
CREATE INDEX idx_words_wcm_score ON words (wcm_score);
CREATE INDEX idx_words_syllable_count ON words (syllable_count);

-- Graph queries
CREATE INDEX idx_edges_type ON word_edges (relation_type);
CREATE INDEX idx_edges_word1 ON word_edges (word1_id);
CREATE INDEX idx_edges_word2 ON word_edges (word2_id);
```

### Query Optimization

**Before**:
```sql
-- Slow: Sequential scan
SELECT * FROM words WHERE word LIKE 'cat%';
```

**After**:
```sql
-- Fast: Index scan
SELECT * FROM words WHERE word >= 'cat' AND word < 'cau';
```

### Connection Pooling

```python
from psycopg2.pool import SimpleConnectionPool

pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL
)

# In FastAPI
@app.on_event("startup")
async def startup():
    app.state.db_pool = pool

@app.on_event("shutdown")
async def shutdown():
    app.state.db_pool.closeall()
```

---

## Monitoring

### Query Performance

```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Database Size

```sql
-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Cost Estimates

### Netlify + Neon (Recommended)

**Free tier**:
- 26K words × ~200 bytes = ~5 MB (structured data)
- 26K words × 384-dim × 4 bytes = ~40 MB (vectors)
- Total: ~50 MB << 10 GB free ✅

**Estimated monthly cost (free tier)**:
- $0 for typical usage (<10K requests/day)

**Paid tier** (if scaling):
- Netlify: $19/month (100K requests/month)
- Neon: $19/month (5 GB storage, 100 hours compute)
- Total: ~$40/month for moderate traffic

### Cloudflare

**Free tier**:
- 100K requests/day ✅
- D1: 10 GB storage ✅

**Estimated monthly cost**:
- $0 for typical usage

---

## Checklist

- [ ] Choose deployment option (Netlify + Neon recommended)
- [ ] Set up frontend deployment
- [ ] Set up database (Neon/D1/self-hosted)
- [ ] Run migration script
- [ ] Deploy backend (Netlify Functions/Workers)
- [ ] Configure environment variables
- [ ] Test API endpoints
- [ ] Verify database indexes
- [ ] Set up monitoring
- [ ] Configure custom domain (optional)

---

## Troubleshooting

### "pgvector extension not found"

**Fix**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If still failing, ensure your Postgres version supports pgvector (14+).

### "HNSW index build too slow"

**Fix**: Reduce `ef_construction`:
```sql
CREATE INDEX CONCURRENTLY idx_words_syllable_embedding
ON words USING hnsw (syllable_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 32);  -- Lower = faster build, slightly worse recall
```

### "Out of memory during migration"

**Fix**: Batch inserts:
```python
# Instead of inserting all at once
for i in range(0, len(words), 1000):
    batch = words[i:i+1000]
    cur.executemany(insert_query, batch)
    conn.commit()
```

---

## Further Reading

- **[ARCHITECTURE_V2.md](ARCHITECTURE_V2.md)** - Detailed architecture design
- **[Neon Documentation](https://neon.tech/docs)** - Neon database docs
- **[pgvector Documentation](https://github.com/pgvector/pgvector)** - Vector extension for Postgres
- **[Netlify Functions](https://docs.netlify.com/functions/overview/)** - Serverless functions guide

---

**Last Updated**: 2025-10-28
