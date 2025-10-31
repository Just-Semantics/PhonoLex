# PhonoLex v2.0 Deployment Guide

This guide evaluates deployment options for PhonoLex v2.0, which consists of:
- **Frontend**: React + TypeScript + Vite + MUI
- **Backend**: FastAPI (Python 3.10+) with SQLAlchemy ORM
- **Database**: PostgreSQL 15+ with pgvector extension
- **Data**: 26K words with syllable embeddings (1GB), phonological features, psycholinguistic properties

## Executive Summary

**Recommended Deployment Strategy**: Railway or Render

| Platform | Frontend | Backend | Database | Cost | Verdict |
|----------|----------|---------|----------|------|---------|
| **Railway** ⭐ | ✅ Static | ✅ Full Python | ✅ PostgreSQL | $5/mo + usage | **RECOMMENDED** |
| **Render** | ✅ Static | ✅ Full Python | ✅ PostgreSQL | $7/mo (90-day free trial) | **RECOMMENDED** |
| Netlify | ✅ Excellent | ❌ No Python | ⚠️ Neon (via extension) | Free + paid tiers | ❌ **Backend incompatible** |
| Cloudflare | ✅ Pages | ⚠️ Limited Python | ⚠️ D1 or external | Free + paid tiers | ❌ **PostgreSQL drivers incompatible** |
| Vercel | ✅ Excellent | ⚠️ Limited Python | ❌ External only | Free + paid tiers | ⚠️ **Not ideal for backend** |

---

## Platform Analysis

### 1. Railway ⭐ RECOMMENDED

**Pros**:
- ✅ One-click deployment for FastAPI + PostgreSQL
- ✅ Full Python support (no WebAssembly limitations)
- ✅ Automatic PostgreSQL provisioning with pgvector support
- ✅ Simple GitHub integration with auto-deploy
- ✅ Built-in monitoring and logging
- ✅ Environment variables managed automatically
- ✅ Docker support (can deploy custom images)
- ✅ $5 trial credit to evaluate platform

**Cons**:
- 💰 Usage-based pricing (CPU: $20/vCPU/mo, RAM: $10/GB/mo)
- 📊 Costs can scale with traffic

**Pricing** (2025):
- **Trial**: $5 one-time credit (30-day expiration)
- **Hobby**: $5/month (includes $5 usage credit)
- **Usage-based**: Pay for CPU, RAM, storage, bandwidth consumed
- **Committed Spend Tiers**: Available for consistent usage

**Deployment Process**:
1. Connect GitHub repository
2. Railway auto-detects FastAPI (or use Dockerfile)
3. Add PostgreSQL service (one-click)
4. Deploy frontend as static site
5. Configure environment variables
6. Done!

**Best For**: Production deployments, predictable traffic, teams wanting full control

---

### 2. Render ⭐ RECOMMENDED (Alternative)

**Pros**:
- ✅ Full Python/FastAPI support
- ✅ Managed PostgreSQL with automatic daily backups (paid tiers)
- ✅ Docker support out of the box
- ✅ 90-day free PostgreSQL database (for testing)
- ✅ Connection pooling and read replicas available
- ✅ Flat-rate pricing (easier to predict than usage-based)

**Cons**:
- ⏰ Free database expires after 90 days
- 💰 $7/month minimum for always-on services
- ❌ Free databases don't include backups

**Pricing** (2025):
- **Free Tier**: Available for testing (database expires in 90 days)
- **Paid Instances**: Start at $7/month for always-on services
- **PostgreSQL**: Paid plans include automatic daily backups

**Deployment Process**:
1. Connect GitHub repository
2. Create PostgreSQL database (select free or paid tier)
3. Deploy FastAPI backend (Render auto-detects or use Dockerfile)
4. Deploy frontend as static site
5. Configure environment variables
6. Done!

**Best For**: Developers wanting predictable flat-rate pricing, Docker-first workflows

---

### 3. Netlify ❌ NOT COMPATIBLE

**Why it doesn't work**:
- ❌ **No Python support** for serverless functions
- ❌ Only supports JavaScript, TypeScript, Go
- ⚠️ Has Neon PostgreSQL extension, but can't run Python backend

**What Netlify IS good for**:
- ✅ Excellent frontend hosting (React/Vite)
- ✅ Netlify DB (Neon-powered PostgreSQL, one-click provisioning)
- ✅ Great for JAMstack apps with JavaScript backends

**Possible Workaround** (not recommended):
- Host React frontend on Netlify
- Host FastAPI backend on Railway/Render
- Use Netlify DB (Neon) for database
- **Complexity**: High - requires CORS configuration, multiple platforms

**Verdict**: Use Netlify for frontend-only projects or JavaScript backends

---

### 4. Cloudflare Workers/Pages ⚠️ LIMITED COMPATIBILITY

**Python Support Status**:
- ⚠️ Python Workers available (Beta as of 2025)
- ⚠️ Uses Pyodide (Python compiled to WebAssembly)
- ❌ **Critical Limitation**: psycopg2 and SQLAlchemy PostgreSQL drivers DO NOT WORK
- ❌ Packages with C extensions incompatible with WebAssembly
- ❌ **As of 2025, packages don't run in production** (beta limitation)

**What Works**:
- ✅ Cloudflare Pages: Excellent for React frontend
- ✅ Python Workers: Pure Python code (standard library)
- ✅ FastAPI framework supported (limited use cases)
- ✅ Hyperdrive: Connection pooling for external databases

**What Doesn't Work**:
- ❌ psycopg2 (PostgreSQL driver with C extensions)
- ❌ SQLAlchemy with PostgreSQL backend
- ❌ Most database libraries with native dependencies

**Database Options**:
1. **Cloudflare D1** (SQLite-based, serverless)
   - ⚠️ Not PostgreSQL (would require full rewrite)
   - ⚠️ No pgvector support
2. **External Neon with Hyperdrive** (connection pooling)
   - ⚠️ Requires pure Python HTTP client (can't use psycopg2)
   - ⚠️ Would need PostgREST or similar HTTP API wrapper

**Pricing**:
- **Workers Free**: 100K requests/day
- **Workers Paid**: $5/month (10M requests/month)
- **Pages**: Free (unlimited bandwidth)
- **Hyperdrive**: Free tier available

**Possible Architecture** (complex):
```
Frontend (Cloudflare Pages)
    ↓
Python Worker (pure Python only)
    ↓
PostgREST HTTP API (wrapper for PostgreSQL)
    ↓
Neon PostgreSQL (external)
```

**Verdict**: Not suitable for PhonoLex v2.0 due to PostgreSQL driver incompatibility. Would require significant architecture changes.

---

### 5. Vercel ⚠️ NOT IDEAL

**Status**:
- ✅ Excellent for React/Next.js frontends
- ⚠️ Limited Python support (serverless functions only)
- ❌ Not optimized for FastAPI backends
- ❌ No managed PostgreSQL (must use external)

**Why it's not ideal**:
- Vercel is frontend-first (Next.js ecosystem)
- Python serverless functions have limitations (similar to Netlify)
- Not designed for full FastAPI applications
- Better suited for lightweight APIs or Next.js API routes

**Pricing**:
- **Free Tier**: Good for demos
- **Usage-Based**: Can scale quickly with bandwidth/execution time

**Verdict**: Use Vercel for frontend, host backend elsewhere (like Railway/Render)

---

## Recommended Architecture: Railway (Full-Stack)

```
┌─────────────────────────────────────────────────┐
│              Railway Project                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐    ┌──────────────┐         │
│  │   Frontend   │    │   Backend    │         │
│  │   (React)    │───▶│   (FastAPI)  │         │
│  │   Static     │    │   Python     │         │
│  └──────────────┘    └──────┬───────┘         │
│                              │                  │
│                              ▼                  │
│                      ┌──────────────┐          │
│                      │  PostgreSQL  │          │
│                      │  + pgvector  │          │
│                      └──────────────┘          │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Services**:
1. **Frontend Service**: Static site (React build)
2. **Backend Service**: Python web service (FastAPI)
3. **Database Service**: PostgreSQL with pgvector

**Environment Variables** (auto-configured by Railway):
- `DATABASE_URL`: PostgreSQL connection string
- `FRONTEND_URL`: Frontend URL for CORS
- Other FastAPI settings

---

## Recommended Architecture: Render (Full-Stack Alternative)

```
┌─────────────────────────────────────────────────┐
│               Render Project                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐    ┌──────────────┐         │
│  │   Frontend   │    │   Backend    │         │
│  │   (Static)   │───▶│   (Web Svc)  │         │
│  │              │    │   FastAPI    │         │
│  └──────────────┘    └──────┬───────┘         │
│                              │                  │
│                              ▼                  │
│                      ┌──────────────┐          │
│                      │  PostgreSQL  │          │
│                      │  (Managed)   │          │
│                      └──────────────┘          │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Services**:
1. **Static Site**: React frontend
2. **Web Service**: Python/FastAPI backend (Docker-based)
3. **PostgreSQL**: Managed database with automatic backups (paid)

---

## Hybrid Architecture: Cloudflare Pages + Railway Backend

If you already have a Cloudflare subscription and want to use it:

```
┌──────────────────┐         ┌─────────────────────────┐
│  Cloudflare      │         │     Railway             │
│  Pages           │         ├─────────────────────────┤
│  (React)         │────────▶│  Backend (FastAPI)      │
│                  │  CORS   │          │              │
└──────────────────┘         │          ▼              │
                             │  PostgreSQL + pgvector  │
                             └─────────────────────────┘
```

**Pros**:
- ✅ Cloudflare's global CDN for frontend (fast worldwide)
- ✅ Railway handles complex backend/database (full Python support)
- ✅ Cloudflare Pages is free (unlimited bandwidth)

**Cons**:
- 🔧 Requires CORS configuration
- 💰 Pay for two platforms
- 🔗 More complex deployment workflow

**Cost Estimate**:
- Cloudflare Pages: **Free**
- Railway: **$5-20/month** (depending on usage)

---

## Cost Comparison (Monthly)

### Low Traffic (< 10K requests/day, < 1GB data transfer)

| Platform | Cost | Notes |
|----------|------|-------|
| Railway | $5-10 | $5 subscription + minimal usage |
| Render | $7 | Flat rate, always-on |
| Cloudflare + Railway | $5-10 | Pages free, Railway backend |

### Medium Traffic (100K requests/day, 10GB data transfer)

| Platform | Cost | Notes |
|----------|------|-------|
| Railway | $15-30 | Usage-based scaling |
| Render | $7-25 | May need higher tier |
| Cloudflare + Railway | $15-30 | Pages free, Railway scales |

---

## Database Considerations

### PostgreSQL Requirements for PhonoLex

- **Version**: PostgreSQL 15+
- **Extension**: pgvector (for syllable embeddings)
- **Storage**: ~2GB (26K words + embeddings + psycholinguistic data)
- **Connections**: Moderate (web app traffic)

### pgvector Extension

- ✅ **Railway**: Supports custom PostgreSQL extensions
- ✅ **Render**: Supports custom PostgreSQL extensions
- ⚠️ **Neon** (Netlify/external): Check pgvector support
- ❌ **Cloudflare D1**: SQLite-based (no PostgreSQL)

### Database Migration

Both Railway and Render support running migration scripts on deployment:
1. Add `migrations/` directory with SQL scripts
2. Configure deployment hooks to run migrations
3. Use Alembic (SQLAlchemy migration tool) for version control

---

## Deployment Checklist

### Pre-Deployment

- [ ] Verify all environment variables are documented
- [ ] Test database migrations locally
- [ ] Ensure frontend build works (`npm run build`)
- [ ] Test backend API locally (`uvicorn main:app`)
- [ ] Verify CORS settings for production domains
- [ ] Check `.gitignore` excludes large files (embeddings, models)
- [ ] Prepare `requirements.txt` (or `pyproject.toml`)
- [ ] Create `Dockerfile` (optional, but recommended for Render)

### Railway Deployment

- [ ] Create Railway account
- [ ] Create new project
- [ ] Add PostgreSQL service
- [ ] Deploy backend (connect GitHub repo)
- [ ] Deploy frontend (static site)
- [ ] Configure environment variables
- [ ] Run database migrations
- [ ] Test API endpoints
- [ ] Test frontend → backend connectivity
- [ ] Configure custom domain (optional)

### Render Deployment

- [ ] Create Render account
- [ ] Create PostgreSQL database (free or paid)
- [ ] Create web service for backend
- [ ] Create static site for frontend
- [ ] Configure environment variables
- [ ] Run database migrations
- [ ] Test API endpoints
- [ ] Test frontend → backend connectivity
- [ ] Configure custom domain (optional)

---

## Environment Variables

### Backend (FastAPI)

```bash
# Database
DATABASE_URL=postgresql://user:password@host:port/dbname

# Frontend (for CORS)
FRONTEND_URL=https://phonolex.pages.dev

# Optional: API keys, secrets
SECRET_KEY=your-secret-key
```

### Frontend (React/Vite)

```bash
# API endpoint
VITE_API_URL=https://api.phonolex.com

# Optional: analytics, monitoring
VITE_ANALYTICS_ID=your-analytics-id
```

---

## Large File Handling

PhonoLex has large embeddings (1GB+) that should NOT be committed to Git.

### Option 1: Build Embeddings on Deployment
- Store raw data in Git (CMU dictionary, PHOIBLE features)
- Run embedding build scripts during deployment
- ⚠️ Slow initial deployment (~10-15 minutes)
- ✅ No large file storage issues

### Option 2: External Storage (Recommended)
- Store embeddings in S3/R2/Cloud Storage
- Download during startup (cached locally)
- ✅ Fast deployment
- 💰 Small storage cost (~$0.50/month)

### Option 3: Railway/Render Persistent Volumes
- Store embeddings in persistent volume
- Build once, reuse across deployments
- ✅ Fast deployments after initial build
- 💰 Volume storage cost (~$0.10/GB/month)

---

## Next Steps

1. **Choose Platform**: Railway (recommended) or Render
2. **Prepare Repository**:
   - Add `Dockerfile` (optional but recommended)
   - Update `.gitignore` for large files
   - Document environment variables
3. **Create Deployment Guide**: Step-by-step instructions for chosen platform
4. **Test Deployment**: Use trial/free tier first
5. **Monitor Costs**: Set up billing alerts
6. **Configure Domain**: Point custom domain to deployment

---

## References

- [Railway Documentation](https://docs.railway.com/)
- [Render Documentation](https://render.com/docs)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Cloudflare Workers Python Docs](https://developers.cloudflare.com/workers/languages/python/)
- [Netlify Functions Overview](https://docs.netlify.com/build/functions/overview/)

---

## Questions to Consider

Before finalizing deployment strategy:

1. **Traffic expectations**: How many users/requests per day?
2. **Budget**: Comfortable with usage-based vs flat-rate pricing?
3. **Existing subscriptions**: Already using Cloudflare/Netlify/Vercel?
4. **DevOps complexity**: Want simple deployment or more control?
5. **Database needs**: Size, backup frequency, read replicas?
6. **Global distribution**: Need multi-region deployment?

---

## Conclusion

**For PhonoLex v2.0, we recommend Railway** for the following reasons:

1. ✅ Full Python support (no WebAssembly limitations)
2. ✅ One-click PostgreSQL with pgvector
3. ✅ Simple deployment workflow
4. ✅ Good documentation and community
5. ✅ Affordable for low-medium traffic
6. ✅ $5 trial to test platform

**Alternative: Render** if you prefer flat-rate pricing and want to test with the 90-day free PostgreSQL database.

Both platforms will work well for PhonoLex v2.0. Choose based on pricing preference (usage-based vs flat-rate) and trial experience.
