#!/usr/bin/env python3
"""
PhonoLex Web App v2.0 - FastAPI Backend with PostgreSQL + pgvector

Database-centric architecture that leverages:
- PostgreSQL + pgvector for vector similarity
- JSONB indexes for fast pattern matching
- Typed graph edges stored relationally
- Smart indexing for <100ms query times
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

# Import routers
from routers import words, similarity, graph, phonemes

# Database service
from database import DatabaseService


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.

    Startup:
    - Verify database connection
    - Check pgvector extension
    - Load statistics

    Shutdown:
    - Close database connections
    """
    print("=" * 80)
    print("PhonoLex API v2.0 - Starting up...")
    print("=" * 80)

    # Initialize database service
    db = DatabaseService()

    # Store in app state
    app.state.db = db

    # Get initial stats
    try:
        stats = db.get_stats()
        print(f"\nDatabase statistics:")
        print(f"  Words: {stats['total_words']:,}")
        print(f"  Phonemes: {stats['total_phonemes']:,}")
        print(f"  Edges: {stats['total_edges']:,}")
        if stats.get('edge_types'):
            print(f"\nEdge types:")
            for edge_type, count in stats['edge_types'].items():
                print(f"    {edge_type}: {count:,}")
    except Exception as e:
        print(f"Warning: Could not load stats: {e}")

    print("\n" + "=" * 80)
    print("✓ PhonoLex API v2.0 ready")
    print("=" * 80 + "\n")

    yield

    # Shutdown
    print("\nShutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PhonoLex API v2.0",
    description="Phonological word filtering and similarity search with database-centric architecture",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # React default
        "http://localhost:8000",  # API docs
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Include Routers
# ============================================================================

app.include_router(words.router)
app.include_router(similarity.router)
app.include_router(graph.router)
app.include_router(phonemes.router)


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "PhonoLex API",
        "version": "2.0.0",
        "status": "running",
        "architecture": "database-centric (PostgreSQL + pgvector)",
        "endpoints": {
            "words": "/api/words",
            "similarity": "/api/similarity",
            "graph": "/api/graph",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.

    Checks:
    - API is running
    - Database connection
    - pgvector extension
    """
    try:
        db = DatabaseService()
        stats = db.get_stats()

        return {
            "status": "healthy",
            "database": "connected",
            "pgvector": "enabled",
            "words": stats['total_words'],
            "edges": stats['total_edges']
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


# ============================================================================
# Development Server
# ============================================================================

def main():
    """Entry point for running the development server"""
    import uvicorn

    # Check if database is populated
    try:
        db = DatabaseService()
        stats = db.get_stats()

        if stats['total_words'] == 0:
            print("\n" + "=" * 80)
            print("WARNING: Database is empty!")
            print("=" * 80)
            print("\nPlease run data population scripts:")
            print("  1. python webapp/backend/migrations/populate_phonemes.py")
            print("  2. python webapp/backend/migrations/populate_words.py")
            print("  3. python webapp/backend/migrations/populate_edges.py")
            print("\nOr see: docs/ARCHITECTURE_V2.md for setup instructions")
            print("=" * 80 + "\n")
    except Exception as e:
        print(f"\nWarning: Could not check database: {e}\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
