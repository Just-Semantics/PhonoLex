"""
API router for vector similarity search using pgvector.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from database import DatabaseService
from routers.words import WordResponse
from models import Word

router = APIRouter(prefix="/api/similarity", tags=["similarity"])


# Dependency injection
def get_db():
    """Get database service instance"""
    return DatabaseService()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class SimilarityResult(BaseModel):
    """Single similarity search result"""
    word: WordResponse
    similarity: float


class SimilaritySearchRequest(BaseModel):
    """Request for similarity search"""
    word: str
    threshold: float = 0.85
    limit: int = 50
    filters: Optional[Dict] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/word/{word}", response_model=List[SimilarityResult])
async def similarity_search_by_word(
    word: str,
    threshold: float = 0.85,
    limit: int = 50,
    db: DatabaseService = Depends(get_db)
):
    """
    Find words similar to given word using syllable embeddings.

    Uses pgvector's HNSW index for fast approximate nearest neighbor search.

    Args:
        word: Target word
        threshold: Minimum similarity (0-1), default 0.85
        limit: Max results, default 50

    Returns:
        List of similar words with similarity scores, sorted by similarity (highest first)
    """
    results = db.find_similar_words_by_embedding(
        word=word,
        threshold=threshold,
        limit=limit
    )

    if not results:
        # Check if word exists but has no embedding
        word_obj = db.get_word_by_word(word)
        if not word_obj:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found")
        elif not word_obj.syllable_embedding:
            raise HTTPException(status_code=400, detail=f"Word '{word}' has no embedding")
        else:
            # No similar words above threshold
            return []

    return [
        SimilarityResult(
            word=WordResponse.from_orm(word_obj),
            similarity=sim
        )
        for word_obj, sim in results
    ]


@router.post("/search", response_model=List[SimilarityResult])
async def similarity_search(
    request: SimilaritySearchRequest,
    db: DatabaseService = Depends(get_db)
):
    """
    Similarity search with additional filters.

    Args:
        request: Search request with word, threshold, limit, and optional filters

    Returns:
        List of similar words matching filters
    """
    results = db.find_similar_words_by_embedding(
        word=request.word,
        threshold=request.threshold,
        limit=request.limit
    )

    if not results:
        word_obj = db.get_word_by_word(request.word)
        if not word_obj:
            raise HTTPException(status_code=404, detail=f"Word '{request.word}' not found")
        return []

    # Apply additional filters if specified
    filtered_results = results
    if request.filters:
        filtered_results = []
        for word_obj, sim in results:
            # Apply filters
            if request.filters.get('max_wcm') and word_obj.wcm_score:
                if word_obj.wcm_score > request.filters['max_wcm']:
                    continue
            if request.filters.get('complexity') and word_obj.complexity:
                if word_obj.complexity != request.filters['complexity']:
                    continue

            filtered_results.append((word_obj, sim))

    return [
        SimilarityResult(
            word=WordResponse.from_orm(word_obj),
            similarity=sim
        )
        for word_obj, sim in filtered_results
    ]
