#!/usr/bin/env python3
"""
PhonoLex Web App - FastAPI Backend

Serves phonological word filtering and similarity search APIs for SLPs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
from services.phoneme_filter import (
    PhonemeFilterService,
    PhonemeConstraint,
    WordFilter
)

# Initialize FastAPI app
app = FastAPI(
    title="PhonoLex API",
    description="Phonological word filtering and similarity search for SLPs",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance (loaded on startup)
filter_service: Optional[PhonemeFilterService] = None


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class PhonemeConstraintSchema(BaseModel):
    """Schema for a single phoneme constraint (position-based or contains)"""
    position: Optional[int] = None  # None = "contains" mode (phoneme anywhere in word)
    phoneme_type: Optional[Literal['vowel', 'consonant']] = None
    allowed_phonemes: Optional[List[str]] = None
    required_features: Optional[Dict[str, str]] = None


class WordFilterRequest(BaseModel):
    """Request schema for word filtering"""
    pattern: List[PhonemeConstraintSchema]
    min_syllables: Optional[int] = None
    max_syllables: Optional[int] = None
    min_phonemes: Optional[int] = None
    max_phonemes: Optional[int] = None
    limit: Optional[int] = 500  # Max results to return (pagination)
    offset: Optional[int] = 0   # Pagination offset


class WordResult(BaseModel):
    """Schema for a single word result"""
    word: str
    phonemes: List[str]
    stress: List[Optional[int]]
    syllables: int
    ipa: str
    frequency: float  # Log10 word frequency from SUBTLEX


class FilterResponse(BaseModel):
    """Response schema for word filtering"""
    count: int
    words: List[WordResult]


class PhonemeInfo(BaseModel):
    """Schema for phoneme information"""
    phoneme: str
    type: Literal['vowel', 'consonant']
    features: Dict[str, str]


class PhonemeListResponse(BaseModel):
    """Response schema for phoneme lists"""
    vowels: List[str]
    consonants: List[str]


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global filter_service
    print("Starting PhonoLex API...")
    filter_service = PhonemeFilterService()
    print("✓ Ready to serve requests")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "PhonoLex API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/api/phonemes", response_model=PhonemeListResponse)
async def get_phonemes():
    """
    Get lists of all available phonemes.

    Returns:
        Lists of vowels and consonants
    """
    if filter_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "vowels": sorted(list(filter_service.vowel_phonemes)),
        "consonants": sorted(list(filter_service.consonant_phonemes))
    }


@app.get("/api/phonemes/{phoneme}")
async def get_phoneme_info(phoneme: str):
    """
    Get detailed information about a specific phoneme.

    Args:
        phoneme: IPA phoneme symbol

    Returns:
        Phoneme type and Phoible features
    """
    if filter_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Check if phoneme exists
    if phoneme not in filter_service.phoneme_features:
        raise HTTPException(status_code=404, detail=f"Phoneme '{phoneme}' not found")

    phoneme_type = 'vowel' if phoneme in filter_service.vowel_phonemes else 'consonant'
    features = filter_service.phoneme_features[phoneme]

    return {
        "phoneme": phoneme,
        "type": phoneme_type,
        "features": features
    }


@app.post("/api/phonemes/by-features")
async def get_phonemes_by_features(features: Dict[str, str]):
    """
    Find phonemes matching feature specification.

    Args:
        features: Dict of feature names and values, e.g.:
                  {"periodicGlottalSource": "+", "nasal": "-"}

    Returns:
        List of matching phonemes
    """
    if filter_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    matching = filter_service.get_phonemes_by_features(features)

    return {
        "features": features,
        "matching_phonemes": sorted(list(matching)),
        "count": len(matching)
    }


@app.post("/api/filter", response_model=FilterResponse)
async def filter_words(request: WordFilterRequest):
    """
    Filter words based on phonological constraints.

    Args:
        request: WordFilterRequest with pattern and constraints

    Returns:
        List of matching words with phonological information
    """
    if filter_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # DEBUG: Log the incoming request
    print(f"\n=== FILTER REQUEST ===")
    print(f"Pattern constraints: {len(request.pattern)}")
    for i, c in enumerate(request.pattern):
        print(f"  Constraint {i}: pos={c.position}, type={c.phoneme_type}, phonemes={c.allowed_phonemes}")
    print(f"=====================\n")

    # Convert request to internal types
    pattern = []
    for constraint_schema in request.pattern:
        constraint = PhonemeConstraint(
            position=constraint_schema.position,
            phoneme_type=constraint_schema.phoneme_type,
            allowed_phonemes=set(constraint_schema.allowed_phonemes) if constraint_schema.allowed_phonemes else None,
            required_features=constraint_schema.required_features
        )
        pattern.append(constraint)

    word_filter = WordFilter(
        pattern=pattern,
        min_syllables=request.min_syllables,
        max_syllables=request.max_syllables,
        min_phonemes=request.min_phonemes,
        max_phonemes=request.max_phonemes
    )

    # Execute filter
    results = filter_service.filter_words(word_filter)

    # Apply pagination
    total_count = len(results)
    offset = request.offset or 0
    limit = request.limit or 500
    paginated_results = results[offset:offset + limit]

    # DEBUG: Log first few results
    print(f"Returning {len(paginated_results)} words (total: {total_count})")
    if paginated_results:
        print("First 5 results:")
        for i, w in enumerate(paginated_results[:5]):
            first_phoneme = w['phonemes'][0] if w['phonemes'] else 'NONE'
            print(f"  {i+1}. {w['word']}: first phoneme = {first_phoneme}")
    print()

    return {
        "count": total_count,
        "words": paginated_results
    }


@app.get("/api/stats")
async def get_stats():
    """
    Get corpus statistics.

    Returns:
        Counts and statistics about the lexicon
    """
    if filter_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "total_words": len(filter_service.words),
        "total_phonemes": len(filter_service.vowel_phonemes) + len(filter_service.consonant_phonemes),
        "vowels": len(filter_service.vowel_phonemes),
        "consonants": len(filter_service.consonant_phonemes),
        "indexed_positions": len(filter_service.position_index)
    }


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
