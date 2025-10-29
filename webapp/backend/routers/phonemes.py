"""
API router for phoneme queries.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from database import DatabaseService
from sqlalchemy import text

router = APIRouter(prefix="/api/phonemes", tags=["phonemes"])

# Dependency injection for database service
def get_db():
    """Get database service instance"""
    return DatabaseService()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class PhonemeResponse(BaseModel):
    """Response schema for a phoneme"""
    phoneme_id: int
    ipa: str
    segment_class: str
    features: Dict[str, str]
    has_trajectory: bool


class PhonemeListResponse(BaseModel):
    """Response schema for list of phonemes"""
    phonemes: List[PhonemeResponse]
    total_count: int


class PhonemeSearchRequest(BaseModel):
    """Request schema for searching phonemes by features"""
    features: Dict[str, str]  # e.g., {"voice": "+", "consonantal": "+"}


class PhonemeSearchResponse(BaseModel):
    """Response schema for phoneme feature search"""
    features: Dict[str, str]
    matching_phonemes: List[str]
    count: int


# ============================================================================
# Endpoints
# ============================================================================

@router.get("", response_model=PhonemeListResponse)
async def list_phonemes(
    used_in_words: bool = True,
    db: DatabaseService = Depends(get_db)
):
    """
    Get list of phonemes with their Phoible features.

    Args:
        used_in_words: If True, only return phonemes that appear in the words table (default: True)

    Returns:
        List of phonemes with features
    """
    if used_in_words:
        # Only return phonemes that are actually used in our word dataset
        query = text("""
            WITH used_phonemes AS (
                SELECT DISTINCT jsonb_array_elements(phonemes_json)->>'ipa' as ipa
                FROM words
            )
            SELECT p.phoneme_id, p.ipa, p.segment_class, p.features, p.has_trajectory
            FROM phonemes p
            INNER JOIN used_phonemes u ON p.ipa = u.ipa
            ORDER BY p.ipa
        """)
    else:
        # Return all phonemes from Phoible database
        query = text("""
            SELECT phoneme_id, ipa, segment_class, features, has_trajectory
            FROM phonemes
            ORDER BY ipa
        """)

    with db.get_session() as session:
        result = session.execute(query).fetchall()

    phonemes = [
        PhonemeResponse(
            phoneme_id=row[0],
            ipa=row[1],
            segment_class=row[2],
            features=row[3] or {},
            has_trajectory=row[4] or False
        )
        for row in result
    ]

    return PhonemeListResponse(
        phonemes=phonemes,
        total_count=len(phonemes)
    )


@router.get("/{ipa}")
async def get_phoneme(ipa: str, db: DatabaseService = Depends(get_db)):
    """
    Get a specific phoneme by IPA symbol.

    Args:
        ipa: IPA symbol (e.g., "æ", "t", "ŋ")

    Returns:
        Phoneme details with Phoible features
    """
    query = text("""
        SELECT phoneme_id, ipa, segment_class, features, has_trajectory
        FROM phonemes
        WHERE ipa = :ipa
        LIMIT 1
    """)

    with db.get_session() as session:
        result = session.execute(query, {"ipa": ipa}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail=f"Phoneme '{ipa}' not found")

    return {
        "phoneme": result[1],
        "type": result[2],
        "features": result[3] or {}
    }


@router.post("/by-features", response_model=PhonemeSearchResponse)
async def search_phonemes_by_features(
    request: PhonemeSearchRequest,
    db: DatabaseService = Depends(get_db)
):
    """
    Search for phonemes matching specific feature values.

    Args:
        request: Dictionary of feature constraints (e.g., {"voice": "+", "nasal": "+"})

    Returns:
        List of matching phoneme IPA symbols
    """
    import json

    if not request.features:
        raise HTTPException(status_code=400, detail="At least one feature constraint required")

    # Build JSONB query using PostgreSQL @> operator
    # Convert features dict to JSON string for query
    feature_json_str = json.dumps(request.features)

    query = text("""
        SELECT ipa
        FROM phonemes
        WHERE features @> CAST(:feature_json AS jsonb)
        ORDER BY ipa
    """)

    with db.get_session() as session:
        result = session.execute(query, {"feature_json": feature_json_str}).fetchall()

    matching_phonemes = [row[0] for row in result]

    return PhonemeSearchResponse(
        features=request.features,
        matching_phonemes=matching_phonemes,
        count=len(matching_phonemes)
    )
