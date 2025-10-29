"""
API router for word queries.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from database import DatabaseService
from models import Word

router = APIRouter(prefix="/api/words", tags=["words"])

# Dependency injection for database service
def get_db():
    """Get database service instance"""
    return DatabaseService()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class WordResponse(BaseModel):
    """Response schema for a single word"""
    word_id: int
    word: str
    ipa: str
    phonemes: List[Dict]
    syllables: List[Dict]
    phoneme_count: int
    syllable_count: int
    wcm_score: Optional[int]
    word_length: Optional[str]
    complexity: Optional[str]
    # Psycholinguistic properties
    frequency: Optional[float]
    log_frequency: Optional[float]
    aoa: Optional[float]
    imageability: Optional[float]
    familiarity: Optional[float]
    concreteness: Optional[float]
    valence: Optional[float]
    arousal: Optional[float]
    dominance: Optional[float]

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, word: Word):
        """Convert SQLAlchemy Word to Pydantic model"""
        return cls(
            word_id=word.word_id,
            word=word.word,
            ipa=word.ipa,
            phonemes=word.phonemes_json,
            syllables=word.syllables_json,
            phoneme_count=word.phoneme_count,
            syllable_count=word.syllable_count,
            wcm_score=word.wcm_score,
            word_length=word.word_length,
            complexity=word.complexity,
            frequency=word.frequency,
            log_frequency=word.log_frequency,
            aoa=word.aoa,
            imageability=word.imageability,
            familiarity=word.familiarity,
            concreteness=word.concreteness,
            valence=word.valence,
            arousal=word.arousal,
            dominance=word.dominance
        )


class WordFilterRequest(BaseModel):
    """Request schema for word filtering"""
    min_syllables: Optional[int] = None
    max_syllables: Optional[int] = None
    min_phonemes: Optional[int] = None
    max_phonemes: Optional[int] = None
    word_length: Optional[str] = None  # 'short', 'medium', 'long'
    complexity: Optional[str] = None  # 'low', 'medium', 'high'
    limit: int = 500
    offset: int = 0


class PatternSearchRequest(BaseModel):
    """Request schema for phoneme pattern search"""
    starts_with: Optional[str] = None
    ends_with: Optional[str] = None
    contains: Optional[str] = None
    contains_medial_only: bool = False  # If True, "contains" excludes initial/final positions
    word_length: Optional[str] = None
    complexity: Optional[str] = None
    min_syllables: Optional[int] = None
    max_syllables: Optional[int] = None
    limit: int = 500


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/{word}", response_model=WordResponse)
async def get_word(word: str, db: DatabaseService = Depends(get_db)):
    """
    Get word by string.

    Args:
        word: Word string

    Returns:
        Word information with phonological properties
    """
    word_obj = db.get_word_by_word(word)

    if not word_obj:
        raise HTTPException(status_code=404, detail=f"Word '{word}' not found")

    return WordResponse.from_orm(word_obj)


@router.post("/filter", response_model=List[WordResponse])
async def filter_words(request: WordFilterRequest, db: DatabaseService = Depends(get_db)):
    """
    Filter words by properties.

    Args:
        request: Filter criteria

    Returns:
        List of matching words
    """
    words = db.get_words_by_filters(
        min_syllables=request.min_syllables,
        max_syllables=request.max_syllables,
        min_phonemes=request.min_phonemes,
        max_phonemes=request.max_phonemes,
        word_length=request.word_length,
        complexity=request.complexity,
        limit=request.limit,
        offset=request.offset
    )

    return [WordResponse.from_orm(w) for w in words]


@router.post("/pattern-search", response_model=List[WordResponse])
async def pattern_search(request: PatternSearchRequest, db: DatabaseService = Depends(get_db)):
    """
    Search words by phoneme patterns.

    Supports:
    - starts_with: Word must start with this phoneme (IPA)
    - ends_with: Word must end with this phoneme (IPA)
    - contains: Word must contain this phoneme (IPA)
    - contains_medial_only: If True, "contains" excludes initial/final positions (medial only)

    Args:
        request: Pattern search criteria

    Returns:
        List of matching words
    """
    filters = {}
    if request.word_length:
        filters['word_length'] = request.word_length
    if request.complexity:
        filters['complexity'] = request.complexity
    if request.min_syllables:
        filters['min_syllables'] = request.min_syllables
    if request.max_syllables:
        filters['max_syllables'] = request.max_syllables

    words = db.get_words_by_phoneme_pattern(
        starts_with=request.starts_with,
        ends_with=request.ends_with,
        contains=request.contains,
        contains_medial_only=request.contains_medial_only,
        filters=filters if filters else None,
        limit=request.limit
    )

    return [WordResponse.from_orm(w) for w in words]


@router.get("/stats/summary")
async def get_stats(db: DatabaseService = Depends(get_db)):
    """
    Get database statistics.

    Returns:
        Statistics about words, phonemes, and edges
    """
    return db.get_stats()


@router.get("/stats/property-ranges")
async def get_property_ranges(db: DatabaseService = Depends(get_db)):
    """
    Get min/max ranges for all numeric properties in the database.

    Returns:
        Dict with min/max values for syllables, phonemes, WCM, MSH, frequency, AoA, etc.
    """
    from sqlalchemy import text, func
    from models import Word

    with db.get_session() as session:
        # Get min/max for each property
        result = session.query(
            func.min(Word.syllable_count).label('min_syllables'),
            func.max(Word.syllable_count).label('max_syllables'),
            func.min(Word.phoneme_count).label('min_phonemes'),
            func.max(Word.phoneme_count).label('max_phonemes'),
            func.min(Word.wcm_score).label('min_wcm'),
            func.max(Word.wcm_score).label('max_wcm'),
            func.min(Word.msh_stage).label('min_msh'),
            func.max(Word.msh_stage).label('max_msh'),
            func.min(Word.frequency).label('min_frequency'),
            func.max(Word.frequency).label('max_frequency'),
            func.min(Word.aoa).label('min_aoa'),
            func.max(Word.aoa).label('max_aoa'),
            func.min(Word.imageability).label('min_imageability'),
            func.max(Word.imageability).label('max_imageability'),
            func.min(Word.familiarity).label('min_familiarity'),
            func.max(Word.familiarity).label('max_familiarity'),
            func.min(Word.concreteness).label('min_concreteness'),
            func.max(Word.concreteness).label('max_concreteness'),
            func.min(Word.valence).label('min_valence'),
            func.max(Word.valence).label('max_valence'),
            func.min(Word.arousal).label('min_arousal'),
            func.max(Word.arousal).label('max_arousal'),
            func.min(Word.dominance).label('min_dominance'),
            func.max(Word.dominance).label('max_dominance'),
        ).one()

        return {
            'syllables': [result.min_syllables or 1, result.max_syllables or 5],
            'phonemes': [result.min_phonemes or 1, result.max_phonemes or 10],
            'wcm': [result.min_wcm or 0, result.max_wcm or 15],
            'msh': [result.min_msh or 1, result.max_msh or 6],
            'frequency': [result.min_frequency or 0, result.max_frequency or 1000],
            'aoa': [result.min_aoa or 2, result.max_aoa or 10],
            'imageability': [result.min_imageability or 1, result.max_imageability or 7],
            'familiarity': [result.min_familiarity or 1, result.max_familiarity or 7],
            'concreteness': [result.min_concreteness or 1, result.max_concreteness or 5],
            'valence': [result.min_valence or 1, result.max_valence or 9],
            'arousal': [result.min_arousal or 1, result.max_arousal or 9],
            'dominance': [result.min_dominance or 1, result.max_dominance or 9],
        }


@router.get("/phonemes/unique")
async def get_unique_phonemes(db: DatabaseService = Depends(get_db)):
    """
    Get list of all unique phonemes used in the dataset.

    Returns:
        List of unique IPA phonemes with counts
    """
    from sqlalchemy import text

    query = text("""
        WITH phoneme_list AS (
            SELECT jsonb_array_elements(phonemes_json)->>'ipa' as ipa
            FROM words
        )
        SELECT ipa, COUNT(*) as word_count
        FROM phoneme_list
        GROUP BY ipa
        ORDER BY word_count DESC
    """)

    with db.get_session() as session:
        result = session.execute(query).fetchall()

    return {
        "phonemes": [{"ipa": row[0], "word_count": row[1]} for row in result],
        "total_unique": len(result)
    }
