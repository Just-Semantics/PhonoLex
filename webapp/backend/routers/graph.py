"""
API router for graph operations and export.
"""

from fastapi import APIRouter, HTTPException, Depends, Response
from pydantic import BaseModel
from typing import List, Optional, Dict
import gzip
import json
from database import DatabaseService
from routers.words import WordResponse

router = APIRouter(prefix="/api/graph", tags=["graph"])


# Dependency injection
def get_db():
    """Get database service instance"""
    return DatabaseService()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class EdgeResponse(BaseModel):
    """Single edge response"""
    word1: str
    word2: str
    relation_type: str
    metadata: Dict
    weight: float


class NeighborResult(BaseModel):
    """Result for neighbor query"""
    neighbor: WordResponse
    edge: EdgeResponse


class MinimalPairResult(BaseModel):
    """Result for minimal pair query"""
    word1: WordResponse
    word2: WordResponse
    position: int
    phoneme1: str
    phoneme2: str
    feature_diff: Optional[int]


class RhymeResult(BaseModel):
    """Result for rhyme query"""
    rhyme: WordResponse
    rhyme_type: str
    nucleus: str
    coda: List[str]
    quality: float


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/neighbors/{word}", response_model=List[NeighborResult])
async def get_neighbors(
    word: str,
    relation_type: Optional[str] = None,
    limit: int = 100,
    db: DatabaseService = Depends(get_db)
):
    """
    Get neighboring words in the phonological graph.

    Args:
        word: Source word
        relation_type: Optional filter by edge type (MINIMAL_PAIR, RHYME, NEIGHBOR, etc.)
        limit: Max results

    Returns:
        List of neighbors with edge information
    """
    neighbors = db.get_word_neighbors(
        word=word,
        relation_type=relation_type,
        limit=limit
    )

    if not neighbors:
        # Check if word exists
        word_obj = db.get_word_by_word(word)
        if not word_obj:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found")
        return []

    results = []
    for neighbor_word, edge in neighbors:
        # Determine which word is word1 and word2 for correct labeling
        if edge.word1_id == neighbor_word.word_id:
            w1, w2 = neighbor_word.word, word
        else:
            w1, w2 = word, neighbor_word.word

        results.append(
            NeighborResult(
                neighbor=WordResponse.from_orm(neighbor_word),
                edge=EdgeResponse(
                    word1=w1,
                    word2=w2,
                    relation_type=edge.relation_type,
                    metadata=edge.edge_metadata,
                    weight=edge.weight
                )
            )
        )

    return results


@router.get("/minimal-pairs", response_model=List[MinimalPairResult])
async def get_minimal_pairs(
    phoneme1: str,
    phoneme2: str,
    min_syllables: Optional[int] = None,
    max_syllables: Optional[int] = None,
    min_wcm: Optional[int] = None,
    max_wcm: Optional[int] = None,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
    limit: int = 50,
    db: DatabaseService = Depends(get_db)
):
    """
    Get minimal pairs for specific phoneme contrast.

    Args:
        phoneme1: First phoneme (IPA)
        phoneme2: Second phoneme (IPA)
        min_syllables: Minimum syllable count
        max_syllables: Maximum syllable count
        min_wcm: Minimum WCM score
        max_wcm: Maximum WCM score
        min_frequency: Minimum frequency
        max_frequency: Maximum frequency
        limit: Max results

    Returns:
        List of minimal pairs with metadata
    """
    filters = {}
    if min_syllables is not None:
        filters['min_syllables'] = min_syllables
    if max_syllables is not None:
        filters['max_syllables'] = max_syllables
    if min_wcm is not None:
        filters['min_wcm'] = min_wcm
    if max_wcm is not None:
        filters['max_wcm'] = max_wcm
    if min_frequency is not None:
        filters['min_frequency'] = min_frequency
    if max_frequency is not None:
        filters['max_frequency'] = max_frequency

    pairs = db.get_minimal_pairs(
        phoneme1=phoneme1,
        phoneme2=phoneme2,
        filters=filters if filters else None,
        limit=limit
    )

    return [
        MinimalPairResult(
            word1=WordResponse.from_orm(w1),
            word2=WordResponse.from_orm(w2),
            position=metadata.get('position', 0),
            phoneme1=metadata.get('phoneme1', phoneme1),
            phoneme2=metadata.get('phoneme2', phoneme2),
            feature_diff=metadata.get('feature_diff')
        )
        for w1, w2, metadata in pairs
    ]


@router.get("/rhymes/{word}", response_model=List[RhymeResult])
async def get_rhymes(
    word: str,
    rhyme_mode: Optional[str] = 'last_1',
    use_embeddings: Optional[bool] = False,
    word_length: Optional[str] = None,
    complexity: Optional[str] = None,
    limit: int = 50,
    db: DatabaseService = Depends(get_db)
):
    """
    Get rhyming words for a given word.

    Args:
        word: Source word
        rhyme_mode: Rhyme matching mode ('last_1', 'last_2', 'last_3', 'assonance', 'consonance')
        use_embeddings: Include near-matches using embeddings (quality < 1.0)
        word_length: Optional filter ('short', 'medium', 'long')
        complexity: Optional filter ('low', 'medium', 'high')
        limit: Max results

    Returns:
        List of rhyming words with rhyme metadata and quality scores
    """
    # Validate rhyme_mode
    valid_modes = ['last_1', 'last_2', 'last_3', 'assonance', 'consonance']
    if rhyme_mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rhyme_mode '{rhyme_mode}'. Must be one of: {valid_modes}"
        )

    filters = {}
    if word_length:
        filters['word_length'] = word_length
    if complexity:
        filters['complexity'] = complexity

    rhymes = db.get_rhymes(
        word=word,
        rhyme_mode=rhyme_mode,
        use_embeddings=use_embeddings,
        filters=filters if filters else None,
        limit=limit
    )

    if not rhymes:
        # Check if word exists
        word_obj = db.get_word_by_word(word)
        if not word_obj:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found")
        return []

    return [
        RhymeResult(
            rhyme=WordResponse.from_orm(rhyme_word),
            rhyme_type=metadata.get('rhyme_type', 'unknown'),
            nucleus=metadata.get('nucleus', ''),
            coda=metadata.get('coda', []),
            quality=metadata.get('quality', 1.0)
        )
        for rhyme_word, metadata in rhymes
    ]


@router.get("/export")
async def export_graph(
    include_embeddings: bool = False,
    db: DatabaseService = Depends(get_db)
):
    """
    Export full graph data for client caching.

    This endpoint returns the entire phonological graph (nodes + edges) as compressed JSON.
    Clients can cache this data locally for instant queries.

    Args:
        include_embeddings: Whether to include 384-dim embedding vectors (increases size significantly)

    Returns:
        Gzipped JSON containing:
        - nodes: All words with properties
        - edges: All typed edges
        - phonemes: All phonemes with features
        - version: Data version
        - stats: Database statistics
    """
    print("Exporting graph data...")

    # Export data from database
    graph_data = db.export_graph_data(include_embeddings=include_embeddings)

    # Convert to JSON
    json_data = json.dumps(graph_data, separators=(',', ':'))  # Compact JSON

    # Compress with gzip
    compressed = gzip.compress(json_data.encode('utf-8'), compresslevel=6)

    print(f"✓ Exported {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    print(f"  Size: {len(json_data):,} bytes (uncompressed), {len(compressed):,} bytes (compressed)")
    print(f"  Compression ratio: {len(json_data) / len(compressed):.1f}x")

    # Return compressed response
    return Response(
        content=compressed,
        media_type='application/gzip',
        headers={
            'Content-Encoding': 'gzip',
            'Content-Disposition': 'attachment; filename="phonolex_graph.json.gz"',
            'X-Uncompressed-Size': str(len(json_data)),
            'X-Compressed-Size': str(len(compressed)),
            'X-Node-Count': str(len(graph_data['nodes'])),
            'X-Edge-Count': str(len(graph_data['edges']))
        }
    )
