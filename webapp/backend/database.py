"""
Database service layer for PhonoLex v2.0.

Provides high-level interface for database operations using SQLAlchemy.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker, Session, aliased, defer
from sqlalchemy.pool import NullPool
import numpy as np

from models import Base, Word, Phoneme, WordEdge, Syllable, WordSyllable

# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/phonolex")


class DatabaseService:
    """
    Database service for PhonoLex.

    Handles all database operations including:
    - Vector similarity search (pgvector)
    - Pattern matching (JSONB queries)
    - Graph queries (typed edges)
    - Word/phoneme lookups
    """

    def __init__(self, database_url: str = DATABASE_URL):
        """
        Initialize database service.

        Args:
            database_url: PostgreSQL connection string
        """
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL debugging
        )

        # Create session factory
        # expire_on_commit=False prevents DetachedInstanceError when returning ORM objects
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine, expire_on_commit=False)

        # Verify connection
        self._verify_connection()

    def _verify_connection(self):
        """Verify database connection and pgvector extension"""
        with self.get_session() as session:
            # Check pgvector extension
            result = session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
            if not result.fetchone():
                raise RuntimeError("pgvector extension not installed")

    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session (context manager).

        Usage:
            with db.get_session() as session:
                words = session.query(Word).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ========================================================================
    # Word Queries
    # ========================================================================

    def get_word_by_id(self, word_id: int) -> Optional[Word]:
        """Get word by ID"""
        with self.get_session() as session:
            return session.query(Word).filter(Word.word_id == word_id).first()

    def get_word_by_word(self, word: str) -> Optional[Word]:
        """Get word by string"""
        with self.get_session() as session:
            return session.query(Word).filter(Word.word == word).first()

    def get_words_by_filters(
        self,
        min_syllables: Optional[int] = None,
        max_syllables: Optional[int] = None,
        min_phonemes: Optional[int] = None,
        max_phonemes: Optional[int] = None,
        word_length: Optional[str] = None,
        complexity: Optional[str] = None,
        limit: int = 500,
        offset: int = 0
    ) -> List[Word]:
        """
        Get words by property filters.

        Args:
            min_syllables: Minimum syllable count
            max_syllables: Maximum syllable count
            min_phonemes: Minimum phoneme count
            max_phonemes: Maximum phoneme count
            word_length: 'short', 'medium', 'long'
            complexity: 'low', 'medium', 'high'
            limit: Max results
            offset: Pagination offset

        Returns:
            List of Word objects
        """
        with self.get_session() as session:
            query = session.query(Word)

            if min_syllables is not None:
                query = query.filter(Word.syllable_count >= min_syllables)
            if max_syllables is not None:
                query = query.filter(Word.syllable_count <= max_syllables)
            if min_phonemes is not None:
                query = query.filter(Word.phoneme_count >= min_phonemes)
            if max_phonemes is not None:
                query = query.filter(Word.phoneme_count <= max_phonemes)
            if word_length:
                query = query.filter(Word.word_length == word_length)
            if complexity:
                query = query.filter(Word.complexity == complexity)

            return query.limit(limit).offset(offset).all()

    def get_words_by_phoneme_pattern(
        self,
        starts_with: Optional[str] = None,
        ends_with: Optional[str] = None,
        contains: Optional[str] = None,
        contains_medial_only: bool = False,
        filters: Optional[Dict] = None,
        limit: int = 500
    ) -> List[Word]:
        """
        Get words matching phoneme patterns using JSONB queries.

        Args:
            starts_with: IPA phoneme word must start with
            ends_with: IPA phoneme word must end with
            contains: IPA phoneme word must contain
            contains_medial_only: If True, "contains" excludes initial/final positions
            filters: Additional property filters
            limit: Max results

        Returns:
            List of Word objects
        """
        with self.get_session() as session:
            query = session.query(Word)

            # Pattern matching using JSONB operators
            if starts_with:
                # First phoneme: phonemes_json[0]->>'ipa' = starts_with
                query = query.filter(
                    text(f"phonemes_json->0->>'ipa' = '{starts_with}'")
                )

            if ends_with:
                # Last phoneme: phonemes_json[-1]->>'ipa' = ends_with
                # Note: PostgreSQL uses negative indexing in JSONB
                query = query.filter(
                    text(f"phonemes_json->-1->>'ipa' = '{ends_with}'")
                )

            if contains:
                if contains_medial_only:
                    # Medial only: contains the phoneme but NOT in first or last position
                    # Word must have at least 3 phonemes and contain phoneme in middle positions
                    query = query.filter(
                        text(f"""
                            jsonb_array_length(phonemes_json) >= 3
                            AND EXISTS (
                                SELECT 1
                                FROM jsonb_array_elements(phonemes_json) WITH ORDINALITY AS elem(val, idx)
                                WHERE elem.val->>'ipa' = '{contains}'
                                AND idx > 1
                                AND idx < jsonb_array_length(phonemes_json)
                            )
                        """)
                    )
                else:
                    # Anywhere: Contains phoneme in any position
                    query = query.filter(
                        text(f"phonemes_json @> '[{{\"ipa\": \"{contains}\"}}]'")
                    )

            # Apply additional filters
            if filters:
                if filters.get('word_length'):
                    query = query.filter(Word.word_length == filters['word_length'])
                if filters.get('complexity'):
                    query = query.filter(Word.complexity == filters['complexity'])
                if filters.get('min_syllables'):
                    query = query.filter(Word.syllable_count >= filters['min_syllables'])
                if filters.get('max_syllables'):
                    query = query.filter(Word.syllable_count <= filters['max_syllables'])

            return query.limit(limit).all()

    # ========================================================================
    # Vector Similarity Search (pgvector)
    # ========================================================================

    def find_similar_words_by_embedding(
        self,
        word: str,
        threshold: float = 0.85,
        limit: int = 50
    ) -> List[Tuple[Word, float]]:
        """
        Find words similar to given word using syllable embeddings.

        Uses pgvector cosine similarity with HNSW index for fast ANN search.

        Args:
            word: Target word
            threshold: Minimum similarity (0-1)
            limit: Max results

        Returns:
            List of (Word, similarity) tuples, sorted by similarity (highest first)
        """
        with self.get_session() as session:
            # Get target word embedding
            target_word = session.query(Word).filter(Word.word == word).first()
            if not target_word or target_word.syllable_embedding is None:
                return []

            # pgvector cosine similarity query
            # <=> is the cosine distance operator (lower = more similar)
            # similarity = 1 - distance
            query = session.query(
                Word,
                (1 - Word.syllable_embedding.cosine_distance(target_word.syllable_embedding)).label('similarity')
            ).filter(
                Word.word_id != target_word.word_id,  # Exclude self
                Word.syllable_embedding.isnot(None),
                (1 - Word.syllable_embedding.cosine_distance(target_word.syllable_embedding)) > threshold
            ).order_by(
                Word.syllable_embedding.cosine_distance(target_word.syllable_embedding)
            ).limit(limit)

            results = query.all()
            return [(word, float(sim)) for word, sim in results]

    def find_similar_words_by_vector(
        self,
        embedding: np.ndarray,
        threshold: float = 0.85,
        limit: int = 50,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Word, float]]:
        """
        Find words similar to given embedding vector.

        Args:
            embedding: 384-dim embedding vector
            threshold: Minimum similarity
            limit: Max results
            filters: Optional property filters (wcm_score, complexity, etc.)

        Returns:
            List of (Word, similarity) tuples
        """
        with self.get_session() as session:
            # Convert numpy array to list for pgvector
            embedding_list = embedding.tolist()

            query = session.query(
                Word,
                (1 - Word.syllable_embedding.cosine_distance(embedding_list)).label('similarity')
            ).filter(
                Word.syllable_embedding.isnot(None),
                (1 - Word.syllable_embedding.cosine_distance(embedding_list)) > threshold
            )

            # Apply filters
            if filters:
                if filters.get('max_wcm'):
                    query = query.filter(Word.wcm_score <= filters['max_wcm'])
                if filters.get('complexity'):
                    query = query.filter(Word.complexity == filters['complexity'])

            query = query.order_by(
                Word.syllable_embedding.cosine_distance(embedding_list)
            ).limit(limit)

            results = query.all()
            return [(word, float(sim)) for word, sim in results]

    def compute_word_similarity(
        self,
        word1: str,
        word2: str
    ) -> float:
        """
        Compute word-to-word similarity using soft Levenshtein distance on syllable sequences.

        Uses the syllable_embeddings array column to get exact embeddings from Layer 4 checkpoint,
        then computes hierarchical soft Levenshtein similarity as described in
        docs/EMBEDDINGS_ARCHITECTURE.md.

        Args:
            word1: First word
            word2: Second word

        Returns:
            float: Similarity score [0.0, 1.0]
        """
        with self.get_session() as session:
            # Use raw SQL to avoid SQLAlchemy ARRAY(Vector) parsing issues
            from sqlalchemy import text as sql_text

            query = sql_text("""
                SELECT word, syllable_embeddings::text
                FROM words
                WHERE word IN (:word1, :word2)
            """)

            result = session.execute(query, {'word1': word1, 'word2': word2}).fetchall()

            if len(result) != 2:
                return 0.0

            # Parse the embeddings manually
            embeddings_dict = {}
            for word, emb_str in result:
                if not emb_str or emb_str == '{}':
                    return 0.0

                # Parse PostgreSQL array format: {"[...]","[...]",...}
                # Remove outer braces and quotes
                emb_str = emb_str.strip('{}')
                if not emb_str:
                    return 0.0

                # Split by "," (quoted comma) to get individual vectors
                vectors_str = []
                in_vector = False
                current_vector = ""

                for char in emb_str:
                    if char == '"':
                        in_vector = not in_vector
                        continue
                    if in_vector:
                        current_vector += char
                    elif char == ',' and current_vector:
                        vectors_str.append(current_vector.strip())
                        current_vector = ""

                if current_vector:
                    vectors_str.append(current_vector.strip())

                # Parse each vector: [val1,val2,...]
                syllable_embs = []
                for vec_str in vectors_str:
                    vec_str = vec_str.strip('[]')
                    values = [float(v) for v in vec_str.split(',')]
                    syllable_embs.append(np.array(values, dtype=np.float32))

                embeddings_dict[word] = np.array(syllable_embs)

            if word1 not in embeddings_dict or word2 not in embeddings_dict:
                return 0.0

            # Compute soft Levenshtein similarity
            return self._soft_levenshtein_similarity(
                embeddings_dict[word1],
                embeddings_dict[word2]
            )

    def _soft_levenshtein_similarity(
        self,
        syllables1: np.ndarray,
        syllables2: np.ndarray
    ) -> float:
        """
        Compute soft Levenshtein similarity between two syllable sequences.

        Implementation from scripts/build_layer4_syllable_embeddings.py (lines 138-189).
        Uses edit distance minimization with soft costs based on syllable similarity.

        Args:
            syllables1: [len1, 384] syllable embeddings (normalized)
            syllables2: [len2, 384] syllable embeddings (normalized)

        Returns:
            float: Similarity score [0.0, 1.0] where 1.0 = identical
        """
        len1, len2 = len(syllables1), len(syllables2)

        # Pre-compute all pairwise similarities (vectorized)
        # Syllables are already normalized in database, so dot product = cosine similarity
        sim_matrix = syllables1 @ syllables2.T  # [len1, len2], values in [-1, 1]

        # Dynamic programming for edit distance with soft costs
        dp = np.zeros((len1 + 1, len2 + 1))

        # Initialize: cost of inserting/deleting syllables
        for i in range(len1 + 1):
            dp[i][0] = i * 1.0  # Deletion cost
        for j in range(len2 + 1):
            dp[0][j] = j * 1.0  # Insertion cost

        # Fill DP table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                # Match/substitute cost: 1 - similarity (0 if identical, 2 if opposite)
                match_cost = 1.0 - sim_matrix[i-1, j-1]

                dp[i][j] = min(
                    dp[i-1][j] + 1.0,           # Delete from s1
                    dp[i][j-1] + 1.0,           # Insert from s2
                    dp[i-1][j-1] + match_cost   # Match/substitute
                )

        # Normalize to [0, 1] similarity
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0

        edit_distance = dp[len1][len2]
        similarity = 1.0 - (edit_distance / max_len)
        return float(max(0.0, min(1.0, similarity)))

    # ========================================================================
    # Graph Queries (Typed Edges)
    # ========================================================================

    def get_word_neighbors(
        self,
        word: str,
        relation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Tuple[Word, WordEdge]]:
        """
        Get neighboring words in the phonological graph.

        Args:
            word: Source word
            relation_type: Filter by edge type (MINIMAL_PAIR, RHYME, etc.)
            limit: Max results

        Returns:
            List of (neighbor_word, edge) tuples
        """
        with self.get_session() as session:
            source_word = session.query(Word).filter(Word.word == word).first()
            if not source_word:
                return []

            # Query bidirectional edges
            query = session.query(Word, WordEdge).join(
                WordEdge,
                ((WordEdge.word1_id == source_word.word_id) & (WordEdge.word2_id == Word.word_id)) |
                ((WordEdge.word2_id == source_word.word_id) & (WordEdge.word1_id == Word.word_id))
            ).filter(
                Word.word_id != source_word.word_id
            )

            if relation_type:
                query = query.filter(WordEdge.relation_type == relation_type)

            return query.limit(limit).all()

    def get_minimal_pairs(
        self,
        phoneme1: str,
        phoneme2: str,
        filters: Optional[Dict] = None,
        limit: int = 50
    ) -> List[Tuple[Word, Word, Dict]]:
        """
        Get minimal pairs for specific phoneme contrast.

        Args:
            phoneme1: First phoneme (IPA)
            phoneme2: Second phoneme (IPA)
            filters: Optional property filters
            limit: Max results

        Returns:
            List of (word1, word2, metadata) tuples
        """
        with self.get_session() as session:
            # Create aliases for the two Word instances
            # Use defer() to exclude syllable_embeddings column (not needed for minimal pairs)
            Word1 = aliased(Word)
            Word2 = aliased(Word)

            query = session.query(Word1, Word2, WordEdge.edge_metadata).options(
                defer(Word1.syllable_embeddings),
                defer(Word2.syllable_embeddings)
            ).join(
                WordEdge,
                WordEdge.word1_id == Word1.word_id
            ).join(
                Word2,
                WordEdge.word2_id == Word2.word_id
            ).filter(
                WordEdge.relation_type == 'MINIMAL_PAIR',
                text(f"edge_metadata->>'phoneme1' = '{phoneme1}' AND edge_metadata->>'phoneme2' = '{phoneme2}'")
            )

            # Apply filters
            if filters:
                # Numeric range filters
                if filters.get('min_syllables') is not None:
                    query = query.filter(Word1.syllable_count >= filters['min_syllables'])
                if filters.get('max_syllables') is not None:
                    query = query.filter(Word1.syllable_count <= filters['max_syllables'])
                if filters.get('min_wcm') is not None:
                    query = query.filter(Word1.wcm_score >= filters['min_wcm'])
                if filters.get('max_wcm') is not None:
                    query = query.filter(Word1.wcm_score <= filters['max_wcm'])
                if filters.get('min_frequency') is not None:
                    query = query.filter(Word1.frequency >= filters['min_frequency'])
                if filters.get('max_frequency') is not None:
                    query = query.filter(Word1.frequency <= filters['max_frequency'])

            results = query.limit(limit).all()
            return [(w1, w2, dict(metadata)) for w1, w2, metadata in results]

    def get_rhymes(
        self,
        word: str,
        rhyme_mode: str = 'last_1',
        use_embeddings: bool = False,
        filters: Optional[Dict] = None,
        limit: int = 50
    ) -> List[Tuple[Word, Dict]]:
        """
        Get rhyming words for a given word with configurable matching strategy.

        Args:
            word: Source word
            rhyme_mode: Rhyme matching mode:
                - 'last_1': Match last syllable (traditional rhyme)
                - 'last_2': Match last 2 syllables
                - 'last_3': Match last 3 syllables
                - 'assonance': Match nucleus (vowel) only
                - 'consonance': Match coda (final consonants) only
            use_embeddings: If False, return only perfect matches (quality=1.0).
                           If True, include near-matches using embeddings (quality<1.0).
            filters: Optional property filters (word_length, complexity)
            limit: Max results

        Returns:
            List of (rhyming_word, metadata) tuples with quality scores
        """
        with self.get_session() as session:
            source_word = session.query(Word).filter(Word.word == word).first()
            if not source_word:
                return []

            # Get source word syllables
            source_syllables = source_word.syllables_json
            if not source_syllables:
                return []

            # Get phoneme-based exact matches (quality = 1.0)
            exact_matches = self._get_rhymes_by_phonemes(
                source_word=source_word,
                source_syllables=source_syllables,
                rhyme_mode=rhyme_mode,
                filters=filters,
                limit=limit,
                session=session
            )

            if not use_embeddings:
                # Return only perfect matches
                return exact_matches

            # If use_embeddings=True, also include near-matches
            # Get embedding-based approximate matches (quality < 1.0)
            near_matches = self._get_rhymes_by_embeddings(
                source_word=source_word,
                rhyme_mode=rhyme_mode,
                filters=filters,
                limit=limit * 2,  # Get more candidates
                session=session
            )

            # Combine exact and near matches, removing duplicates
            exact_word_ids = {w.word_id for w, _ in exact_matches}
            combined = exact_matches + [
                (w, meta) for w, meta in near_matches
                if w.word_id not in exact_word_ids
            ]

            # Sort by quality (exact matches first, then by similarity)
            combined.sort(key=lambda x: x[1]['quality'], reverse=True)

            return combined[:limit]

    def _get_rhymes_by_phonemes(
        self,
        source_word: Word,
        source_syllables: List[Dict],
        rhyme_mode: str,
        filters: Optional[Dict],
        limit: int,
        session: Session
    ) -> List[Tuple[Word, Dict]]:
        """
        Get rhymes using exact phoneme matching.

        Args:
            source_word: Source word object
            source_syllables: Source word syllables_json
            rhyme_mode: Rhyme matching mode
            filters: Optional filters
            limit: Max results
            session: Database session

        Returns:
            List of (rhyming_word, metadata) tuples
        """
        # Build JSONB query based on rhyme mode
        query = session.query(Word).filter(
            Word.word_id != source_word.word_id,
            Word.syllables_json.isnot(None)
        )

        # Apply filters first to reduce search space
        if filters:
            if filters.get('word_length'):
                query = query.filter(Word.word_length == filters['word_length'])
            if filters.get('complexity'):
                query = query.filter(Word.complexity == filters['complexity'])

        if rhyme_mode == 'last_1':
            # Match last syllable (nucleus + coda)
            last_syl = source_syllables[-1]
            nucleus = last_syl.get('nucleus', '')
            coda = last_syl.get('coda', [])

            # JSONB query: last syllable nucleus and coda must match
            query = query.filter(
                text(f"""
                    syllables_json->-1->>'nucleus' = '{nucleus}'
                    AND syllables_json->-1->'coda' = '{json.dumps(coda)}'::jsonb
                """)
            )

        elif rhyme_mode == 'last_2':
            # Match last 2 syllables
            if len(source_syllables) >= 2:
                last_2 = source_syllables[-2:]
                query = query.filter(
                    text(f"""
                        jsonb_array_length(syllables_json) >= 2
                        AND syllables_json->-2 = '{json.dumps(last_2[0])}'::jsonb
                        AND syllables_json->-1 = '{json.dumps(last_2[1])}'::jsonb
                    """)
                )
            else:
                # If source has < 2 syllables, fall back to last_1
                return self._get_rhymes_by_phonemes(
                    source_word, source_syllables, 'last_1', filters, limit, session
                )

        elif rhyme_mode == 'last_3':
            # Match last 3 syllables
            if len(source_syllables) >= 3:
                last_3 = source_syllables[-3:]
                query = query.filter(
                    text(f"""
                        jsonb_array_length(syllables_json) >= 3
                        AND syllables_json->-3 = '{json.dumps(last_3[0])}'::jsonb
                        AND syllables_json->-2 = '{json.dumps(last_3[1])}'::jsonb
                        AND syllables_json->-1 = '{json.dumps(last_3[2])}'::jsonb
                    """)
                )
            else:
                # Fall back to matching all available syllables
                if len(source_syllables) == 2:
                    return self._get_rhymes_by_phonemes(
                        source_word, source_syllables, 'last_2', filters, limit, session
                    )
                else:
                    return self._get_rhymes_by_phonemes(
                        source_word, source_syllables, 'last_1', filters, limit, session
                    )

        elif rhyme_mode == 'assonance':
            # Match nucleus (vowel) only from last syllable
            last_syl = source_syllables[-1]
            nucleus = last_syl.get('nucleus', '')

            query = query.filter(
                text(f"syllables_json->-1->>'nucleus' = '{nucleus}'")
            )

        elif rhyme_mode == 'consonance':
            # Match coda (final consonants) only from last syllable
            last_syl = source_syllables[-1]
            coda = last_syl.get('coda', [])

            query = query.filter(
                text(f"syllables_json->-1->'coda' = '{json.dumps(coda)}'::jsonb")
            )

        # Execute query
        candidates = query.limit(limit * 2).all()  # Get more candidates for ranking

        # Build results with metadata
        results = []
        for candidate in candidates[:limit]:
            metadata = {
                'rhyme_type': rhyme_mode,
                'nucleus': source_syllables[-1].get('nucleus', ''),
                'coda': source_syllables[-1].get('coda', []),
                'quality': 1.0,  # Perfect match for phoneme-based
                'match_method': 'phoneme'
            }
            results.append((candidate, metadata))

        return results

    def _get_rhymes_by_embeddings(
        self,
        source_word: Word,
        rhyme_mode: str,
        filters: Optional[Dict],
        limit: int,
        session: Session
    ) -> List[Tuple[Word, Dict]]:
        """
        Get rhymes using syllable embedding similarity (near-rhyme).

        This method finds near-matches that RESPECT the rhyme_mode constraint.
        For example, if rhyme_mode='last_2', it will only return words where the
        last 2 syllables are similar (not just overall phonologically similar).

        Strategy:
        1. Use phoneme-based matching to constrain candidates (with relaxed matching)
        2. Then rank by embedding similarity to find best near-matches

        Args:
            source_word: Source word object
            rhyme_mode: Rhyme matching mode
            filters: Optional filters
            limit: Max results
            session: Database session

        Returns:
            List of (rhyming_word, metadata) tuples with quality < 1.0
        """
        if source_word.syllable_embedding is None:
            return []

        source_syllables = source_word.syllables_json or []
        if not source_syllables:
            return []

        # Build base query with syllable embedding similarity
        threshold = 0.70
        query = session.query(
            Word,
            (1 - Word.syllable_embedding.cosine_distance(source_word.syllable_embedding)).label('similarity')
        ).filter(
            Word.word_id != source_word.word_id,
            Word.syllables_json.isnot(None),
            Word.syllable_embedding.isnot(None),
            (1 - Word.syllable_embedding.cosine_distance(source_word.syllable_embedding)) > threshold
        )

        # Apply rhyme_mode constraint using relaxed phoneme matching
        # For near-matches, we match at least the nucleus (vowel), allowing coda variation
        if rhyme_mode == 'last_1':
            # For near-rhyme last_1: match nucleus of last syllable (allow coda variation)
            last_syl = source_syllables[-1]
            nucleus = last_syl.get('nucleus', '')
            query = query.filter(
                text(f"syllables_json->-1->>'nucleus' = '{nucleus}'")
            )

        elif rhyme_mode == 'last_2':
            # For near-rhyme last_2: match nuclei of last 2 syllables (allow coda variation)
            if len(source_syllables) >= 2:
                nucleus_2 = source_syllables[-2].get('nucleus', '')
                nucleus_1 = source_syllables[-1].get('nucleus', '')
                query = query.filter(
                    text(f"""
                        jsonb_array_length(syllables_json) >= 2
                        AND syllables_json->-2->>'nucleus' = '{nucleus_2}'
                        AND syllables_json->-1->>'nucleus' = '{nucleus_1}'
                    """)
                )
            else:
                # Fall back to last_1 if source has < 2 syllables
                return self._get_rhymes_by_embeddings(
                    source_word, 'last_1', filters, limit, session
                )

        elif rhyme_mode == 'last_3':
            # For near-rhyme last_3: match nuclei of last 3 syllables
            if len(source_syllables) >= 3:
                nucleus_3 = source_syllables[-3].get('nucleus', '')
                nucleus_2 = source_syllables[-2].get('nucleus', '')
                nucleus_1 = source_syllables[-1].get('nucleus', '')
                query = query.filter(
                    text(f"""
                        jsonb_array_length(syllables_json) >= 3
                        AND syllables_json->-3->>'nucleus' = '{nucleus_3}'
                        AND syllables_json->-2->>'nucleus' = '{nucleus_2}'
                        AND syllables_json->-1->>'nucleus' = '{nucleus_1}'
                    """)
                )
            else:
                # Fall back appropriately
                if len(source_syllables) == 2:
                    return self._get_rhymes_by_embeddings(
                        source_word, 'last_2', filters, limit, session
                    )
                else:
                    return self._get_rhymes_by_embeddings(
                        source_word, 'last_1', filters, limit, session
                    )

        elif rhyme_mode == 'assonance':
            # Assonance: match nucleus only (already relaxed)
            last_syl = source_syllables[-1]
            nucleus = last_syl.get('nucleus', '')
            query = query.filter(
                text(f"syllables_json->-1->>'nucleus' = '{nucleus}'")
            )

        elif rhyme_mode == 'consonance':
            # Consonance: match coda only
            # For near-match consonance, we can relax by allowing similar codas
            # For now, keep exact coda match but use embedding similarity to rank
            last_syl = source_syllables[-1]
            coda = last_syl.get('coda', [])
            query = query.filter(
                text(f"syllables_json->-1->'coda' = '{json.dumps(coda)}'::jsonb")
            )

        # Apply filters
        if filters:
            if filters.get('word_length'):
                query = query.filter(Word.word_length == filters['word_length'])
            if filters.get('complexity'):
                query = query.filter(Word.complexity == filters['complexity'])

        # Order by similarity
        query = query.order_by(
            Word.syllable_embedding.cosine_distance(source_word.syllable_embedding)
        ).limit(limit)

        candidates = query.all()

        # Build results with metadata
        results = []
        for candidate, similarity in candidates:
            metadata = {
                'rhyme_type': f'{rhyme_mode}_near',
                'nucleus': source_syllables[-1].get('nucleus', '') if source_syllables else '',
                'coda': source_syllables[-1].get('coda', []) if source_syllables else [],
                'quality': float(similarity),
                'match_method': 'embedding'
            }
            results.append((candidate, metadata))

        return results

    # ========================================================================
    # Phoneme Queries
    # ========================================================================

    def get_phoneme_by_ipa(self, ipa: str) -> Optional[Phoneme]:
        """Get phoneme by IPA symbol"""
        with self.get_session() as session:
            return session.query(Phoneme).filter(Phoneme.ipa == ipa).first()

    def get_phonemes_by_class(self, segment_class: str) -> List[Phoneme]:
        """Get all phonemes of a class (consonant/vowel)"""
        with self.get_session() as session:
            return session.query(Phoneme).filter(Phoneme.segment_class == segment_class).all()

    def get_phonemes_by_features(self, features: Dict[str, str]) -> List[Phoneme]:
        """
        Get phonemes matching feature specification.

        Args:
            features: Dict of feature names to values, e.g. {"nasal": "+", "labial": "+"}

        Returns:
            List of Phoneme objects
        """
        with self.get_session() as session:
            query = session.query(Phoneme)

            for feature, value in features.items():
                query = query.filter(
                    text(f"features->>'{feature}' = '{value}'")
                )

            return query.all()

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_session() as session:
            word_count = session.query(func.count(Word.word_id)).scalar()
            phoneme_count = session.query(func.count(Phoneme.phoneme_id)).scalar()
            edge_count = session.query(func.count(WordEdge.edge_id)).scalar()

            # Edge type breakdown
            edge_types = session.query(
                WordEdge.relation_type,
                func.count(WordEdge.edge_id)
            ).group_by(WordEdge.relation_type).all()

            return {
                'total_words': word_count,
                'total_phonemes': phoneme_count,
                'total_edges': edge_count,
                'edge_types': {rel_type: count for rel_type, count in edge_types}
            }

    # ========================================================================
    # Graph Export (for client caching)
    # ========================================================================

    def export_graph_data(self, include_embeddings: bool = False) -> Dict:
        """
        Export full graph data for client caching.

        Args:
            include_embeddings: Whether to include embedding vectors (increases size)

        Returns:
            Dict with nodes, edges, and phonemes
        """
        with self.get_session() as session:
            # Export words (nodes)
            words = session.query(Word).all()
            nodes_data = []
            for word in words:
                node = {
                    'word_id': word.word_id,
                    'word': word.word,
                    'ipa': word.ipa,
                    'phonemes': word.phonemes_json,
                    'syllables': word.syllables_json,
                    'phoneme_count': word.phoneme_count,
                    'syllable_count': word.syllable_count,
                    'wcm_score': word.wcm_score,
                    'word_length': word.word_length,
                    'complexity': word.complexity,
                    'frequency': word.frequency,
                    'aoa': word.aoa,
                }

                if include_embeddings and word.syllable_embedding:
                    node['syllable_embedding'] = word.syllable_embedding.tolist()

                nodes_data.append(node)

            # Export edges
            edges = session.query(WordEdge).all()
            edges_data = []
            for edge in edges:
                edges_data.append({
                    'word1_id': edge.word1_id,
                    'word2_id': edge.word2_id,
                    'relation_type': edge.relation_type,
                    'metadata': edge.edge_metadata,
                    'weight': edge.weight
                })

            # Export phonemes
            phonemes = session.query(Phoneme).all()
            phonemes_data = []
            for phoneme in phonemes:
                phonemes_data.append({
                    'phoneme_id': phoneme.phoneme_id,
                    'ipa': phoneme.ipa,
                    'segment_class': phoneme.segment_class,
                    'features': phoneme.features
                })

            return {
                'nodes': nodes_data,
                'edges': edges_data,
                'phonemes': phonemes_data,
                'version': '2.0.0',
                'stats': self.get_stats()
            }
