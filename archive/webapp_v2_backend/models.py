"""
SQLAlchemy models for PhonoLex v2.0 database.

Maps to the PostgreSQL schema with pgvector support.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey, CheckConstraint, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, deferred
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Word(Base):
    """
    Word table - core lexicon with phonological structure and embeddings.
    """
    __tablename__ = 'words'

    word_id = Column(Integer, primary_key=True)
    word = Column(String(100), unique=True, nullable=False, index=True)
    ipa = Column(Text, nullable=False)

    # Phonological structure (JSONB for fast queries)
    phonemes_json = Column(JSONB, nullable=False)
    syllables_json = Column(JSONB, nullable=False)

    # Counts
    phoneme_count = Column(Integer, nullable=False, index=True)
    syllable_count = Column(Integer, nullable=False, index=True)

    # Psycholinguistic properties
    frequency = Column(Float, index=True)
    log_frequency = Column(Float)
    aoa = Column(Float, index=True)  # Age of acquisition
    imageability = Column(Float)
    familiarity = Column(Float)
    concreteness = Column(Float)
    valence = Column(Float)
    arousal = Column(Float)
    dominance = Column(Float)

    # Clinical measures
    wcm_score = Column(Integer, index=True)  # Word Complexity Measure
    msh_stage = Column(Integer)  # Motor Speech Hierarchy

    # Embeddings (pgvector)
    # DEPRECATED: Old single-syllable embedding (first syllable only)
    syllable_embedding = deferred(Column(Vector(384)))  # For backwards compatibility, will be removed

    # NEW: Per-syllable embeddings (list of 384-dim vectors, one per syllable)
    # For soft Levenshtein word similarity
    # NOTE: Deferred due to pgvector ARRAY(Vector) parsing issues
    syllable_embeddings = deferred(Column(ARRAY(Vector(384))))  # Variable length array

    word_embedding_flat = deferred(Column(Vector(64)))  # 64-dim simple embedding (optional)

    # Categorical properties
    word_length = Column(String(10), index=True)  # 'short', 'medium', 'long'
    complexity = Column(String(10), index=True)  # 'low', 'medium', 'high'

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))

    # Relationships
    edges_as_word1 = relationship('WordEdge', foreign_keys='WordEdge.word1_id', back_populates='word1')
    edges_as_word2 = relationship('WordEdge', foreign_keys='WordEdge.word2_id', back_populates='word2')

    def __repr__(self):
        return f"<Word(word='{self.word}', ipa='{self.ipa}')>"


class Syllable(Base):
    """
    Syllable table - unique syllables with embeddings (lookup table).
    """
    __tablename__ = 'syllables'

    syllable_id = Column(Integer, primary_key=True)
    ipa = Column(Text, unique=True, nullable=False, index=True)

    # Syllable structure (JSONB)
    structure = Column(JSONB, nullable=False)  # {onset: [...], nucleus: "...", coda: [...]}

    # Phoneme counts
    onset_count = Column(Integer, nullable=False, default=0)
    coda_count = Column(Integer, nullable=False, default=0)
    total_phonemes = Column(Integer, nullable=False, default=1)

    # Embedding (384-dim: onset + nucleus + coda)
    embedding = Column(Vector(384), nullable=False)

    # Statistics
    frequency = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))

    def __repr__(self):
        return f"<Syllable(ipa='{self.ipa}', structure={self.structure})>"


class WordSyllable(Base):
    """
    Junction table mapping words to ordered syllables.
    """
    __tablename__ = 'word_syllables'

    word_id = Column(Integer, ForeignKey('words.word_id', ondelete='CASCADE'), primary_key=True)
    syllable_id = Column(Integer, ForeignKey('syllables.syllable_id', ondelete='CASCADE'), nullable=False)
    position = Column(Integer, primary_key=True)  # 0-indexed position

    # Relationships
    word = relationship('Word', backref='word_syllables')
    syllable = relationship('Syllable', backref='word_syllables')

    def __repr__(self):
        return f"<WordSyllable(word_id={self.word_id}, syllable_id={self.syllable_id}, position={self.position})>"


class Phoneme(Base):
    """
    Phoneme table - inventory with Phoible features and embeddings.
    """
    __tablename__ = 'phonemes'

    phoneme_id = Column(Integer, primary_key=True)
    ipa = Column(String(10), unique=True, nullable=False, index=True)
    segment_class = Column(String(20), nullable=False, index=True)  # consonant, vowel, tone

    # Phoible features (JSONB)
    features = Column(JSONB, nullable=False)

    # Vector representations (multiple granularities)
    raw_features = Column(Vector(38))  # Raw ternary features
    endpoints_76d = Column(Vector(76))  # Start + end states
    trajectories_152d = Column(Vector(152))  # 4 timesteps (diphthongs)
    contextual_128d = Column(Vector(128))  # Context-aware from model

    # Trajectory metadata
    has_trajectory = Column(Boolean, default=False)
    trajectory_features = Column(ARRAY(Text))

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))

    def __repr__(self):
        return f"<Phoneme(ipa='{self.ipa}', class='{self.segment_class}')>"


class WordEdge(Base):
    """
    Word edges table - typed graph relationships between words.
    """
    __tablename__ = 'word_edges'

    edge_id = Column(Integer, primary_key=True)
    word1_id = Column(Integer, ForeignKey('words.word_id', ondelete='CASCADE'), nullable=False, index=True)
    word2_id = Column(Integer, ForeignKey('words.word_id', ondelete='CASCADE'), nullable=False, index=True)

    # Edge type
    relation_type = Column(String(50), nullable=False, index=True)

    # Relationship-specific metadata (JSONB for flexibility)
    # Note: Can't use 'metadata' as it's reserved by SQLAlchemy
    edge_metadata = Column(JSONB, nullable=False, server_default='{}')

    # Precomputed weight for graph algorithms
    weight = Column(Float, default=1.0)

    # Timestamp
    created_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))

    # Relationships
    word1 = relationship('Word', foreign_keys=[word1_id], back_populates='edges_as_word1')
    word2 = relationship('Word', foreign_keys=[word2_id], back_populates='edges_as_word2')

    # Constraints
    __table_args__ = (
        CheckConstraint('word1_id != word2_id', name='no_self_loops'),
        CheckConstraint('word1_id < word2_id', name='ordered_pair'),
    )

    def __repr__(self):
        return f"<WordEdge(word1_id={self.word1_id}, word2_id={self.word2_id}, type='{self.relation_type}')>"
