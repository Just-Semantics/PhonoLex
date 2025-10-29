from typing import Dict, List, Optional, Union, Literal, Set, ClassVar, TypeVar, Type, Generic, Tuple, Any, Callable, Annotated
from pydantic import BaseModel, Field, validator, root_validator, field_validator, model_validator
import numpy as np
from enum import Enum, auto
import json
from datetime import datetime


class DialectCode(str, Enum):
    """Enum for dialect codes using ISO language codes"""
    US_GENERAL = "en_US"
    UK_RP = "en_GB_RP"  # Corrected to standard ISO code
    UK_GENERAL = "en_GB"  # Corrected to standard ISO code
    AUS_GENERAL = "en_AU"
    CAN_GENERAL = "en_CA"
    

class PhoneticContext(str, Enum):
    """Enum for describing phonetic contexts"""
    WORD_INITIAL = "initial"
    WORD_FINAL = "final"
    INTERVOCALIC = "intervocalic"
    CONSONANT_CLUSTER = "cluster"
    STRESSED_SYLLABLE = "stressed"
    UNSTRESSED_SYLLABLE = "unstressed"


class FeatureSystem(str, Enum):
    """Enum for different phonological feature systems"""
    PHOIBLE = "phoible"
    HAYES = "hayes"
    SPE = "spe"  # Sound Pattern of English
    

class PhonemeFeatureVector(BaseModel):
    """Representation of a phoneme's feature vector based on PHOIBLE or other systems"""
    phoneme: str = Field(description="IPA representation of the phoneme")
    arpa_equivalent: Optional[str] = Field(None, description="ARPAbet equivalent if available")
    features: Dict[str, Union[float, bool, None]] = Field(
        description="Feature dictionary (from selected feature system)"
    )
    vector: List[float] = Field(
        description="Normalized vector representation of features"
    )
    feature_system: FeatureSystem = Field(
        FeatureSystem.PHOIBLE, description="The feature system used"
    )
    
    model_config = {
        "arbitrary_types_allowed": True
    }
        
    @field_validator('vector')
    @classmethod
    def check_vector_normalized(cls, v):
        """Validate that the vector is normalized"""
        if not v:
            return v
        norm = np.linalg.norm(v)
        if not np.isclose(norm, 1.0, rtol=1e-3) and norm != 0:
            raise ValueError("Feature vector must be normalized (unit length)")
        return v
    
    def similarity(self, other: 'PhonemeFeatureVector') -> float:
        """Calculate cosine similarity with another phoneme"""
        if len(self.vector) != len(other.vector):
            raise ValueError("Vectors must have the same dimensions")
        
        if not self.vector or not other.vector:
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    @property
    def is_vowel(self) -> bool:
        """Return True if the phoneme is a vowel"""
        return self.features.get('syllabic', False) is True
    
    @property
    def is_consonant(self) -> bool:
        """Return True if the phoneme is a consonant"""
        return self.features.get('consonantal', False) is True


class Syllable(BaseModel):
    """Representation of a syllable with constituent phonemes"""
    phonemes: List[str] = Field(description="List of phonemes in the syllable")
    stress: Optional[int] = Field(None, description="Stress level (0=unstressed, 1=primary, 2=secondary)")
    onset: List[str] = Field(default_factory=list, description="Onset phonemes")
    nucleus: List[str] = Field(default_factory=list, description="Nucleus phonemes")
    coda: List[str] = Field(default_factory=list, description="Coda phonemes")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @field_validator('stress')
    @classmethod
    def check_stress_level(cls, v):
        """Validate that stress level is valid"""
        if v is not None and v not in [0, 1, 2]:
            raise ValueError("Stress must be 0 (unstressed), 1 (primary), or 2 (secondary)")
        return v
    
    @model_validator(mode='after')
    def check_syllable_structure(self):
        """Validate syllable structure if onset/nucleus/coda are provided"""
        if self.onset or self.nucleus or self.coda:
            reconstructed = self.onset + self.nucleus + self.coda
            if self.phonemes != reconstructed:
                self.phonemes = reconstructed
        
        return self
    
    @property
    def is_stressed(self) -> bool:
        """Return True if the syllable has stress level 1 or 2"""
        return self.stress in [1, 2]
    
    @property
    def is_open(self) -> bool:
        """Return True if the syllable is open (no coda)"""
        return len(self.coda) == 0
    
    @property
    def is_heavy(self) -> bool:
        """Return True if the syllable is heavy (has coda or long vowel)"""
        # Simplified - would need more sophisticated analysis in practice
        return len(self.coda) > 0 or len(self.nucleus) > 1


class Pronunciation(BaseModel):
    """Representation of a word's pronunciation with detailed phonological properties"""
    phonemes: List[str] = Field(description="IPA representation of the pronunciation")
    arpa: Optional[List[str]] = Field(None, description="ARPAbet representation if available")
    dialect: DialectCode = Field(DialectCode.US_GENERAL, description="Dialect of this pronunciation")
    syllables: List[Syllable] = Field(default_factory=list, description="Syllable breakdown")
    phoneme_vectors: Optional[List[List[float]]] = Field(
        None, description="Vector representation of each phoneme"
    )
    word_vector: Optional[List[float]] = Field(
        None, description="Aggregated vector representation of the word"
    )
    rhyme_vector: Optional[List[float]] = Field(
        None, description="Vector representation of the rhyming portion"
    )
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @property
    def syllable_count(self) -> int:
        """Return the number of syllables"""
        if self.syllables:
            return len(self.syllables)
        # Fallback: count syllabic segments
        # This is a simplification - would need more sophisticated syllabification
        vowel_count = sum(1 for p in self.phonemes if p in ["a", "e", "i", "o", "u", 
                                                           "æ", "ɑ", "ɔ", "ɛ", "ɪ", 
                                                           "ɯ", "ʊ", "ʌ", "ə", "ɚ"])
        return max(1, vowel_count)  # Ensure at least one syllable
    
    @property
    def stress_pattern(self) -> List[int]:
        """Return the stress pattern as a list of integers"""
        if not self.syllables:
            return []
        return [s.stress if s.stress is not None else 0 for s in self.syllables]
    
    @property
    def rhyme(self) -> List[str]:
        """Return the rhyming portion (last stressed vowel to end)"""
        if not self.phonemes:
            return []
        
        # If syllables are available, use that
        if self.syllables:
            last_stress_idx = -1
            for i, syl in enumerate(self.syllables):
                if syl.is_stressed:
                    last_stress_idx = i
            
            # If no stressed syllable, return the last syllable
            if last_stress_idx == -1:
                return self.syllables[-1].phonemes
            
            # Otherwise return all phonemes from last stressed syllable
            result = []
            for i in range(last_stress_idx, len(self.syllables)):
                result.extend(self.syllables[i].phonemes)
            return result
        
        # Fallback if no syllables: just return last 3 phonemes
        return self.phonemes[-3:]
    
    def similarity(self, other: 'Pronunciation') -> float:
        """Calculate phonological similarity with another pronunciation"""
        # If word vectors are available, use them
        if self.word_vector and other.word_vector:
            # Simple cosine similarity
            dot_product = sum(a * b for a, b in zip(self.word_vector, other.word_vector))
            norm_a = np.linalg.norm(self.word_vector)
            norm_b = np.linalg.norm(other.word_vector)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        
        # Fallback: Levenshtein distance on phonemes (normalized)
        # This is simplified - would use more sophisticated phonological similarity
        max_len = max(len(self.phonemes), len(other.phonemes))
        if max_len == 0:
            return 1.0
        
        # Simple Levenshtein distance implementation
        rows = len(self.phonemes) + 1
        cols = len(other.phonemes) + 1
        distance = [[0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(1, rows):
            distance[i][0] = i
        for j in range(1, cols):
            distance[0][j] = j
            
        for i in range(1, rows):
            for j in range(1, cols):
                if self.phonemes[i-1] == other.phonemes[j-1]:
                    cost = 0
                else:
                    cost = 1
                distance[i][j] = min(
                    distance[i-1][j] + 1,      # deletion
                    distance[i][j-1] + 1,      # insertion
                    distance[i-1][j-1] + cost  # substitution
                )
        
        # Convert to similarity (0-1)
        return 1.0 - (distance[rows-1][cols-1] / max_len)
    
    def rhyme_similarity(self, other: 'Pronunciation') -> float:
        """Calculate rhyming similarity with another pronunciation"""
        # If rhyme vectors are available, use them
        if self.rhyme_vector and other.rhyme_vector:
            # Simple cosine similarity
            dot_product = sum(a * b for a, b in zip(self.rhyme_vector, other.rhyme_vector))
            norm_a = np.linalg.norm(self.rhyme_vector)
            norm_b = np.linalg.norm(other.rhyme_vector)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        
        # Fallback: compare rhyming portions
        self_rhyme = self.rhyme
        other_rhyme = other.rhyme
        
        if not self_rhyme or not other_rhyme:
            return 0.0
        
        # Simple matching coefficient
        min_len = min(len(self_rhyme), len(other_rhyme))
        matches = sum(1 for i in range(min_len) if self_rhyme[-i-1] == other_rhyme[-i-1])
        
        return matches / min_len


class PartOfSpeech(str, Enum):
    """Part of speech enumeration using Universal Dependencies tagset"""
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"  # Adjective
    ADV = "ADV"  # Adverb
    ADP = "ADP"  # Adposition (preposition or postposition)
    AUX = "AUX"  # Auxiliary verb
    CCONJ = "CCONJ"  # Coordinating conjunction
    DET = "DET"  # Determiner
    INTJ = "INTJ"  # Interjection
    NUM = "NUM"  # Numeral
    PART = "PART"  # Particle
    PRON = "PRON"  # Pronoun
    PROPN = "PROPN"  # Proper noun
    PUNCT = "PUNCT"  # Punctuation
    SCONJ = "SCONJ"  # Subordinating conjunction
    SYM = "SYM"  # Symbol
    X = "X"  # Other


class Word(BaseModel):
    """Representation of a word with its phonological and lexical properties"""
    orthography: str = Field(description="Written form of the word")
    pronunciations: Dict[DialectCode, Pronunciation] = Field(
        default_factory=dict, description="Pronunciations by dialect"
    )
    frequency: Optional[float] = Field(
        None, description="Word frequency (normalized 0-1)"
    )
    pos: Optional[PartOfSpeech] = Field(
        None, description="Part of speech"
    )
    syllable_count: Optional[int] = Field(None, description="Number of syllables")
    lemma: Optional[str] = Field(None, description="Base form (for inflected words)")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @field_validator('frequency')
    @classmethod
    def check_frequency(cls, v):
        """Validate that frequency is between 0 and 1"""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Frequency must be normalized between 0 and 1")
        return v
    
    @property
    def default_pronunciation(self) -> Optional[Pronunciation]:
        """Return the default pronunciation (US_GENERAL if available, otherwise first in dict)"""
        if not self.pronunciations:
            return None
        
        if DialectCode.US_GENERAL in self.pronunciations:
            return self.pronunciations[DialectCode.US_GENERAL]
        
        # Fallback to the first pronunciation
        return next(iter(self.pronunciations.values()))
    
    @property
    def computed_syllable_count(self) -> Optional[int]:
        """Return the syllable count based on pronunciations if not explicitly set"""
        if self.syllable_count is not None:
            return self.syllable_count
        
        pron = self.default_pronunciation
        if pron:
            return pron.syllable_count
        
        return None
    
    def pronunciation_similarity(self, other: 'Word') -> float:
        """Calculate pronunciation similarity with another word"""
        self_pron = self.default_pronunciation
        other_pron = other.default_pronunciation
        
        if not self_pron or not other_pron:
            return 0.0
        
        return self_pron.similarity(other_pron)
    
    def rhymes_with(self, other: 'Word', threshold: float = 0.7) -> bool:
        """Check if this word rhymes with another word"""
        self_pron = self.default_pronunciation
        other_pron = other.default_pronunciation
        
        if not self_pron or not other_pron:
            return False
        
        return self_pron.rhyme_similarity(other_pron) >= threshold


class WordCollection(BaseModel):
    """A collection of words with a specific purpose or category"""
    name: str = Field(description="Name of the collection")
    description: Optional[str] = Field(None, description="Description of the collection")
    words: List[str] = Field(description="List of words in the collection")
    source: Optional[str] = Field(None, description="Source of the words (e.g., dictionary name)")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @field_validator('words')
    @classmethod
    def check_unique_words(cls, v):
        """Validate that all words are unique"""
        if len(v) != len(set(v)):
            # Find duplicates for better error message
            seen = set()
            duplicates = [w for w in v if w in seen or seen.add(w)]
            raise ValueError(f"Word collection contains duplicates: {duplicates}")
        return v
    
    def filter_by_phonological_feature(self, feature_name: str, feature_value: Any) -> List[str]:
        """
        Filter words by a phonological feature
        
        Note: This method would require access to a WordRepository or similar to work fully
        """
        # Placeholder - would need to be implemented based on actual data access pattern
        return []
    
    def to_json(self) -> str:
        """Serialize the collection to JSON"""
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "words": self.words,
            "source": self.source,
            "created_at": self.created_at.isoformat()
        })


class PhonologicalPattern(BaseModel):
    """Representation of a phonological pattern for matching"""
    name: Optional[str] = Field(None, description="Optional name for the pattern")
    description: Optional[str] = Field(None, description="Description of what this pattern matches")
    pattern_type: Literal["contains", "starts_with", "ends_with"] = Field(
        "contains", description="Type of pattern matching"
    )
    phoneme_patterns: List[Dict[str, Union[str, float, List[float], None]]] = Field(
        description="List of feature specifications to match"
    )
    similarity_threshold: float = Field(
        0.8, description="Threshold for vector similarity matches (0-1)"
    )
    contexts: List[PhoneticContext] = Field(
        default_factory=list, description="Phonetic contexts where this pattern applies"
    )
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @field_validator('similarity_threshold')
    @classmethod
    def check_threshold(cls, v):
        """Validate that threshold is between 0 and 1"""
        if v < 0 or v > 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v
    
    def match_word(self, word: Word) -> bool:
        """Check if the word matches this pattern"""
        # This would need to be implemented based on your matching logic
        # Placeholder implementation
        return False
    
    def match_phonemes(self, phonemes: List[str]) -> bool:
        """Check if a phoneme sequence matches this pattern"""
        # This would need to be implemented based on your matching logic
        # Placeholder implementation
        return False


class PhonologicalAnalysis(BaseModel):
    """Results of phonological analysis on a word or collection"""
    target: str = Field(description="Word or collection analyzed")
    features: Dict[str, Union[int, float, str, bool, List]] = Field(
        description="Extracted phonological features"
    )
    vector_representation: Optional[List[float]] = Field(
        None, description="Vector representation if applicable"
    )
    patterns_found: List[str] = Field(
        default_factory=list, description="Names of patterns found"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Computed metrics"
    ) 
    
    model_config = {
        "arbitrary_types_allowed": True
    } 