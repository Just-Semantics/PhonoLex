# PhonoLex

A phonological lexicon with vector-based representation for phoneme features

## Phoneme Vector Representation

The PhonoLex system represents phonemes as vectors in a high-dimensional feature space. Each dimension corresponds to a phonological feature (e.g., "voiced", "nasal", "front"). This vector representation enables:

1. **Phonetic similarity measurement** - Similar phonemes have similar vectors
2. **Rhyme detection** - Matching final syllable patterns
3. **Phonological pattern identification** - Finding words with similar sound structures
4. **Statistical analysis** - Calculating average pronunciations for word groups

### Feature Vector Creation

Phoneme vectors are created from feature specifications as follows:

1. **Simple feature values**:
   - `+` (presence of feature) → 1.0
   - `-` (absence of feature) → 0.0
   - Undefined features → 0.0 (no artificial similarity)

2. **Complex/fuzzy feature values**:
   - Mixed values (e.g., `+,-`) → Proportion of positive values (0.5 for equal mix)
   - Vector is normalized to magnitude 1.0 for consistent comparison

3. **Composite phonemes** (e.g., diphthongs):
   - Created by combining component phoneme vectors
   - Primary components weighted more heavily (0.7 vs 0.3)
   - Result is normalized to ensure consistent magnitude

This approach ensures that phonological similarity is accurately captured while maintaining mathematical properties needed for vector operations.

### Example

The phoneme 'p' differs from 'b' primarily in voicing, so their vectors are very similar except in the "voiced" dimension. Both are quite different from 'm', which shares place of articulation but differs in manner.

## Installation and Setup

PhonoLex is available as a Python package. You can install it using [uv](https://github.com/astral-sh/uv):

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from the current directory
uv pip install -e .
```

## Data Processing Pipeline

PhonoLex includes a complete data processing pipeline to build a comprehensive phonological dataset:

1. **Extract English phoneme features** from PHOIBLE
2. **Add missing phonemes** not present in PHOIBLE
3. **Process CMU dictionary** to create pronunciation data
4. **Filter for WordNet words** to get a high-quality lexicon
5. **Create comprehensive dataset** with phonological and semantic information
6. **Filter phoneme data** to include only supported English phonemes

### Running the Pipeline

All processing scripts are included in the package. You can run the entire pipeline with:

```bash
python -m phonolex.pipeline.run
```

This will create all the processed data files in the configured output directory.

## Package Structure

PhonoLex is organized as a Python package with the following components:

- **Core models**: Pydantic models for phonological data structures
- **Data processing**: Pipeline components for processing phonological data
- **Rule builder**: Create and apply phonological rules to words and patterns
- **API**: FastAPI-based web API for accessing phonological data
- **Utils**: Utility functions for vector operations and data manipulation

## Using the Rules System

PhonoLex provides a flexible rule system for defining and applying phonological rules. The rule system is built on top of [seqrule](https://github.com/neumanns-workshop/seqrule), a library for defining rules on sequences of objects.

### Basic Rule Usage

```python
from seqrule import AbstractObject, check_sequence
from phonolex.rules import create_feature_match_rule, create_natural_class_rule

# Create phonemes with features
phonemes = [
    AbstractObject(
        ipa="p", 
        consonantal=True, 
        sonorant=False, 
        continuant=False, 
        voice=False
    ),
    AbstractObject(
        ipa="a", 
        syllabic=True, 
        consonantal=False, 
        sonorant=True
    ),
    AbstractObject(
        ipa="t", 
        consonantal=True, 
        sonorant=False, 
        continuant=False, 
        voice=False
    )
]

# Create a rule for vowels
vowel_rule = create_feature_match_rule({"syllabic": True})

# Check if the vowel in the sequence satisfies the rule
vowels_only = [p for p in phonemes if p["syllabic"]]
result = check_sequence(vowels_only, vowel_rule)
print(f"Vowels only? {result}")  # True

# Create a rule for stops
stop_rule = create_natural_class_rule("stop")

# Check if the consonants in the sequence are stops
consonants = [p for p in phonemes if p["consonantal"]]
result = check_sequence(consonants, stop_rule)
print(f"Stops only? {result}")  # True
```

### Complex Rules

The rule system supports complex rules with environments and contexts:

```python
from phonolex.rules import create_environment_rule

# Create a rule for intervocalic obstruents
intervocalic_obstruents = create_environment_rule(
    {"consonantal": True, "sonorant": False},  # Target
    {"syllabic": True},                        # Left context
    {"syllabic": True}                         # Right context
)

# Check if the sequence contains an intervocalic obstruent
result = check_sequence(phonemes, intervocalic_obstruents)
print(f"Contains intervocalic obstruents? {result}")  # True
```

### Available Rule Types

PhonoLex provides the following rule types:

- **Feature Match**: Check if phonemes match specific features
- **Natural Class**: Check if phonemes belong to a natural class
- **Environment**: Check if phonemes appear in specific environments
- **Assimilation**: Check for feature assimilation between phonemes
- **Dissimilation**: Check for feature dissimilation between phonemes
- **Syllable Structure**: Check if syllables follow specific patterns
- **Sonority**: Check if phonemes follow sonority sequencing principle
- **Harmonization**: Check for vowel harmony in a sequence
- **Phonotactics**: Check if a sequence follows phonotactic constraints

## Running the Examples

PhonoLex includes example scripts that demonstrate how to use the package. You can run them using:

```bash
# Run the phonological rules example
python scripts/run_examples.py
```

## Web API

PhonoLex includes a FastAPI-based web API for accessing phonological data. To start the API server:

```bash
# Start the API server
python scripts/run_api.py
```

The API provides endpoints for:

- Word lookup with phonological information
- Phoneme similarity calculation
- Rule application to words
- Phonological pattern searching

You can access the API documentation at http://localhost:8000/docs after starting the server.

To test the API, run the example client:

```bash
# Run the API client example
python examples/api_client.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license. 