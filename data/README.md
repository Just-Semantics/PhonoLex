# PhonoLex Data Repository

This directory contains curated phonological and lexical data for English.

## Directory Structure

```
data/
├── cmu/                    # CMU Pronouncing Dictionary
├── phoible/               # Phoible phoneme database
│   └── english/          # English-specific extracts
└── mappings/             # ARPAbet ↔ IPA mappings
```

## Datasets

### 1. CMU Pronouncing Dictionary ([cmu/](cmu/))
- **Source**: Carnegie Mellon University
- **Version**: 0.7b
- **Content**: ~134,000 English word pronunciations in ARPAbet
- **Format**: Plain text, one word per line
- **Encoding**: ARPAbet phonemes with stress markers (0, 1, 2)

**Example entry:**
```
HELLO  HH AH0 L OW1
```

### 2. Phoible Database ([phoible/](phoible/))
- **Source**: https://phoible.org/
- **Languages**: 2,716 languages worldwide
- **Content**: Phoneme inventories with distinctive features
- **Format**: CSV with phonological feature matrices

**English subset** ([phoible/english/](phoible/english/)):
- 10 English varieties (American, British, Australian, NZ, Liberian, etc.)
- 406 phoneme entries across all varieties
- Ranges from 36 phonemes (Liberian) to 45 phonemes (NZ)

### 3. ARPAbet-IPA Mappings ([mappings/](mappings/))
- **Content**: Bidirectional phoneme mappings
- **Coverage**: 39 ARPAbet phonemes (14 vowels + 24 consonants + stress)
- **Format**: JSON + Python module
- **Sources**: Verified against Wikipedia and CMU documentation

**Files:**
- `arpa_to_ipa.json` - 86 entries (includes stress variants)
- `ipa_to_arpa.json` - 35 base phoneme mappings
- `phoneme_mappings.py` - Conversion functions and utilities

## Quick Reference

### ARPAbet Vowels
```
AA ɑ    AE æ    AH ʌ/ə   AO ɔ    AW aʊ   AY aɪ   
EH ɛ    ER ɝ/ɚ  EY eɪ   IH ɪ    IY i    OW oʊ   
OY ɔɪ   UH ʊ    UW u
```

### ARPAbet Consonants
```
B b     CH tʃ   D d     DH ð    F f     G g     HH h    JH dʒ
K k     L l     M m     N n     NG ŋ    P p     R ɹ     S s
SH ʃ    T t     TH θ    V v     W w     Y j     Z z     ZH ʒ
```

## Use Cases

1. **Text-to-Speech**: Convert orthography → ARPAbet → IPA → speech synthesis
2. **Phonological Analysis**: Analyze English phonotactics and distributions
3. **Dialect Comparison**: Compare phoneme inventories across English varieties
4. **Linguistic Research**: Access feature-based phoneme representations
5. **Natural Language Processing**: Phoneme-level text processing

## Data Integrity

All data in this directory is:
- ✅ From authoritative sources (CMU, Phoible, Wikipedia)
- ✅ Version-controlled and documented
- ✅ Includes source attribution and licensing
- ✅ Verified against multiple references

## Archive

Previous project files have been moved to [../archive/](../archive/) for reference.

## Next Steps

To start a new PhonoLex project, you have:
- Raw pronunciation data (CMU dict)
- Phonological features (Phoible)
- Conversion utilities (ARPAbet ↔ IPA)
- Multiple English dialects for comparison

Ready to build! 🎯
