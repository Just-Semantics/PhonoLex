# ARPAbet to IPA Phoneme Mappings

This directory contains authoritative mappings between ARPAbet (used by the CMU Pronouncing Dictionary) and the International Phonetic Alphabet (IPA).

## Sources

These mappings are verified against:
- **Wikipedia ARPAbet article**: https://en.wikipedia.org/wiki/ARPABET
- **Wikipedia CMU Pronouncing Dictionary**: https://en.wikipedia.org/wiki/CMU_Pronouncing_Dictionary
- **Original CMU documentation**

## Files

- `arpa_to_ipa.json` - ARPAbet → IPA conversion (86 entries including stress variants)
- `ipa_to_arpa.json` - IPA → ARPAbet conversion (35 base phonemes)
- `phoneme_mappings.py` - Python module with conversion functions and documentation

## Coverage

### Vowels (14 phonemes)
| ARPAbet | IPA | Example | Word |
|---------|-----|---------|------|
| AA | ɑ | balm | "bɑm" |
| AE | æ | bat | "bæt" |
| AH | ʌ | butt | "bʌt" |
| AH0 | ə | comma | "kɑmə" |
| AO | ɔ | caught | "kɔt" |
| AW | aʊ | bout | "baʊt" |
| AY | aɪ | bite | "baɪt" |
| EH | ɛ | bet | "bɛt" |
| ER | ɝ | bird | "bɝd" |
| ER0 | ɚ | letter | "lɛtɚ" |
| EY | eɪ | bait | "beɪt" |
| IH | ɪ | bit | "bɪt" |
| IY | i | beat | "bit" |
| OW | oʊ | boat | "boʊt" |
| OY | ɔɪ | boy | "bɔɪ" |
| UH | ʊ | book | "bʊk" |
| UW | u | boot | "but" |

### Consonants (24 phonemes)
| ARPAbet | IPA | Example | Word |
|---------|-----|---------|------|
| B | b | buy | "baɪ" |
| CH | tʃ | China | "tʃaɪnə" |
| D | d | die | "daɪ" |
| DH | ð | thy | "ðaɪ" |
| F | f | fight | "faɪt" |
| G | g | guy | "gaɪ" |
| HH | h | high | "haɪ" |
| JH | dʒ | jive | "dʒaɪv" |
| K | k | kite | "kaɪt" |
| L | l | lie | "laɪ" |
| M | m | my | "maɪ" |
| N | n | nigh | "naɪ" |
| NG | ŋ | sing | "sɪŋ" |
| P | p | pie | "paɪ" |
| R | ɹ | rye | "ɹaɪ" |
| S | s | sigh | "saɪ" |
| SH | ʃ | shy | "ʃaɪ" |
| T | t | tie | "taɪ" |
| TH | θ | thigh | "θaɪ" |
| V | v | vie | "vaɪ" |
| W | w | wise | "waɪz" |
| Y | j | yacht | "jɑt" |
| Z | z | zoo | "zu" |
| ZH | ʒ | pleasure | "plɛʒɚ" |

### Stress Markers

CMU dict uses numeric stress markers on vowels:
- `0` = no stress (unmarked in IPA or use ə for reduced vowels)
- `1` = primary stress (IPA: ˈ before syllable)
- `2` = secondary stress (IPA: ˌ before syllable)

**Example**: 
- CMU: `P R OW0 N AH2 N S IY0 EY1 SH AH0 N`
- IPA: `pɹoʊˌnʌnsiˈeɪʃən`
- Word: "pronunciation"

## Special Cases

### AH phoneme
- `AH` (stressed) → `ʌ` (as in "cut")
- `AH0` (unstressed) → `ə` (schwa, as in "comma")

### ER phoneme (r-colored vowels)
- `ER` (stressed) → `ɝ` (as in "bird")
- `ER0` (unstressed) → `ɚ` (as in "letter")

### R sound
- CMU uses `R` for the English approximant
- IPA uses `ɹ` (not `r` which is a trill)

## Usage

```python
import json

# Load mappings
with open('arpa_to_ipa.json') as f:
    arpa_to_ipa = json.load(f)

# Convert ARPAbet to IPA
cmu_word = ["HH", "EH1", "L", "OW0"]
ipa_word = "".join(arpa_to_ipa.get(p, p) for p in cmu_word)
# Result: "ˈhɛloʊ"
```

## Notes

- These mappings represent **General American English** phonology
- Regional dialects may have different phoneme inventories (see [../phoible/english/](../phoible/english/))
- The CMU dict does not distinguish certain vowel qualities that vary regionally (e.g., cot-caught merger)

