# Phoible English Phoneme Inventories

This directory contains extracted English phoneme inventories from the Phoible database.

## Source
- Full Phoible dataset: `../phoible.csv` (2,716 languages)
- English extracts: `phoible-english.csv` (407 entries including header)

## English Varieties Included

| Inventory ID | Language Name | Dialect/Region | Phoneme Count |
|--------------|---------------|----------------|---------------|
| 160 | English | General | 40 |
| 1561 | Liberian English | - | 36 |
| 2175 | English (American) | Western and Mid-Western US; Southern California | 39 |
| 2176 | American English | Southeastern Michigan | 39 |
| 2177 | English (Australian) | - | 44 |
| 2178 | English (British) | Liverpool | 40 |
| 2179 | English (New Zealand) | Pākehā | 45 |
| 2180 | English (British) | Tyneside English (Newcastle) | 39 |
| 2252 | English | English (RP) | 44 |
| 2515 | English | English (Liverpool) | 40 |

## Notes

- **Total English inventories**: 10
- **Phoneme count range**: 36-45 phonemes per variety
- **Most phonemes**: New Zealand English (45)
- **Fewest phonemes**: Liberian English (36)

## Key Varieties

- **RP (Received Pronunciation)**: Inventory 2252 - Traditional British prestige accent
- **General American**: Inventories 2175, 2176 - Standard American varieties
- **Regional British**: Liverpool (2178, 2515), Tyneside/Newcastle (2180)
- **Commonwealth**: Australian (2177), New Zealand (2179)
- **African**: Liberian (1561)

## Data Format

The CSV file contains one row per phoneme with features including:
- Basic identification (InventoryID, Language, Dialect)
- Phoneme representation and allophones
- Phonological features (consonantal, sonorant, continuant, etc.)
- Articulation features (labial, coronal, dorsal, etc.)
- Laryngeal features (voice, aspiration, glottalization)

