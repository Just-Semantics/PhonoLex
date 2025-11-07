# PHOIBLE Features Reference

Complete documentation of the 38 distinctive features used in PhonoLex, based on the PHOIBLE database (Moran & McCloy, 2019).

## Overview

**PHOIBLE** (PHOnetics Information Base and Lexicon) is a comprehensive database of phonological inventories from 2,716 languages worldwide. PhonoLex uses PHOIBLE's distinctive feature system to represent English phonemes.

**Feature system**: Hayes (2009) + Moisik & Esling (2011)
- **38 ternary features** (values: +, -, 0)
- **7 feature categories**: Major class, Laryngeal, Manner, Place (articulator), Place (tongue body), Place (detailed), Vowel-specific
- **39 English phonemes** (General American English)

**Feature values**:
- `+` : Feature is present
- `-` : Feature is absent
- `0` : Feature is not applicable to this phoneme class

**Coverage**: 105,484 phonemes across 2,716 languages in PHOIBLE database. PhonoLex uses 39 English phonemes.

---

## Feature Categories

### 1. Major Class Features (4 features)

These features define the broadest phoneme categories: consonants vs. vowels, obstruents vs. sonorants.

#### consonantal

**Definition**: Constriction in the oral cavity sufficient to impede airflow

**Values**:
- `+`: Obstruents, nasals, liquids (p, t, k, s, z, m, n, l, ɹ)
- `-`: Vowels, glides (i, u, w, j, a, e, o)

**Phonemes**:
- **consonantal:+** (22): p, b, t, d, k, g, f, v, θ, ð, s, z, ʃ, ʒ, tʃ, dʒ, m, n, ŋ, l, ɹ, h
- **consonantal:-** (17): All vowels (i, ɪ, e, ɛ, æ, ə, ʌ, a, ɑ, ɔ, o, u, ʊ) + glides (w, j)

**Key distinctions**:
- Consonants have oral constriction, vowels/glides do not
- This is NOT the same as "consonant vs. vowel" in orthography
- /w/ and /j/ are `consonantal:-` despite being written as consonants

---

#### syllabic

**Definition**: Can form the nucleus of a syllable

**Values**:
- `+`: Vowels (i, æ, u, a, e, o)
- `-`: Consonants (p, t, k, s, m, n, l)

**Phonemes**:
- **syllabic:+** (15): All vowels (i, ɪ, e, ɛ, æ, ə, ʌ, a, ɑ, ɔ, o, u, ʊ, aɪ, aʊ, ɔɪ, oʊ, eɪ)
- **syllabic:-** (24): All consonants + glides

**Key distinctions**:
- Syllabic sounds can be syllable nuclei (centers)
- In English, only vowels are syllabic
- Some languages allow syllabic consonants (e.g., syllabic /n/ in "button" in some dialects)

---

#### sonorant

**Definition**: Spontaneous voicing is possible (no turbulent airflow in oral cavity)

**Values**:
- `+`: Vowels, nasals, liquids, glides (i, m, n, l, ɹ, w, j)
- `-`: Obstruents (p, t, k, s, f, θ)

**Phonemes**:
- **sonorant:+** (21): All vowels + m, n, ŋ, l, ɹ, w, j, h
- **sonorant:-** (18): p, b, t, d, k, g, f, v, θ, ð, s, z, ʃ, ʒ, tʃ, dʒ (and h in some analyses)

**Key distinctions**:
- **MAJOR CLASS DIFFERENCE**: sonorant:+ vs. sonorant:- is the distinction used in maximal opposition
- Obstruents (sonorant:-) have turbulent airflow: stops, fricatives, affricates
- Sonorants (sonorant:+) have clear resonance: vowels, nasals, liquids, glides

**Clinical significance**: Maximal opposition pairs must have different sonorant values (e.g., /s/ [sonorant:-] vs. /l/ [sonorant:+]).

---

#### approximant

**Definition**: Close approximation of articulators without turbulence

**Values**:
- `+`: Glides, liquids (w, j, l, ɹ)
- `-`: Obstruents, vowels, nasals

**Phonemes**:
- **approximant:+** (4): w, j, l, ɹ
- **approximant:-** (35): All others

**Key distinctions**:
- Approximants have articulators close together but not touching
- No turbulent airflow (unlike fricatives)
- Not nasals (airflow through oral cavity only)

---

### 2. Laryngeal Features (11 features)

These features describe vocal fold configuration and laryngeal state.

#### voice

**Definition**: Vocal fold vibration during articulation

**Values**:
- `+`: Voiced sounds (b, d, g, z, v, m, n, all vowels)
- `-`: Voiceless sounds (p, t, k, s, f, θ)

**Phonemes**:
- **voice:+** (24): b, d, g, v, ð, z, ʒ, dʒ, m, n, ŋ, l, ɹ, w, j + all vowels
- **voice:-** (15): p, t, k, f, θ, s, ʃ, tʃ, h

**Key distinctions**:
- Voiced/voiceless pairs: p/b, t/d, k/g, f/v, θ/ð, s/z, ʃ/ʒ, tʃ/dʒ
- All vowels are voiced in English
- All sonorants (nasals, liquids, glides) are voiced

**Clinical significance**: Voicing errors are common (e.g., "pig" → "big", "t" → "d").

---

#### spreadGlottis

**Definition**: Glottis spread open for aspiration or breathy voice

**Values**:
- `+`: Aspirated stops (pʰ, tʰ, kʰ in word-initial position), /h/
- `-`: Non-aspirated sounds

**Phonemes**:
- **spreadGlottis:+** (4): p, t, k (in word-initial stressed position), h
- **spreadGlottis:-** (35): All others

**Key distinctions**:
- English voiceless stops are aspirated word-initially (pin = [pʰɪn])
- Not aspirated after /s/ (spin = [spɪn], not [spʰɪn])
- /h/ is always spreadGlottis:+

**Allophonic variation**: spreadGlottis value depends on position (initial vs. post-/s/).

---

#### constrictedGlottis

**Definition**: Glottis constricted (glottal stop, creaky voice)

**Values**:
- `+`: Glottalized sounds (ʔ)
- `-`: Most English sounds

**Phonemes**:
- **constrictedGlottis:+** (1): ʔ (glottal stop, not in standard inventory)
- **constrictedGlottis:-** (38): All others in standard English

**Note**: Glottal stop /ʔ/ occurs in some English dialects (e.g., "button" = [bʌʔən] in Cockney).

---

#### periodicGlottalSource

**Definition**: Periodic vibration of vocal folds (modal voice)

**Values**:
- `+`: Voiced sounds with modal phonation (b, d, g, all vowels, m, n, l)
- `-`: Voiceless sounds (p, t, k, s, f)

**Phonemes**:
- **periodicGlottalSource:+** (24): Same as voice:+
- **periodicGlottalSource:-** (15): Same as voice:-

**Key distinctions**:
- Highly correlated with `voice` feature
- Distinguishes modal voice from breathy/creaky voice
- All voiced sounds have periodic glottal source

---

#### stiffVocalFolds

**Definition**: Vocal folds stiffened (voiceless obstruents)

**Values**:
- `+`: Voiceless obstruents (p, t, k, s, f, θ)
- `-`: Voiced obstruents, sonorants
- `0`: Vowels, sonorants (not applicable)

**Phonemes**:
- **stiffVocalFolds:+** (9): p, t, k, f, θ, s, ʃ, tʃ, h
- **stiffVocalFolds:-** (9): b, d, g, v, ð, z, ʒ, dʒ
- **stiffVocalFolds:0** (21): All vowels + sonorant consonants

**Key distinctions**:
- Voiceless obstruents have stiff vocal folds (prevent vibration)
- Voiced obstruents have slack vocal folds (allow vibration)
- Sonorants are not specified (0)

---

#### fortis

**Definition**: Greater articulatory force (fortis vs. lenis)

**Values**:
- `+`: Voiceless obstruents (p, t, k, s, f)
- `-`: Voiced obstruents (b, d, g, z, v)
- `0`: Sonorants

**Phonemes**:
- **fortis:+** (9): p, t, k, f, θ, s, ʃ, tʃ
- **fortis:-** (9): b, d, g, v, ð, z, ʒ, dʒ
- **fortis:0** (21): All vowels + sonorant consonants

**Key distinctions**:
- Fortis = stronger, longer, more forceful articulation
- Lenis = weaker, shorter, less forceful
- In English, fortis/lenis aligns with voiceless/voiced

**Note**: Other features (slackVocalFolds, epilaryngealSource, raisedLarynxEjective, loweredLarynxImplosive, constricted) are less relevant for English and have limited variation. See [Lookup](../user-guide/lookup.md#laryngeal-features-11-features) for complete descriptions.

---

### 3. Manner Features (6 features)

These features describe how airflow is modified during articulation.

#### continuant

**Definition**: Airflow continues through the oral cavity

**Values**:
- `+`: Fricatives, approximants, vowels (f, s, ʃ, w, ɹ, l, all vowels)
- `-`: Stops, affricates, nasals (p, t, k, tʃ, m, n)

**Phonemes**:
- **continuant:+** (23): f, v, θ, ð, s, z, ʃ, ʒ, h, w, j, l, ɹ + all vowels
- **continuant:-** (16): p, b, t, d, k, g, tʃ, dʒ, m, n, ŋ

**Key distinctions**:
- Stops/affricates: complete closure, then release (`continuant:-`)
- Fricatives: continuous turbulent airflow (`continuant:+`)
- Approximants/vowels: continuous laminar airflow (`continuant:+`)
- Nasals: oral closure, continuous nasal airflow (`continuant:-`)

**Clinical significance**: Stopping errors (replacing fricatives with stops: "sun" → "tun") involve changing `continuant:+` to `continuant:-`.

---

#### nasal

**Definition**: Airflow through nasal cavity

**Values**:
- `+`: Nasal consonants (m, n, ŋ)
- `-`: Oral sounds (all others)

**Phonemes**:
- **nasal:+** (3): m, n, ŋ
- **nasal:-** (36): All others

**Key distinctions**:
- Nasals have lowered velum, airflow through nose
- Oral sounds have raised velum, airflow through mouth
- All English nasals are voiced

**Clinical significance**: Denasalization errors (replacing nasals with stops: "mom" → "bob").

---

#### strident

**Definition**: High-amplitude frication noise (sibilants, labiodentals)

**Values**:
- `+`: Sibilants (s, z, ʃ, ʒ, tʃ, dʒ), labiodentals (f, v)
- `-`: Non-sibilants (θ, ð), stops
- `0`: Sonorants, vowels

**Phonemes**:
- **strident:+** (10): f, v, s, z, ʃ, ʒ, tʃ, dʒ
- **strident:-** (8): θ, ð, p, b, t, d, k, g
- **strident:0** (21): All vowels + sonorants

**Key distinctions**:
- Sibilants (s, z, ʃ, ʒ) have high-frequency noise
- Non-sibilant fricatives (θ, ð) have lower-amplitude noise
- Labiodentals (f, v) are strident despite not being sibilants

**Clinical significance**: Lisping errors often involve strident sounds (s→θ substitution).

---

#### lateral

**Definition**: Airflow along sides of tongue

**Values**:
- `+`: Lateral approximants (l)
- `-`: Central sounds (all others)

**Phonemes**:
- **lateral:+** (1): l
- **lateral:-** (38): All others

**Key distinctions**:
- Only /l/ is lateral in English
- Central sounds have airflow down the center of the oral cavity
- Laterals are always sonorants

**Clinical significance**: Lateral lisps (airflow along sides during /s/ production).

---

#### delayedRelease

**Definition**: Gradual release from closure (affricates)

**Values**:
- `+`: Affricates (tʃ, dʒ)
- `-`: Stops, fricatives

**Phonemes**:
- **delayedRelease:+** (2): tʃ, dʒ
- **delayedRelease:-** (37): All others

**Key distinctions**:
- Affricates = stop closure + fricative release (t + ʃ = tʃ)
- Stops have abrupt release
- Fricatives have no closure

**Clinical significance**: Deaffrication errors (replacing affricates with fricatives: "ch" → "sh").

---

#### tap

**Definition**: Ballistic tongue movement (flaps)

**Values**:
- `+`: Flaps/taps (ɾ in some American English dialects)
- `-`: All others

**Phonemes**:
- **tap:+** (0-1): ɾ (allophone of /t/, /d/ in some environments, e.g., "butter" = [bʌɾɚ])
- **tap:-** (39): All others in standard phonemic inventory

**Note**: The tap [ɾ] is an allophone of /t/ and /d/ in American English (intervocalic environment), not a separate phoneme.

---

### 4. Place Features - Articulator (3 features)

These features describe which articulators are active.

#### labial

**Definition**: Lips involved in articulation

**Values**:
- `+`: Bilabials (p, b, m), labiodentals (f, v), labio-velars (w)
- `-`: Non-labials (t, d, k, s, n, l)

**Phonemes**:
- **labial:+** (7): p, b, m, f, v, w
- **labial:-** (32): All others

**Subdivisions**:
- Bilabials (both lips): p, b, m
- Labiodentals (lower lip + upper teeth): f, v
- Labio-velars (lips + velum): w

**Clinical significance**: Labialization errors (adding lip rounding to non-labial sounds).

---

#### coronal

**Definition**: Tongue blade/tip raised toward alveolar ridge or palate

**Values**:
- `+`: Alveolars (t, d, s, z, n, l, ɹ), dentals (θ, ð), palatals (ʃ, ʒ, tʃ, dʒ, j)
- `-`: Labials (p, b, m, f, v), velars (k, g, ŋ)

**Phonemes**:
- **coronal:+** (18): t, d, θ, ð, s, z, ʃ, ʒ, tʃ, dʒ, n, l, ɹ, j
- **coronal:-** (21): p, b, m, f, v, k, g, ŋ, w, h + all vowels

**Subdivisions**:
- Dentals (tongue tip at teeth): θ, ð
- Alveolars (tongue tip at alveolar ridge): t, d, s, z, n, l, ɹ
- Palatals (tongue blade at hard palate): ʃ, ʒ, tʃ, dʒ, j

**Clinical significance**: Common place errors involve coronal sounds (fronting: k→t, backing: t→k).

---

#### dorsal

**Definition**: Tongue body raised toward velum

**Values**:
- `+`: Velars (k, g, ŋ, w), back vowels (u, ʊ, o, ɔ, ɑ)
- `-`: Non-dorsals (p, t, s, i, e, æ)

**Phonemes**:
- **dorsal:+** (11): k, g, ŋ, w, j + back vowels (u, ʊ, o, ɔ, ɑ)
- **dorsal:-** (28): All others

**Note**: Multiple place features can be active simultaneously:
- /w/: `labial:+, dorsal:+` (lips + tongue back)
- /j/: `coronal:+, dorsal:+` (tongue blade + body)

**Clinical significance**: Velar fronting (k→t) involves changing `dorsal:+` to `dorsal:-` and `coronal:-` to `coronal:+`.

---

### 5. Place Features - Tongue Body (8 features)

These features describe tongue body position for vowels (and some consonants).

#### high

**Definition**: Tongue body raised

**Values**:
- `+`: High vowels (i, ɪ, u, ʊ), palatals (j, k, g)
- `-`: Mid/low vowels (e, ɛ, æ, ɑ, ɔ, o)
- `0`: Most consonants

**Phonemes**:
- **high:+** (7): i, ɪ, u, ʊ, j + k, g, ŋ (in some analyses)
- **high:-** (8): e, ɛ, æ, ə, ʌ, a, ɑ, ɔ, o
- **high:0** (24): Most consonants

**Vowel space**:
```
        Front    Central    Back
High:   i  ɪ                u  ʊ
Mid:    e  ɛ     ə  ʌ       o  ɔ
Low:          æ   a          ɑ
```

---

#### low

**Definition**: Tongue body lowered

**Values**:
- `+`: Low vowels (æ, a, ɑ, ɔ)
- `-`: High/mid vowels (i, e, u, o)
- `0`: Most consonants

**Phonemes**:
- **low:+** (4): æ, a, ɑ, ɔ
- **low:-** (11): i, ɪ, e, ɛ, ə, ʌ, u, ʊ, o
- **low:0** (24): Most consonants

**Note**: `high` and `low` are mutually exclusive for vowels (`high:+` → `low:-`, `low:+` → `high:-`).

---

#### front

**Definition**: Tongue body fronted

**Values**:
- `+`: Front vowels (i, ɪ, e, ɛ, æ), palatals (j)
- `-`: Back vowels (u, ʊ, o, ɔ, ɑ)
- `0`: Most consonants

**Phonemes**:
- **front:+** (6): i, ɪ, e, ɛ, æ, j
- **front:-** (9): u, ʊ, o, ɔ, ɑ, w
- **front:0** (24): Most consonants

**Vowel space** (front-back dimension):
```
Front: i, ɪ, e, ɛ, æ
Central: ə, ʌ, a
Back: u, ʊ, o, ɔ, ɑ
```

---

#### back

**Definition**: Tongue body backed

**Values**:
- `+`: Back vowels (u, ʊ, o, ɔ, ɑ), velars (k, g, ŋ, w)
- `-`: Front vowels (i, ɪ, e, ɛ, æ)
- `0`: Most consonants

**Phonemes**:
- **back:+** (9): u, ʊ, o, ɔ, ɑ + k, g, ŋ, w
- **back:-** (6): i, ɪ, e, ɛ, æ, j
- **back:0** (24): Most consonants

**Note**: `front` and `back` are mutually exclusive for vowels.

---

#### tense

**Definition**: Greater muscular tension (tense vs. lax vowels)

**Values**:
- `+`: Tense vowels (i, e, u, o)
- `-`: Lax vowels (ɪ, ɛ, ʊ, ɔ, æ)
- `0`: Consonants

**Phonemes**:
- **tense:+** (5): i, e, u, o, ɔ (varies by dialect)
- **tense:-** (10): ɪ, ɛ, æ, ə, ʌ, a, ʊ
- **tense:0** (24): All consonants

**Tense-lax pairs**:
- i (tense) / ɪ (lax)
- e (tense) / ɛ (lax)
- u (tense) / ʊ (lax)
- o (tense) / ɔ (lax)

**Correlates**: Tense vowels are typically longer, higher, and more peripheral in vowel space.

---

**Other tongue body features** (retractedTongueRoot, advancedTongueRoot, raisedLarynx): Less variation in English. See [Lookup](../user-guide/lookup.md#place-features-tongue-body-8-features) for complete descriptions.

---

### 6. Place Features - Detailed (5 features)

#### anterior

**Definition**: Articulated at or in front of alveolar ridge

**Values**:
- `+`: Labials, dentals, alveolars (p, t, s, θ, m, n, l, f, v)
- `-`: Palatals, velars (ʃ, k, g, ŋ, j)
- `0`: Vowels

**Phonemes**:
- **anterior:+** (15): p, b, m, f, v, t, d, θ, ð, s, z, n, l, ɹ
- **anterior:-** (7): ʃ, ʒ, tʃ, dʒ, k, g, ŋ, j
- **anterior:0** (17): All vowels + w, h

**Key distinctions**:
- Anterior sounds articulated further forward
- /s/ (anterior:+) vs. /ʃ/ (anterior:-) differ primarily in this feature
- /t/ (anterior:+) vs. /k/ (anterior:-) differ in anterior + coronal/dorsal

---

#### distributed

**Definition**: Longer constriction along the midline of the oral cavity

**Values**:
- `+`: Palatals (ʃ, ʒ, tʃ, dʒ), dentals (θ, ð), laterals (l)
- `-`: Alveolars (t, d, s, z)
- `0`: Vowels, labials

**Phonemes**:
- **distributed:+** (7): θ, ð, ʃ, ʒ, tʃ, dʒ, l
- **distributed:-** (11): t, d, s, z, n, ɹ
- **distributed:0** (21): Vowels + labials

**Key distinctions**:
- Distributed sounds have longer tongue-palate contact
- /s/ (distributed:-) vs. /ʃ/ (distributed:+): /ʃ/ has broader constriction
- Laterals (l) have long contact along tongue sides

---

**Other detailed features** (labialDental, retroflexed): Limited variation in English. See [Lookup](../user-guide/lookup.md#place-features-detailed-5-features) for complete descriptions.

---

### 7. Vowel-Specific Features (1 feature)

#### retractedTongueRoot

**Definition**: Tongue root retracted (pharyngealized vowels)

**Values**:
- `+`: RTR vowels (rare in English)
- `-`: Most English vowels

**Note**: Minimal variation in English. More relevant for languages with ATR/RTR harmony (e.g., West African languages).

---

## Complete Feature Matrix: English Phonemes

### Consonants

| Phoneme | Major Class | Laryngeal | Manner | Place |
|---------|-------------|-----------|--------|-------|
| **/p/** | cons:+, son:-, syl:- | voice:-, spreadG:+ | cont:-, nas:-, strid:- | lab:+, cor:-, dor:-, ant:+ |
| **/b/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:-, nas:-, strid:- | lab:+, cor:-, dor:-, ant:+ |
| **/t/** | cons:+, son:-, syl:- | voice:-, spreadG:+ | cont:-, nas:-, strid:- | lab:-, cor:+, dor:-, ant:+ |
| **/d/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:-, nas:-, strid:- | lab:-, cor:+, dor:-, ant:+ |
| **/k/** | cons:+, son:-, syl:- | voice:-, spreadG:+ | cont:-, nas:-, strid:- | lab:-, cor:-, dor:+, ant:- |
| **/g/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:-, nas:-, strid:- | lab:-, cor:-, dor:+, ant:- |
| **/m/** | cons:+, son:+, syl:- | voice:+ | cont:-, nas:+, lateral:- | lab:+, cor:-, dor:-, ant:+ |
| **/n/** | cons:+, son:+, syl:- | voice:+ | cont:-, nas:+, lateral:- | lab:-, cor:+, dor:-, ant:+ |
| **/ŋ/** | cons:+, son:+, syl:- | voice:+ | cont:-, nas:+, lateral:- | lab:-, cor:-, dor:+, ant:- |
| **/f/** | cons:+, son:-, syl:- | voice:-, spreadG:- | cont:+, nas:-, strid:+ | lab:+, cor:-, dor:-, ant:+ |
| **/v/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:+, nas:-, strid:+ | lab:+, cor:-, dor:-, ant:+ |
| **/θ/** | cons:+, son:-, syl:- | voice:-, spreadG:- | cont:+, nas:-, strid:- | lab:-, cor:+, dor:-, ant:+, dist:+ |
| **/ð/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:+, nas:-, strid:- | lab:-, cor:+, dor:-, ant:+, dist:+ |
| **/s/** | cons:+, son:-, syl:- | voice:-, spreadG:- | cont:+, nas:-, strid:+ | lab:-, cor:+, dor:-, ant:+, dist:- |
| **/z/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:+, nas:-, strid:+ | lab:-, cor:+, dor:-, ant:+, dist:- |
| **/ʃ/** | cons:+, son:-, syl:- | voice:-, spreadG:- | cont:+, nas:-, strid:+ | lab:-, cor:+, dor:-, ant:-, dist:+ |
| **/ʒ/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:+, nas:-, strid:+ | lab:-, cor:+, dor:-, ant:-, dist:+ |
| **/tʃ/** | cons:+, son:-, syl:- | voice:-, spreadG:- | cont:-, nas:-, strid:+, delR:+ | lab:-, cor:+, dor:-, ant:-, dist:+ |
| **/dʒ/** | cons:+, son:-, syl:- | voice:+, spreadG:- | cont:-, nas:-, strid:+, delR:+ | lab:-, cor:+, dor:-, ant:-, dist:+ |
| **/l/** | cons:+, son:+, syl:- | voice:+ | cont:+, nas:-, lateral:+ | lab:-, cor:+, dor:-, ant:+ |
| **/ɹ/** | cons:+, son:+, syl:- | voice:+ | cont:+, nas:-, lateral:-, approx:+ | lab:-, cor:+, dor:-, ant:+ |
| **/w/** | cons:-, son:+, syl:- | voice:+ | cont:+, nas:-, approx:+ | lab:+, cor:-, dor:+, back:+ |
| **/j/** | cons:-, son:+, syl:- | voice:+ | cont:+, nas:-, approx:+ | lab:-, cor:+, dor:+, front:+, high:+ |
| **/h/** | cons:+, son:+, syl:- | voice:-, spreadG:+ | cont:+ | (glottal, minimal place) |

### Vowels

| Phoneme | Height | Frontness | Tenseness |
|---------|--------|-----------|-----------|
| **/i/** | high:+, low:- | front:+, back:- | tense:+ |
| **/ɪ/** | high:+, low:- | front:+, back:- | tense:- |
| **/e/** | high:-, low:- | front:+, back:- | tense:+ |
| **/ɛ/** | high:-, low:- | front:+, back:- | tense:- |
| **/æ/** | high:-, low:+ | front:+, back:- | tense:- |
| **/ə/** | high:-, low:- | (central) | tense:- |
| **/ʌ/** | high:-, low:- | (central) | tense:- |
| **/a/** | high:-, low:+ | (central) | tense:- |
| **/ɑ/** | high:-, low:+ | front:-, back:+ | tense:- |
| **/ɔ/** | high:-, low:- | front:-, back:+ | tense:+/- (varies) |
| **/o/** | high:-, low:- | front:-, back:+ | tense:+ |
| **/u/** | high:+, low:- | front:-, back:+ | tense:+ |
| **/ʊ/** | high:+, low:- | front:-, back:+ | tense:- |

**Notes**:
- All vowels have: `consonantal:-, syllabic:+, sonorant:+, voice:+, continuant:+`
- Diphthongs (/aɪ/, /aʊ/, /ɔɪ/, /oʊ/, /eɪ/) combine features from two vowels
- This is a simplified matrix; complete matrix has 38 features per phoneme

---

## Using Features in PhonoLex

### Phoneme Lookup

View all 38 features for any phoneme:

**Example**: Lookup /k/
```
consonantal: +
sonorant: -
voice: -
dorsal: +
high: +
back: +
... (32 more features)
```

See [Lookup - Phoneme Lookup](../user-guide/lookup.md#phoneme-lookup) for details.

---

### Phoneme Comparison

Compare two phonemes feature-by-feature:

**Example**: Compare /t/ vs. /d/
```
Shared features (36): consonantal:+, sonorant:-, continuant:-, dorsal:-, ...
Different features (2): voice (- vs +), periodicGlottalSource (- vs +)
Similarity: 0.947
Major class difference: NO
```

See [Lookup - Phoneme Comparison](../user-guide/lookup.md#phoneme-comparison) for details.

---

### Search by Features

Find all phonemes matching specific features:

**Example**: Find all voiced stops
```
Features:
  consonantal: +
  sonorant: -
  continuant: -
  voice: +

Results: b, d, g
```

See [Lookup - Search by Features](../user-guide/lookup.md#search-by-features) for common searches.

---

### Maximal Opposition Scoring

Maximal opposition pairs are scored by:
1. Count feature differences
2. Add +100 if major class difference (`sonorant` values differ)

**Example**: /s/ vs. /l/
```
Feature differences: 14
sonorant: /s/ = -, /l/ = + (MAJOR CLASS DIFFERENCE)
Score: 14 + 100 = 114
```

See [Contrastive Sets - Maximal Opposition](../user-guide/contrastive-sets.md#maximal-opposition) for details.

---

## Clinical Applications

### Understanding Error Patterns

Compare target phoneme to substitution to identify which features differ:

**Example**: Fronting error (k → t)
```
/k/ vs. /t/:
  Different: dorsal (+ vs -), coronal (- vs +), anterior (- vs +)
  Shared: consonantal:+, sonorant:-, voice:-, continuant:-

Interpretation: Child changing place of articulation (velar → alveolar)
```

---

### Planning Intervention

**Minimal pairs**: Find pairs differing in 1-2 features (e.g., /p/ vs. /b/ differ only in voice)

**Maximal opposition**: Find pairs with major class difference + many feature differences (e.g., /s/ vs. /l/)

**Multiple opposition**: Find sets where all phonemes differ maximally (e.g., /t/, /s/, /ʃ/, /θ/ all produced as [t])

See [Contrastive Sets](../user-guide/contrastive-sets.md) for complete documentation.

---

### Sound Class Identification

Use feature search to find sound classes for treatment hierarchies:

**Early-developing**: Stops (continuant:-), nasals (nasal:+), glides (approximant:+)

**Late-developing**: Fricatives (continuant:+), liquids (approximant:+, nasal:-), affricates (delayedRelease:+)

---

## Research Applications

### Cross-linguistic Phonology

PHOIBLE features allow comparison of English phonemes to inventories from 2,716 languages:
- Universal features apply to all languages
- Identify typologically rare vs. common sounds
- Analyze phonological universals

---

### Feature Geometry

Features are organized hierarchically (Feature Geometry theory):
```
Root
├── Major class (consonantal, sonorant, syllabic)
├── Laryngeal (voice, spreadGlottis, etc.)
├── Place
│   ├── Articulator (labial, coronal, dorsal)
│   └── Tongue body (high, low, front, back)
└── Manner (continuant, nasal, lateral, etc.)
```

---

### Phonological Acquisition

Track which features are acquired at different ages:
- Early: voice, place (labial, coronal)
- Middle: continuant, strident
- Late: distributed, lateral

---

## References

**PHOIBLE Database**:
- Moran, S., & McCloy, D. (2019). PHOIBLE 2.0. Max Planck Institute for Evolutionary Anthropology. https://phoible.org/

**Feature Theory**:
- Hayes, B. (2009). *Introductory Phonology*. Wiley-Blackwell.
- Moisik, S. R., & Esling, J. H. (2011). The 'whole larynx' approach to laryngeal features. In *Proceedings of the 17th International Congress of Phonetic Sciences* (pp. 1406-1409).

**Feature Geometry**:
- Clements, G. N. (1985). The geometry of phonological features. *Phonology Yearbook*, 2, 225-252.
- McCarthy, J. J. (1988). Feature geometry and dependency: A review. *Phonetica*, 45(2-4), 84-108.

---

## See Also

- [Lookup](../user-guide/lookup.md) - View features, compare phonemes, search by features
- [Contrastive Sets](../user-guide/contrastive-sets.md) - Use features for intervention planning
- [Technical Architecture](../technical/architecture.md) - How features are extracted and stored
