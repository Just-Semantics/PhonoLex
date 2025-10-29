# Reusable Components Across Tasks

## 🔧 What We've Built (Universal Infrastructure)

Our PhonoLex infrastructure supports **any** phonological learning task. Here's what's reusable:

---

## 1. Phoneme Vectorization Pipeline ✅

**What**: Convert any phoneme to vector representation
**Where**: `data/mappings/phoneme_vectorizer.py`

**Reusable for:**
- Rule learning (get features for stem-final phoneme)
- Similarity tasks (compare phoneme vectors)
- Typology (analyze feature distributions)
- G2P (phoneme embeddings as targets)
- Dialectal analysis (compare phoneme vectors across dialects)

**Example**:
```python
from data.mappings.phoneme_vectorizer import PhonemeVectorizer

vectorizer = PhonemeVectorizer()
phoneme_data = {'Phoneme': 't', 'voiced': '-', 'consonantal': '+', ...}
vec = vectorizer.vectorize(phoneme_data)

# Get 76-dim endpoints OR 152-dim trajectory
features = vec.endpoints_76d  # Use for any task
```

---

## 2. Phoible Feature Database ✅

**What**: 105K phonemes with 38 distinctive features
**Where**:
- `data/phoible/phoible.csv` (full, 2,716 languages)
- `data/phoible/english/phoible-english.csv` (English subset)

**Reusable for:**
- **Any task needing phonological features**
- Lookup phoneme properties
- Compare phonemes
- Analyze typological patterns
- Train feature-based models

**Example**:
```python
# Look up features for /t/
import csv
with open('data/phoible/phoible.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Phoneme'] == 't':
            print(f"voiced: {row['voiced']}")  # '-'
            print(f"consonantal: {row['consonantal']}")  # '+'
            break
```

---

## 3. Vector Database (ChromaDB) ✅

**What**: Similarity search over phoneme feature space
**Where**:
- `data/mappings/chroma_db/` (English)
- `data/mappings/phoible_full_db/` (all languages)

**Reusable for:**
- Find similar phonemes (any task needing similarity)
- K-nearest neighbors lookup
- Clustering analysis
- Visualization (t-SNE input)
- Anomaly detection

**Example**:
```python
import chromadb
client = chromadb.PersistentClient(path='data/mappings/chroma_db')
collection = client.get_collection('phonemes_endpoints_76d')

# Find phonemes similar to /s/
s_phoneme = collection.get(where={"phoneme": "s"}, include=['embeddings'], limit=1)
similar = collection.query(
    query_embeddings=[s_phoneme['embeddings'][0]],
    n_results=10
)
# Returns: /z/, /ʃ/, /ʒ/, ... (sibilants!)
```

---

## 4. ARPAbet ↔ IPA Mappings ✅

**What**: Verified conversion between notation systems
**Where**:
- `data/mappings/arpa_to_ipa.json`
- `data/mappings/ipa_to_arpa.json`
- `data/mappings/phoneme_mappings.py`

**Reusable for:**
- Convert CMU dict (ARPAbet) → IPA → Phoible features
- Any task using CMU dict data
- Standardize notation across datasets
- G2P (pronunciation as IPA)

**Example**:
```python
import json
with open('data/mappings/arpa_to_ipa.json') as f:
    arpa_to_ipa = json.load(f)

# Convert CMU pronunciation
cmu_word = ["K", "AE1", "T"]  # "cat"
ipa_word = "".join(arpa_to_ipa[p] for p in cmu_word)
# Result: "kˈæt"
```

---

## 5. Learning Datasets ✅

**What**: Multiple datasets for different tasks
**Where**: `data/learning_datasets/`

**Breakdown**:

| Dataset | Size | Use For |
|---------|------|---------|
| SIGMORPHON | 92K examples | Morphology, rule learning |
| UniMorph | 1.6M examples | Held-out testing, augmentation |
| ipa-dict | 191K words | Pronunciation lookup, G2P |
| CMU Dict | 134K words | Pronunciation lookup, G2P |

**All interoperable** - can link via word forms

---

## 6. Data Processing Utilities ✅

**What**: Scripts to combine/process datasets
**Where**: `data/learning_datasets/analyze_datasets.py` (starting point)

**Can extend for:**
- Merge SIGMORPHON + pronunciations
- Link morphology to phonology
- Create train/dev/test splits
- Generate novel word test sets
- Cross-dataset evaluation

---

## 🎯 Task-Specific Pipelines

Now let's see how to **compose** these components for each task:

### Task A: Phonological Rule Learning

**Pipeline**:
```python
# 1. Load morphological data (SIGMORPHON)
lemma, inflected, features = load_sigmorphon()

# 2. Get pronunciations (CMU Dict)
lemma_phones = cmu_dict[lemma]  # ARPAbet

# 3. Convert to IPA
lemma_ipa = arpa_to_ipa(lemma_phones)

# 4. Get phonological features (Phoible)
final_phoneme = lemma_ipa[-1]
phon_features = phoible_vectorize(final_phoneme)  # 76-dim

# 5. Train model
model.fit(phon_features, allomorph_choice)

# 6. Evaluate on novel words
predict_allomorph(novel_word_features)
```

**Reuses**: #1, #2, #4, #5

---

### Task B: Cross-Linguistic Typology

**Pipeline**:
```python
# 1. Load full Phoible (all languages)
all_phonemes = load_phoible_full()

# 2. Get vector representations
vectors = [vectorize_phoneme(p) for p in all_phonemes]

# 3. Use vector database for similarity
chroma_full_db = load_chroma('phoible_full_db')

# 4. Cluster analysis
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10).fit(vectors)

# 5. Analyze patterns
analyze_feature_cooccurrence(clusters)
```

**Reuses**: #1, #2, #3

---

### Task C: Phoneme Similarity (Embeddings)

**Pipeline**:
```python
# 1. Start with Phoible features as initialization
init_embeddings = phoible_features  # 38-dim

# 2. Load context from datasets
contexts = extract_phoneme_contexts(cmu_dict)

# 3. Train embeddings (e.g., skip-gram)
model = train_embeddings(init_embeddings, contexts)

# 4. Evaluate on downstream tasks
rhyme_accuracy = eval_rhyme_detection(model)
minimal_pair_accuracy = eval_minimal_pairs(model)

# 5. Compare to Phoible features
compare_to_baseline(model, phoible_features)
```

**Reuses**: #1, #2, #3, #4, #5

---

### Task D: Dialectal Variation

**Pipeline**:
```python
# 1. Load English phonemes by dialect (Phoible)
am_english = load_phoible_english(dialect='American')
uk_english = load_phoible_english(dialect='British')

# 2. Vectorize both
am_vectors = vectorize(am_english)
uk_vectors = vectorize(uk_english)

# 3. Find systematic differences using vector DB
for phoneme in am_english:
    am_vec = get_vector(phoneme, 'American')
    uk_similar = find_similar(am_vec, dialect='British')
    analyze_difference(phoneme, uk_similar)

# 4. Link to pronunciations (ipa-dict US vs UK)
us_pronunciations = ipa_dict['en_US']
uk_pronunciations = ipa_dict['en_UK']
compare_pronunciations(us_pronunciations, uk_pronunciations)
```

**Reuses**: #1, #2, #3, #5

---

### Task E: Grapheme-to-Phoneme

**Pipeline**:
```python
# 1. Load G2P data (CMU Dict or ipa-dict)
word, pronunciation = load_cmu_dict()

# 2. Convert pronunciation to IPA
ipa_pronunciation = arpa_to_ipa(pronunciation)

# 3. Get phoneme feature vectors (targets)
target_vectors = [vectorize_phoneme(p) for p in ipa_pronunciation]

# 4. Train G2P model
#    Input: character sequence
#    Output: phoneme feature vectors (then decode to phonemes)
model = train_g2p(
    input=word_chars,
    targets=target_vectors  # Use Phoible features!
)

# 5. Decode predictions to phonemes
predicted_features = model.predict(novel_word)
predicted_phonemes = nearest_phoneme(predicted_features, phoible_db)
```

**Reuses**: #1, #2, #3, #4, #5

---

## 🔥 Key Insight: Everything Connects

The **common thread** across all tasks:

1. **Phonemes** are the atomic unit
2. **Features** (from Phoible) are the representation
3. **Vector space** enables computation (similarity, clustering, prediction)
4. **Datasets** provide training signal
5. **Mappings** allow interoperability

```
            ┌─────────────┐
            │   Phoible   │ (features)
            │  105K phon  │
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │ Vectorizer  │ (76d / 152d)
            └──────┬──────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │  Task  │ │  Task  │ │  Task  │
   │   A    │ │   B    │ │   C    │
   └────────┘ └────────┘ └────────┘
   Rule        Typology    Similarity
   Learning    Analysis    Learning
```

---

## 📦 What You'd Build for Each Task

### Common Code (Reuse 100%)
```python
# phonolex/core/
├── phoneme_vectorizer.py  # Already have ✅
├── phoible_loader.py      # Already have (CSV reading) ✅
├── mappings.py            # Already have ✅
└── vector_db.py           # Already have (ChromaDB) ✅
```

### Task-Specific Code (New for each)
```python
# phonolex/tasks/
├── rule_learning/
│   ├── sigmorphon_loader.py  # Load SIGMORPHON format
│   ├── allomorph_model.py    # Predict allomorph from features
│   └── evaluate.py            # Wug test, accuracy metrics
│
├── typology/
│   ├── cluster_analysis.py   # K-means, hierarchical
│   ├── universal_patterns.py # Find implications
│   └── visualize.py          # t-SNE, plots
│
├── embeddings/
│   ├── contrastive_learning.py  # Train embeddings
│   ├── downstream_tasks.py      # Rhyme, minimal pairs
│   └── evaluate.py              # Compare to baselines
│
├── dialectology/
│   ├── compare_dialects.py   # Systematic differences
│   ├── correspondence_rules.py # Sound changes
│   └── classify.py            # Dialect identification
│
└── g2p/
    ├── seq2seq_model.py      # Char → phoneme features
    ├── decode.py             # Features → phoneme
    └── evaluate.py           # PER, WER
```

**Estimate**:
- Common code: Already done ✅
- Task-specific: 200-500 lines per task
- Total per task: ~1 week of coding

---

## 💡 Strategic Recommendation

**Do them sequentially, build on learnings:**

### Phase 1: Rule Learning (2 weeks)
- Learn what works for phonological prediction
- Build evaluation framework
- **Outcome**: Baseline for comparing feature representations

### Phase 2: Embeddings (2 weeks)
- Use insights from Phase 1
- Train better representations
- **Outcome**: Improved features for all tasks

### Phase 3: Choose Your Adventure (2 weeks)
- Typology (exploratory)
- Dialectology (applied)
- G2P (benchmarkable)

**Each phase informs the next** and shares the infrastructure!

---

## 🎯 Immediate Next Step

Regardless of which task we pick first, the **next concrete action** is the same:

**Create a unified data loader that combines:**
1. Morphological examples (SIGMORPHON)
2. Pronunciations (CMU or ipa-dict)
3. Phonological features (Phoible vectors)

This loader will be **reused across all tasks**.

Want me to build that? It's the foundation for everything else.
