# PhonoLex Project Plan

**Status**: Planning Phase
**Date**: 2025-10-26

---

## 🎯 Core Question

**What do we actually want to learn/discover/build?**

Let's think carefully about the objectives before diving into implementation.

---

## 💭 Potential Objectives

### Option A: Phonological Rule Learning
**Goal**: Learn phonologically-conditioned morphological alternations

**Example**:
- English plurals: cat→cats /-s/, dog→dogs /-z/, bus→buses /-ɪz/
- Past tense: walk→walked /-t/, hug→hugged /-d/, pat→patted /-ɪd/

**Research Question**: Can a model learn these rules from phonological features alone?

**Why interesting**: Tests if phonological features (from Phoible) are sufficient to predict allomorph selection

**Datasets needed**: SIGMORPHON + Phoible features

**Evaluation**: Accuracy on novel words (wug test)

---

### Option B: Cross-Linguistic Phonological Typology
**Goal**: Discover universal patterns in phoneme inventories

**Example**:
- Which features commonly co-occur?
- Are there implicational universals? (if /p/ then /b/?)
- Cluster languages by phonological similarity

**Research Question**: What structure exists in the space of possible phoneme systems?

**Why interesting**: Connects phonology to typology, could reveal constraints

**Datasets needed**: Full Phoible (2,716 languages)

**Evaluation**: Qualitative analysis, clustering metrics

---

### Option C: Phoneme Similarity for NLP Applications
**Goal**: Build better phonological representations for downstream tasks

**Example**:
- Rhyme detection
- Pronunciation error detection (L2 learning)
- Speech recognition features
- Text-to-speech improvements

**Research Question**: Do learned phoneme embeddings outperform hand-crafted features?

**Why interesting**: Practical applications

**Datasets needed**: CMU Dict + tasks with evaluation metrics

**Evaluation**: Task-specific metrics (rhyme accuracy, etc.)

---

### Option D: Dialectal Variation Analysis
**Goal**: Quantify and predict dialectal pronunciation differences

**Example**:
- How does /r/ differ between American, British, Australian English?
- Can we predict dialect from phoneme inventory?
- Map systematic sound correspondences

**Research Question**: What are the systematic differences between English dialects?

**Why interesting**: Sociolinguistics + phonology

**Datasets needed**: Phoible English (10 dialects) + ipa-dict (US/UK)

**Evaluation**: Classification accuracy, correspondence rules

---

### Option E: Grapheme-to-Phoneme (G2P) Learning
**Goal**: Learn spelling → pronunciation with phonological features

**Example**:
- "through" → /θɹu/ (irregular)
- Predict stress patterns
- Handle loan words

**Research Question**: Do phonological features help G2P models?

**Why interesting**: Classic NLP problem, benchmarks exist

**Datasets needed**: CMU Dict or ipa-dict (orthography + pronunciation)

**Evaluation**: Phoneme error rate, word error rate

---

## 🤔 Decision Criteria

Before choosing, consider:

### 1. **Scientific Interest**
- What would we learn that's new?
- What hypotheses can we test?
- What insights would be valuable?

### 2. **Technical Feasibility**
- Do we have the right data?
- Can we build/train the models?
- Are there existing baselines to compare against?

### 3. **Practical Value**
- Would this be useful for applications?
- Could others build on this?
- Is there demand for this?

### 4. **Time Investment**
- How long would this take?
- What's the minimum viable version?
- Can we iterate and improve?

---

## 📊 What We Have (Assets)

1. **Phoible Database**
   - 105K phonemes, 2,716 languages
   - 38 distinctive features per phoneme
   - Vector representations (76-dim, 152-dim)
   - Trajectory encoding for diphthongs

2. **CMU Pronouncing Dictionary**
   - 134K words with ARPAbet pronunciations
   - ARPAbet ↔ IPA verified mappings
   - General American English

3. **Learning Datasets**
   - SIGMORPHON: 92K morphological examples
   - UniMorph: 1.6M morphological examples
   - ipa-dict: 191K IPA pronunciations (US + UK)

4. **Infrastructure**
   - ChromaDB vector databases (English + Full)
   - Phoneme vectorization pipeline
   - Data processing scripts

---

## 🎲 My Initial Thoughts

### Most Scientifically Interesting: **Option A (Rule Learning)**

**Why:**
- Clear hypothesis: phonological features predict allomorph selection
- Connects formal phonology with ML
- Generalizable (wug test)
- Interpretable (can extract learned rules)

**How it would work:**
```python
# Training example
Input:
  lemma: "cat" /kæt/
  features: N;PL
  phonological context: final segment /t/ [voiced:-, continuant:-, ...]

Output:
  inflected: "cats" /kæts/
  allomorph: /-s/ (not /-z/ or /-ɪz/)

# Model learns:
# Rule: If stem ends in voiceless non-sibilant → use /-s/
```

**Evaluation:**
- Accuracy on held-out words
- Generalization to novel words (wug test style)
- Compare: rule-based baseline vs neural model
- Ablation: with/without phonological features

**Timeline:** 2-3 weeks for MVP
- Week 1: Data prep, baseline
- Week 2: Model training
- Week 3: Evaluation, analysis

---

### Most Practically Useful: **Option C (Phoneme Similarity)**

**Why:**
- Many downstream applications
- Can benchmark against existing work
- Reusable embeddings

**How it would work:**
- Train embeddings using contrastive learning
- Tasks: rhyme detection, minimal pairs, etc.
- Compare to hand-crafted features (Phoible)

**Timeline:** 3-4 weeks

---

### Most Exploratory: **Option B (Typology)**

**Why:**
- Could discover new patterns
- Visualization would be beautiful
- Connects to theoretical linguistics

**How it would work:**
- Cluster languages by phoneme inventories
- Find implicational universals
- Analyze feature co-occurrence

**Timeline:** 2 weeks for analysis

---

## ❓ Questions to Consider

1. **What's the end goal?**
   - Research paper?
   - Practical tool?
   - Learning exercise?
   - Exploration?

2. **What excites YOU most?**
   - Learning ML techniques?
   - Understanding phonology better?
   - Building something useful?
   - Publishing results?

3. **What would success look like?**
   - A model that beats baselines?
   - New insights about phonology?
   - A reusable tool/library?
   - Understanding how it all works?

4. **Constraints?**
   - Time available?
   - Computational resources?
   - Need for novelty vs learning?

---

## 🚀 Recommendation: Start Small, Iterate

### Phase 1: Proof of Concept (1 week)
Pick ONE simple task:
- English plural allomorphy (just 3 rules: -s, -z, -ɪz)
- Small dataset (1000 examples)
- Simple baseline (rule-based)
- Simple model (logistic regression on features)

**Deliverable**: Working end-to-end pipeline

### Phase 2: Scale Up (1-2 weeks)
- Full SIGMORPHON dataset
- Add past tense
- Neural model
- Proper evaluation

**Deliverable**: Competitive results

### Phase 3: Publish/Share (1 week)
- Write up results
- Create notebook/tutorial
- Share on GitHub
- Blog post?

---

## 💬 What Do You Think?

Before we proceed, let's discuss:

1. Which objective resonates most with you?
2. What would you want to learn from this project?
3. How much time do you want to invest?
4. What would make you proud of the result?

**Let's be intentional about what we build and why.**

---

## 📝 Next Steps (Once Decided)

1. Write clear problem statement
2. Define evaluation metrics
3. Create baseline
4. Design model architecture
5. Implement training pipeline
6. Run experiments
7. Analyze results
8. Document findings

**But first: Let's decide on the objective!**
