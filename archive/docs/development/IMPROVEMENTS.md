# Improvements for Contextual Phoneme Embeddings

## Current Status
✅ Contextual phoneme embeddings via transformer (92.8% MLM accuracy)
✅ Position-aware aggregation (concat first-middle-last works best)
✅ Distinguishes anagrams (cat vs act: 0.31)
✅ Captures rhymes (cat vs bat: 0.75)

## Key Improvements to Consider

### 1. **Contrastive Learning for Word-Level Representations**
**Problem**: Model trained only on masked phoneme prediction - doesn't learn word-level similarity
**Solution**: Add contrastive objective during training

```python
# Positive pairs:
- Rhymes: (cat, bat), (dog, fog)
- Minimal pairs: (cat, bat), (sit, sit)
- Morphological variants: (run, running), (walk, walked)

# Negative pairs:
- Unrelated words: (cat, dog)
- Anagrams: (cat, act) ← should be pushed apart!
```

**Benefits**:
- CLS token learns meaningful representations
- Word embeddings optimized for similarity tasks
- Can tune what "similar" means (rhyme vs morphology vs semantics)

### 2. **Syllable Structure Awareness**
**Problem**: Current model treats phonemes as flat sequence
**Solution**: Encode syllable boundaries and structure

```
cat = /kæt/ = onset(k) + nucleus(æ) + coda(t)
running = /ɹʌnɪŋ/ = [onset(ɹ)+nucleus(ʌ)+coda(n)] + [onset(ø)+nucleus(ɪ)+coda(ŋ)]
```

**Implementation**:
- Add syllable boundary tokens: `<SYL>`
- Add position-in-syllable embeddings (onset/nucleus/coda)
- Hierarchical attention: phoneme → syllable → word

**Benefits**:
- Captures phonotactic constraints
- Better rhyme detection (based on syllable nucleus/coda)
- More linguistically grounded

### 3. **Phonological Feature Integration**
**Problem**: Model learns phoneme embeddings from scratch, ignores phonological features
**Solution**: Initialize phoneme embeddings with feature vectors

```python
# Features from Phoible:
/k/ = [consonantal, dorsal, voiceless, ...]
/æ/ = [vocalic, front, low, ...]

# Hybrid approach:
phoneme_emb = learned_emb + feature_projection(phoible_features)
```

**Benefits**:
- Faster training (warm start)
- Better generalization (similar phonemes have similar embeddings)
- Cross-lingual transfer (same features across languages)

### 4. **Multi-Task Learning with Phonological Tasks**
**Current**: Only masked phoneme prediction
**Add**:
- **Stress prediction**: Predict which syllable is stressed
- **Phonotactic validity**: Binary classification (valid vs invalid phoneme sequence)
- **Rhyme prediction**: Given two words, predict if they rhyme
- **Allomorph selection**: Predict correct allomorph (walk→walked vs run→ran)

**Benefits**:
- Richer phonological knowledge
- Better word-level representations
- Task-specific heads on top of shared encoder

### 5. **Attention Pooling (Learned Aggregation)**
**Problem**: Fixed aggregation (concat, mean) may not be optimal
**Solution**: Learn how to aggregate phoneme embeddings

```python
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        self.query = nn.Parameter(torch.randn(d_model))

    def forward(self, phoneme_embeddings):
        # phoneme_embeddings: (seq_len, d_model)
        # Compute attention scores
        scores = torch.matmul(phoneme_embeddings, self.query)  # (seq_len,)
        weights = torch.softmax(scores, dim=0)
        # Weighted sum
        return torch.sum(phoneme_embeddings * weights.unsqueeze(1), dim=0)
```

**Benefits**:
- Model learns which positions matter most
- Different from mean (not uniform weighting)
- Single vector output (like CLS but learned)

### 6. **Distance Metric Learning**
**Problem**: Cosine similarity may not be ideal for phonological distance
**Solution**: Learn a distance metric specifically for phonology

```python
# Siamese network approach:
class PhonologicalDistance(nn.Module):
    def __init__(self, d_model):
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, word_emb1, word_emb2):
        # Concatenate embeddings
        combined = torch.cat([word_emb1, word_emb2], dim=-1)
        # Output: probability of similarity
        return self.projection(combined)
```

Train with phonological similarity labels (rhyme, minimal pair, etc.)

### 7. **Sub-Phoneme (Feature) Level Attention**
**Problem**: Phonemes are atomic - but they have internal structure
**Solution**: Decompose phonemes into features, attend over features

```python
# Instead of: phoneme_id → embedding
# Use: phoneme_id → [voice, place, manner, ...] → feature_embeddings → aggregate

/k/ = voiceless + dorsal + stop
/g/ = voiced + dorsal + stop  # differs only in voicing!
```

**Benefits**:
- Model learns that /k/ and /g/ are similar (differ by one feature)
- Captures phonological naturalness
- Better generalization

### 8. **Cross-Lingual Pre-Training**
**Current**: English only
**Improvement**: Pre-train on multiple languages, fine-tune on English

- Use IPA (cross-lingual phonemes)
- Leverage all Phoible inventories (2,716 languages!)
- Transfer learning from rich phonological diversity

**Benefits**:
- Better phoneme representations
- Captures universal phonological patterns
- Can handle code-switching and loanwords

### 9. **Explicit Onset-Nucleus-Coda Decomposition**
**Problem**: Model doesn't know syllable structure
**Solution**: Explicitly encode ONC structure

```python
# Parse phonemes into syllable structure
cat = /kæt/ → Syllable(onset=[k], nucleus=[æ], coda=[t])

# Separate embeddings for each component
word_emb = concat([
    onset_pooling([k]),
    nucleus_pooling([æ]),
    coda_pooling([t])
])
```

**Benefits**:
- Linguistically principled
- Perfect for rhyme detection (match nucleus+coda)
- Captures phonotactic constraints within syllable positions

### 10. **Variational Phoneme Embeddings**
**Problem**: Deterministic embeddings - no uncertainty
**Solution**: Learn distributions over phoneme embeddings (VAE-style)

```python
# Instead of: phoneme → embedding vector
# Use: phoneme → (μ, σ) → sample embedding ~ N(μ, σ²)
```

**Benefits**:
- Captures phonological ambiguity
- Allophones (variations) naturally represented
- Regularization effect

## Priority Ranking

**High Priority** (biggest impact, reasonable effort):
1. ✅ **Contrastive learning** - Critical for word-level similarity
2. ✅ **Attention pooling** - Better than fixed aggregation
3. ✅ **Syllable structure** - Core to phonology

**Medium Priority** (good improvements):
4. Multi-task learning with phonological tasks
5. Phonological feature integration
6. Distance metric learning

**Low Priority** (research-y, complex):
7. Sub-phoneme feature attention
8. Cross-lingual pre-training
9. Variational embeddings

## Next Steps

**Immediate**:
- Add contrastive learning objective
- Train with positive/negative pairs (rhymes, morphology, anagrams)
- Evaluate on phonological similarity benchmarks

**Short-term**:
- Add syllable boundary information
- Implement attention pooling
- Multi-task with stress/phonotactics

**Long-term**:
- Cross-lingual expansion
- Feature-level modeling
- Generative capabilities (phoneme sequence generation)
