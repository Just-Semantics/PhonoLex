# PhonoLex Cleanup Plan

**Date**: 2025-10-28
**Goal**: Remove outdated documentation and consolidate to current v2.0 architecture

---

## Current State Analysis

### Documentation Files

#### ✅ KEEP (Essential & Current)
- `README.md` - Main project README (needs updating to v2.0)
- `docs/ARCHITECTURE_V2.md` - **NEW! Current architecture (database-centric)**
- `src/phonolex/embeddings/README.md` - Embeddings framework documentation
- `data/mappings/README.md` - Phoneme vectorization docs
- `webapp/README.md` - Web app documentation (needs updating)

#### ⚠️ ARCHIVE (Research/Historical Value)
Move to `archive/docs/`:
- `docs/VECTOR_ENCODING_STRATEGIES.md` - Historical research
- `docs/NORMALIZATION_ANALYSIS.md` - Historical research
- `docs/EDIT_DISTANCE_OPTIMIZATION.md` - Optimization notes (keep for reference)
- `docs/HIERARCHICAL_SIMILARITY_OPTIMIZATION.md` - Optimization notes
- `docs/development/CURRICULUM_LEARNING_RESULTS.md` - Model training results
- `docs/development/FINAL_MODEL.md` - Old model documentation
- `docs/development/IMPROVEMENTS.md` - Historical dev notes
- `docs/development/SESSION_SUMMARY.md` - Historical session notes
- `docs/development/PROJECT_PLAN.md` - Old project plan (superseded by ARCHITECTURE_V2.md)
- `docs/development/REUSABLE_COMPONENTS.md` - Old component plan
- `docs/development/PHONEME_EMBEDDINGS_FRAMEWORK.md` - Old framework docs
- `docs/development/LEARNING_DATASETS.md` - Dataset info (keep for reference)
- `research/*.md` - All research papers (historical value)

#### ❌ DELETE (Superseded/Obsolete)
- `docs/README.md` - Just links to other docs, not needed
- `docs/development/README.md` - Just directory index
- `webapp/BACKEND_COMPLETE.md` - Old backend status (superseded)

---

## Model Files

### Analysis of models/ Directory

#### ✅ KEEP (Current Production Models)
1. **models/hierarchical/final.pt**
   - Current production model for syllable embeddings
   - Used in phonological graph
   - **Status**: KEEP

2. **models/curriculum/phoible_initialized_final.pt**
   - Tuned phoneme embeddings (128-dim)
   - Used for contextual phoneme representations
   - **Status**: KEEP

3. **models/english_multitask/model.pt** (MAYBE)
   - 32-dim phoneme embeddings
   - Check if still used
   - **Status**: VERIFY USAGE, then KEEP or ARCHIVE

#### ⚠️ ARCHIVE (Experimental/Old Versions)
Move to `archive/models/`:
- `models/contextual_phoneme_embeddings/` - Superseded by curriculum models
- `models/curriculum/phase1_phoible_init.pt` - Training checkpoint (keep for reproducibility)
- `models/curriculum/phase2_mlm_finetuned.pt` - Training checkpoint
- `models/curriculum/phases34_final.pt` - Training checkpoint
- `models/english_phoneme_embeddings/` - Old experiment
- `models/mlm_only/` - Ablation experiment
- `models/phoneme_embeddings/` - Old training runs
- `models/phonolex_bert/` - Old experiment
- `models/phonolex_bert_quick/` - Old experiment
- `models/sequential/` - Old experiment
- `models/word_embeddings/` - Old experiment (64-dim word embeddings)
- `models/word_embeddings_sequential/` - Old experiment
- `models/word_embeddings_skipgram/` - Old experiment

#### ❌ DELETE (Not Used)
- None (archive instead for reproducibility)

---

## Cleanup Actions

### Phase 1: Create Archive Structure
```bash
mkdir -p archive/docs/development
mkdir -p archive/docs/research
mkdir -p archive/models
```

### Phase 2: Move Historical Docs
```bash
# Move old development docs
mv docs/VECTOR_ENCODING_STRATEGIES.md archive/docs/
mv docs/NORMALIZATION_ANALYSIS.md archive/docs/
mv docs/EDIT_DISTANCE_OPTIMIZATION.md archive/docs/
mv docs/HIERARCHICAL_SIMILARITY_OPTIMIZATION.md archive/docs/
mv docs/development/CURRICULUM_LEARNING_RESULTS.md archive/docs/development/
mv docs/development/FINAL_MODEL.md archive/docs/development/
mv docs/development/IMPROVEMENTS.md archive/docs/development/
mv docs/development/SESSION_SUMMARY.md archive/docs/development/
mv docs/development/PROJECT_PLAN.md archive/docs/development/
mv docs/development/REUSABLE_COMPONENTS.md archive/docs/development/
mv docs/development/PHONEME_EMBEDDINGS_FRAMEWORK.md archive/docs/development/

# Move research docs
mv research/*.md archive/docs/research/

# Remove empty READMEs
rm docs/README.md
rm docs/development/README.md
```

### Phase 3: Archive Old Models
```bash
# Archive experimental models
mv models/contextual_phoneme_embeddings archive/models/
mv models/english_phoneme_embeddings archive/models/
mv models/mlm_only archive/models/
mv models/phoneme_embeddings archive/models/
mv models/phonolex_bert archive/models/
mv models/phonolex_bert_quick archive/models/
mv models/sequential archive/models/
mv models/word_embeddings archive/models/
mv models/word_embeddings_sequential archive/models/
mv models/word_embeddings_skipgram archive/models/

# Archive curriculum checkpoints (keep final.pt in place)
mkdir -p archive/models/curriculum_checkpoints
mv models/curriculum/phase1_phoible_init.pt archive/models/curriculum_checkpoints/
mv models/curriculum/phase2_mlm_finetuned.pt archive/models/curriculum_checkpoints/
mv models/curriculum/phases34_final.pt archive/models/curriculum_checkpoints/
```

### Phase 4: Update Main Documentation
1. Update `README.md` to reference ARCHITECTURE_V2.md
2. Update `webapp/README.md` with v2.0 architecture
3. Create `docs/INDEX.md` with clear navigation
4. Remove `webapp/BACKEND_COMPLETE.md` (outdated)

### Phase 5: Create Clean Documentation Structure
```
docs/
├── ARCHITECTURE_V2.md          ← **PRIMARY REFERENCE**
├── INDEX.md                     ← Navigation guide
├── EMBEDDINGS.md                ← Consolidated embedding docs
├── DEPLOYMENT.md                ← Deployment guide (extracted from ARCHITECTURE_V2.md)
└── development/
    └── LEARNING_DATASETS.md     ← Dataset reference (keep)
```

---

## Post-Cleanup State

### Active Documentation
- `README.md` - Updated main README
- `docs/ARCHITECTURE_V2.md` - **Single source of truth** for architecture
- `docs/INDEX.md` - Navigation guide
- `docs/EMBEDDINGS.md` - Embedding documentation
- `docs/DEPLOYMENT.md` - Deployment guide
- `docs/development/LEARNING_DATASETS.md` - Dataset reference
- `webapp/README.md` - Updated webapp docs

### Active Models
```
models/
├── hierarchical/
│   └── final.pt                    ← Production model (syllable embeddings)
├── curriculum/
│   └── phoible_initialized_final.pt ← Production model (phoneme embeddings)
└── english_multitask/
    └── model.pt                     ← Verify if used, keep if yes
```

### Archive (Historical Reference)
```
archive/
├── docs/
│   ├── VECTOR_ENCODING_STRATEGIES.md
│   ├── NORMALIZATION_ANALYSIS.md
│   ├── EDIT_DISTANCE_OPTIMIZATION.md
│   ├── HIERARCHICAL_SIMILARITY_OPTIMIZATION.md
│   ├── development/
│   │   ├── CURRICULUM_LEARNING_RESULTS.md
│   │   ├── FINAL_MODEL.md
│   │   ├── IMPROVEMENTS.md
│   │   ├── SESSION_SUMMARY.md
│   │   ├── PROJECT_PLAN.md
│   │   ├── REUSABLE_COMPONENTS.md
│   │   └── PHONEME_EMBEDDINGS_FRAMEWORK.md
│   └── research/
│       ├── CROSS_PAPER_SYNTHESIS.md
│       ├── KEY_PAPERS_TO_DOWNLOAD.md
│       └── RESEARCH_SYNTHESIS.md
└── models/
    ├── contextual_phoneme_embeddings/
    ├── english_phoneme_embeddings/
    ├── mlm_only/
    ├── phoneme_embeddings/
    ├── phonolex_bert/
    ├── phonolex_bert_quick/
    ├── sequential/
    ├── word_embeddings/
    ├── word_embeddings_sequential/
    ├── word_embeddings_skipgram/
    └── curriculum_checkpoints/
        ├── phase1_phoible_init.pt
        ├── phase2_mlm_finetuned.pt
        └── phases34_final.pt
```

---

## Benefits

### Before Cleanup
- 15 documentation files (many outdated)
- 15 model directories (many experimental)
- Confusion about which docs to follow
- Unclear which models are production

### After Cleanup
- 6 core documentation files (all current)
- 2-3 production model directories
- Single source of truth: `ARCHITECTURE_V2.md`
- Clear separation: active vs. archive
- Easy onboarding for new developers

---

## Execution Checklist

- [ ] Review this plan
- [ ] Create archive directory structure
- [ ] Move historical docs to archive
- [ ] Archive old models (keep final production models)
- [ ] Update README.md
- [ ] Create docs/INDEX.md
- [ ] Create docs/EMBEDDINGS.md
- [ ] Create docs/DEPLOYMENT.md
- [ ] Update webapp/README.md
- [ ] Remove outdated status files
- [ ] Verify active models are correct
- [ ] Test that phonological graph still works
- [ ] Commit cleanup with clear message
- [ ] Update .gitignore if needed

---

**Ready to execute?** Review this plan, then we'll run the cleanup commands.
