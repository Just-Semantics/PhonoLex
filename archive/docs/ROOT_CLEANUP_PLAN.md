# Root Directory Cleanup Plan

## Files to Archive

### Temporary Status/Plan Documents (archive/docs/)
- ARCHIVE_PLAN.md → archive/docs/
- CLEANUP_PLAN.md → archive/docs/
- CORRECTED_V2_STATUS.md → archive/docs/
- V2_COMPLETE_SUMMARY.md → archive/docs/
- V2_FINAL_STATUS.md → archive/docs/
- V2_FIXES_SUMMARY.md → archive/docs/

### Analysis/Testing Scripts (keep in root - still useful)
- analyze_syllable_frequency.py ✓ Keep
- benchmark_edit_distance.py ✓ Keep
- check_graph_structure.py ✓ Keep
- test_all_models.py ✓ Keep
- test_optimizations.py ✓ Keep

### Demo Scripts (keep in root - still useful)
- demo_phonological_graph.py ✓ Keep
- demo_word_similarity.py ✓ Keep
- quick_demo.py ✓ Keep

### Canonical Files (KEEP IN ROOT)
- CLAUDE.md ✓ Keep (project instructions)
- README.md ✓ Keep (main readme)
- EMBEDDINGS_ARCHITECTURE.md ✓ Keep (canonical spec)
- TRAINING_SCRIPTS_SUMMARY.md ✓ Keep (quick reference)
- compute_layer1_phoible_features.py ✓ Keep (Layer 1)
- demo_layer2_normalized_vectors.py ✓ Keep (Layer 2)
- train_layer3_contextual_embeddings.py ✓ Keep (Layer 3)
- train_layer4_hierarchical.py ✓ Keep (Layer 4)

## Cleanup Commands

```bash
# Archive temporary status docs
mkdir -p archive/docs
mv ARCHIVE_PLAN.md archive/docs/
mv CLEANUP_PLAN.md archive/docs/
mv CORRECTED_V2_STATUS.md archive/docs/
mv V2_COMPLETE_SUMMARY.md archive/docs/
mv V2_FINAL_STATUS.md archive/docs/
mv V2_FIXES_SUMMARY.md archive/docs/

# Clean up this plan too
mv ROOT_CLEANUP_PLAN.md archive/docs/
```

## Final Root Structure

```
PhonoLex/
├── README.md                                  # Main readme
├── CLAUDE.md                                  # Project instructions
├── EMBEDDINGS_ARCHITECTURE.md                 # Canonical architecture spec
├── TRAINING_SCRIPTS_SUMMARY.md                # Quick reference
│
├── compute_layer1_phoible_features.py         # Layer 1
├── demo_layer2_normalized_vectors.py          # Layer 2
├── train_layer3_contextual_embeddings.py      # Layer 3 ⭐
├── train_layer4_hierarchical.py               # Layer 4 ⭐
│
├── demo_phonological_graph.py                 # Demos
├── demo_word_similarity.py
├── quick_demo.py
│
├── analyze_syllable_frequency.py              # Analysis tools
├── benchmark_edit_distance.py
├── check_graph_structure.py
├── test_all_models.py
├── test_optimizations.py
│
├── data/                                      # Data directory
├── models/                                    # Models directory
├── src/                                       # Source code
├── docs/                                      # Documentation
├── experiments/                               # Experiments
└── archive/                                   # Archived files
```

Clean, organized, and canonical!
