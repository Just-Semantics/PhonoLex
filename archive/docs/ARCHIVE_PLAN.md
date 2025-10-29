# Files to Archive

## Old Training Scripts (Superseded by Canonical Versions)

### To Archive:
1. **train_hierarchical_final.py** 
   - Superseded by: train_layer4_hierarchical.py
   - Issue: Trained Layer 3 from scratch instead of using pre-trained

2. **train_sequential_all_datasets.py**
   - Superseded by: train_layer3_contextual_embeddings.py
   - Issue: Used all datasets including SIGMORPHON/UniMorph (not needed)

3. **train_sequential_cmu_ipadict.py**
   - Superseded by: train_layer3_contextual_embeddings.py
   - Issue: This WAS the correct version, but now copied to canonical name

### Keep (Still Useful):
- **demo_phonological_graph.py** - Still useful demo
- **demo_word_similarity.py** - Still useful demo  
- **quick_demo.py** - Quick testing
- **test_all_models.py** - Model testing utility
- **test_optimizations.py** - Optimization verification
- **check_graph_structure.py** - Graph validation
- **analyze_syllable_frequency.py** - Analysis tool
- **benchmark_edit_distance.py** - Benchmarking tool

## Archive Location
Move to: `archive/training_scripts_old/`

## Command
```bash
mkdir -p archive/training_scripts_old
mv train_hierarchical_final.py archive/training_scripts_old/
mv train_sequential_all_datasets.py archive/training_scripts_old/
mv train_sequential_cmu_ipadict.py archive/training_scripts_old/
```
