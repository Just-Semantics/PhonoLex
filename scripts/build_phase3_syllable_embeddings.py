#!/usr/bin/env python3
"""
Build Phase 3 syllable embeddings directly from Phase 2 Phoible features.

No training required - purely feature-based embeddings.
Users can adjust onset/nucleus/coda weights at query time in the UI.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify
from src.phonolex.word_filter import WordFilter


def build_phoible_syllable_embeddings(word, phonemes_with_stress, phoible_features):
    """
    Build syllable embeddings directly from Phase 2 Phoible features.

    Structure: onset(76) + nucleus(76) + coda(76) = 228-dim per syllable
    Each component normalized separately for equal weighting.
    """
    try:
        # Syllabify
        syllables = syllabify(phonemes_with_stress)
        if not syllables:
            return None

        # Extract plain phonemes
        word_phonemes = [p.phoneme for p in phonemes_with_stress]

        syllable_embeddings = []
        phoneme_idx = 0

        for syl in syllables:
            # Onset (76-dim) - normalize individually
            if syl.onset:
                onset_vecs = []
                for _ in syl.onset:
                    phoneme = word_phonemes[phoneme_idx]
                    if phoneme not in phoible_features:
                        return None  # Unknown phoneme
                    onset_vecs.append(phoible_features[phoneme])
                    phoneme_idx += 1

                onset_emb = np.mean(onset_vecs, axis=0)  # Average if multiple consonants
                onset_norm = np.linalg.norm(onset_emb)
                if onset_norm > 0:
                    onset_emb = onset_emb / onset_norm  # Normalize separately
            else:
                onset_emb = np.zeros(76)  # Zero vector for empty onset

            # Nucleus (76-dim) - normalize individually
            nucleus_phoneme = word_phonemes[phoneme_idx]
            if nucleus_phoneme not in phoible_features:
                return None
            nucleus_emb = phoible_features[nucleus_phoneme].copy()
            nucleus_norm = np.linalg.norm(nucleus_emb)
            if nucleus_norm > 0:
                nucleus_emb = nucleus_emb / nucleus_norm
            phoneme_idx += 1

            # Coda (76-dim) - normalize individually
            if syl.coda:
                coda_vecs = []
                for _ in syl.coda:
                    phoneme = word_phonemes[phoneme_idx]
                    if phoneme not in phoible_features:
                        return None
                    coda_vecs.append(phoible_features[phoneme])
                    phoneme_idx += 1

                coda_emb = np.mean(coda_vecs, axis=0)
                coda_norm = np.linalg.norm(coda_emb)
                if coda_norm > 0:
                    coda_emb = coda_emb / coda_norm
            else:
                coda_emb = np.zeros(76)  # Zero vector for empty coda

            # Concatenate: 76 + 76 + 76 = 228-dim syllable embedding
            syllable_emb = np.concatenate([onset_emb, nucleus_emb, coda_emb])
            syllable_embeddings.append(syllable_emb)

        return syllable_embeddings

    except Exception as e:
        return None


def main():
    print("=" * 80)
    print("Building Phase 3 Syllable Embeddings from Pure Phoible Features")
    print("=" * 80)
    print()
    print("Benefits:")
    print("  • No training required (instant)")
    print("  • Smaller embeddings (228-dim vs 384-dim per syllable)")
    print("  • Linguistically transparent (pure phonological features)")
    print("  • User-adjustable onset/nucleus/coda weights in UI")
    print()
    print("=" * 80)
    print()

    # Load Phase 2 Phoible features
    print("[1/4] Loading Phase 2 Phoible features...")
    phase2_path = Path('embeddings/phase2/normalized_76d.pkl')
    with open(phase2_path, 'rb') as f:
        phoible_features = pickle.load(f)
    print(f"✓ Loaded {len(phoible_features)} phoneme feature vectors (76-dim)")

    # Load CMU dictionary
    print("\n[2/4] Loading CMU dictionary...")
    loader = EnglishPhonologyLoader()
    print(f"✓ Loaded {len(loader.lexicon)} words")

    # Load word filter
    print("\n[3/4] Loading word filter...")
    word_filter = WordFilter()
    word_filter.load_all_norms()
    eligible_words = word_filter.get_eligible_words()
    print(f"✓ {len(eligible_words):,} words meet filtering criterion")

    # Build syllable embeddings
    print("\n[4/4] Computing syllable embeddings...")
    word_to_syllable_embeddings = {}
    skipped = 0

    for word, phonemes_with_stress in tqdm(loader.lexicon_with_stress.items(), desc="Processing words"):
        # Apply filter
        if word.lower() not in eligible_words:
            skipped += 1
            continue

        # Compute syllable embeddings
        syllable_embs = build_phoible_syllable_embeddings(word, phonemes_with_stress, phoible_features)

        if syllable_embs is not None:
            word_to_syllable_embeddings[word] = syllable_embs

    print(f"\n✓ Computed embeddings for {len(word_to_syllable_embeddings):,} words")
    print(f"  Filtered out: {skipped:,} words (no norms)")
    print(f"  Reduction: {100*skipped/(len(word_to_syllable_embeddings)+skipped):.1f}%")

    # Save embeddings
    print("\n[5/5] Saving embeddings...")
    output_path = Path("embeddings/phase3/syllable_embeddings_phoible.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    checkpoint = {
        'word_to_syllable_embeddings': word_to_syllable_embeddings,
        'source': 'phase2_phoible_features',
        'filter_criterion': 'frequency + at least one psycholinguistic norm',
        'num_words': len(word_to_syllable_embeddings),
        'embedding_dim': 228,
        'syllable_structure': 'onset(76) + nucleus(76) + coda(76)',
        'normalization': 'component-wise (onset, nucleus, coda normalized separately)',
        'weighting': 'user-adjustable at query time',
        'training': 'none (pure feature-based)',
    }

    torch.save(checkpoint, output_path)

    # Report file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved to {output_path}")
    print(f"  File size: {size_mb:.1f} MB")

    # Compare to MLM approach
    mlm_path = Path("embeddings/phase3/syllable_embeddings_filtered.pt")
    if mlm_path.exists():
        mlm_size_mb = mlm_path.stat().st_size / (1024 * 1024)
        print(f"\n  Comparison:")
        print(f"    MLM (384-dim):     {mlm_size_mb:.1f} MB")
        print(f"    Phoible (228-dim): {size_mb:.1f} MB")
        print(f"    Reduction: {100*(mlm_size_mb-size_mb)/mlm_size_mb:.1f}%")

    print("\n" + "=" * 80)
    print("✓ SUCCESS: Phoible-based Phase 3 embeddings created")
    print("=" * 80)

    # Show example
    print("\nExample words with embeddings:")
    for word in list(word_to_syllable_embeddings.keys())[:5]:
        num_syllables = len(word_to_syllable_embeddings[word])
        print(f"  {word}: {num_syllables} syllable(s), {num_syllables*228} dimensions total")

    print()
    print("Next steps:")
    print("1. Export client-side data:")
    print("   python scripts/export_clientside_data.py")
    print()
    print("2. Update frontend to expose onset/nucleus/coda weight controls")
    print()


if __name__ == "__main__":
    main()
