#!/usr/bin/env python3
"""
Build filtered Layer 4 syllable embeddings with vocabulary reduction.

This script creates syllable embeddings for only the filtered vocabulary
(words with frequency + at least one other psycholinguistic norm).

Reduces from 125K words → 24K words (80% reduction, 1.0GB → 0.2GB).
"""

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify
from src.phonolex.word_filter import WordFilter


def load_layer3_model(model_path="models/layer3/model.pt"):
    """Load trained Layer 3 contextual phoneme model"""
    print(f"Loading Layer 3 model from {model_path}...")

    # Import model class
    from src.phonolex.models.phonolex_bert import PhonoLexBERT

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Get model configuration
    phoneme_to_id = checkpoint['phoneme_to_id']
    num_phonemes = len(phoneme_to_id)

    # Get max_len from checkpoint if available
    state_dict = checkpoint['model_state_dict']
    pe_shape = state_dict['pos_encoder.pe'].shape
    max_len = pe_shape[0]  # Extract from saved positional encoding

    # Model configuration (from training)
    model = PhonoLexBERT(
        num_phonemes=num_phonemes,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        max_len=max_len
    )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ Model loaded ({num_phonemes} phonemes, 128-dim embeddings)")
    return model, phoneme_to_id


def compute_syllable_embeddings_for_word(phonemes_with_stress, model, phoneme_to_id):
    """
    Compute syllable embeddings for a word using Layer 3 model.

    Args:
        phonemes_with_stress: List of PhonemeWithStress objects
        model: Trained PhonoLexBERT model
        phoneme_to_id: Dict mapping IPA to model indices

    Returns:
        List of numpy arrays (one per syllable, each 384-dim)
        Returns None if word cannot be processed
    """
    try:
        # Syllabify word (needs PhonemeWithStress objects)
        syllables = syllabify(phonemes_with_stress)
        if not syllables:
            return None

        # Extract plain phonemes for model input
        word_phonemes = [p.phoneme for p in phonemes_with_stress]

        # Convert phonemes to indices
        phoneme_indices = []
        for phoneme in word_phonemes:
            if phoneme not in phoneme_to_id:
                return None  # Unknown phoneme
            phoneme_indices.append(phoneme_to_id[phoneme])

        # Create input tensor and attention mask
        input_tensor = torch.tensor([phoneme_indices], dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor)  # All positions are valid

        # Forward pass to get contextual phoneme embeddings
        with torch.no_grad():
            _, _, phoneme_embeddings = model(input_tensor, attention_mask)  # Shape: (1, num_phonemes, 128)

        # Remove batch dimension
        phoneme_embeddings = phoneme_embeddings.squeeze(0)  # Shape: (num_phonemes, 128)

        # Aggregate phoneme embeddings by syllable structure
        syllable_embeddings = []
        phoneme_idx = 0

        for syl in syllables:
            # Onset (128-dim)
            if syl.onset:
                onset_embs = phoneme_embeddings[phoneme_idx:phoneme_idx+len(syl.onset)]
                onset_emb = onset_embs.mean(dim=0)  # Average if multiple consonants
                phoneme_idx += len(syl.onset)
            else:
                onset_emb = torch.zeros(128)  # Zero vector for empty onset

            # Nucleus (128-dim)
            nucleus_emb = phoneme_embeddings[phoneme_idx]
            phoneme_idx += 1

            # Coda (128-dim)
            if syl.coda:
                coda_embs = phoneme_embeddings[phoneme_idx:phoneme_idx+len(syl.coda)]
                coda_emb = coda_embs.mean(dim=0)  # Average if multiple consonants
                phoneme_idx += len(syl.coda)
            else:
                coda_emb = torch.zeros(128)  # Zero vector for empty coda

            # Concatenate: [onset(128) + nucleus(128) + coda(128)] = 384-dim
            syllable_emb = torch.cat([onset_emb, nucleus_emb, coda_emb])

            # Normalize to unit length
            syllable_emb = syllable_emb / (syllable_emb.norm() + 1e-8)

            syllable_embeddings.append(syllable_emb.numpy())

        return syllable_embeddings

    except Exception as e:
        # Silently skip words with errors
        return None


def main():
    print("=" * 80)
    print("Building Filtered Layer 4 Syllable Embeddings")
    print("=" * 80)

    # Load CMU dictionary
    print("\n[1/5] Loading CMU dictionary...")
    loader = EnglishPhonologyLoader()
    print(f"✓ Loaded {len(loader.lexicon)} words")

    # Load word filter
    print("\n[2/5] Loading word filter...")
    word_filter = WordFilter()
    word_filter.load_all_norms()
    eligible_words = word_filter.get_eligible_words()
    print(f"✓ {len(eligible_words):,} words meet filtering criterion")

    # Load Layer 3 model
    print("\n[3/5] Loading Layer 3 contextual phoneme model...")
    model, phoneme_to_id = load_layer3_model()

    # Build syllable embeddings
    print("\n[4/5] Computing syllable embeddings...")
    word_to_syllable_embeddings = {}
    skipped = 0

    for word, phonemes_with_stress in tqdm(loader.lexicon_with_stress.items(), desc="Processing words"):
        # Apply filter
        if word.lower() not in eligible_words:
            skipped += 1
            continue

        # Compute syllable embeddings (pass phonemes_with_stress for syllabification)
        syllable_embs = compute_syllable_embeddings_for_word(phonemes_with_stress, model, phoneme_to_id)

        if syllable_embs is not None:
            word_to_syllable_embeddings[word] = syllable_embs

    print(f"\n✓ Computed embeddings for {len(word_to_syllable_embeddings):,} words")
    print(f"  Filtered out: {skipped:,} words (no norms)")
    print(f"  Reduction: {100*skipped/(len(word_to_syllable_embeddings)+skipped):.1f}%")

    # Save embeddings
    print("\n[5/5] Saving filtered embeddings...")
    output_path = Path("embeddings/layer4/syllable_embeddings_filtered.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'word_to_syllable_embeddings': word_to_syllable_embeddings,
        'model_path': 'models/layer3/model.pt',
        'filter_criterion': 'frequency + at least one psycholinguistic norm',
        'num_words': len(word_to_syllable_embeddings),
        'embedding_dim': 384,
        'syllable_structure': 'onset(128) + nucleus(128) + coda(128)'
    }

    torch.save(checkpoint, output_path)

    # Report file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved to {output_path}")
    print(f"  File size: {size_mb:.1f} MB")

    # Compare to unfiltered
    print(f"\n  Comparison:")
    print(f"    Original (125K words): ~1.0 GB")
    print(f"    Filtered ({len(word_to_syllable_embeddings)/1000:.0f}K words): ~{size_mb:.0f} MB")
    print(f"    Reduction: ~{100*(1000-size_mb)/1000:.0f}%")

    print("\n" + "=" * 80)
    print("✓ SUCCESS: Filtered Layer 4 embeddings created")
    print("=" * 80)

    # Show example
    print("\nExample words with embeddings:")
    for word in list(word_to_syllable_embeddings.keys())[:5]:
        num_syllables = len(word_to_syllable_embeddings[word])
        print(f"  {word}: {num_syllables} syllable(s), {num_syllables*384} dimensions total")


if __name__ == "__main__":
    main()
