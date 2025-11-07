#!/usr/bin/env python3
"""
Export all data needed for client-side PhonoLex app.

This script creates a comprehensive data package for running PhonoLex entirely
in the browser, eliminating the need for backend servers and databases.

Outputs:
1. word_metadata.json - All word properties, IPA, syllables, psycholinguistic norms
2. embeddings.json - Phoible-based syllable embeddings (onset/nucleus/coda, 228-dim)
3. minimal_pairs.json - Precomputed minimal pairs relationships
4. phoneme_features.json - Phoneme inventory with Phoible features

Total size: ~50MB uncompressed, ~2MB gzipped (perfect for browser loading)
No quantization needed - Phoible embeddings compress extremely well!
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify


def load_filtered_embeddings():
    """Load the Phoible-based embeddings (no quantization needed)"""
    print("\n[1/5] Loading Phoible-based embeddings...")
    emb_path = project_root / "embeddings/phase3/syllable_embeddings_phoible.pt"

    checkpoint = torch.load(emb_path, map_location='cpu', weights_only=False)

    print(f"  ✓ Loaded {len(checkpoint['word_to_syllable_embeddings']):,} words")
    print(f"  ✓ Embedding dim: {checkpoint['embedding_dim']}")
    print(f"  ✓ Source: {checkpoint['source']}")
    print(f"  ✓ No quantization needed (small size + high compressibility)")

    return checkpoint


def load_word_metadata(filtered_words):
    """Load word metadata from CMU dictionary and psycholinguistic norms"""
    print("\n[2/5] Loading word metadata...")

    loader = EnglishPhonologyLoader()

    # Load psycholinguistic norms
    print("  Loading psycholinguistic norms...")
    norms = loader.load_psycholinguistic_properties()

    word_metadata = {}

    for word in tqdm(filtered_words, desc="  Processing words"):
        # Get CMU pronunciation with stress markers
        phonemes_with_stress = loader.lexicon_with_stress.get(word)
        if not phonemes_with_stress:
            continue

        # Get syllables
        syllables_list = syllabify(phonemes_with_stress)

        # Also get plain IPA phonemes for metadata
        ipa_phones = loader.lexicon.get(word, [])
        syllables_data = [
            {
                'onset': syl.onset,
                'nucleus': syl.nucleus,
                'coda': syl.coda
            }
            for syl in syllables_list
        ]

        # Get psycholinguistic properties
        word_norms = norms.get(word, {})

        word_metadata[word] = {
            'word': word,
            'ipa': ' '.join(ipa_phones),
            'phonemes': ipa_phones,
            'syllables': syllables_data,
            'phoneme_count': len(ipa_phones),
            'syllable_count': len(syllables_list),
            # Psycholinguistic norms
            'frequency': word_norms.get('frequency'),
            'log_frequency': word_norms.get('log_frequency'),
            'concreteness': word_norms.get('concreteness'),
            'aoa': word_norms.get('aoa'),
            'imageability': word_norms.get('imageability'),
            'familiarity': word_norms.get('familiarity'),
            'valence': word_norms.get('valence'),
            'arousal': word_norms.get('arousal'),
            'dominance': word_norms.get('dominance'),
        }

    print(f"  ✓ Processed {len(word_metadata):,} words with metadata")
    return word_metadata


def compute_minimal_pairs(word_metadata):
    """Compute minimal pairs for phoneme contrasts"""
    print("\n[3/5] Computing minimal pairs...")

    # Group words by phoneme length for efficiency
    by_length = defaultdict(list)
    for word, data in word_metadata.items():
        length = data['phoneme_count']
        by_length[length].append(word)

    minimal_pairs = []

    for length, words in tqdm(by_length.items(), desc="  By length"):
        if length < 2:  # Skip single-phoneme words
            continue

        # Compare all pairs of same length
        for i, word1 in enumerate(words):
            phonemes1 = word_metadata[word1]['phonemes']

            for word2 in words[i+1:]:
                phonemes2 = word_metadata[word2]['phonemes']

                # Count differences
                diff_count = sum(p1 != p2 for p1, p2 in zip(phonemes1, phonemes2))

                if diff_count == 1:
                    # Find the position of difference
                    diff_pos = next(i for i, (p1, p2) in enumerate(zip(phonemes1, phonemes2)) if p1 != p2)

                    minimal_pairs.append({
                        'word1': word1,
                        'word2': word2,
                        'position': diff_pos,
                        'phoneme1': phonemes1[diff_pos],
                        'phoneme2': phonemes2[diff_pos]
                    })

    print(f"  ✓ Found {len(minimal_pairs):,} minimal pairs")
    return minimal_pairs


def load_phoneme_features():
    """Load phoneme features from existing phonemes.json"""
    print("\n[4/5] Loading phoneme features...")

    # Load from the already-exported phonemes.json in public/data
    features_path = project_root / "webapp/frontend/public/data/phonemes.json"

    with open(features_path, 'r') as f:
        phonemes_data = json.load(f)

    # Convert from array format to dict format (ipa -> features)
    features_dict = {}
    for phoneme in phonemes_data['phonemes']:
        features_dict[phoneme['ipa']] = phoneme

    print(f"  ✓ Loaded features for {len(features_dict):,} phonemes")
    return features_dict


def export_data(embeddings_checkpoint, word_metadata, minimal_pairs, phoneme_features, output_dir):
    """Export all data to files and compress them"""
    import gzip

    print("\n[5/5] Exporting data...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export word metadata as JSON
    metadata_path = output_dir / "word_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(word_metadata, f, separators=(',', ':'))
    size_mb = metadata_path.stat().st_size / 1024 / 1024
    print(f"  ✓ word_metadata.json ({size_mb:.1f} MB)")

    # 2. Export embeddings in browser-friendly format
    # No quantization needed - Phoible embeddings are already small and compress well
    embeddings_path = output_dir / "embeddings.json"

    # Convert numpy arrays to lists for JSON serialization
    embeddings_json = {}
    for word, syllable_arrays in embeddings_checkpoint['word_to_syllable_embeddings'].items():
        embeddings_json[word] = [arr.tolist() for arr in syllable_arrays]

    export_data = {
        'embeddings': embeddings_json,
        'embedding_dim': int(embeddings_checkpoint['embedding_dim']),
        'syllable_structure': embeddings_checkpoint['syllable_structure'],
        'source': embeddings_checkpoint['source'],
        'normalization': embeddings_checkpoint['normalization'],
    }

    with open(embeddings_path, 'w') as f:
        json.dump(export_data, f, separators=(',', ':'))
    size_mb = embeddings_path.stat().st_size / 1024 / 1024
    print(f"  ✓ embeddings.json ({size_mb:.1f} MB)")

    # 3. Export minimal pairs
    pairs_path = output_dir / "minimal_pairs.json"
    with open(pairs_path, 'w') as f:
        json.dump(minimal_pairs, f, separators=(',', ':'))
    size_mb = pairs_path.stat().st_size / 1024 / 1024
    print(f"  ✓ minimal_pairs.json ({size_mb:.1f} MB)")

    # 4. Export phoneme features
    features_path = output_dir / "phoneme_features.json"
    with open(features_path, 'w') as f:
        json.dump(phoneme_features, f, separators=(',', ':'))
    size_mb = features_path.stat().st_size / 1024 / 1024
    print(f"  ✓ phoneme_features.json ({size_mb:.1f} MB)")

    # 5. Create manifest
    manifest = {
        'version': '2.0.0',
        'created': str(Path(__file__).stat().st_mtime),
        'vocabulary_size': len(word_metadata),
        'minimal_pairs_count': len(minimal_pairs),
        'phoneme_count': len(phoneme_features),
        'filter_criterion': 'frequency + at least one psycholinguistic norm',
        'files': {
            'word_metadata.json': 'Word properties, IPA, syllables, psycholinguistic norms',
            'embeddings.json': 'Phoible-based syllable embeddings (onset/nucleus/coda)',
            'minimal_pairs.json': 'Precomputed minimal pair relationships',
            'phoneme_features.json': 'Phoneme inventory with Phoible features'
        }
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ manifest.json")

    # Calculate total size before compression
    total_size = sum(p.stat().st_size for p in output_dir.glob("*.json")) / 1024 / 1024
    print(f"\n  Total uncompressed size: {total_size:.1f} MB")

    # Compress all JSON files
    print("\n  Compressing files...")
    for json_file in output_dir.glob("*.json"):
        gz_file = Path(str(json_file) + '.gz')
        with open(json_file, 'rb') as f_in:
            with gzip.open(gz_file, 'wb', compresslevel=9) as f_out:
                f_out.writelines(f_in)

        original_size = json_file.stat().st_size / 1024 / 1024
        compressed_size = gz_file.stat().st_size / 1024 / 1024
        ratio = (1 - compressed_size / original_size) * 100
        print(f"    {json_file.name} → {gz_file.name} ({compressed_size:.1f} MB, {ratio:.0f}% reduction)")

        # Delete large uncompressed files (>50MB GitHub limit) - only keep .gz
        if original_size > 50 and json_file.name == 'embeddings.json':
            json_file.unlink()
            print(f"    Deleted uncompressed {json_file.name} (exceeds 50MB GitHub limit)")

    # Calculate compressed total
    total_compressed = sum(p.stat().st_size for p in output_dir.glob("*.json.gz")) / 1024 / 1024
    total_ratio = (1 - total_compressed / total_size) * 100
    print(f"\n  Total compressed size: {total_compressed:.1f} MB ({total_ratio:.0f}% reduction)")


def main():
    print("=" * 80)
    print("PhonoLex Client-Side Data Export")
    print("=" * 80)

    # Load filtered embeddings
    embeddings_checkpoint = load_filtered_embeddings()
    filtered_words = sorted(embeddings_checkpoint['word_to_syllable_embeddings'].keys())

    # Load word metadata
    word_metadata = load_word_metadata(filtered_words)

    # Compute minimal pairs
    minimal_pairs = compute_minimal_pairs(word_metadata)

    # Load phoneme features
    phoneme_features = load_phoneme_features()

    # Export everything
    output_dir = project_root / "webapp/frontend/public/data"
    export_data(embeddings_checkpoint, word_metadata, minimal_pairs, phoneme_features, output_dir)

    print("\n" + "=" * 80)
    print("✓ SUCCESS: Client-side data package created!")
    print("=" * 80)
    print(f"\nData exported to: {output_dir}")
    print("\nNext steps:")
    print("1. Update frontend to load data from public/data/*.json")
    print("2. Implement client-side similarity computation using quantized embeddings")
    print("3. Remove backend API dependencies")
    print("4. Deploy as static site (Netlify, Cloudflare Pages, etc.)")


if __name__ == "__main__":
    main()
