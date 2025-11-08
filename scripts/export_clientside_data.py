#!/usr/bin/env python3
"""
Export all data needed for client-side PhonoLex app.

This script creates a comprehensive data package for running PhonoLex entirely
in the browser, eliminating the need for backend servers and databases.

Outputs (English-specific):
1. word_metadata.json - All word properties, IPA, syllables, psycholinguistic norms (24,744 words)
2. embeddings.json - Phoible-based syllable embeddings (onset/nucleus/coda, 228-dim)
3. minimal_pairs.json - Precomputed minimal pairs relationships
4. phoneme_features.json - Phoneme inventory with Phoible features

Outputs (Cross-linguistic for difficulty tool):
5. phoible_phonemes.json - 3,142 phonemes with 76-dim feature vectors (all languages)
6. phoible_languages.json - 2,095 language phoneme inventories

Total size: ~58MB uncompressed, ~2.5MB gzipped (perfect for browser loading)
No quantization needed - Phoible embeddings compress extremely well!
"""

import sys
import json
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify


def compute_wcm_score(phonemes, syllables):
    """
    Compute Word Complexity Measure (WCM) score.

    8 parameters from Stoel-Gammon (2010):
    1. >2 syllables: +1
    2. Non-initial stress: +1
    3. Word-final consonant: +1
    4. Consonant cluster: +1 per cluster
    5. Velar: +1 per velar
    6. Liquid/rhotic: +1 each
    7. Fricative/affricate: +1 each
    8. Voiced fricative/affricate: +1 additional
    """
    # Define phoneme categories
    velars = {"k", "g", "ŋ"}
    liquids_rhotics = {"l", "ɹ", "r", "ɚ", "ɝ"}
    fricatives_affricates = {"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "tʃ", "dʒ"}
    voiced_fric_affric = {"v", "ð", "z", "ʒ", "dʒ"}
    vowels = {
        "i",
        "ɪ",
        "e",
        "ɛ",
        "æ",
        "ɑ",
        "ɔ",
        "o",
        "ʊ",
        "u",
        "ʌ",
        "ə",
        "ɚ",
        "ɝ",
        "eɪ",
        "aɪ",
        "ɔɪ",
        "aʊ",
        "oʊ",
    }

    score = 0

    # 1. More than 2 syllables
    if len(syllables) > 2:
        score += 1

    # 2. Non-initial stress (get stress from syllable objects)
    stress_positions = [
        i for i, syl in enumerate(syllables) if getattr(syl, "stress", 0) in [1, 2]
    ]
    if stress_positions and stress_positions[0] > 0:
        score += 1

    # 3. Word-final consonant
    if phonemes and phonemes[-1] not in vowels:
        score += 1

    # 4. Consonant clusters (onset or coda with 2+ consonants)
    for syl in syllables:
        if len(syl.onset) >= 2:
            score += 1
        if len(syl.coda) >= 2:
            score += 1

    # 5-8. Sound class counts
    for p in phonemes:
        # Strip stress markers for classification
        p_base = p.replace("ˈ", "").replace("ˌ", "")

        if p_base in velars:
            score += 1
        if p_base in liquids_rhotics:
            score += 1
        if p_base in fricatives_affricates:
            score += 1
        if p_base in voiced_fric_affric:
            score += 1  # Additional point for voiced

    return score


def load_filtered_embeddings():
    """Load the Phoible-based embeddings (no quantization needed)"""
    print("\n[1/5] Loading Phoible-based embeddings...")
    emb_path = project_root / "embeddings/phase3/syllable_embeddings_phoible.pt"

    checkpoint = torch.load(emb_path, map_location="cpu", weights_only=False)

    print(f"  ✓ Loaded {len(checkpoint['word_to_syllable_embeddings']):,} words")
    print(f"  ✓ Embedding dim: {checkpoint['embedding_dim']}")
    print(f"  ✓ Source: {checkpoint['source']}")
    print("  ✓ No quantization needed (small size + high compressibility)")

    return checkpoint


def compute_percentiles(word_metadata):
    """Compute percentiles for all numeric properties"""
    print("\n  Computing percentiles for all properties...")

    # Properties to compute percentiles for
    properties = [
        # Phonological
        "syllable_count",
        "phoneme_count",
        "wcm_score",
        # Phonotactic
        "phono_prob_avg",
        "phono_prob_sum_log",
        "positional_prob_avg",
        # Lexical
        "frequency",
        "aoa",
        # Semantic
        "imageability",
        "familiarity",
        "concreteness",
        # Affective
        "valence",
        "arousal",
        "dominance",
    ]

    # For each property, collect all non-null values and sort them
    property_values = {}
    for prop in properties:
        values = []
        for word_data in word_metadata.values():
            val = word_data.get(prop)
            if val is not None:
                values.append(val)

        if values:
            values.sort()
            property_values[prop] = values
            print(f"    {prop}: {len(values):,} values")

    # Now compute percentile for each word
    for word, word_data in word_metadata.items():
        for prop in properties:
            val = word_data.get(prop)
            if val is not None and prop in property_values:
                sorted_vals = property_values[prop]
                # Find percentile rank (0-100)
                # Use bisect to find insertion point
                import bisect
                idx = bisect.bisect_left(sorted_vals, val)
                percentile = (idx / len(sorted_vals)) * 100
                word_data[f"{prop}_percentile"] = round(percentile, 1)
            else:
                word_data[f"{prop}_percentile"] = None

    print(f"  ✓ Computed percentiles for {len(properties)} properties")


def load_word_metadata(filtered_words):
    """Load word metadata from CMU dictionary and psycholinguistic norms"""
    print("\n[2/5] Loading word metadata...")

    loader = EnglishPhonologyLoader()

    # Load psycholinguistic norms
    print("  Loading psycholinguistic norms...")
    norms = loader.load_psycholinguistic_properties()

    # Load phonotactic probability
    print("  Loading phonotactic probability...")
    phono_prob_path = project_root / "data/phonotactic_probability_24k.json"
    with open(phono_prob_path, "r") as f:
        phono_prob_data = json.load(f)
    phono_probs = phono_prob_data["word_probabilities"]
    print(f"  ✓ Loaded phonotactic probability for {len(phono_probs):,} words")

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
            {"onset": syl.onset, "nucleus": syl.nucleus, "coda": syl.coda}
            for syl in syllables_list
        ]

        # Get psycholinguistic properties
        word_norms = norms.get(word, {})

        # Get phonotactic probability
        word_phono_prob = phono_probs.get(word, {})

        # Compute clinical measures (WCM)
        wcm_score = compute_wcm_score(ipa_phones, syllables_list)

        word_metadata[word] = {
            "word": word,
            "ipa": " ".join(ipa_phones),
            "phonemes": ipa_phones,
            "syllables": syllables_data,
            "phoneme_count": len(ipa_phones),
            "syllable_count": len(syllables_list),
            # Clinical measures
            "wcm_score": wcm_score,
            # Psycholinguistic norms
            "frequency": word_norms.get("frequency"),
            "log_frequency": word_norms.get("log_frequency"),
            "concreteness": word_norms.get("concreteness"),
            "aoa": word_norms.get("aoa"),
            "imageability": word_norms.get("imageability"),
            "familiarity": word_norms.get("familiarity"),
            "valence": word_norms.get("valence"),
            "arousal": word_norms.get("arousal"),
            "dominance": word_norms.get("dominance"),
            # Phonotactic probability (Vitevitch & Luce 2004)
            "phono_prob_avg": word_phono_prob.get("phono_prob_avg"),
            "phono_prob_sum_log": word_phono_prob.get("phono_prob_sum_log"),
            "positional_prob_avg": word_phono_prob.get("positional_prob_avg"),
        }

    # Compute percentiles for all properties
    compute_percentiles(word_metadata)

    print(f"  ✓ Processed {len(word_metadata):,} words with metadata and percentiles")
    return word_metadata


def compute_minimal_pairs(word_metadata):
    """Compute minimal pairs for phoneme contrasts"""
    print("\n[3/5] Computing minimal pairs...")

    # Group words by phoneme length for efficiency
    by_length = defaultdict(list)
    for word, data in word_metadata.items():
        length = data["phoneme_count"]
        by_length[length].append(word)

    minimal_pairs = []

    for length, words in tqdm(by_length.items(), desc="  By length"):
        if length < 2:  # Skip single-phoneme words
            continue

        # Compare all pairs of same length
        for i, word1 in enumerate(words):
            phonemes1 = word_metadata[word1]["phonemes"]

            for word2 in words[i + 1 :]:
                phonemes2 = word_metadata[word2]["phonemes"]

                # Count differences
                diff_count = sum(p1 != p2 for p1, p2 in zip(phonemes1, phonemes2))

                if diff_count == 1:
                    # Find the position of difference
                    diff_pos = next(
                        i
                        for i, (p1, p2) in enumerate(zip(phonemes1, phonemes2))
                        if p1 != p2
                    )

                    minimal_pairs.append(
                        {
                            "word1": word1,
                            "word2": word2,
                            "position": diff_pos,
                            "phoneme1": phonemes1[diff_pos],
                            "phoneme2": phonemes2[diff_pos],
                        }
                    )

    print(f"  ✓ Found {len(minimal_pairs):,} minimal pairs")
    return minimal_pairs


def load_phoneme_features():
    """Load phoneme features from existing phonemes.json"""
    print("\n[4/5] Loading phoneme features...")

    # Load from the already-exported phonemes.json in public/data
    features_path = project_root / "webapp/frontend/public/data/phonemes.json"

    with open(features_path, "r") as f:
        phonemes_data = json.load(f)

    # Convert from array format to dict format (ipa -> features)
    features_dict = {}
    for phoneme in phonemes_data["phonemes"]:
        features_dict[phoneme["ipa"]] = phoneme

    print(f"  ✓ Loaded features for {len(features_dict):,} phonemes")
    return features_dict


def load_phoible_data():
    """Load PHOIBLE phoneme vectors and language inventories for difficulty tool"""
    print("\n[5/6] Loading PHOIBLE data for difficulty tool...")
    import pickle

    # Load 76-dim vectors (Phase 2)
    vectors_path = project_root / "embeddings/phase2/phoible_all_phonemes_76d.pkl"
    with open(vectors_path, "rb") as f:
        vectors_76d = pickle.load(f)

    # Load language inventories
    inventories_path = (
        project_root / "embeddings/phase2/phoible_language_inventories.json"
    )
    with open(inventories_path, "r") as f:
        language_inventories = json.load(f)

    # Load phoneme metadata
    metadata_path = project_root / "embeddings/phase2/phoible_phoneme_metadata.json"
    with open(metadata_path, "r") as f:
        phoneme_metadata = json.load(f)

    print(f"  ✓ Loaded {len(vectors_76d):,} phoneme vectors")
    print(f"  ✓ Loaded {len(language_inventories):,} language inventories")
    print(f"  ✓ Loaded metadata for {len(phoneme_metadata):,} phonemes")

    # Convert numpy arrays to lists for JSON
    vectors_json = {}
    for phoneme, vec in vectors_76d.items():
        vectors_json[phoneme] = vec.tolist()

    return {
        "phonemes": vectors_json,
        "languages": language_inventories,
        "metadata": phoneme_metadata,
    }



def export_data(
    embeddings_checkpoint,
    word_metadata,
    minimal_pairs,
    phoneme_features,
    phoible_data,
    output_dir,
):
    """Export all data to files and compress them"""
    import gzip

    print("\n[6/6] Exporting data...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export word metadata as JSON
    metadata_path = output_dir / "word_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(word_metadata, f, separators=(",", ":"))
    size_mb = metadata_path.stat().st_size / 1024 / 1024
    print(f"  ✓ word_metadata.json ({size_mb:.1f} MB)")

    # 2. Export embeddings in browser-friendly format
    # No quantization needed - Phoible embeddings are already small and compress well
    embeddings_path = output_dir / "embeddings.json"

    # Convert numpy arrays to lists for JSON serialization
    embeddings_json = {}
    for word, syllable_arrays in embeddings_checkpoint[
        "word_to_syllable_embeddings"
    ].items():
        embeddings_json[word] = [arr.tolist() for arr in syllable_arrays]

    export_data = {
        "embeddings": embeddings_json,
        "embedding_dim": int(embeddings_checkpoint["embedding_dim"]),
        "syllable_structure": embeddings_checkpoint["syllable_structure"],
        "source": embeddings_checkpoint["source"],
        "normalization": embeddings_checkpoint["normalization"],
    }

    with open(embeddings_path, "w") as f:
        json.dump(export_data, f, separators=(",", ":"))
    size_mb = embeddings_path.stat().st_size / 1024 / 1024
    print(f"  ✓ embeddings.json ({size_mb:.1f} MB)")

    # 3. Export minimal pairs
    pairs_path = output_dir / "minimal_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(minimal_pairs, f, separators=(",", ":"))
    size_mb = pairs_path.stat().st_size / 1024 / 1024
    print(f"  ✓ minimal_pairs.json ({size_mb:.1f} MB)")

    # 4. Export phoneme features
    features_path = output_dir / "phoneme_features.json"
    with open(features_path, "w") as f:
        json.dump(phoneme_features, f, separators=(",", ":"))
    size_mb = features_path.stat().st_size / 1024 / 1024
    print(f"  ✓ phoneme_features.json ({size_mb:.1f} MB)")

    # 5. Export PHOIBLE phoneme vectors
    phoible_phonemes_path = output_dir / "phoible_phonemes.json"
    phoible_phonemes_export = {
        "phonemes": phoible_data["phonemes"],
        "metadata": phoible_data["metadata"],
        "vector_dim": 76,
        "source": "PHOIBLE + Phase 2 normalized vectors",
        "description": "3,142 phonemes from 2,095 languages with 76-dim feature vectors",
    }
    with open(phoible_phonemes_path, "w") as f:
        json.dump(phoible_phonemes_export, f, separators=(",", ":"))
    size_mb = phoible_phonemes_path.stat().st_size / 1024 / 1024
    print(f"  ✓ phoible_phonemes.json ({size_mb:.1f} MB)")

    # 6. Export PHOIBLE language inventories
    phoible_languages_path = output_dir / "phoible_languages.json"
    phoible_languages_export = {
        "languages": phoible_data["languages"],
        "language_count": len(phoible_data["languages"]),
        "description": "2,095 language phoneme inventories from PHOIBLE",
    }
    with open(phoible_languages_path, "w") as f:
        json.dump(phoible_languages_export, f, separators=(",", ":"))
    size_mb = phoible_languages_path.stat().st_size / 1024 / 1024
    print(f"  ✓ phoible_languages.json ({size_mb:.1f} MB)")

    # 7. Create manifest
    manifest = {
        "version": "2.3.0",
        "created": str(Path(__file__).stat().st_mtime),
        "vocabulary_size": len(word_metadata),
        "minimal_pairs_count": len(minimal_pairs),
        "phoneme_count": len(phoneme_features),
        "phoible_phoneme_count": len(phoible_data["phonemes"]),
        "phoible_language_count": len(phoible_data["languages"]),
        "filter_criterion": "frequency + at least one psycholinguistic norm",
        "files": {
            "word_metadata.json": "Word properties, IPA, syllables, psycholinguistic norms (English)",
            "embeddings.json": "Phoible-based syllable embeddings (onset/nucleus/coda, English)",
            "minimal_pairs.json": "Precomputed minimal pair relationships (English)",
            "phoneme_features.json": "Phoneme inventory with Phoible features (English)",
            "phoible_phonemes.json": "3,142 phonemes with 76-dim vectors (all languages)",
            "phoible_languages.json": "2,095 language phoneme inventories (all languages)",
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("  ✓ manifest.json")

    # Calculate total size before compression
    total_size = sum(p.stat().st_size for p in output_dir.glob("*.json")) / 1024 / 1024
    print(f"\n  Total uncompressed size: {total_size:.1f} MB")

    # Compress all JSON files
    print("\n  Compressing files...")
    for json_file in output_dir.glob("*.json"):
        gz_file = Path(str(json_file) + ".gz")
        with open(json_file, "rb") as f_in:
            with gzip.open(gz_file, "wb", compresslevel=9) as f_out:
                f_out.writelines(f_in)

        original_size = json_file.stat().st_size / 1024 / 1024
        compressed_size = gz_file.stat().st_size / 1024 / 1024
        ratio = (1 - compressed_size / original_size) * 100
        print(
            f"    {json_file.name} → {gz_file.name} ({compressed_size:.1f} MB, {ratio:.0f}% reduction)"
        )

        # Delete large uncompressed files (>50MB GitHub limit) - only keep .gz
        if original_size > 50 and json_file.name == "embeddings.json":
            json_file.unlink()
            print(
                f"    Deleted uncompressed {json_file.name} (exceeds 50MB GitHub limit)"
            )

    # Calculate compressed total
    total_compressed = (
        sum(p.stat().st_size for p in output_dir.glob("*.json.gz")) / 1024 / 1024
    )
    total_ratio = (1 - total_compressed / total_size) * 100
    print(
        f"\n  Total compressed size: {total_compressed:.1f} MB ({total_ratio:.0f}% reduction)"
    )


def main():
    print("=" * 80)
    print("PhonoLex Client-Side Data Export")
    print("=" * 80)

    # Load filtered embeddings
    embeddings_checkpoint = load_filtered_embeddings()
    filtered_words = sorted(embeddings_checkpoint["word_to_syllable_embeddings"].keys())

    # Load word metadata
    word_metadata = load_word_metadata(filtered_words)

    # Compute minimal pairs
    minimal_pairs = compute_minimal_pairs(word_metadata)

    # Load phoneme features
    phoneme_features = load_phoneme_features()

    # Load PHOIBLE data for difficulty tool
    phoible_data = load_phoible_data()

    # Export everything
    output_dir = project_root / "webapp/frontend/public/data"
    export_data(
        embeddings_checkpoint,
        word_metadata,
        minimal_pairs,
        phoneme_features,
        phoible_data,
        output_dir,
    )

    print("\n" + "=" * 80)
    print("✓ SUCCESS: Client-side data package created!")
    print("=" * 80)
    print(f"\nData exported to: {output_dir}")
    print("\nNew in v2.3:")
    print("  • 3,142 phonemes with 76-dim vectors")
    print("  • 2,095 language phoneme inventories")
    print("  • Ready for phoneme difficulty tool (Flege's SLM)")
    print("\nNext steps:")
    print("1. Existing tools (Builder, Contrastive Intervention) filter to English only")
    print("2. New difficulty tool uses all 2,095 languages")
    print("3. Deploy as static site (Netlify, Cloudflare Pages, etc.)")


if __name__ == "__main__":
    main()
