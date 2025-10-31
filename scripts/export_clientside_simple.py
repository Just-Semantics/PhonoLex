#!/usr/bin/env python3
"""
Simple client-side data export using existing phonological graph.

Exports the existing graph filtered to match our 24,744-word filtered vocabulary.
Includes both IPA and ARPAbet pronunciations for user preference.
"""

import sys
import json
import torch
import pickle
import gzip
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


def main():
    print("=" * 80)
    print("PhonoLex Client-Side Data Export (Simple)")
    print("=" * 80)

    # Load filtered vocabulary from quantized embeddings
    print("\n[1/4] Loading filtered vocabulary...")
    emb_path = project_root / "embeddings/layer4/syllable_embeddings_filtered_quantized.pt"
    checkpoint = torch.load(emb_path, map_location='cpu', weights_only=False)
    filtered_words = set(checkpoint['quantized_embeddings'].keys())
    print(f"  ✓ {len(filtered_words):,} words in filtered vocabulary")

    # Load CMU dictionary for ARPAbet
    print("\n[2/5] Loading CMU dictionary for ARPAbet...")
    loader = EnglishPhonologyLoader()

    # Create IPA to ARPAbet reverse mapping
    ipa_to_arpa = {v: k for k, v in loader.arpa_to_ipa.items() if k.rstrip('012') == k}  # Base forms only
    print(f"  ✓ Loaded ARPAbet mappings")

    # Load existing phonological graph
    print("\n[3/5] Loading phonological graph...")
    graph_path = project_root / "data/phonological_graph.pkl"
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)

    G = graph_data['graph']
    print(f"  ✓ Graph has {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Filter graph to only include filtered words
    print("\n[4/5] Filtering graph and adding ARPAbet...")
    word_metadata = {}
    for node in tqdm(G.nodes(), desc="  Processing nodes"):
        if node in filtered_words:
            node_data = G.nodes[node]

            # Convert IPA phonemes to ARPAbet
            ipa_phonemes = node_data.get('phonemes', [])
            arpa_phonemes = []
            for ipa in ipa_phonemes:
                arpa = ipa_to_arpa.get(ipa, ipa)  # Fallback to IPA if no mapping
                arpa_phonemes.append(arpa)

            word_metadata[node] = {
                'word': node,
                'ipa': node_data.get('ipa', ''),
                'arpa': ' '.join(arpa_phonemes),  # ARPAbet pronunciation
                'phonemes': ipa_phonemes,  # IPA phonemes
                'phonemes_arpa': arpa_phonemes,  # ARPAbet phonemes
                'syllables': node_data.get('syllables', []),
                'phoneme_count': node_data.get('phoneme_count', 0),
                'syllable_count': node_data.get('syllable_count', 0),
                'wcm_score': node_data.get('wcm_score'),
                'msh_stage': node_data.get('msh_stage'),
                'frequency': node_data.get('frequency'),
                'log_frequency': node_data.get('log_frequency'),
                'concreteness': node_data.get('concreteness'),
                'aoa': node_data.get('aoa'),
                'imageability': node_data.get('imageability'),
                'familiarity': node_data.get('familiarity'),
                'valence': node_data.get('valence'),
                'arousal': node_data.get('arousal'),
                'dominance': node_data.get('dominance'),
            }

    print(f"  ✓ Filtered to {len(word_metadata):,} words with ARPAbet")

    # Export embeddings in JSON format
    print("\n[5/5] Exporting data...")
    output_dir = project_root / "webapp/frontend/public/data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also copy ARPAbet mapping file for reference
    import shutil
    arpa_mapping_src = project_root / "data/mappings/arpa_to_ipa.json"
    arpa_mapping_dst = output_dir / "arpa_to_ipa.json"
    shutil.copy(arpa_mapping_src, arpa_mapping_dst)
    print(f"  ✓ arpa_to_ipa.json (mapping reference)")

    # 1. Word metadata
    metadata_path = output_dir / "word_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(word_metadata, f, separators=(',', ':'))
    size_mb = metadata_path.stat().st_size / 1024 / 1024
    print(f"  ✓ word_metadata.json ({size_mb:.1f} MB)")

    # 2. Quantized embeddings
    embeddings_json = {}
    for word, syllable_arrays in tqdm(checkpoint['quantized_embeddings'].items(), desc="  Converting embeddings"):
        embeddings_json[word] = [arr.tolist() for arr in syllable_arrays]

    # Convert scales (handles both single values and lists)
    scales_converted = {}
    for k, v in checkpoint['scales'].items():
        if isinstance(v, list):
            scales_converted[k] = [float(x) for x in v]
        else:
            scales_converted[k] = float(v)

    embeddings_data = {
        'embeddings': embeddings_json,
        'scales': scales_converted,
        'embedding_dim': int(checkpoint['embedding_dim']),
        'quantization': checkpoint['quantization']
    }

    embeddings_path = output_dir / "embeddings_quantized.json"
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings_data, f, separators=(',', ':'))
    size_mb = embeddings_path.stat().st_size / 1024 / 1024
    print(f"  ✓ embeddings_quantized.json ({size_mb:.1f} MB)")

    # 3. Create manifest
    manifest = {
        'version': '2.0.0',
        'vocabulary_size': len(word_metadata),
        'filter_criterion': 'frequency + at least one psycholinguistic norm',
        'files': {
            'word_metadata.json': 'Word properties, IPA, syllables, psycholinguistic norms',
            'embeddings_quantized.json': 'Int8 quantized syllable embeddings for similarity'
        }
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ manifest.json")

    total_size = sum(p.stat().st_size for p in output_dir.glob("*.json")) / 1024 / 1024
    print(f"\n  Total data package size: {total_size:.1f} MB")

    print("\n" + "=" * 80)
    print("✓ SUCCESS!")
    print("=" * 80)
    print(f"\nData exported to: {output_dir}")


if __name__ == "__main__":
    main()
