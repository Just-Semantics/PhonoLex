"""
Utility to load quantized embeddings for deployment.

Usage:
    from load_quantized_embeddings import load_quantized

    embeddings = load_quantized('embeddings/layer4/syllable_embeddings_filtered_quantized.pt')
    cat_syllables = embeddings['cat']  # List of 384-dim numpy arrays
"""

import torch
import numpy as np


def dequantize_vector(quantized_vec: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize int8 vector back to float32"""
    return quantized_vec.astype(np.float32) * scale


def load_quantized(checkpoint_path: str):
    """
    Load and dequantize embeddings.

    Args:
        checkpoint_path: Path to quantized checkpoint

    Returns:
        Dict[str, List[np.ndarray]] - word -> syllable embeddings (float32)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    quantized_data = checkpoint['quantized_embeddings']
    scales = checkpoint['scales']

    # Dequantize on-demand (lazy loading possible)
    dequantized = {}
    for word, quantized_sylls in quantized_data.items():
        word_scales = scales[word]

        dequantized_sylls = []
        for quant_syl, scale in zip(quantized_sylls, word_scales):
            deq_syl = dequantize_vector(quant_syl, scale)
            dequantized_sylls.append(deq_syl)

        dequantized[word] = dequantized_sylls

    return dequantized


if __name__ == "__main__":
    # Test loading
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "embeddings/layer4/syllable_embeddings_filtered_quantized.pt"

    print(f"Loading quantized embeddings from {path}...")
    embeddings = load_quantized(path)

    print(f"✓ Loaded {len(embeddings):,} words")

    # Show sample
    sample_word = list(embeddings.keys())[0]
    print(f"\nExample: '{sample_word}'")
    print(f"  Syllables: {len(embeddings[sample_word])}")
    print(f"  First syllable shape: {embeddings[sample_word][0].shape}")
