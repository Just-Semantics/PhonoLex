#!/usr/bin/env python3
"""
Quantize Layer 4 embeddings from float32 to int8 for deployment.

Reduces size by 75%:
- float32: 4 bytes per value
- int8: 1 byte per value

Example: 24K words × 384 dims × 4 bytes = ~37MB → ~9MB (per syllable avg)
With compression: ~170MB → ~60MB total
"""

import sys
from pathlib import Path
import torch
import numpy as np
import gzip
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def quantize_vector(vec: np.ndarray, scale: float = None, zero_point: int = None):
    """
    Quantize float32 vector to int8.

    Uses symmetric quantization: int8_val = round(float_val / scale)

    Args:
        vec: Float32 numpy array
        scale: Quantization scale (computed if None)
        zero_point: Zero point (always 0 for symmetric)

    Returns:
        (quantized_vec, scale) tuple
    """
    if scale is None:
        # Compute scale: max absolute value maps to 127
        max_val = np.abs(vec).max()
        scale = max_val / 127.0 if max_val > 0 else 1.0

    # Quantize: round(float / scale)
    quantized = np.round(vec / scale).astype(np.int8)

    return quantized, scale


def dequantize_vector(quantized_vec: np.ndarray, scale: float) -> np.ndarray:
    """
    Dequantize int8 vector back to float32.

    Args:
        quantized_vec: int8 numpy array
        scale: Quantization scale

    Returns:
        Float32 numpy array (approximation of original)
    """
    return quantized_vec.astype(np.float32) * scale


def quantize_syllable_embeddings(embeddings_path: str, output_path: str = None):
    """
    Quantize all syllable embeddings in checkpoint.

    Args:
        embeddings_path: Path to filtered embeddings (.pt file)
        output_path: Output path (defaults to *_quantized.pt)

    Returns:
        Path to quantized embeddings
    """
    print("=" * 80)
    print("Quantizing Layer 4 Syllable Embeddings")
    print("=" * 80)

    # Load embeddings
    print(f"\n[1/4] Loading embeddings from {embeddings_path}...")
    checkpoint = torch.load(embeddings_path, map_location='cpu', weights_only=False)
    word_to_syllable_embeddings = checkpoint['word_to_syllable_embeddings']

    print(f"✓ Loaded {len(word_to_syllable_embeddings):,} words")

    # Count total syllables
    total_syllables = sum(len(sylls) for sylls in word_to_syllable_embeddings.values())
    print(f"✓ Total syllables: {total_syllables:,}")

    # Quantize all embeddings
    print(f"\n[2/4] Quantizing embeddings (float32 → int8)...")

    quantized_data = {}
    scales = {}

    for word, syllable_embs in word_to_syllable_embeddings.items():
        quantized_syllables = []
        word_scales = []

        for syl_emb in syllable_embs:
            # Quantize this syllable embedding (384-dim)
            quantized, scale = quantize_vector(syl_emb)
            quantized_syllables.append(quantized)
            word_scales.append(scale)

        quantized_data[word] = quantized_syllables
        scales[word] = word_scales

    print(f"✓ Quantized {len(quantized_data):,} words")

    # Create output checkpoint
    print(f"\n[3/4] Creating quantized checkpoint...")

    quantized_checkpoint = {
        'quantized_embeddings': quantized_data,
        'scales': scales,
        'embedding_dim': 384,
        'quantization': 'int8_symmetric',
        'num_words': len(quantized_data),
        'original_path': embeddings_path,
        'filter_criterion': checkpoint.get('filter_criterion', 'unknown')
    }

    # Save quantized embeddings
    if output_path is None:
        output_path = Path(embeddings_path).parent / (Path(embeddings_path).stem + '_quantized.pt')

    print(f"\n[4/4] Saving to {output_path}...")
    torch.save(quantized_checkpoint, output_path)

    # Get file sizes
    original_size = Path(embeddings_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = 100 * (1 - quantized_size / original_size)

    print(f"\n✓ Saved quantized embeddings")
    print(f"\n" + "=" * 80)
    print("SIZE COMPARISON")
    print("=" * 80)
    print(f"Original (float32):  {original_size:>6.1f} MB")
    print(f"Quantized (int8):    {quantized_size:>6.1f} MB")
    print(f"Reduction:           {reduction:>6.1f}%")

    # Test compression
    print(f"\n" + "=" * 80)
    print("COMPRESSION TEST (gzip)")
    print("=" * 80)

    # Compress original
    with open(embeddings_path, 'rb') as f:
        original_data = f.read()
    compressed_original = gzip.compress(original_data, compresslevel=9)

    # Compress quantized
    with open(output_path, 'rb') as f:
        quantized_file_data = f.read()
    compressed_quantized = gzip.compress(quantized_file_data, compresslevel=9)

    original_compressed_size = len(compressed_original) / (1024 * 1024)
    quantized_compressed_size = len(compressed_quantized) / (1024 * 1024)

    print(f"Original compressed:  {original_compressed_size:>6.1f} MB")
    print(f"Quantized compressed: {quantized_compressed_size:>6.1f} MB")
    print(f"Total reduction:      {100*(1-quantized_compressed_size/original_compressed_size):>6.1f}%")

    # Test reconstruction error
    print(f"\n" + "=" * 80)
    print("QUALITY TEST (Reconstruction Error)")
    print("=" * 80)

    # Sample 100 random words
    sample_words = list(word_to_syllable_embeddings.keys())[:100]

    errors = []
    for word in sample_words:
        original = word_to_syllable_embeddings[word]
        quantized_sylls = quantized_data[word]
        word_scales_list = scales[word]

        for orig_syl, quant_syl, scale in zip(original, quantized_sylls, word_scales_list):
            reconstructed = dequantize_vector(quant_syl, scale)

            # Compute relative error
            mse = np.mean((orig_syl - reconstructed) ** 2)
            relative_error = np.sqrt(mse) / (np.linalg.norm(orig_syl) + 1e-8)
            errors.append(relative_error)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"Mean relative error:  {mean_error:.4f} ({mean_error*100:.2f}%)")
    print(f"Max relative error:   {max_error:.4f} ({max_error*100:.2f}%)")
    print(f"\n✓ Quantization quality: {'Excellent' if mean_error < 0.01 else 'Good' if mean_error < 0.05 else 'Acceptable'}")

    print("\n" + "=" * 80)
    print("✓ SUCCESS: Embeddings quantized")
    print("=" * 80)
    print(f"\nUsage:")
    print(f"  from load_quantized_embeddings import load_quantized")
    print(f"  embeddings = load_quantized('{output_path}')")

    return output_path


def load_quantized_embeddings(checkpoint_path: str):
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

    # Dequantize
    dequantized = {}
    for word, quantized_sylls in quantized_data.items():
        word_scales = scales[word]

        dequantized_sylls = []
        for quant_syl, scale in zip(quantized_sylls, word_scales):
            deq_syl = dequantize_vector(quant_syl, scale)
            dequantized_sylls.append(deq_syl)

        dequantized[word] = dequantized_sylls

    return dequantized


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quantize Layer 4 embeddings to int8")
    parser.add_argument(
        "--input",
        type=str,
        default="embeddings/layer4/syllable_embeddings_filtered.pt",
        help="Path to input embeddings"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output quantized embeddings (auto-generated if not specified)"
    )
    args = parser.parse_args()

    quantize_syllable_embeddings(args.input, args.output)


if __name__ == "__main__":
    main()
