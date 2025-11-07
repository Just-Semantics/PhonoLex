#!/usr/bin/env python3
"""
Debug script to examine syllable embedding structure
"""

import torch
import numpy as np

def main():
    # Load embeddings
    checkpoint = torch.load('embeddings/layer4/syllable_embeddings_filtered.pt',
                           map_location='cpu', weights_only=False)

    word_embeddings = checkpoint['word_to_syllable_embeddings']

    # Get embeddings for test words
    cat = word_embeddings['cat'][0]  # First syllable
    bat = word_embeddings['bat'][0]
    make = word_embeddings['make'][0]
    bake = word_embeddings['bake'][0]
    take = word_embeddings['take'][0]

    print("=" * 80)
    print("Syllable Embedding Analysis")
    print("=" * 80)
    print()

    # Split into components
    def split_syllable(syll):
        onset = syll[:128]
        nucleus = syll[128:256]
        coda = syll[256:384]
        return onset, nucleus, coda

    cat_o, cat_n, cat_c = split_syllable(cat)
    bat_o, bat_n, bat_c = split_syllable(bat)
    make_o, make_n, make_c = split_syllable(make)
    bake_o, bake_n, bake_c = split_syllable(bake)
    take_o, take_n, take_c = split_syllable(take)

    # Check if components are normalized
    print("Component Norms (should be ~1.0 if normalized separately):")
    print(f"  cat:  onset={np.linalg.norm(cat_o):.4f}, nucleus={np.linalg.norm(cat_n):.4f}, coda={np.linalg.norm(cat_c):.4f}")
    print(f"  bat:  onset={np.linalg.norm(bat_o):.4f}, nucleus={np.linalg.norm(bat_n):.4f}, coda={np.linalg.norm(cat_c):.4f}")
    print(f"  make: onset={np.linalg.norm(make_o):.4f}, nucleus={np.linalg.norm(make_n):.4f}, coda={np.linalg.norm(make_c):.4f}")
    print()

    # Check full syllable norm
    print("Full Syllable Norms:")
    print(f"  cat:  {np.linalg.norm(cat):.4f}")
    print(f"  bat:  {np.linalg.norm(bat):.4f}")
    print(f"  make: {np.linalg.norm(make):.4f}")
    print()

    # Component similarities
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("Component-wise similarities:")
    print()
    print("cat vs bat:")
    print(f"  Onset:   {cosine_sim(cat_o, bat_o):.4f}  (k vs b - should be different)")
    print(f"  Nucleus: {cosine_sim(cat_n, bat_n):.4f}  (æ vs æ - should be ~1.0)")
    print(f"  Coda:    {cosine_sim(cat_c, bat_c):.4f}  (t vs t - should be ~1.0)")
    print(f"  Full:    {cosine_sim(cat, bat):.4f}")
    print()

    print("make vs bake (/m/ vs /b/ - both bilabial, voiced):")
    print(f"  Onset:   {cosine_sim(make_o, bake_o):.4f}")
    print(f"  Nucleus: {cosine_sim(make_n, bake_n):.4f}")
    print(f"  Coda:    {cosine_sim(make_c, bake_c):.4f}")
    print(f"  Full:    {cosine_sim(make, bake):.4f}")
    print()

    print("make vs take (/m/ vs /t/ - different place, different voicing):")
    print(f"  Onset:   {cosine_sim(make_o, take_o):.4f}")
    print(f"  Nucleus: {cosine_sim(make_n, take_n):.4f}")
    print(f"  Coda:    {cosine_sim(make_c, take_c):.4f}")
    print(f"  Full:    {cosine_sim(make, take):.4f}")
    print()

    print("=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print("If onset similarity make-bake > make-take but full similarity shows opposite,")
    print("then the problem is that 256 dims (nucleus+coda) dominate over 128 dims (onset).")
    print()
    print("Solution: Use weighted or sliced similarity at query time.")
    print("=" * 80)

if __name__ == "__main__":
    main()
