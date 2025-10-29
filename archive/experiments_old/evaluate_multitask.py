#!/usr/bin/env python3
"""
Evaluate Multi-Task English Phonology Model

Tests:
1. Phoneme similarity (do similar phonemes cluster?)
2. Allomorph prediction accuracy
3. Rhyme detection
4. Phonological neighborhoods
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


def load_model(model_path="models/english_multitask/model.pt"):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')

    # Simple wrapper
    class Model:
        def __init__(self, checkpoint):
            self.phoneme_to_id = checkpoint['phoneme_to_id']
            self.id_to_phoneme = checkpoint['id_to_phoneme']
            self.embeddings = None

            # Load embeddings from state dict
            if 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
                if 'embeddings.weight' in state:
                    self.embeddings = state['embeddings.weight'].numpy()

        def get_embedding(self, phoneme):
            if phoneme not in self.phoneme_to_id:
                return None
            idx = self.phoneme_to_id[phoneme]
            return self.embeddings[idx]

    return Model(checkpoint)


def cosine_sim(e1, e2):
    """Cosine similarity"""
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)


def eval_phoneme_similarity(model):
    """Test if similar phonemes have similar embeddings"""
    print("\n" + "=" * 70)
    print("1. PHONEME SIMILARITY")
    print("=" * 70)

    # Test pairs that should be similar
    similar_pairs = [
        ('p', 'b', 'voiceless/voiced bilabial stops'),
        ('t', 'd', 'voiceless/voiced alveolar stops'),
        ('k', 'g', 'voiceless/voiced velar stops'),
        ('s', 'z', 'voiceless/voiced fricatives'),
        ('f', 'v', 'voiceless/voiced fricatives'),
        ('θ', 'ð', 'voiceless/voiced fricatives'),
    ]

    # Test pairs that should be dissimilar
    dissimilar_pairs = [
        ('p', 'a', 'stop vs vowel'),
        ('t', 'i', 'stop vs vowel'),
        ('s', 'u', 'fricative vs vowel'),
    ]

    print("\nSimilar pairs (should have HIGH similarity):")
    similar_scores = []
    for p1, p2, desc in similar_pairs:
        e1 = model.get_embedding(p1)
        e2 = model.get_embedding(p2)

        if e1 is not None and e2 is not None:
            sim = cosine_sim(e1, e2)
            similar_scores.append(sim)
            print(f"  /{p1}/ - /{p2}/ ({desc}): {sim:.4f}")

    print("\nDissimilar pairs (should have LOW similarity):")
    dissimilar_scores = []
    for p1, p2, desc in dissimilar_pairs:
        e1 = model.get_embedding(p1)
        e2 = model.get_embedding(p2)

        if e1 is not None and e2 is not None:
            sim = cosine_sim(e1, e2)
            dissimilar_scores.append(sim)
            print(f"  /{p1}/ - /{p2}/ ({desc}): {sim:.4f}")

    if similar_scores and dissimilar_scores:
        print(f"\n✓ Similar pairs avg: {np.mean(similar_scores):.4f}")
        print(f"✓ Dissimilar pairs avg: {np.mean(dissimilar_scores):.4f}")
        print(f"✓ Separation: {np.mean(similar_scores) - np.mean(dissimilar_scores):.4f}")


def eval_nearest_neighbors(model):
    """Find nearest neighbors for common phonemes"""
    print("\n" + "=" * 70)
    print("2. NEAREST NEIGHBORS")
    print("=" * 70)

    test_phonemes = ['p', 'b', 't', 's', 'z', 'æ', 'i', 'u']

    for phone in test_phonemes:
        emb = model.get_embedding(phone)
        if emb is None:
            continue

        # Find similar
        sims = []
        for other_phone in model.phoneme_to_id.keys():
            if other_phone == phone:
                continue
            other_emb = model.get_embedding(other_phone)
            if other_emb is not None:
                sim = cosine_sim(emb, other_emb)
                sims.append((other_phone, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

        print(f"\n/{phone}/ nearest neighbors:")
        for p, s in sims[:5]:
            print(f"  /{p}/  {s:.4f}")


def eval_word_similarity(model, loader):
    """Test word-level phonological similarity"""
    print("\n" + "=" * 70)
    print("3. WORD SIMILARITY")
    print("=" * 70)

    def get_word_embedding(word):
        """Average phoneme embeddings for a word"""
        if word not in loader.lexicon:
            return None

        phonemes = loader.lexicon[word]
        embs = []
        for p in phonemes:
            e = model.get_embedding(p)
            if e is not None:
                embs.append(e)

        if not embs:
            return None

        return np.mean(embs, axis=0)

    # Test rhyming words
    rhyme_pairs = [
        ('cat', 'bat'),
        ('dog', 'fog'),
        ('make', 'take'),
        ('light', 'fight'),
    ]

    # Test non-rhyming
    non_rhyme_pairs = [
        ('cat', 'dog'),
        ('make', 'light'),
        ('bat', 'fog'),
    ]

    print("\nRhyming pairs (should be similar):")
    rhyme_scores = []
    for w1, w2 in rhyme_pairs:
        e1 = get_word_embedding(w1)
        e2 = get_word_embedding(w2)

        if e1 is not None and e2 is not None:
            sim = cosine_sim(e1, e2)
            rhyme_scores.append(sim)
            print(f"  {w1} - {w2}: {sim:.4f}")

    print("\nNon-rhyming pairs (should be dissimilar):")
    non_rhyme_scores = []
    for w1, w2 in non_rhyme_pairs:
        e1 = get_word_embedding(w1)
        e2 = get_word_embedding(w2)

        if e1 is not None and e2 is not None:
            sim = cosine_sim(e1, e2)
            non_rhyme_scores.append(sim)
            print(f"  {w1} - {w2}: {sim:.4f}")

    if rhyme_scores and non_rhyme_scores:
        print(f"\n✓ Rhyming avg: {np.mean(rhyme_scores):.4f}")
        print(f"✓ Non-rhyming avg: {np.mean(non_rhyme_scores):.4f}")
        print(f"✓ Rhyme detection signal: {np.mean(rhyme_scores) - np.mean(non_rhyme_scores):.4f}")


def find_similar_words(model, loader, target_word, top_k=10):
    """Find phonologically similar words"""

    def get_word_embedding(word):
        if word not in loader.lexicon:
            return None
        phonemes = loader.lexicon[word]
        embs = [model.get_embedding(p) for p in phonemes if model.get_embedding(p) is not None]
        return np.mean(embs, axis=0) if embs else None

    target_emb = get_word_embedding(target_word)
    if target_emb is None:
        return []

    sims = []
    for word in loader.lexicon.keys():
        if word == target_word:
            continue
        word_emb = get_word_embedding(word)
        if word_emb is not None:
            sim = cosine_sim(target_emb, word_emb)
            sims.append((word, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def eval_similar_words(model, loader):
    """Find similar words for test cases"""
    print("\n" + "=" * 70)
    print("4. PHONOLOGICALLY SIMILAR WORDS")
    print("=" * 70)

    test_words = ['cat', 'dog', 'phone', 'make']

    for word in test_words:
        print(f"\nWords similar to '{word}':")
        similar = find_similar_words(model, loader, word, top_k=10)
        for w, sim in similar:
            phones = ''.join(loader.lexicon[w])
            print(f"  {w:15s} /{phones:15s}  {sim:.4f}")


def main():
    print("=" * 70)
    print("MULTI-TASK ENGLISH PHONOLOGY EVALUATION")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = load_model()
    print(f"  ✓ Loaded {len(model.phoneme_to_id)} phoneme embeddings")

    # Load data
    print("\nLoading data...")
    loader = EnglishPhonologyLoader()

    # Run evaluations
    eval_phoneme_similarity(model)
    eval_nearest_neighbors(model)
    eval_word_similarity(model, loader)
    eval_similar_words(model, loader)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
