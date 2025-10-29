"""
Evaluate contextual phoneme embeddings and word-level aggregations.

Like sentence transformers:
- Word = sentence (sequence of phonemes)
- Contextual phoneme embeddings (like contextual word embeddings)
- Aggregate to get word-level representation (CLS, mean pooling, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from train_contextual_phoneme_embeddings import ContextualPhonemeEmbedding, PositionalEncoding


def load_model():
    """Load the trained contextual phoneme embedding model."""
    checkpoint = torch.load('models/contextual_phoneme_embeddings/model.pt', map_location='cpu')

    model = ContextualPhonemeEmbedding(
        num_phonemes=len(checkpoint['phoneme_to_id']),
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        dim_feedforward=512,
        max_len=checkpoint['max_length']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def get_word_embedding(model, word, phoneme_to_id, loader, method='cls'):
    """
    Get word-level embedding from contextual phoneme embeddings.

    Args:
        model: The contextual phoneme embedding model
        word: The word to embed
        phoneme_to_id: Mapping from phonemes to IDs
        loader: Data loader for getting pronunciations
        method: 'cls' (CLS token), 'mean' (mean pool), or 'max' (max pool)

    Returns:
        Word embedding vector
    """
    if word not in loader.lexicon:
        return None

    # Get phonemes
    phonemes = loader.lexicon[word]

    # Create input sequence: [CLS] + phonemes + [SEP]
    CLS = phoneme_to_id['<CLS>']
    SEP = phoneme_to_id['<SEP>']
    PAD = phoneme_to_id['<PAD>']

    input_ids = [CLS] + [phoneme_to_id[p] for p in phonemes] + [SEP]
    seq_len = len(input_ids)

    # Create tensors
    input_tensor = torch.LongTensor(input_ids).unsqueeze(0)  # (1, seq_len)
    attention_mask = torch.ones(1, seq_len).long()  # (1, seq_len)

    # Get contextual embeddings
    with torch.no_grad():
        _, contextual_emb = model(input_tensor, attention_mask)  # (1, seq_len, d_model)

    contextual_emb = contextual_emb.squeeze(0)  # (seq_len, d_model)

    # Aggregate to word-level embedding
    if method == 'cls':
        # Use CLS token representation
        word_emb = contextual_emb[0]  # (d_model,)
    elif method == 'mean':
        # Mean pool over phonemes (excluding CLS and SEP)
        word_emb = contextual_emb[1:-1].mean(dim=0)  # (d_model,)
    elif method == 'max':
        # Max pool over phonemes
        word_emb = contextual_emb[1:-1].max(dim=0)[0]  # (d_model,)
    else:
        raise ValueError(f"Unknown method: {method}")

    return word_emb.numpy()


def cosine_sim(e1, e2):
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)


def find_similar_words(target_word, all_words, word_embeddings, top_k=10):
    """Find most similar words based on word embeddings."""
    if target_word not in word_embeddings:
        return []

    target_emb = word_embeddings[target_word]

    # Compute similarities
    similarities = []
    for word, emb in word_embeddings.items():
        if word != target_word:
            sim = cosine_sim(target_emb, emb)
            similarities.append((word, sim))

    # Sort and return top K
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def main():
    print("=" * 70)
    print("CONTEXTUAL PHONEME EMBEDDINGS EVALUATION")
    print("=" * 70)

    # Load model and data
    model, checkpoint = load_model()
    phoneme_to_id = checkpoint['phoneme_to_id']
    loader = EnglishPhonologyLoader()

    print(f"\nModel loaded:")
    print(f"  - {len(phoneme_to_id)} phonemes (including special tokens)")
    print(f"  - {checkpoint['d_model']}-dim contextual embeddings")
    print(f"  - {checkpoint['num_layers']} transformer layers")

    # Test different aggregation methods
    methods = ['cls', 'mean', 'max']

    for method in methods:
        print(f"\n{'=' * 70}")
        print(f"WORD EMBEDDING METHOD: {method.upper()}")
        print(f"{'=' * 70}")

        # Compute word embeddings for test words
        test_words = ['cat', 'bat', 'mat', 'sat', 'act', 'dog', 'fog', 'log',
                      'make', 'take', 'cake', 'run', 'running', 'phone', 'clone']

        word_embeddings = {}
        for word in test_words:
            emb = get_word_embedding(model, word, phoneme_to_id, loader, method=method)
            if emb is not None:
                word_embeddings[word] = emb

        # Test similarity pairs
        print("\nWord similarity scores:")
        test_pairs = [
            ('cat', 'bat', 'minimal pair'),
            ('cat', 'mat', 'rhyme'),
            ('cat', 'act', 'anagram'),
            ('dog', 'fog', 'rhyme'),
            ('make', 'take', 'rhyme'),
            ('run', 'running', 'morphology'),
            ('cat', 'dog', 'unrelated'),
        ]

        for w1, w2, desc in test_pairs:
            if w1 in word_embeddings and w2 in word_embeddings:
                sim = cosine_sim(word_embeddings[w1], word_embeddings[w2])
                p1 = ''.join(loader.lexicon.get(w1, []))
                p2 = ''.join(loader.lexicon.get(w2, []))
                print(f'  {w1:10s} /{p1:15s}  -  {w2:10s} /{p2:15s}  [{desc:15s}]: {sim:.4f}')

        # Find nearest neighbors
        print("\nNearest neighbors:")
        for word in ['cat', 'dog', 'run']:
            if word in word_embeddings:
                similar = find_similar_words(word, test_words, word_embeddings, top_k=5)
                print(f'\n  "{word}" (/{"".join(loader.lexicon.get(word, []))}) →')
                for w, s in similar[:5]:
                    phones = ''.join(loader.lexicon.get(w, []))
                    print(f'    {w:15s} /{phones:20s}  {s:.4f}')

    print(f"\n{'=' * 70}")
    print("KEY INSIGHT")
    print("=" * 70)
    print("✓ Each phoneme gets a CONTEXTUAL representation based on position")
    print("✓ Same phoneme in different contexts has different embeddings")
    print("  - /t/ in 'cat' (word-final) vs 'take' (word-initial)")
    print("  - /æ/ in 'cat' (middle) vs 'bat' (middle, different context)")
    print("✓ Aggregate contextual phoneme embeddings → word embeddings")
    print("  - CLS token: learned summary representation")
    print("  - Mean pooling: average of phoneme contexts")
    print("  - Max pooling: max activation across phonemes")
    print("✓ Similar to sentence transformers: word = sentence, phoneme = word")


if __name__ == "__main__":
    main()
