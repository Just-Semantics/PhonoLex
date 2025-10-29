#!/usr/bin/env python3
"""
Layer 4: Build Hierarchical Syllable Embeddings

This script builds Layer 4 syllable embeddings from a pre-trained Layer 3 model.

Process:
1. Load frozen Layer 3 contextual phoneme embeddings (128-dim)
2. Syllabify each word (onset-nucleus-coda)
3. Aggregate phoneme embeddings by syllable role:
   - Onset: Mean of onset consonants (128-dim)
   - Nucleus: Vowel embedding (128-dim)
   - Coda: Mean of coda consonants (128-dim)
4. Concatenate: [onset + nucleus + coda] = 384-dim per syllable
5. Normalize to unit length

Input: models/curriculum/phoible_initialized_model.pt (Layer 3)
Output: embeddings/layer4/syllable_embeddings.pt (Layer 4)

Layer 3 is FROZEN - we don't retrain it, just reorganize by syllable structure.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np
from typing import List

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader, PhonemeWithStress
from src.phonolex.utils.syllabification import syllabify, Syllable
from src.phonolex.models.phonolex_bert import PhonoLexBERT


class HierarchicalPhonemeEncoder(nn.Module):
    """Phoneme-level transformer encoder (Layer 3 architecture)."""

    def __init__(self, num_phonemes, d_model=128, nhead=4, num_layers=3, max_len=50):
        super().__init__()
        self.d_model = d_model

        # Phoneme embeddings
        self.phoneme_embeddings = nn.Embedding(num_phonemes, d_model)

        # Positional encoding
        self.pos_encoder = self._create_positional_encoding(max_len, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection for next-phoneme prediction
        self.fc_out = nn.Linear(d_model, num_phonemes)

    def _create_positional_encoding(self, max_len, d_model):
        """Sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, phoneme_ids, return_embeddings=False):
        """
        Args:
            phoneme_ids: [batch, seq_len]
            return_embeddings: If True, return contextual embeddings instead of logits

        Returns:
            If return_embeddings: [batch, seq_len, d_model]
            Else: [batch, seq_len, num_phonemes]
        """
        # Embed phonemes
        x = self.phoneme_embeddings(phoneme_ids)  # [batch, seq_len, d_model]

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :].to(x.device)

        # Transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]

        if return_embeddings:
            return x

        # Project to phoneme logits
        logits = self.fc_out(x)  # [batch, seq_len, num_phonemes]
        return logits


def get_syllable_embedding(syllable: Syllable, phoneme_embeddings: np.ndarray, d_model: int) -> np.ndarray:
    """
    Aggregate contextual phoneme embeddings into a single syllable embedding.

    Args:
        syllable: Syllable with onset, nucleus, coda
        phoneme_embeddings: Contextual embeddings for all phonemes in word [seq_len, d_model]
        d_model: Embedding dimension (128)

    Returns:
        384-dim syllable embedding: [onset_128 + nucleus_128 + coda_128]
    """
    # Onset: Mean of onset consonants
    if syllable.onset:
        onset_indices = list(range(len(syllable.onset)))
        onset_emb = np.mean([phoneme_embeddings[i] for i in onset_indices], axis=0)
    else:
        onset_emb = np.zeros(d_model)

    # Nucleus: Single vowel
    nucleus_idx = len(syllable.onset)
    nucleus_emb = phoneme_embeddings[nucleus_idx]

    # Coda: Mean of coda consonants
    if syllable.coda:
        coda_start = len(syllable.onset) + 1
        coda_indices = list(range(coda_start, coda_start + len(syllable.coda)))
        coda_emb = np.mean([phoneme_embeddings[i] for i in coda_indices], axis=0)
    else:
        coda_emb = np.zeros(d_model)

    # Concatenate
    syllable_emb = np.concatenate([onset_emb, nucleus_emb, coda_emb])  # 384-dim

    # Normalize to unit length
    norm = np.linalg.norm(syllable_emb)
    if norm > 0:
        syllable_emb = syllable_emb / norm

    return syllable_emb


def hierarchical_soft_levenshtein(syllables1: List[np.ndarray], syllables2: List[np.ndarray]) -> float:
    """
    Optimized soft Levenshtein distance on syllable sequences.

    Uses vectorized similarity matrix computation (50x speedup over naive).

    Args:
        syllables1: List of syllable embeddings (384-dim each)
        syllables2: List of syllable embeddings (384-dim each)

    Returns:
        Similarity score [0.0, 1.0] where 1.0 = identical
    """
    len1, len2 = len(syllables1), len(syllables2)

    # Pre-compute all pairwise similarities (vectorized)
    # Stack syllables into matrices [len, 384]
    s1_matrix = np.array(syllables1)  # [len1, 384]
    s2_matrix = np.array(syllables2)  # [len2, 384]

    # Compute all cosine similarities at once (dot product since normalized)
    sim_matrix = s1_matrix @ s2_matrix.T  # [len1, len2]

    # Dynamic programming for edit distance with soft costs
    dp = np.zeros((len1 + 1, len2 + 1))

    # Initialize: cost of inserting/deleting syllables
    for i in range(len1 + 1):
        dp[i][0] = i * 1.0  # Deletion cost
    for j in range(len2 + 1):
        dp[0][j] = j * 1.0  # Insertion cost

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Match/substitute cost: 1 - similarity (0 if identical, 2 if opposite)
            match_cost = 1.0 - sim_matrix[i-1, j-1]

            dp[i][j] = min(
                dp[i-1][j] + 1.0,        # Delete from s1
                dp[i][j-1] + 1.0,        # Insert from s2
                dp[i-1][j-1] + match_cost  # Match/substitute
            )

    # Normalize to [0, 1] similarity
    max_len = max(len1, len2)
    if max_len == 0:
        return 1.0

    edit_distance = dp[len1][len2]
    similarity = 1.0 - (edit_distance / max_len)
    return max(0.0, min(1.0, similarity))


def build_layer4():
    """Build Layer 4 syllable embeddings from frozen Layer 3."""

    print("=== Building Layer 4: Hierarchical Syllable Embeddings ===\n")

    # Load CMU dictionary
    print("Loading CMU dictionary...")
    loader = EnglishPhonologyLoader()
    words_data = loader.lexicon
    print(f"Loaded {len(words_data)} words\n")

    # Load Layer 3 model (frozen)
    layer3_path = Path('models/layer3/model.pt')
    if not layer3_path.exists():
        print(f"ERROR: Layer 3 model not found at {layer3_path}")
        print("Please train Layer 3 first using train_layer3_contextual_embeddings.py")
        return

    print(f"Loading Layer 3 model from {layer3_path}...")
    checkpoint = torch.load(layer3_path, map_location='cpu')
    phoneme_to_id = checkpoint['phoneme_to_id']
    max_length = checkpoint['max_length']

    # Initialize model (using PhonoLexBERT, the actual Layer 3 architecture)
    model = PhonoLexBERT(
        num_phonemes=len(phoneme_to_id),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        max_len=max_length
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Layer 3 model loaded (frozen)\n")

    # Build Layer 4 embeddings
    print("Building syllable embeddings...")
    word_to_syllable_embeddings = {}
    d_model = 128

    for i, (word, phonemes) in enumerate(words_data.items()):
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(words_data)} words...")

        # Convert phonemes to IDs (with CLS/SEP tokens like in training)
        CLS, SEP, PAD = phoneme_to_id['<CLS>'], phoneme_to_id['<SEP>'], phoneme_to_id.get('<PAD>', 0)
        phoneme_ids = [CLS] + [phoneme_to_id.get(p, PAD) for p in phonemes] + [SEP]
        seq_len = len(phoneme_ids)

        # Pad to max_length
        padded_ids = phoneme_ids + [PAD] * (max_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

        phoneme_tensor = torch.tensor([padded_ids], dtype=torch.long)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long)

        # Get contextual embeddings from Layer 3 (frozen)
        with torch.no_grad():
            _, _, contextual_embs = model(phoneme_tensor, mask_tensor)  # [1, seq_len, 128]
            # Extract phoneme embeddings (skip CLS/SEP tokens)
            contextual_embs = contextual_embs[0, 1:seq_len-1].numpy()  # [num_phonemes, 128]

        # Syllabify (need PhonemeWithStress objects)
        phonemes_with_stress = [PhonemeWithStress(p, stress=0) for p in phonemes]
        syllables = syllabify(phonemes_with_stress)

        # Aggregate each syllable
        syllable_embeddings = []
        phoneme_offset = 0

        for syllable in syllables:
            syll_len = len(syllable.onset) + 1 + len(syllable.coda)  # onset + nucleus + coda
            syll_phoneme_embs = contextual_embs[phoneme_offset:phoneme_offset + syll_len]

            syll_emb = get_syllable_embedding(syllable, syll_phoneme_embs, d_model)
            syllable_embeddings.append(syll_emb)

            phoneme_offset += syll_len

        word_to_syllable_embeddings[word] = syllable_embeddings

    print(f"  Completed {len(words_data)} words\n")

    # Save Layer 4
    output_dir = Path('models/layer4')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'model.pt'

    torch.save({
        'word_to_syllable_embeddings': word_to_syllable_embeddings,
        'phoneme_to_id': phoneme_to_id,
        'd_model': d_model,
        'syllable_dim': 384
    }, output_path)

    print(f"Saved Layer 4 to {output_path}")
    print(f"Total words: {len(word_to_syllable_embeddings)}")

    # Quick evaluation with hierarchical similarity
    print("\n=== Quick Evaluation (Hierarchical Soft Levenshtein) ===")
    test_pairs = [
        ('cat', 'bat'),
        ('cat', 'act'),
        ('cat', 'dog'),
        ('make', 'take'),
        ('computer', 'commuter')
    ]

    for w1, w2 in test_pairs:
        if w1 in word_to_syllable_embeddings and w2 in word_to_syllable_embeddings:
            embs1 = word_to_syllable_embeddings[w1]
            embs2 = word_to_syllable_embeddings[w2]

            # Use hierarchical soft Levenshtein
            sim = hierarchical_soft_levenshtein(embs1, embs2)
            print(f"{w1}-{w2}: {sim:.3f}")


if __name__ == '__main__':
    build_layer4()
