#!/usr/bin/env python3
"""
Curriculum Learning Pipeline for Phonological Word Embeddings

Phase 1: Phoible features (universal prior)
Phase 2: Contextual phoneme embeddings (MLM fine-tuning)
Phase 3: Syllable structure (hierarchical)
Phase 4: Word embeddings (contrastive learning)

Each phase builds on the previous, with checkpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import json

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv


# ============================================================================
# PHASE 1: Load Phoible Feature Vectors
# ============================================================================

def load_phoible_features_for_english(loader: EnglishPhonologyLoader):
    """
    Load Phoible articulatory features for English phonemes.

    Returns:
        phoneme_to_features: dict mapping phoneme -> 76-dim feature vector
    """
    print("=" * 70)
    print("PHASE 1: LOADING PHOIBLE FEATURES")
    print("=" * 70)
    print()

    # Load Phoible English data
    phoible_csv = Path('data/phoible/english/phoible-english.csv')
    print(f"Loading Phoible features from {phoible_csv}...")

    phoible_data = load_phoible_csv(str(phoible_csv))
    print(f"Loaded {len(phoible_data)} phoneme entries")

    # Vectorize
    vectorizer = PhonemeVectorizer(encoding_scheme='three_way')

    phoneme_to_features = {}

    for phoneme_data in phoible_data:
        vec = vectorizer.vectorize(phoneme_data)
        phoneme = vec.phoneme

        # Use 76-dim endpoint features (start + end states)
        if phoneme not in phoneme_to_features:
            phoneme_to_features[phoneme] = vec.endpoints_76d

    print(f"✓ Extracted features for {len(phoneme_to_features)} unique phonemes")
    print(f"  Feature dimension: 76 (articulatory endpoints)")
    print()

    # Map to our English phonemes
    english_phoneme_features = {}
    missing = []

    for phoneme in loader.english_phonemes:
        if phoneme in phoneme_to_features:
            english_phoneme_features[phoneme] = phoneme_to_features[phoneme]
        else:
            missing.append(phoneme)

    print(f"✓ Mapped {len(english_phoneme_features)}/{len(loader.english_phonemes)} English phonemes to Phoible")

    if missing:
        print(f"  ⚠ Missing features for: {missing}")
        print(f"    (Will initialize randomly)")

    return english_phoneme_features


def create_phoneme_embedding_matrix(phoneme_to_id, phoible_features, d_model=128):
    """
    Create phoneme embedding matrix initialized with Phoible features.

    Args:
        phoneme_to_id: dict mapping phoneme -> id
        phoible_features: dict mapping phoneme -> 76-dim feature vector
        d_model: target embedding dimension

    Returns:
        embedding_matrix: [num_phonemes, d_model]
        feature_projection: nn.Linear to project 76->d_model
    """
    print("Creating phoneme embedding matrix...")

    num_phonemes = len(phoneme_to_id)
    embedding_matrix = torch.randn(num_phonemes, d_model) * 0.01  # Small random init

    # Project Phoible features to d_model dimension
    feature_projection = nn.Linear(76, d_model)

    # Initialize embeddings for phonemes we have features for
    with torch.no_grad():
        for phoneme, idx in phoneme_to_id.items():
            if phoneme in phoible_features:
                # Project Phoible features
                features_tensor = torch.FloatTensor(phoible_features[phoneme])
                projected = feature_projection(features_tensor)
                embedding_matrix[idx] = projected

    print(f"✓ Initialized {num_phonemes} phoneme embeddings")
    print(f"  - {len(phoible_features)} from Phoible features")
    print(f"  - {num_phonemes - len(phoible_features)} randomly initialized")
    print()

    return embedding_matrix, feature_projection


# ============================================================================
# PHASE 2: Contextual Phoneme Embeddings (MLM)
# ============================================================================

class PhoibleInitializedTransformer(nn.Module):
    """
    Transformer initialized with Phoible features.
    Phase 2: Fine-tune phoneme embeddings with masked language modeling.
    """

    def __init__(self, embedding_matrix, feature_projection, num_phonemes,
                 d_model=128, nhead=4, num_layers=3, max_len=50):
        super().__init__()

        self.d_model = d_model

        # Initialize phoneme embeddings with Phoible features
        self.phoneme_embeddings = nn.Embedding(num_phonemes, d_model)
        self.phoneme_embeddings.weight.data = embedding_matrix

        # Keep feature projection for potential later use
        self.feature_projection = feature_projection

        # Positional encoding
        self.pos_encoder = self._create_positional_encoding(max_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLM head
        self.mlm_head = nn.Linear(d_model, num_phonemes)

    def _create_positional_encoding(self, max_len, d_model):
        """Create fixed positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            mlm_logits: [batch, seq_len, num_phonemes]
            contextual_embeddings: [batch, seq_len, d_model]
        """
        # Phoneme embeddings (initialized with Phoible)
        x = self.phoneme_embeddings(input_ids)  # [batch, seq_len, d_model]

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :].to(x.device)

        # Create attention mask for transformer (True = mask out)
        src_key_padding_mask = (attention_mask == 0)

        # Transformer
        contextual = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # MLM prediction
        mlm_logits = self.mlm_head(contextual)

        return mlm_logits, contextual


def train_phase2_mlm(model, dataloader, device, num_epochs=5):
    """
    Phase 2: Fine-tune phoneme embeddings with MLM.
    """
    print("=" * 70)
    print("PHASE 2: FINE-TUNING PHONEME EMBEDDINGS (MLM)")
    print("=" * 70)
    print()

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is PAD

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            mlm_logits, _ = model(input_ids, attention_mask)

            loss = criterion(mlm_logits.view(-1, mlm_logits.size(-1)), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            preds = mlm_logits.argmax(dim=-1)
            mask = target_ids != 0
            correct += ((preds == target_ids) & mask).sum().item()
            total += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    print()
    print("✓ Phase 2 complete - phoneme embeddings fine-tuned!")
    print()

    return model


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING PIPELINE")
    print("=" * 70)
    print()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        loader = EnglishPhonologyLoader()

    print(f"Loaded {len(loader.lexicon):,} English words")
    print(f"English phonemes: {len(loader.english_phonemes)}\n")

    # ========================================================================
    # PHASE 1: Load Phoible Features
    # ========================================================================

    phoible_features = load_phoible_features_for_english(loader)

    # Create phoneme vocabulary
    phonemes = ['<PAD>', '<CLS>', '<SEP>', '<MASK>'] + sorted(loader.english_phonemes)
    phoneme_to_id = {p: i for i, p in enumerate(phonemes)}

    # Initialize embedding matrix with Phoible features
    embedding_matrix, feature_projection = create_phoneme_embedding_matrix(
        phoneme_to_id, phoible_features, d_model=128
    )

    # Save Phase 1 checkpoint
    checkpoint_dir = Path('models/curriculum')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'embedding_matrix': embedding_matrix,
        'feature_projection': feature_projection.state_dict(),
        'phoneme_to_id': phoneme_to_id,
        'phoible_features': phoible_features
    }, checkpoint_dir / 'phase1_phoible_init.pt')

    print(f"✓ Phase 1 checkpoint saved: {checkpoint_dir}/phase1_phoible_init.pt\n")

    # ========================================================================
    # PHASE 2: Fine-tune with MLM
    # ========================================================================

    # Create dataset (simplified for now - reuse from previous code)
    from train_phonology_bert_v2 import ImprovedPhonologyDataset

    with contextlib.redirect_stdout(io.StringIO()):
        dataset = ImprovedPhonologyDataset(loader, mask_prob=0.15)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    # Create model
    model = PhoibleInitializedTransformer(
        embedding_matrix=embedding_matrix,
        feature_projection=feature_projection,
        num_phonemes=len(phoneme_to_id),
        d_model=128, nhead=4, num_layers=3
    ).to(device)

    # Train Phase 2
    model = train_phase2_mlm(model, dataloader, device, num_epochs=5)

    # Save Phase 2 checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': phoneme_to_id
    }, checkpoint_dir / 'phase2_mlm_finetuned.pt')

    print(f"✓ Phase 2 checkpoint saved: {checkpoint_dir}/phase2_mlm_finetuned.pt\n")

    print("=" * 70)
    print("PHASES 1-2 COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - Phase 3: Add syllable structure")
    print("  - Phase 4: Contrastive learning for word embeddings")
    print()


if __name__ == '__main__':
    main()
