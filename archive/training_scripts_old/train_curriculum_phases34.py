#!/usr/bin/env python3
"""
Curriculum Learning - Phases 3 & 4

Phase 3: Syllable-aware hierarchical model
Phase 4: Word embeddings with contrastive learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import numpy as np
import random

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from src.phonolex.utils.syllabification import syllabify, get_rhyme_part
from train_phonology_bert_v2 import contrastive_loss_v2


# ============================================================================
# PHASE 3: Syllable-Aware Model
# ============================================================================

class SyllableAwareModel(nn.Module):
    """
    Hierarchical model: phoneme → syllable → word

    Uses Phase 2's contextual phoneme embeddings + syllable structure
    """

    def __init__(self, phoneme_embeddings_weight, num_phonemes,
                 d_model=128, nhead=4, num_layers=3, max_len=50):
        super().__init__()

        self.d_model = d_model

        # Phoneme embeddings (from Phase 2)
        self.phoneme_embeddings = nn.Embedding(num_phonemes, d_model)
        self.phoneme_embeddings.weight.data = phoneme_embeddings_weight

        # Positional encoding
        self.pos_encoder = self._create_positional_encoding(max_len, d_model)

        # Phoneme-level transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.phoneme_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Syllable-level attention pooling (aggregate phonemes → syllable)
        self.syllable_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Word-level attention pooling (aggregate syllables → word)
        self.word_attention = nn.Parameter(torch.randn(d_model))

        # MLM head (for Phase 3 training)
        self.mlm_head = nn.Linear(d_model, num_phonemes)

    def _create_positional_encoding(self, max_len, d_model):
        """Fixed positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids, attention_mask, syllable_boundaries=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            syllable_boundaries: [batch, num_syllables, 2] (start, end indices) - optional

        Returns:
            mlm_logits: [batch, seq_len, num_phonemes]
            word_embedding: [batch, d_model]
            syllable_embeddings: [batch, num_syllables, d_model] or None
        """
        # Phoneme embeddings
        x = self.phoneme_embeddings(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :].to(x.device)

        # Phoneme-level context
        src_key_padding_mask = (attention_mask == 0)
        contextual_phonemes = self.phoneme_transformer(x, src_key_padding_mask=src_key_padding_mask)

        # MLM head
        mlm_logits = self.mlm_head(contextual_phonemes)

        # Syllable-level aggregation (if boundaries provided)
        if syllable_boundaries is not None:
            # TODO: Implement syllable pooling using boundaries
            # For now, use simple attention pooling over all phonemes
            pass

        # Word-level: attention pooling over phonemes
        scores = torch.matmul(contextual_phonemes, self.word_attention)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        word_embedding = torch.sum(contextual_phonemes * weights.unsqueeze(-1), dim=1)

        return mlm_logits, word_embedding, None


# ============================================================================
# PHASE 4: Contrastive Dataset with Syllable Info
# ============================================================================

class SyllableAwareDataset(Dataset):
    """Dataset with syllable structure + contrastive pairs"""

    def __init__(self, loader: EnglishPhonologyLoader, phoneme_to_id, mask_prob=0.15):
        self.loader = loader
        self.phoneme_to_id = phoneme_to_id
        self.mask_prob = mask_prob

        self.PAD = '<PAD>'
        self.CLS = '<CLS>'
        self.SEP = '<SEP>'
        self.MASK = '<MASK>'

        self.max_length = 50

        # Build word list
        self.words = list(loader.lexicon_with_stress.keys())

        # Build word data (with syllables)
        self.word_data = []
        for word in self.words:
            phonemes_with_stress = loader.lexicon_with_stress[word]
            phoneme_ids = [self.phoneme_to_id['<CLS>']] + \
                         [self.phoneme_to_id.get(p.phoneme, self.phoneme_to_id['<PAD>'])
                          for p in phonemes_with_stress] + \
                         [self.phoneme_to_id['<SEP>']]

            # Extract syllables
            syllables = syllabify(phonemes_with_stress)

            self.word_data.append({
                'word': word,
                'phoneme_ids': phoneme_ids,
                'syllables': syllables
            })

        # Generate contrastive pairs
        self._generate_contrastive_pairs()

    def _generate_contrastive_pairs(self):
        """Generate positive and negative pairs"""
        self.positive_pairs = []
        self.negative_pairs = []

        # Positive: rhymes (share nucleus+coda of last syllable)
        rhyme_groups = {}
        for i, data in enumerate(self.word_data):
            if data['syllables']:
                rhyme = get_rhyme_part(data['syllables'])
                if rhyme:
                    if rhyme not in rhyme_groups:
                        rhyme_groups[rhyme] = []
                    rhyme_groups[rhyme].append(i)

        for rhyme, indices in rhyme_groups.items():
            if len(indices) >= 2:
                for _ in range(min(len(indices), 5)):
                    i, j = random.sample(indices, 2)
                    self.positive_pairs.append((i, j, 1.0))  # rhyme

        # Positive: morphology
        for pair in self.loader.morphology:
            if pair.lemma in self.loader.lexicon and pair.inflected in self.loader.lexicon:
                try:
                    i = self.words.index(pair.lemma)
                    j = self.words.index(pair.inflected)
                    self.positive_pairs.append((i, j, 1.0))  # morphology
                except ValueError:
                    continue

        # Negative: unrelated words
        for _ in range(len(self.positive_pairs)):
            i, j = random.sample(range(len(self.words)), 2)
            self.negative_pairs.append((i, j, 0.0))

        print(f"Contrastive pairs: {len(self.positive_pairs)} positive, {len(self.negative_pairs)} negative")

    def __len__(self):
        return len(self.word_data)

    def __getitem__(self, idx):
        data = self.word_data[idx]
        phoneme_ids = data['phoneme_ids']

        # Apply masking for MLM
        input_ids = phoneme_ids.copy()
        target_ids = [0] * len(phoneme_ids)

        for i in range(1, len(phoneme_ids) - 1):  # Skip CLS and SEP
            if random.random() < self.mask_prob:
                target_ids[i] = input_ids[i]
                input_ids[i] = self.phoneme_to_id[self.MASK]

        # Pad
        seq_len = len(input_ids)
        padded_input = input_ids + [self.phoneme_to_id[self.PAD]] * (self.max_length - seq_len)
        padded_target = target_ids + [0] * (self.max_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)

        return {
            'input_ids': torch.LongTensor(padded_input),
            'target_ids': torch.LongTensor(padded_target),
            'attention_mask': torch.LongTensor(attention_mask),
            'word_idx': idx
        }

    def get_contrastive_batch(self, batch_size=64):
        """Sample contrastive pairs"""
        all_pairs = self.positive_pairs + self.negative_pairs
        if len(all_pairs) == 0:
            return [], []

        batch = random.sample(all_pairs, min(batch_size, len(all_pairs)))
        pairs = [(i, j) for i, j, _ in batch]
        labels = [label for _, _, label in batch]

        return pairs, labels


# ============================================================================
# Training
# ============================================================================

def train_phases34(model, dataset, device, num_epochs=10, contra_weight=0.5, temperature=0.05):
    """
    Phase 3: Train with syllable structure (MLM)
    Phase 4: Add contrastive learning for word embeddings
    """
    print("=" * 70)
    print("PHASES 3-4: SYLLABLE STRUCTURE + WORD EMBEDDINGS")
    print("=" * 70)
    print()

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(num_epochs):
        total_loss = 0
        mlm_loss_sum = 0
        contra_loss_sum = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward
            mlm_logits, word_emb, _ = model(input_ids, attention_mask)

            # MLM loss
            mlm_loss = mlm_criterion(mlm_logits.view(-1, mlm_logits.size(-1)), target_ids.view(-1))

            # Contrastive loss
            pairs, labels = dataset.get_contrastive_batch(batch_size=64)
            if len(pairs) > 0:
                # Get embeddings for pairs
                word_indices1 = [p[0] for p in pairs]
                word_indices2 = [p[1] for p in pairs]

                data1 = [dataset[i] for i in word_indices1]
                data2 = [dataset[i] for i in word_indices2]

                input_ids1 = torch.stack([d['input_ids'] for d in data1]).to(device)
                mask1 = torch.stack([d['attention_mask'] for d in data1]).to(device)
                input_ids2 = torch.stack([d['input_ids'] for d in data2]).to(device)
                mask2 = torch.stack([d['attention_mask'] for d in data2]).to(device)

                _, emb1, _ = model(input_ids1, mask1)
                _, emb2, _ = model(input_ids2, mask2)

                contra_loss = contrastive_loss_v2(emb1, emb2, torch.FloatTensor(labels).to(device), temperature)
            else:
                contra_loss = torch.tensor(0.0).to(device)

            # Combined loss
            loss = mlm_loss + contra_weight * contra_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mlm_loss_sum += mlm_loss.item()
            contra_loss_sum += contra_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_mlm = mlm_loss_sum / len(dataloader)
        avg_contra = contra_loss_sum / len(dataloader)

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f} (MLM={avg_mlm:.4f}, Contra={avg_contra:.4f})")

    print()
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("CURRICULUM PHASES 3-4")
    print("=" * 70)
    print()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load Phase 2 checkpoint
    checkpoint = torch.load('models/curriculum/phase2_mlm_finetuned.pt')
    phoneme_to_id = checkpoint['phoneme_to_id']

    # Load data
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        loader = EnglishPhonologyLoader()

    print(f"Loaded {len(loader.lexicon):,} words\n")

    # Create dataset with syllables
    dataset = SyllableAwareDataset(loader, phoneme_to_id, mask_prob=0.15)

    # Load Phase 2 model and extract phoneme embeddings
    from train_curriculum import PhoibleInitializedTransformer

    phase2_model = PhoibleInitializedTransformer(
        embedding_matrix=torch.randn(len(phoneme_to_id), 128),  # Dummy, will be overwritten
        feature_projection=nn.Linear(76, 128),  # Dummy
        num_phonemes=len(phoneme_to_id)
    )
    phase2_model.load_state_dict(checkpoint['model_state_dict'])

    # Create Phase 3-4 model with Phase 2's phoneme embeddings
    model = SyllableAwareModel(
        phoneme_embeddings_weight=phase2_model.phoneme_embeddings.weight.data,
        num_phonemes=len(phoneme_to_id),
        d_model=128, nhead=4, num_layers=3
    ).to(device)

    # Train
    model = train_phases34(model, dataset, device, num_epochs=10, contra_weight=0.5, temperature=0.05)

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': phoneme_to_id
    }, 'models/curriculum/phases34_final.pt')

    print("✓ Phases 3-4 checkpoint saved: models/curriculum/phases34_final.pt\n")

    # Quick evaluation
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()

    def get_word_emb(word):
        if word not in loader.lexicon_with_stress:
            return None
        phonemes = loader.lexicon_with_stress[word]
        ids = [phoneme_to_id['<CLS>']] + [phoneme_to_id.get(p.phoneme, phoneme_to_id['<PAD>']) for p in phonemes] + [phoneme_to_id['<SEP>']]
        seq_len = len(ids)
        padded = ids + [phoneme_to_id['<PAD>']] * (50 - seq_len)
        mask = [1] * seq_len + [0] * (50 - seq_len)

        with torch.no_grad():
            _, word_emb, _ = model(
                torch.LongTensor([padded]).to(device),
                torch.LongTensor([mask]).to(device)
            )
        return word_emb.squeeze(0).cpu().numpy()

    def sim(w1, w2):
        e1, e2 = get_word_emb(w1), get_word_emb(w2)
        if e1 is None or e2 is None:
            return None
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)

    tests = [
        ('cat', 'bat', 'rhyme'),
        ('cat', 'mat', 'rhyme'),
        ('cat', 'act', 'ANAGRAM'),
        ('dog', 'fog', 'rhyme'),
        ('run', 'running', 'morph'),
        ('cat', 'dog', 'unrelated'),
    ]

    print()
    for w1, w2, typ in tests:
        s = sim(w1, w2)
        if s is not None:
            print(f"{w1}-{w2:<12} {s:.3f}  ({typ})")

    print("\n" + "=" * 70)
    print("CURRICULUM COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
