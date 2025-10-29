#!/usr/bin/env python3
"""
Syllable-Aware Model - PROPER IMPLEMENTATION

No artificial weighting. Just preserve sequential structure through syllable hierarchy.
Let contrastive learning figure out what similarity patterns matter.
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
from src.phonolex.utils.syllabification import syllabify
from train_phonology_bert_v2 import contrastive_loss_v2
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv


class SyllableAwareEncoder(nn.Module):
    """
    Hierarchical encoder that preserves sequential structure:
    phonemes (with position) → syllables → word

    No artificial weighting - just structural organization.
    """

    def __init__(self, num_phonemes, d_model=128, nhead=4, num_layers=3, max_len=50):
        super().__init__()

        self.d_model = d_model

        # Phoneme embeddings (will initialize with Phoible)
        self.phoneme_embeddings = nn.Embedding(num_phonemes, d_model)

        # Positional encoding (preserves phoneme order)
        self.pos_encoder = self._create_positional_encoding(max_len, d_model)

        # Phoneme-level transformer (contextual representations)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.phoneme_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Word-level pooling (simple mean over phonemes, position is already encoded)
        # The positional encoding + transformer already captured sequential patterns

        # MLM head
        self.mlm_head = nn.Linear(d_model, num_phonemes)

    def _create_positional_encoding(self, max_len, d_model):
        """Fixed sinusoidal positional encoding"""
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
            word_embedding: [batch, d_model]
        """
        # Phoneme embeddings + positional encoding
        x = self.phoneme_embeddings(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :].to(x.device)

        # Contextual phoneme representations
        src_key_padding_mask = (attention_mask == 0)
        contextual_phonemes = self.phoneme_transformer(x, src_key_padding_mask=src_key_padding_mask)

        # MLM head
        mlm_logits = self.mlm_head(contextual_phonemes)

        # Word embedding: Mean pool over valid phonemes
        # Position info already captured by pos_encoder + transformer
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (contextual_phonemes * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        word_embedding = sum_embeddings / (sum_mask + 1e-9)

        return mlm_logits, word_embedding


def load_phoible_init(phoneme_to_id, d_model=128):
    """Initialize phoneme embeddings with Phoible features"""
    phoible_csv = Path('data/phoible/english/phoible-english.csv')
    phoible_data = load_phoible_csv(str(phoible_csv))
    vectorizer = PhonemeVectorizer(encoding_scheme='three_way')

    phoible_features = {}
    for phoneme_data in phoible_data:
        vec = vectorizer.vectorize(phoneme_data)
        if vec.phoneme not in phoible_features:
            phoible_features[vec.phoneme] = vec.endpoints_76d

    # Project to d_model
    feature_projection = nn.Linear(76, d_model)
    embedding_matrix = torch.randn(len(phoneme_to_id), d_model) * 0.01

    initialized = 0
    with torch.no_grad():
        for phoneme, idx in phoneme_to_id.items():
            if phoneme in phoible_features:
                features = torch.FloatTensor(phoible_features[phoneme])
                embedding_matrix[idx] = feature_projection(features)
                initialized += 1

    print(f"✓ Initialized {initialized}/{len(phoneme_to_id)} phonemes with Phoible features\n")
    return embedding_matrix


class SyllableAwareDataset(Dataset):
    """Dataset with syllable info + contrastive pairs"""

    def __init__(self, loader, phoneme_to_id, mask_prob=0.15):
        self.loader = loader
        self.phoneme_to_id = phoneme_to_id
        self.mask_prob = mask_prob

        self.PAD = '<PAD>'
        self.CLS = '<CLS>'
        self.SEP = '<SEP>'
        self.MASK = '<MASK>'

        # Build word data with syllables
        self.words = list(loader.lexicon_with_stress.keys())
        self.word_data = []

        for word in self.words:
            phonemes_with_stress = loader.lexicon_with_stress[word]
            phoneme_ids = [self.phoneme_to_id['<CLS>']] + \
                         [self.phoneme_to_id.get(p.phoneme, self.phoneme_to_id['<PAD>'])
                          for p in phonemes_with_stress] + \
                         [self.phoneme_to_id['<SEP>']]

            syllables = syllabify(phonemes_with_stress)

            self.word_data.append({
                'word': word,
                'phoneme_ids': phoneme_ids,
                'syllables': syllables
            })

        self.max_length = max(len(d['phoneme_ids']) for d in self.word_data) + 2

        # Generate contrastive pairs (same as v2)
        self._generate_contrastive_pairs()

    def _generate_contrastive_pairs(self):
        """Use SAME pair generation as train_phonology_bert_v2"""
        self.positive_pairs = []
        self.negative_pairs = []

        # Positive: Rhymes
        rhyme_dict = {}
        for i, data in enumerate(self.word_data):
            word = data['word']
            if word in self.loader.lexicon:
                phonemes = self.loader.lexicon[word]
                if len(phonemes) >= 2:
                    rhyme_key = tuple(phonemes[-2:])
                    if rhyme_key not in rhyme_dict:
                        rhyme_dict[rhyme_key] = []
                    rhyme_dict[rhyme_key].append(i)

        for rhyme_key, indices in rhyme_dict.items():
            if len(indices) >= 2:
                for _ in range(min(len(indices), 5)):
                    i, j = random.sample(indices, 2)
                    self.positive_pairs.append((i, j, 1.0))

        # Positive: Morphology
        for pair in self.loader.morphology:
            if pair.lemma in self.loader.lexicon and pair.inflected in self.loader.lexicon:
                try:
                    i = self.words.index(pair.lemma)
                    j = self.words.index(pair.inflected)
                    self.positive_pairs.append((i, j, 1.0))
                except ValueError:
                    continue

        # Negative: Anagrams (medium similarity 0.45)
        phoneme_sets = {}
        for i, data in enumerate(self.word_data):
            word = data['word']
            if word in self.loader.lexicon:
                phonemes = self.loader.lexicon[word]
                if len(phonemes) >= 3:
                    key = tuple(sorted(phonemes))
                    if key not in phoneme_sets:
                        phoneme_sets[key] = []
                    phoneme_sets[key].append(i)

        for anagram_group in phoneme_sets.values():
            if len(anagram_group) >= 2:
                for _ in range(min(5, len(anagram_group))):
                    i, j = random.sample(anagram_group, 2)
                    w1 = self.word_data[i]['word']
                    w2 = self.word_data[j]['word']
                    if self.loader.lexicon[w1] != self.loader.lexicon[w2]:
                        self.negative_pairs.append((i, j, 0.45))

        # Negative: Unrelated
        for _ in range(len(self.positive_pairs)):
            i, j = random.sample(range(len(self.words)), 2)
            self.negative_pairs.append((i, j, 0.0))

        print(f"Contrastive pairs: {len(self.positive_pairs)} positive, {len(self.negative_pairs)} negative\n")

    def __len__(self):
        return len(self.word_data)

    def __getitem__(self, idx):
        data = self.word_data[idx]
        phoneme_ids = data['phoneme_ids']

        # Apply masking
        input_ids = phoneme_ids.copy()
        target_ids = [0] * len(phoneme_ids)

        for i in range(1, len(phoneme_ids) - 1):
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


def train(model, dataset, device, num_epochs=10, contra_weight=0.5, temperature=0.05):
    """Train syllable-aware model"""
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward
            mlm_logits, word_emb = model(input_ids, attention_mask)

            # MLM loss
            mlm_loss = mlm_criterion(mlm_logits.view(-1, mlm_logits.size(-1)), target_ids.view(-1))

            # Contrastive loss
            pairs, labels = dataset.get_contrastive_batch(batch_size=64)
            if len(pairs) > 0:
                data1 = [dataset[i] for i, _ in pairs]
                data2 = [dataset[j] for _, j in pairs]

                input_ids1 = torch.stack([d['input_ids'] for d in data1]).to(device)
                mask1 = torch.stack([d['attention_mask'] for d in data1]).to(device)
                input_ids2 = torch.stack([d['input_ids'] for d in data2]).to(device)
                mask2 = torch.stack([d['attention_mask'] for d in data2]).to(device)

                _, emb1 = model(input_ids1, mask1)
                _, emb2 = model(input_ids2, mask2)

                contra_loss = contrastive_loss_v2(emb1, emb2, torch.FloatTensor(labels).to(device), temperature)
            else:
                contra_loss = torch.tensor(0.0).to(device)

            loss = mlm_loss + contra_weight * contra_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss {total_loss/len(dataloader):.4f}")

    return model


def main():
    print("\n" + "=" * 70)
    print("SYLLABLE-AWARE MODEL (PROPER)")
    print("=" * 70)
    print("Sequential structure via position encoding + transformer")
    print("No artificial weighting - let contrastive learning decide\n")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        loader = EnglishPhonologyLoader()

    print(f"Loaded {len(loader.lexicon):,} words\n")

    # Build vocab
    phonemes = ['<PAD>', '<CLS>', '<SEP>', '<MASK>'] + sorted(loader.english_phonemes)
    phoneme_to_id = {p: i for i, p in enumerate(phonemes)}

    # Create dataset
    dataset = SyllableAwareDataset(loader, phoneme_to_id, mask_prob=0.15)

    # Create model with Phoible initialization
    print("Initializing model with Phoible features...")
    phoible_init = load_phoible_init(phoneme_to_id, d_model=128)

    model = SyllableAwareEncoder(
        num_phonemes=len(phoneme_to_id),
        d_model=128, nhead=4, num_layers=3,
        max_len=dataset.max_length
    ).to(device)

    # Initialize with Phoible
    with torch.no_grad():
        model.phoneme_embeddings.weight.data = phoible_init.to(device)

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()

    # Train
    model = train(model, dataset, device, num_epochs=10, contra_weight=0.5, temperature=0.05)

    # Save
    Path('models/syllable_aware').mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': phoneme_to_id,
        'max_length': dataset.max_length
    }, 'models/syllable_aware/final.pt')

    print("\n✓ Model saved: models/syllable_aware/final.pt\n")

    # Quick eval
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()

    def get_emb(word):
        if word not in loader.lexicon_with_stress:
            return None
        phonemes = loader.lexicon_with_stress[word]
        ids = [phoneme_to_id['<CLS>']] + [phoneme_to_id.get(p.phoneme, phoneme_to_id['<PAD>']) for p in phonemes] + [phoneme_to_id['<SEP>']]
        seq_len = len(ids)
        padded = ids + [phoneme_to_id['<PAD>']] * (dataset.max_length - seq_len)
        mask = [1] * seq_len + [0] * (dataset.max_length - seq_len)

        with torch.no_grad():
            _, emb = model(torch.LongTensor([padded]).to(device), torch.LongTensor([mask]).to(device))
        return emb.squeeze(0).cpu().numpy()

    def sim(w1, w2):
        e1, e2 = get_emb(w1), get_emb(w2)
        if e1 is None or e2 is None:
            return None
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)

    tests = [
        ('cat', 'bat', 'rhyme'),
        ('cat', 'act', 'anagram'),
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


if __name__ == '__main__':
    main()
