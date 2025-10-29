"""
PhonoLex-BERT: Transformer-based phonological embeddings with improvements.

Features:
1. Masked phoneme prediction (like BERT's MLM)
2. Contrastive learning for word-level similarity
3. Attention pooling (learned aggregation)
4. Multi-task learning (rhyme detection, phonotactics)

Architecture: Phoneme → Contextual Embedding → Attention Pool → Word Embedding
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys
import math
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


class PhonologyDataset(Dataset):
    """Dataset with multiple training objectives."""

    def __init__(self, loader: EnglishPhonologyLoader, mask_prob: float = 0.15):
        self.loader = loader
        self.mask_prob = mask_prob

        # Create phoneme vocabulary
        all_phonemes = set()
        for phonemes in loader.lexicon.values():
            all_phonemes.update(phonemes)

        # Special tokens
        self.PAD = '<PAD>'
        self.MASK = '<MASK>'
        self.CLS = '<CLS>'
        self.SEP = '<SEP>'

        self.phoneme_to_id = {
            self.PAD: 0,
            self.MASK: 1,
            self.CLS: 2,
            self.SEP: 3,
        }
        self.phoneme_to_id.update({p: i + 4 for i, p in enumerate(sorted(all_phonemes))})
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

        # Word vocabulary
        self.word_to_id = {word: i for i, word in enumerate(loader.lexicon.keys())}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        self.words = list(loader.lexicon.keys())

        # Max length
        self.max_length = max(len(phonemes) for phonemes in loader.lexicon.values()) + 2

        print(f"\n{'='*70}")
        print(f"PHONOLOGY DATASET")
        print(f"{'='*70}")
        print(f"Vocabulary: {len(self.word_to_id):,} words")
        print(f"Phonemes: {len(self.phoneme_to_id)} (including special tokens)")
        print(f"Max sequence length: {self.max_length}")

        # Prepare data
        self.word_data = []
        for word, phonemes in loader.lexicon.items():
            word_id = self.word_to_id[word]
            phoneme_ids = [self.phoneme_to_id[self.CLS]] + \
                          [self.phoneme_to_id[p] for p in phonemes] + \
                          [self.phoneme_to_id[self.SEP]]
            self.word_data.append((word_id, phoneme_ids, word))

        # Create contrastive pairs
        print("\nGenerating contrastive pairs...")
        self._create_contrastive_pairs()

    def _create_contrastive_pairs(self):
        """Generate positive and negative word pairs."""
        self.positive_pairs = []
        self.negative_pairs = []

        # Positive pairs: rhymes
        print("  - Finding rhymes...")
        rhyme_dict = {}
        for word, phonemes in self.loader.lexicon.items():
            if len(phonemes) >= 2:
                # Rhyme = last 2 phonemes
                rhyme_key = tuple(phonemes[-2:])
                if rhyme_key not in rhyme_dict:
                    rhyme_dict[rhyme_key] = []
                rhyme_dict[rhyme_key].append(word)

        for rhyme_group in rhyme_dict.values():
            if len(rhyme_group) >= 2:
                # Sample pairs within rhyme group
                for _ in range(min(5, len(rhyme_group))):
                    w1, w2 = random.sample(rhyme_group, 2)
                    if w1 != w2:
                        self.positive_pairs.append((w1, w2, 'rhyme'))

        # Positive pairs: morphological variants (from SIGMORPHON)
        print("  - Finding morphological variants...")
        for pair in self.loader.morphology:
            if pair.lemma in self.loader.lexicon and pair.inflected in self.loader.lexicon:
                self.positive_pairs.append((pair.lemma, pair.inflected, 'morphology'))

        # Negative pairs: unrelated words
        print("  - Sampling negative pairs...")
        for _ in range(len(self.positive_pairs)):
            w1, w2 = random.sample(self.words, 2)
            # Check they don't rhyme
            p1 = self.loader.lexicon[w1]
            p2 = self.loader.lexicon[w2]
            if len(p1) >= 2 and len(p2) >= 2:
                if tuple(p1[-2:]) != tuple(p2[-2:]):
                    self.negative_pairs.append((w1, w2, 'unrelated'))

        # Negative pairs: anagrams (same phonemes, different order)
        print("  - Finding anagrams...")
        phoneme_sets = {}
        for word, phonemes in self.loader.lexicon.items():
            key = tuple(sorted(phonemes))
            if key not in phoneme_sets:
                phoneme_sets[key] = []
            phoneme_sets[key].append(word)

        for anagram_group in phoneme_sets.values():
            if len(anagram_group) >= 2:
                for _ in range(min(2, len(anagram_group))):
                    w1, w2 = random.sample(anagram_group, 2)
                    if w1 != w2 and self.loader.lexicon[w1] != self.loader.lexicon[w2]:
                        self.negative_pairs.append((w1, w2, 'anagram'))

        print(f"  ✓ {len(self.positive_pairs):,} positive pairs")
        print(f"  ✓ {len(self.negative_pairs):,} negative pairs")

    def __len__(self):
        return len(self.word_data)

    def __getitem__(self, idx):
        word_id, phoneme_ids, word = self.word_data[idx]
        seq_len = len(phoneme_ids)

        # Masked language modeling
        input_ids = phoneme_ids.copy()
        target_ids = phoneme_ids.copy()

        for i in range(1, seq_len - 1):
            if random.random() < self.mask_prob:
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = self.phoneme_to_id[self.MASK]
                elif rand < 0.9:
                    input_ids[i] = random.randint(4, len(self.phoneme_to_id) - 1)

        # Pad
        padding_length = self.max_length - seq_len
        input_ids += [self.phoneme_to_id[self.PAD]] * padding_length
        target_ids += [self.phoneme_to_id[self.PAD]] * padding_length
        attention_mask = [1] * seq_len + [0] * padding_length

        return {
            'word_id': word_id,
            'input_ids': torch.LongTensor(input_ids),
            'target_ids': torch.LongTensor(target_ids),
            'attention_mask': torch.LongTensor(attention_mask),
        }

    def get_contrastive_batch(self, batch_size):
        """Get a batch of contrastive pairs."""
        positive_batch = random.sample(self.positive_pairs, min(batch_size // 2, len(self.positive_pairs)))
        negative_batch = random.sample(self.negative_pairs, min(batch_size // 2, len(self.negative_pairs)))

        pairs = []
        labels = []

        for w1, w2, pair_type in positive_batch:
            pairs.append((self.word_to_id[w1], self.word_to_id[w2]))
            labels.append(1.0)  # Positive

        for w1, w2, pair_type in negative_batch:
            pairs.append((self.word_to_id[w1], self.word_to_id[w2]))
            labels.append(0.0)  # Negative

        return pairs, labels


class AttentionPooling(nn.Module):
    """Learned attention pooling over phoneme sequences."""

    def __init__(self, d_model: int):
        super().__init__()
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(d_model))

    def forward(self, phoneme_embeddings, attention_mask):
        """
        Args:
            phoneme_embeddings: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len)

        Returns:
            pooled: (batch, d_model)
        """
        # Compute attention scores
        scores = torch.matmul(phoneme_embeddings, self.query)  # (batch, seq_len)

        # Mask padding
        scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax attention weights
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        pooled = torch.sum(phoneme_embeddings * weights.unsqueeze(-1), dim=1)  # (batch, d_model)

        return pooled


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class PhonoLexBERT(nn.Module):
    """
    Transformer-based phonological embeddings with:
    - Masked phoneme prediction
    - Attention pooling
    - Contrastive learning
    """

    def __init__(self, num_phonemes: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 512, max_len: int = 50):
        super().__init__()

        self.d_model = d_model

        # Phoneme embeddings
        self.phoneme_embeddings = nn.Embedding(num_phonemes, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling (for word-level embeddings)
        self.attention_pooling = AttentionPooling(d_model)

        # Task heads
        self.mlm_head = nn.Linear(d_model, num_phonemes)  # Masked language modeling

        # Initialize
        nn.init.xavier_uniform_(self.phoneme_embeddings.weight)
        nn.init.xavier_uniform_(self.mlm_head.weight)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            mlm_logits: (batch, seq_len, num_phonemes)
            word_embeddings: (batch, d_model)
            contextual_embeddings: (batch, seq_len, d_model)
        """
        # Embed phonemes
        x = self.phoneme_embeddings(input_ids)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        src_key_padding_mask = (attention_mask == 0)
        contextual_embeddings = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, d_model)

        # MLM predictions
        mlm_logits = self.mlm_head(contextual_embeddings)  # (batch, seq_len, num_phonemes)

        # Word-level embeddings via attention pooling
        word_embeddings = self.attention_pooling(contextual_embeddings, attention_mask)  # (batch, d_model)

        return mlm_logits, word_embeddings, contextual_embeddings

    def get_word_embedding(self, input_ids, attention_mask):
        """Get word embedding only (no MLM)."""
        with torch.no_grad():
            _, word_emb, _ = self.forward(input_ids, attention_mask)
        return word_emb


def contrastive_loss(emb1, emb2, labels, temperature=0.07):
    """
    Contrastive loss for word pairs.

    Args:
        emb1, emb2: (batch, d_model) word embeddings
        labels: (batch,) 1.0 for similar, 0.0 for dissimilar
        temperature: scaling factor

    Returns:
        loss: scalar
    """
    # Normalize embeddings
    emb1 = nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = nn.functional.normalize(emb2, p=2, dim=1)

    # Cosine similarity
    similarity = (emb1 * emb2).sum(dim=1) / temperature  # (batch,)

    # Binary cross-entropy with logits
    # If label=1 (similar), we want high similarity
    # If label=0 (dissimilar), we want low similarity
    loss = nn.functional.binary_cross_entropy_with_logits(
        similarity,
        labels,
        reduction='mean'
    )

    return loss


def train():
    # Configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = PhonologyDataset(loader, mask_prob=0.15)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0
    )

    # Create model
    model = PhonoLexBERT(
        num_phonemes=len(dataset.phoneme_to_id),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        max_len=dataset.max_length
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=dataset.phoneme_to_id[dataset.PAD])

    # Training loop
    num_epochs = 15
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")

    for epoch in range(num_epochs):
        model.train()
        total_mlm_loss = 0
        total_contrastive_loss = 0
        total_correct = 0
        total_predictions = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # MLM training
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)

            mlm_logits, word_embeddings, _ = model(input_ids, attention_masks)

            # MLM loss
            mlm_loss = mlm_criterion(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                target_ids.view(-1)
            )

            # Contrastive loss (every 2nd batch)
            if batch_idx % 2 == 0:
                pairs, labels = dataset.get_contrastive_batch(batch_size=32)
                if len(pairs) > 0:
                    # Get embeddings for pairs
                    pair_word_ids1 = []
                    pair_word_ids2 = []
                    for w1_id, w2_id in pairs:
                        pair_word_ids1.append(w1_id)
                        pair_word_ids2.append(w2_id)

                    # Get phoneme sequences
                    seqs1 = [dataset.word_data[wid][1] for wid in pair_word_ids1]
                    seqs2 = [dataset.word_data[wid][1] for wid in pair_word_ids2]

                    # Pad and tensorize
                    max_len = dataset.max_length
                    input_ids1 = []
                    masks1 = []
                    for seq in seqs1:
                        padded = seq + [dataset.phoneme_to_id[dataset.PAD]] * (max_len - len(seq))
                        input_ids1.append(padded)
                        masks1.append([1] * len(seq) + [0] * (max_len - len(seq)))

                    input_ids2 = []
                    masks2 = []
                    for seq in seqs2:
                        padded = seq + [dataset.phoneme_to_id[dataset.PAD]] * (max_len - len(seq))
                        input_ids2.append(padded)
                        masks2.append([1] * len(seq) + [0] * (max_len - len(seq)))

                    input_ids1 = torch.LongTensor(input_ids1).to(device)
                    masks1 = torch.LongTensor(masks1).to(device)
                    input_ids2 = torch.LongTensor(input_ids2).to(device)
                    masks2 = torch.LongTensor(masks2).to(device)
                    labels_tensor = torch.FloatTensor(labels).to(device)

                    # Get word embeddings
                    _, word_emb1, _ = model(input_ids1, masks1)
                    _, word_emb2, _ = model(input_ids2, masks2)

                    # Contrastive loss
                    contra_loss = contrastive_loss(word_emb1, word_emb2, labels_tensor)
                else:
                    contra_loss = torch.tensor(0.0).to(device)
            else:
                contra_loss = torch.tensor(0.0).to(device)

            # Combined loss
            loss = mlm_loss + 0.5 * contra_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_mlm_loss += mlm_loss.item()
            total_contrastive_loss += contra_loss.item()

            predictions = mlm_logits.argmax(dim=-1)
            mask = attention_masks.bool()
            correct = (predictions == target_ids) & mask
            total_correct += correct.sum().item()
            total_predictions += mask.sum().item()

            accuracy = 100.0 * total_correct / total_predictions if total_predictions > 0 else 0
            pbar.set_postfix({
                'mlm_loss': f'{mlm_loss.item():.4f}',
                'contra_loss': f'{contra_loss.item():.4f}',
                'acc': f'{accuracy:.2f}%'
            })

        avg_mlm_loss = total_mlm_loss / len(dataloader)
        avg_contra_loss = total_contrastive_loss / (len(dataloader) // 2)
        avg_accuracy = 100.0 * total_correct / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - MLM Loss: {avg_mlm_loss:.4f}, "
              f"Contrastive Loss: {avg_contra_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")

    # Save model
    output_dir = Path("models/phonolex_bert")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_id': dataset.word_to_id,
        'id_to_word': dataset.id_to_word,
        'phoneme_to_id': dataset.phoneme_to_id,
        'id_to_phoneme': dataset.id_to_phoneme,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'max_length': dataset.max_length,
    }, output_dir / "model.pt")

    print(f"\n{'='*70}")
    print(f"Model saved to: {output_dir / 'model.pt'}")
    print(f"{'='*70}")
    print("\n✨ PhonoLex-BERT trained with:")
    print("  ✓ Masked phoneme prediction")
    print("  ✓ Contrastive learning (rhymes, morphology, anagrams)")
    print("  ✓ Attention pooling (learned aggregation)")


if __name__ == "__main__":
    train()
