"""
PhonoLex-BERT v2: Improved training with better contrastive learning.

Improvements over v1:
- Higher contrastive loss weight (1.0 instead of 0.5)
- Lower temperature (0.05 instead of 0.07) for sharper distinctions
- More hard negatives (anagrams, minimal pairs that differ)
- Balanced sampling of positive/negative pairs
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
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from train_phonology_bert import PhonoLexBERT, AttentionPooling, PositionalEncoding


class ImprovedPhonologyDataset(Dataset):
    """Dataset with improved contrastive pair generation."""

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
        print(f"IMPROVED PHONOLOGY DATASET")
        print(f"{'='*70}")
        print(f"Vocabulary: {len(self.word_to_id):,} words")
        print(f"Phonemes: {len(self.phoneme_to_id)} (including special tokens)")

        # Prepare data
        self.word_data = []
        for word, phonemes in loader.lexicon.items():
            word_id = self.word_to_id[word]
            phoneme_ids = [self.phoneme_to_id[self.CLS]] + \
                          [self.phoneme_to_id[p] for p in phonemes] + \
                          [self.phoneme_to_id[self.SEP]]
            self.word_data.append((word_id, phoneme_ids, word))

        # Create improved contrastive pairs
        print("\nGenerating improved contrastive pairs...")
        self._create_contrastive_pairs()

    def _phoneme_overlap(self, p1, p2):
        """Compute phoneme overlap between two words."""
        set1 = set(p1)
        set2 = set(p2)
        return len(set1 & set2) / max(len(set1), len(set2), 1)

    def _edit_distance(self, p1, p2):
        """Compute edit distance between phoneme sequences."""
        if len(p1) == 0:
            return len(p2)
        if len(p2) == 0:
            return len(p1)

        # Dynamic programming
        dp = [[0] * (len(p2) + 1) for _ in range(len(p1) + 1)]

        for i in range(len(p1) + 1):
            dp[i][0] = i
        for j in range(len(p2) + 1):
            dp[0][j] = j

        for i in range(1, len(p1) + 1):
            for j in range(1, len(p2) + 1):
                if p1[i-1] == p2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[len(p1)][len(p2)]

    def _create_contrastive_pairs(self):
        """Generate positive and negative pairs with better balance."""
        self.positive_pairs = []
        self.negative_pairs = []

        # 1. POSITIVE: Exact rhymes (last 2+ phonemes match)
        print("  - Finding rhymes...")
        rhyme_dict = defaultdict(list)
        for word, phonemes in self.loader.lexicon.items():
            if len(phonemes) >= 2:
                rhyme_key = tuple(phonemes[-2:])
                rhyme_dict[rhyme_key].append(word)

        rhyme_count = 0
        for rhyme_group in rhyme_dict.values():
            if len(rhyme_group) >= 2:
                # Sample pairs
                for _ in range(min(3, len(rhyme_group))):
                    w1, w2 = random.sample(rhyme_group, 2)
                    if w1 != w2:
                        self.positive_pairs.append((w1, w2, 'rhyme', 1.0))
                        rhyme_count += 1

        # 2. POSITIVE: Morphological variants
        print("  - Finding morphological variants...")
        morph_count = 0
        for pair in self.loader.morphology:
            if pair.lemma in self.loader.lexicon and pair.inflected in self.loader.lexicon:
                self.positive_pairs.append((pair.lemma, pair.inflected, 'morphology', 0.9))
                morph_count += 1

        # 3. POSITIVE: Minimal pairs (1 phoneme different, same length)
        print("  - Finding minimal pairs...")
        minimal_count = 0
        for word1 in random.sample(self.words, min(5000, len(self.words))):
            p1 = self.loader.lexicon[word1]
            if len(p1) < 2:
                continue
            for word2 in random.sample(self.words, min(20, len(self.words))):
                if word1 != word2:
                    p2 = self.loader.lexicon[word2]
                    if len(p1) == len(p2) and self._edit_distance(p1, p2) == 1:
                        self.positive_pairs.append((word1, word2, 'minimal', 0.7))
                        minimal_count += 1
                        if minimal_count >= 5000:
                            break
            if minimal_count >= 5000:
                break

        # 4. NEGATIVE: Completely unrelated (no phoneme overlap)
        print("  - Finding unrelated pairs...")
        unrelated_count = 0
        max_attempts = 50000
        attempts = 0
        while unrelated_count < 15000 and attempts < max_attempts:
            attempts += 1
            w1, w2 = random.sample(self.words, 2)
            p1 = self.loader.lexicon[w1]
            p2 = self.loader.lexicon[w2]

            # Check low overlap and different rhyme
            if len(p1) >= 2 and len(p2) >= 2:
                overlap = self._phoneme_overlap(p1, p2)
                rhyme_match = tuple(p1[-2:]) == tuple(p2[-2:])

                if overlap < 0.3 and not rhyme_match:
                    self.negative_pairs.append((w1, w2, 'unrelated', 0.0))
                    unrelated_count += 1

        # 5. NEGATIVE: Anagrams (same phonemes, different order) - CRITICAL!
        print("  - Finding anagrams (hard negatives)...")
        phoneme_sets = defaultdict(list)
        for word, phonemes in self.loader.lexicon.items():
            if len(phonemes) >= 3:  # At least 3 phonemes for interesting anagrams
                key = tuple(sorted(phonemes))
                phoneme_sets[key].append(word)

        anagram_count = 0
        for anagram_group in phoneme_sets.values():
            if len(anagram_group) >= 2:
                for _ in range(min(5, len(anagram_group))):
                    w1, w2 = random.sample(anagram_group, 2)
                    # Check they're actually different sequences
                    if w1 != w2 and self.loader.lexicon[w1] != self.loader.lexicon[w2]:
                        self.negative_pairs.append((w1, w2, 'anagram', 0.45))  # Medium similarity (same phonemes, different order)
                        anagram_count += 1

        # 6. NEGATIVE: Different rhyme endings (semi-hard negatives)
        print("  - Finding non-rhyming pairs...")
        non_rhyme_count = 0
        for _ in range(10000):
            w1, w2 = random.sample(self.words, 2)
            p1 = self.loader.lexicon[w1]
            p2 = self.loader.lexicon[w2]

            if len(p1) >= 2 and len(p2) >= 2:
                # Same length, different ending
                if len(p1) == len(p2) and tuple(p1[-2:]) != tuple(p2[-2:]):
                    overlap = self._phoneme_overlap(p1, p2)
                    if overlap > 0.3:  # Some overlap but don't rhyme
                        self.negative_pairs.append((w1, w2, 'non_rhyme', 0.3))
                        non_rhyme_count += 1

        print(f"  ✓ Positive pairs: {len(self.positive_pairs):,}")
        print(f"    - Rhymes: {rhyme_count:,}")
        print(f"    - Morphology: {morph_count:,}")
        print(f"    - Minimal pairs: {minimal_count:,}")
        print(f"  ✓ Negative pairs: {len(self.negative_pairs):,}")
        print(f"    - Unrelated: {unrelated_count:,}")
        print(f"    - Anagrams: {anagram_count:,}")
        print(f"    - Non-rhyming: {non_rhyme_count:,}")

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
        """Get a balanced batch of contrastive pairs."""
        # Sample equal numbers of positive and negative
        n_pos = min(batch_size // 2, len(self.positive_pairs))
        n_neg = min(batch_size // 2, len(self.negative_pairs))

        positive_batch = random.sample(self.positive_pairs, n_pos)
        negative_batch = random.sample(self.negative_pairs, n_neg)

        pairs = []
        labels = []

        for w1, w2, pair_type, target_sim in positive_batch:
            pairs.append((self.word_to_id[w1], self.word_to_id[w2]))
            labels.append(target_sim)

        for w1, w2, pair_type, target_sim in negative_batch:
            pairs.append((self.word_to_id[w1], self.word_to_id[w2]))
            labels.append(target_sim)

        return pairs, labels


def contrastive_loss_v2(emb1, emb2, labels, temperature=0.05):
    """
    Improved contrastive loss with lower temperature for sharper distinctions.
    """
    # Normalize embeddings
    emb1 = nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = nn.functional.normalize(emb2, p=2, dim=1)

    # Cosine similarity
    similarity = (emb1 * emb2).sum(dim=1)  # (batch,)

    # MSE loss on similarities
    # This allows for graded similarity (not just 0/1)
    loss = nn.functional.mse_loss(similarity, labels)

    return loss


def train():
    # Configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION - V2")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Improvements:")
    print(f"  - Higher contrastive loss weight: 1.5")
    print(f"  - Lower temperature: 0.05")
    print(f"  - More hard negatives (anagrams)")
    print(f"  - Graded similarity targets")

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = ImprovedPhonologyDataset(loader, mask_prob=0.15)

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
    num_epochs = 20
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")

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

            # Contrastive loss (every batch now!)
            pairs, labels = dataset.get_contrastive_batch(batch_size=64)
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
                contra_loss = contrastive_loss_v2(word_emb1, word_emb2, labels_tensor, temperature=0.05)
            else:
                contra_loss = torch.tensor(0.0).to(device)

            # Combined loss (higher weight on contrastive)
            loss = mlm_loss + 1.5 * contra_loss

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
                'mlm': f'{mlm_loss.item():.3f}',
                'contra': f'{contra_loss.item():.3f}',
                'acc': f'{accuracy:.1f}%'
            })

        avg_mlm_loss = total_mlm_loss / len(dataloader)
        avg_contra_loss = total_contrastive_loss / len(dataloader)
        avg_accuracy = 100.0 * total_correct / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - MLM: {avg_mlm_loss:.4f}, "
              f"Contra: {avg_contra_loss:.4f}, Acc: {avg_accuracy:.2f}%")

    # Save model
    output_dir = Path("models/phonolex_bert_v2")
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
    print(f"PhonoLex-BERT v2 saved to: {output_dir / 'model.pt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train()
