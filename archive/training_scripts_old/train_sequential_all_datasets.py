#!/usr/bin/env python3
"""
Sequential Phonological Embeddings - ALL DATASETS

Combines the best architecture (sequential model with Phoible initialization)
with training on ALL English datasets:
- CMU Dictionary (125K words)
- SIGMORPHON morphology (80K pairs)
- ipa-dict (US + UK pronunciations)
- UniMorph derivations

Training objective: Next-phoneme prediction (teaches adjacency patterns)
Similarity: Soft Levenshtein distance on contextual embedding sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import numpy as np
import random
from scipy.spatial.distance import cosine as cosine_dist

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from experiments.training_scripts.train_phonology_bert import PhonoLexBERT
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv
from experiments.training_scripts.train_english_multitask import MultiTaskEnglishDataLoader


def load_phoible_init(phoneme_to_id, d_model=128):
    """Initialize with Phoible features (76-dim → d_model)"""
    print("=" * 70)
    print("INITIALIZING WITH PHOIBLE FEATURES")
    print("=" * 70)

    phoible_csv = Path('data/phoible/english/phoible-english.csv')
    phoible_data = load_phoible_csv(str(phoible_csv))
    vectorizer = PhonemeVectorizer(encoding_scheme='three_way')

    phoible_features = {}
    for phoneme_data in phoible_data:
        vec = vectorizer.vectorize(phoneme_data)
        if vec.phoneme not in phoible_features:
            phoible_features[vec.phoneme] = vec.endpoints_76d

    # Create projection layer
    feature_projection = nn.Linear(76, d_model)
    embedding_matrix = torch.randn(len(phoneme_to_id), d_model) * 0.01

    # Initialize phonemes we have features for
    initialized = 0
    with torch.no_grad():
        for phoneme, idx in phoneme_to_id.items():
            if phoneme in phoible_features:
                features = torch.FloatTensor(phoible_features[phoneme])
                embedding_matrix[idx] = feature_projection(features)
                initialized += 1

    print(f"✓ Initialized {initialized}/{len(phoneme_to_id)} phonemes with Phoible features")
    print()
    return embedding_matrix


class AllDatasetsSequentialDataset(Dataset):
    """
    Next-phoneme prediction dataset using ALL English data sources:
    - CMU Dictionary
    - SIGMORPHON morphology
    - ipa-dict (US + UK)
    - UniMorph derivations
    """

    def __init__(self, phoneme_to_id):
        self.phoneme_to_id = phoneme_to_id

        self.PAD = '<PAD>'
        self.CLS = '<CLS>'
        self.SEP = '<SEP>'

        print("=" * 70)
        print("LOADING ALL ENGLISH DATASETS")
        print("=" * 70)
        print()

        # Load all datasets via MultiTaskEnglishDataLoader
        multitask_loader = MultiTaskEnglishDataLoader(data_dir="data")

        print(f"\n✓ Combined lexicon: {len(multitask_loader.lexicon):,} unique words")
        print(f"✓ Phoneme inventory: {len(multitask_loader.phonemes)} phonemes")

        # Build training examples from all words
        self.word_data = []

        for word, phonemes in multitask_loader.lexicon.items():
            # Convert to IDs
            phoneme_ids = [self.phoneme_to_id['<CLS>']] + \
                         [self.phoneme_to_id.get(p, self.phoneme_to_id['<PAD>']) for p in phonemes] + \
                         [self.phoneme_to_id['<SEP>']]

            self.word_data.append({
                'word': word,
                'phoneme_ids': phoneme_ids
            })

        self.max_length = max(len(d['phoneme_ids']) for d in self.word_data) + 2

        print(f"\n✓ Dataset: {len(self.word_data):,} training examples")
        print(f"✓ Max sequence length: {self.max_length}")
        print()

    def __len__(self):
        return len(self.word_data)

    def __getitem__(self, idx):
        data = self.word_data[idx]
        phoneme_ids = data['phoneme_ids']

        # Next-phoneme prediction
        input_ids = phoneme_ids[:-1]  # All but last
        target_ids = phoneme_ids[1:]   # Shifted by 1

        seq_len = len(input_ids)
        padded_input = input_ids + [self.phoneme_to_id[self.PAD]] * (self.max_length - seq_len)
        padded_target = target_ids + [0] * (self.max_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)

        return {
            'input_ids': torch.LongTensor(padded_input),
            'target_ids': torch.LongTensor(padded_target),
            'attention_mask': torch.LongTensor(attention_mask),
        }


def sequence_similarity(seq1, seq2):
    """
    Soft Levenshtein distance on embedding sequences.

    Handles:
    - Insertions/deletions (different word lengths)
    - Substitutions with soft costs (embedding similarity)
    - Preserves order (cat ≠ act)
    """
    len1 = len(seq1)
    len2 = len(seq2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Dynamic programming for edit distance
    dp = np.zeros((len1 + 1, len2 + 1))

    # Initialize
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill DP table with soft costs
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # Soft substitution cost based on embedding similarity
            dist = cosine_dist(seq1[i-1], seq2[j-1])
            match_cost = dist

            delete_cost = dp[i-1][j] + 1.0
            insert_cost = dp[i][j-1] + 1.0
            match_or_substitute = dp[i-1][j-1] + match_cost

            dp[i][j] = min(delete_cost, insert_cost, match_or_substitute)

    # Normalize to [0, 1] similarity
    max_len = max(len1, len2)
    normalized_distance = dp[len1][len2] / max_len if max_len > 0 else 0
    similarity = max(0.0, 1.0 - normalized_distance)

    return similarity


def train(model, dataset, device, num_epochs=20):
    """Train with next-phoneme prediction"""
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("=" * 70)
    print("TRAINING: NEXT-PHONEME PREDICTION")
    print("=" * 70)
    print("Task: Predict next phoneme from context")
    print("Learns: Phonotactic patterns across all datasets")
    print()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            mlm_logits, _, _ = model(input_ids, attention_mask)

            loss = criterion(mlm_logits.view(-1, mlm_logits.size(-1)), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = mlm_logits.argmax(dim=-1)
            mask = target_ids != 0
            correct += ((preds == target_ids) & mask).sum().item()
            total += mask.sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    print()
    return model


def main():
    print("\n" + "=" * 70)
    print("SEQUENTIAL EMBEDDINGS - ALL DATASETS")
    print("=" * 70)
    print("Architecture: Phoible-initialized transformer")
    print("Training: Next-phoneme prediction")
    print("Data: CMU + SIGMORPHON + ipa-dict + UniMorph")
    print()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Build phoneme vocabulary from all datasets
    print("Building phoneme vocabulary from all datasets...")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        multitask_loader = MultiTaskEnglishDataLoader(data_dir="data")

    phonemes = ['<PAD>', '<CLS>', '<SEP>', '<MASK>'] + sorted(multitask_loader.phonemes)
    phoneme_to_id = {p: i for i, p in enumerate(phonemes)}

    print(f"✓ Phoneme vocabulary: {len(phoneme_to_id)} phonemes\n")

    # Create dataset
    dataset = AllDatasetsSequentialDataset(phoneme_to_id)

    # Initialize with Phoible
    print("Initializing embeddings with Phoible features...")
    phoible_init = load_phoible_init(phoneme_to_id, d_model=128)

    # Create model
    model = PhonoLexBERT(
        num_phonemes=len(phoneme_to_id),
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        max_len=dataset.max_length
    ).to(device)

    # Set Phoible initialization
    with torch.no_grad():
        model.phoneme_embeddings.weight.data = phoible_init.to(device)

    print("✓ Model initialized\n")

    # Train
    model = train(model, dataset, device, num_epochs=20)

    # Save
    Path('models/sequential').mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': phoneme_to_id,
        'max_length': dataset.max_length
    }, 'models/sequential/all_datasets_final.pt')

    print("✓ Model saved: models/sequential/all_datasets_final.pt\n")

    # Evaluate
    print("=" * 70)
    print("EVALUATION: SEQUENTIAL SIMILARITY")
    print("=" * 70)
    print()

    model.eval()

    # Load base lexicon for evaluation
    with contextlib.redirect_stdout(io.StringIO()):
        base_loader = EnglishPhonologyLoader()

    def get_seq(word):
        """Get contextual phoneme embedding sequence for a word"""
        if word not in base_loader.lexicon:
            return None

        phonemes = base_loader.lexicon[word]
        CLS, SEP, PAD = phoneme_to_id['<CLS>'], phoneme_to_id['<SEP>'], phoneme_to_id['<PAD>']
        ids = [CLS] + [phoneme_to_id.get(p, PAD) for p in phonemes] + [SEP]
        seq_len = len(ids)
        padded = ids + [PAD] * (dataset.max_length - seq_len)
        mask = [1] * seq_len + [0] * (dataset.max_length - seq_len)

        with torch.no_grad():
            _, _, contextual = model(
                torch.LongTensor([padded]).to(device),
                torch.LongTensor([mask]).to(device)
            )

        # Extract phoneme embeddings (skip CLS/SEP)
        phoneme_embs = contextual[0, 1:seq_len-1].cpu().numpy()
        return phoneme_embs

    # Test cases
    tests = [
        ('cat', 'bat', 'rhyme'),
        ('cat', 'act', 'anagram'),
        ('cat', 'mat', 'rhyme'),
        ('dog', 'fog', 'rhyme'),
        ('run', 'running', 'morphology'),
        ('cat', 'dog', 'unrelated'),
        ('computer', 'commuter', 'sound-alike'),
        ('make', 'take', 'rhyme'),
    ]

    print(f'{"Word 1":<12} {"Word 2":<12} {"Similarity":<12} {"Type":<15}')
    print('-' * 60)

    for w1, w2, test_type in tests:
        seq1 = get_seq(w1)
        seq2 = get_seq(w2)

        if seq1 is not None and seq2 is not None:
            sim = sequence_similarity(seq1, seq2)
            print(f'{w1:<12} {w2:<12} {sim:.3f}        {test_type:<15}')

    print()
    print("=" * 70)
    print("KEY PROPERTIES:")
    print("=" * 70)
    print("✓ Trained on ALL English datasets (not just CMU)")
    print("✓ Phoible-initialized for phonological grounding")
    print("✓ Sequential similarity preserves order (cat ≠ act)")
    print("✓ Soft Levenshtein handles insertions/deletions")
    print("✓ Learns phonotactic patterns from diverse data")
    print()


if __name__ == '__main__':
    main()
