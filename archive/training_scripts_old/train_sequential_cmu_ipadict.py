#!/usr/bin/env python3
"""
Sequential Phonological Embeddings - CMU + ipa-dict

Combines the best architecture (sequential model with Phoible initialization)
with training on:
- CMU Dictionary (125K words, General American English)
- ipa-dict (US + UK pronunciations for additional word coverage)

Training objective: Next-phoneme prediction (teaches adjacency patterns)
Similarity: Soft Levenshtein distance on contextual embedding sequences

Note: SIGMORPHON and UniMorph excluded - they mostly add suffix patterns,
not genuinely new phoneme sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from experiments.training_scripts.train_phonology_bert import PhonoLexBERT
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv


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


def parse_ipa_dict_pronunciation(pron_str):
    """
    Parse ipa-dict pronunciation format: /ˈbaʊt/, /kəz/
    Returns list of phonemes (stripped of stress marks)
    """
    # Remove slashes and spaces
    pron = pron_str.strip().strip('/').strip()

    phonemes = []

    # Common IPA multi-char phonemes
    multi_char = {
        'tʃ': 'tʃ',
        'dʒ': 'dʒ',
        'ʃ': 'ʃ',
        'ʒ': 'ʒ',
        'θ': 'θ',
        'ð': 'ð',
        'ŋ': 'ŋ',
    }

    i = 0
    while i < len(pron):
        # Skip stress marks
        if pron[i] in 'ˈˌ':
            i += 1
            continue

        # Check for multi-char phonemes
        matched = False
        for multi, single in multi_char.items():
            if pron[i:i+len(multi)] == multi:
                phonemes.append(single)
                i += len(multi)
                matched = True
                break

        if not matched:
            phonemes.append(pron[i])
            i += 1

    return phonemes


def load_ipa_dict(data_dir="data"):
    """Load ipa-dict pronunciations (US + UK)"""
    print("=" * 70)
    print("LOADING IPA-DICT")
    print("=" * 70)

    lexicon = {}

    for lang in ['en_US', 'en_UK']:
        path = Path(data_dir) / "learning_datasets" / "ipa-dict" / "data" / f"{lang}.txt"

        if not path.exists():
            print(f"⚠ {lang}.txt not found, skipping")
            continue

        added = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                word, pron = parts
                word = word.lower()

                # Handle multiple pronunciations (some entries have "pron1, pron2")
                prons = pron.split(', ')

                for p in prons:
                    try:
                        phonemes = parse_ipa_dict_pronunciation(p)

                        # Only add if not already in lexicon (CMU takes priority)
                        if phonemes and word not in lexicon:
                            lexicon[word] = phonemes
                            added += 1
                    except Exception:
                        pass

        print(f"✓ Added {added:,} words from {lang}")

    print(f"✓ Total ipa-dict words: {len(lexicon):,}")
    print()
    return lexicon


class CMUIPADictDataset(Dataset):
    """
    Next-phoneme prediction dataset using CMU + ipa-dict.

    CMU provides base coverage (125K words).
    ipa-dict adds additional word coverage (especially UK variants).
    """

    def __init__(self, phoneme_to_id):
        self.phoneme_to_id = phoneme_to_id

        self.PAD = '<PAD>'
        self.CLS = '<CLS>'
        self.SEP = '<SEP>'

        print("=" * 70)
        print("LOADING CMU + IPA-DICT")
        print("=" * 70)
        print()

        # Load CMU (base)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            cmu_loader = EnglishPhonologyLoader()

        print(f"✓ CMU Dictionary: {len(cmu_loader.lexicon):,} words")

        # Load ipa-dict
        ipa_dict_lexicon = load_ipa_dict(data_dir="data")

        # Combine: CMU takes priority
        combined_lexicon = dict(cmu_loader.lexicon)

        added = 0
        for word, phonemes in ipa_dict_lexicon.items():
            if word not in combined_lexicon:
                combined_lexicon[word] = phonemes
                added += 1

        print(f"✓ Added {added:,} new words from ipa-dict")
        print(f"✓ Combined lexicon: {len(combined_lexicon):,} words")

        # Extract phoneme inventory
        all_phonemes = set()
        for phonemes in combined_lexicon.values():
            all_phonemes.update(phonemes)

        print(f"✓ Phoneme inventory: {len(all_phonemes)} unique phonemes")
        print()

        # Build training examples
        self.word_data = []

        for word, phonemes in combined_lexicon.items():
            # Convert to IDs
            phoneme_ids = [self.phoneme_to_id['<CLS>']] + \
                         [self.phoneme_to_id.get(p, self.phoneme_to_id['<PAD>']) for p in phonemes] + \
                         [self.phoneme_to_id['<SEP>']]

            self.word_data.append({
                'word': word,
                'phoneme_ids': phoneme_ids
            })

        self.max_length = max(len(d['phoneme_ids']) for d in self.word_data) + 2

        print(f"✓ Dataset: {len(self.word_data):,} training examples")
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
    print("Learns: Phonotactic patterns from CMU + ipa-dict")
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
    print("SEQUENTIAL EMBEDDINGS - CMU + IPA-DICT")
    print("=" * 70)
    print("Architecture: Phoible-initialized transformer")
    print("Training: Next-phoneme prediction")
    print("Data: CMU Dictionary + ipa-dict (US + UK)")
    print()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Build phoneme vocabulary
    print("Building phoneme vocabulary...")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cmu_loader = EnglishPhonologyLoader()

    ipa_dict_lexicon = load_ipa_dict(data_dir="data")

    # Collect all phonemes from both sources
    all_phonemes = set(cmu_loader.english_phonemes)
    for phonemes in ipa_dict_lexicon.values():
        all_phonemes.update(phonemes)

    phonemes = ['<PAD>', '<CLS>', '<SEP>', '<MASK>'] + sorted(all_phonemes)
    phoneme_to_id = {p: i for i, p in enumerate(phonemes)}

    print(f"✓ Phoneme vocabulary: {len(phoneme_to_id)} phonemes\n")

    # Create dataset
    dataset = CMUIPADictDataset(phoneme_to_id)

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
    }, 'models/sequential/cmu_ipadict_final.pt')

    print("✓ Model saved: models/sequential/cmu_ipadict_final.pt\n")

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
    print("✓ CMU Dictionary (125K words) + ipa-dict (additional coverage)")
    print("✓ Phoible-initialized for phonological grounding")
    print("✓ Sequential similarity preserves order (cat ≠ act)")
    print("✓ Soft Levenshtein handles insertions/deletions")
    print("✓ Learns from diverse English phoneme sequences")
    print("✓ No morphology datasets (just genuine word sequences)")
    print()


if __name__ == '__main__':
    main()
