#!/usr/bin/env python3
"""
Multi-Task English Phonology Learning

Combines ALL English data sources:
1. CMU Dict (125K words) - phonotactics
2. ipa-dict (191K words) - more pronunciations
3. SIGMORPHON (80K) - inflectional morphology
4. UniMorph (225K) - derivational morphology

Tasks:
- Skip-gram context prediction
- Allomorph prediction
- Rhyme detection
- Phonological neighborhood effects
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import re

from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader


def parse_ipa_dict_pronunciation(pron_str):
    """
    Parse ipa-dict pronunciation format: /ˈbaʊt/, /kəz/

    Returns list of phonemes (stripped of stress marks for now)
    """
    # Remove slashes and spaces
    pron = pron_str.strip().strip('/').strip()

    # For now, split into individual characters
    # (This is crude - IPA has multi-char phonemes like tʃ, but we'll align with CMU)
    # Better would be to use a proper IPA parser
    phonemes = []

    # Common IPA mappings to single phonemes
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


class MultiTaskEnglishDataLoader:
    """Load ALL English phonological data"""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

        # Load base CMU data
        self.base_loader = EnglishPhonologyLoader(data_dir=data_dir)

        # Combined lexicon
        self.lexicon = dict(self.base_loader.lexicon)

        print("\n" + "=" * 70)
        print("LOADING ADDITIONAL DATA SOURCES")
        print("=" * 70)

        # Load ipa-dict
        self._load_ipa_dict()

        # Load UniMorph derivations
        self._load_unimorph()

        # Get final phoneme inventory
        self.phonemes = set()
        for phones in self.lexicon.values():
            self.phonemes.update(phones)

        print(f"\n✓ Final lexicon: {len(self.lexicon):,} words")
        print(f"✓ Final phoneme inventory: {len(self.phonemes)} phonemes")

    def _load_ipa_dict(self):
        """Load ipa-dict pronunciations"""
        print("\nLoading ipa-dict...")

        added = 0
        for lang in ['en_US', 'en_UK']:
            path = self.data_dir / "learning_datasets" / "ipa-dict" / "data" / f"{lang}.txt"

            if not path.exists():
                continue

            with open(path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        continue

                    word, pron = parts
                    word = word.lower()

                    # Handle multiple pronunciations
                    prons = pron.split(', ')

                    for p in prons:
                        try:
                            phonemes = parse_ipa_dict_pronunciation(p)

                            if phonemes and word not in self.lexicon:
                                self.lexicon[word] = phonemes
                                added += 1
                        except:
                            pass

        print(f"  ✓ Added {added:,} new words from ipa-dict")

    def _load_unimorph(self):
        """Load UniMorph derivations"""
        print("\nLoading UniMorph derivations...")

        path = self.data_dir / "learning_datasets" / "unimorph-eng" / "eng.derivations.tsv"

        if not path.exists():
            print("  ✗ UniMorph not found")
            return

        self.derivations = []

        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 4:
                    continue

                source, target, relation, affix = parts
                source = source.lower()
                target = target.lower()

                # Get pronunciations
                source_phones = self.lexicon.get(source)
                target_phones = self.lexicon.get(target)

                if source_phones and target_phones:
                    self.derivations.append({
                        'source': source,
                        'target': target,
                        'source_phones': source_phones,
                        'target_phones': target_phones,
                        'relation': relation,
                        'affix': affix
                    })

        print(f"  ✓ Loaded {len(self.derivations):,} derivations")
        with_phones = sum(1 for d in self.derivations if d['source_phones'] and d['target_phones'])
        print(f"  ✓ {with_phones:,} derivations have pronunciations")


class MultiTaskEnglishDataset(Dataset):
    """
    Dataset for multi-task English phonology learning

    Tasks:
    1. Context prediction (skip-gram)
    2. Allomorph prediction
    3. Phonological similarity
    """

    def __init__(self, loader: MultiTaskEnglishDataLoader, window_size: int = 2):
        self.loader = loader
        self.window_size = window_size

        # Build phoneme vocabulary
        self.phoneme_to_id = {p: i for i, p in enumerate(sorted(loader.phonemes))}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

        print(f"\nPhoneme vocabulary: {len(self.phoneme_to_id)} phonemes")

        # Generate skip-gram examples
        print("Generating context examples...")
        self.context_examples = []

        for word, phonemes in loader.lexicon.items():
            for i, center in enumerate(phonemes):
                start = max(0, i - window_size)
                end = min(len(phonemes), i + window_size + 1)

                context = []
                for j in range(start, end):
                    if j != i and phonemes[j] in self.phoneme_to_id:
                        context.append(phonemes[j])

                if context and center in self.phoneme_to_id:
                    center_id = self.phoneme_to_id[center]
                    context_ids = [self.phoneme_to_id[c] for c in context]

                    if context_ids:
                        self.context_examples.append((center_id, context_ids))

        print(f"  ✓ Generated {len(self.context_examples):,} context examples")

        # Generate allomorph examples from SIGMORPHON
        print("Generating allomorph examples...")
        self.allomorph_examples = []

        for pair in loader.base_loader.morphology:
            if not pair.lemma_phonemes or not pair.inflected_phonemes:
                continue

            # Get final phoneme of lemma
            final_phone = pair.lemma_phonemes[-1]
            if final_phone not in self.phoneme_to_id:
                continue

            # Extract suffix
            lemma_len = len(pair.lemma_phonemes)
            inflected_len = len(pair.inflected_phonemes)

            if inflected_len > lemma_len:
                suffix = tuple(pair.inflected_phonemes[lemma_len:])

                self.allomorph_examples.append({
                    'final_phoneme_id': self.phoneme_to_id[final_phone],
                    'suffix': suffix,
                    'features': pair.features
                })

        # Build suffix vocabulary
        suffix_counts = defaultdict(int)
        for ex in self.allomorph_examples:
            suffix_counts[ex['suffix']] += 1

        # Keep suffixes that appear at least 5 times
        self.suffix_to_id = {}
        for suffix, count in suffix_counts.items():
            if count >= 5:
                self.suffix_to_id[suffix] = len(self.suffix_to_id)

        self.id_to_suffix = {i: s for s, i in self.suffix_to_id.items()}

        # Filter examples to only include known suffixes
        self.allomorph_examples = [
            ex for ex in self.allomorph_examples
            if ex['suffix'] in self.suffix_to_id
        ]

        # Add suffix IDs
        for ex in self.allomorph_examples:
            ex['suffix_id'] = self.suffix_to_id[ex['suffix']]

        print(f"  ✓ Generated {len(self.allomorph_examples):,} allomorph examples")
        print(f"  ✓ {len(self.suffix_to_id)} unique suffixes")

    def __len__(self):
        return len(self.context_examples)

    def __getitem__(self, idx):
        # Primary: context example
        center, context = self.context_examples[idx]

        # Pad/truncate context
        if len(context) > 4:
            context = context[:4]
        else:
            context = context + [context[0]] * (4 - len(context))

        item = {
            'center_id': center,
            'context_ids': context
        }

        # Secondary: maybe add allomorph example
        if self.allomorph_examples and np.random.random() < 0.3:
            allo_ex = self.allomorph_examples[np.random.randint(len(self.allomorph_examples))]
            item['allomorph_final_id'] = allo_ex['final_phoneme_id']
            item['allomorph_suffix_id'] = allo_ex['suffix_id']

        return item


def collate_fn(batch):
    """Collate batch"""
    collated = {
        'center_ids': [],
        'context_ids': [],
        'allomorph_final_ids': [],
        'allomorph_suffix_ids': []
    }

    for item in batch:
        collated['center_ids'].append(item['center_id'])
        collated['context_ids'].append(item['context_ids'])

        if 'allomorph_final_id' in item:
            collated['allomorph_final_ids'].append(item['allomorph_final_id'])
            collated['allomorph_suffix_ids'].append(item['allomorph_suffix_id'])

    # Convert to tensors
    result = {
        'center_ids': torch.tensor(collated['center_ids'], dtype=torch.long),
        'context_ids': torch.tensor(collated['context_ids'], dtype=torch.long)
    }

    if collated['allomorph_final_ids']:
        result['allomorph_final_ids'] = torch.tensor(collated['allomorph_final_ids'], dtype=torch.long)
        result['allomorph_suffix_ids'] = torch.tensor(collated['allomorph_suffix_ids'], dtype=torch.long)

    return result


class MultiTaskModel(nn.Module):
    """Multi-task phoneme embedding model"""

    def __init__(self, num_phonemes, num_suffixes, embedding_dim=32):
        super().__init__()

        # Phoneme embeddings
        self.embeddings = nn.Embedding(num_phonemes, embedding_dim)

        # Task heads
        self.context_proj = nn.Linear(embedding_dim, num_phonemes)
        self.allomorph_proj = nn.Linear(embedding_dim, num_suffixes)

        # Init
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, phoneme_ids):
        return self.embeddings(phoneme_ids)

    def compute_loss(self, batch):
        """Compute multi-task loss"""
        losses = {}

        # Context prediction
        center_emb = self(batch['center_ids'])
        context_logits = self.context_proj(center_emb)

        context_loss = 0
        for i in range(batch['context_ids'].size(1)):
            context_loss += nn.functional.cross_entropy(context_logits, batch['context_ids'][:, i])
        losses['context'] = context_loss / batch['context_ids'].size(1)

        # Allomorph prediction
        if 'allomorph_final_ids' in batch:
            final_emb = self(batch['allomorph_final_ids'])
            suffix_logits = self.allomorph_proj(final_emb)
            losses['allomorph'] = nn.functional.cross_entropy(suffix_logits, batch['allomorph_suffix_ids'])

        # Total loss
        total = sum(losses.values())

        return total, losses


def train():
    """Train multi-task English phonology model"""

    print("=" * 70)
    print("MULTI-TASK ENGLISH PHONOLOGY LEARNING")
    print("=" * 70)

    # Load data
    loader = MultiTaskEnglishDataLoader()
    dataset = MultiTaskEnglishDataset(loader, window_size=2)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create model
    model = MultiTaskModel(
        num_phonemes=len(dataset.phoneme_to_id),
        num_suffixes=len(dataset.suffix_to_id),
        embedding_dim=32
    )

    print(f"\nModel: {len(dataset.phoneme_to_id)} phonemes → 32-dim embeddings")
    print(f"  Suffix vocabulary: {len(dataset.suffix_to_id)} suffixes")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_context = 0
        total_allomorph = 0
        num_batches = 0
        num_allomorph_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            loss, losses = model.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_context += losses['context'].item()
            if 'allomorph' in losses:
                total_allomorph += losses['allomorph'].item()
                num_allomorph_batches += 1

            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        avg_context = total_context / num_batches
        avg_allomorph = total_allomorph / num_allomorph_batches if num_allomorph_batches > 0 else 0

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, context={avg_context:.4f}, allomorph={avg_allomorph:.4f}")

    # Save
    output_dir = Path("models/english_multitask")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'phoneme_to_id': dataset.phoneme_to_id,
        'id_to_phoneme': dataset.id_to_phoneme,
        'suffix_to_id': dataset.suffix_to_id,
        'id_to_suffix': dataset.id_to_suffix,
        'embedding_dim': 32
    }, output_dir / "model.pt")

    print(f"\n✓ Model saved to {output_dir / 'model.pt'}")

    # Test
    print("\n" + "=" * 70)
    print("TESTING EMBEDDINGS")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        all_embeddings = model(torch.arange(len(dataset.phoneme_to_id))).numpy()

    def cosine_sim(e1, e2):
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)

    test_phonemes = ['p', 'b', 't', 's', 'z']

    for phone in test_phonemes:
        if phone not in dataset.phoneme_to_id:
            continue

        idx = dataset.phoneme_to_id[phone]
        emb = all_embeddings[idx]

        sims = []
        for other_phone, other_idx in dataset.phoneme_to_id.items():
            if other_phone == phone:
                continue
            sim = cosine_sim(emb, all_embeddings[other_idx])
            sims.append((other_phone, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

        print(f"\n/{phone}/ most similar:")
        for p, s in sims[:5]:
            print(f"  /{p}/  {s:.4f}")


if __name__ == '__main__':
    train()
