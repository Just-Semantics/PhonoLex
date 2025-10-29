#!/usr/bin/env python3
"""
Curriculum Learning - FIXED

Just use the EXACT working code (train_phonology_bert_v2.py)
but initialize phoneme embeddings with Phoible features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from train_phonology_bert_v2 import ImprovedPhonologyDataset, PhonoLexBERT, contrastive_loss_v2
from data.mappings.phoneme_vectorizer import PhonemeVectorizer, load_phoible_csv


def load_phoible_init(phoneme_to_id, d_model=128):
    """Load Phoible features and create initialization matrix"""
    print("=" * 70)
    print("LOADING PHOIBLE FEATURES FOR INITIALIZATION")
    print("=" * 70)

    # Load Phoible
    phoible_csv = Path('data/phoible/english/phoible-english.csv')
    phoible_data = load_phoible_csv(str(phoible_csv))
    vectorizer = PhonemeVectorizer(encoding_scheme='three_way')

    phoible_features = {}
    for phoneme_data in phoible_data:
        vec = vectorizer.vectorize(phoneme_data)
        if vec.phoneme not in phoible_features:
            phoible_features[vec.phoneme] = vec.endpoints_76d  # 76-dim

    print(f"✓ Loaded {len(phoible_features)} phoneme feature vectors (76-dim)")

    # Create projection layer and embedding matrix
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

    print(f"✓ Initialized {initialized}/{len(phoneme_to_id)} phoneme embeddings with Phoible")
    print()

    return embedding_matrix


def main():
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING (FIXED)")
    print("=" * 70)
    print("Same code as working model, but Phoible-initialized")
    print()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        loader = EnglishPhonologyLoader()
        dataset = ImprovedPhonologyDataset(loader, mask_prob=0.15)

    print(f"Loaded {len(loader.lexicon):,} words")
    print(f"Phonemes: {len(dataset.phoneme_to_id)}\n")

    # Load Phoible initialization
    phoible_init = load_phoible_init(dataset.phoneme_to_id, d_model=128)

    # Create model (same as working version)
    model = PhonoLexBERT(
        num_phonemes=len(dataset.phoneme_to_id),
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        max_len=dataset.max_length
    ).to(device)

    # INITIALIZE with Phoible features
    with torch.no_grad():
        model.phoneme_embeddings.weight.data = phoible_init.to(device)

    print("✓ Model initialized with Phoible features\n")

    # Train (exact same as working model)
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print()

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=dataset.phoneme_to_id[dataset.PAD])

    contra_weight = 0.5
    temperature = 0.05
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)

            mlm_logits, _, _ = model(input_ids, attention_masks)
            mlm_loss = mlm_criterion(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                target_ids.view(-1)
            )

            # Contrastive loss
            pairs, labels = dataset.get_contrastive_batch(batch_size=64)
            if len(pairs) > 0:
                pair_word_ids1 = [p[0] for p in pairs]
                pair_word_ids2 = [p[1] for p in pairs]

                seqs1 = [dataset.word_data[wid][1] for wid in pair_word_ids1]
                seqs2 = [dataset.word_data[wid][1] for wid in pair_word_ids2]

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

                _, word_emb1, _ = model(input_ids1, masks1)
                _, word_emb2, _ = model(input_ids2, masks2)

                contra_loss = contrastive_loss_v2(word_emb1, word_emb2, labels_tensor, temperature=temperature)
            else:
                contra_loss = torch.tensor(0.0).to(device)

            loss = mlm_loss + contra_weight * contra_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/{num_epochs}: Loss {total_loss/len(dataloader):.4f}")

    # Save
    Path('models/curriculum').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'models/curriculum/phoible_initialized_final.pt')
    print()
    print("✓ Model saved: models/curriculum/phoible_initialized_final.pt\n")

    # Evaluate
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()

    def get_emb(word):
        if word not in dataset.loader.lexicon:
            return None
        phonemes = dataset.loader.lexicon[word]
        CLS, SEP, PAD = dataset.phoneme_to_id['<CLS>'], dataset.phoneme_to_id['<SEP>'], dataset.phoneme_to_id['<PAD>']
        ids = [CLS] + [dataset.phoneme_to_id[p] for p in phonemes] + [SEP]
        seq_len = len(ids)
        padded = ids + [PAD] * (dataset.max_length - seq_len)
        mask = [1] * seq_len + [0] * (dataset.max_length - seq_len)

        with torch.no_grad():
            _, emb, _ = model(torch.LongTensor([padded]).to(device), torch.LongTensor([mask]).to(device))
        return emb.squeeze(0).cpu().numpy()

    def sim(w1, w2):
        e1, e2 = get_emb(w1), get_emb(w2)
        if e1 is None or e2 is None:
            return None
        return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)

    tests = [
        ('cat', 'bat', 'rhyme'),
        ('cat', 'mat', 'rhyme'),
        ('cat', 'act', 'ANAGRAM'),
        ('dog', 'fog', 'rhyme'),
        ('make', 'take', 'rhyme'),
        ('run', 'running', 'morph'),
        ('cat', 'dog', 'unrelated'),
        ('cat', 'phone', 'unrelated'),
    ]

    print()
    rhyme_sims, morph_sims, unrel_sims, anagram_sim = [], [], [], None

    for w1, w2, typ in tests:
        s = sim(w1, w2)
        if s is not None:
            good = False
            if typ in ['rhyme', 'morph'] and s > 0.7:
                good = True
            elif typ in ['ANAGRAM', 'unrelated'] and s < 0.5:
                good = True

            result = '✓' if good else '✗'
            print(f"{w1}-{w2:<14} {s:.3f}  ({typ:<10}) {result}")

            if typ == 'rhyme':
                rhyme_sims.append(s)
            elif typ == 'morph':
                morph_sims.append(s)
            elif typ == 'unrelated':
                unrel_sims.append(s)
            elif typ == 'ANAGRAM':
                anagram_sim = s

    print()
    print("SUMMARY:")
    print(f"  Rhyme avg:     {np.mean(rhyme_sims):.3f} (want >0.70)")
    print(f"  Morph avg:     {np.mean(morph_sims):.3f} (want >0.70)")
    print(f"  Unrelated avg: {np.mean(unrel_sims):.3f} (want <0.50)")
    print(f"  Anagram:       {anagram_sim:.3f} (want <0.40)")
    print()
    print("="*70)
    print("DONE!")
    print("="*70)


if __name__ == '__main__':
    main()
