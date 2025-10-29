"""
Grid search for PhonoLex-BERT hyperparameters.

Search over:
- Contrastive loss weight: [0.5, 1.0, 2.0]
- Temperature: [0.03, 0.05, 0.07, 0.1]
- Epochs: 10 (fast iteration)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))
from src.phonolex.embeddings.english_data_loader import EnglishPhonologyLoader
from train_phonology_bert_v2 import ImprovedPhonologyDataset, PhonoLexBERT, contrastive_loss_v2


def evaluate_model(model, dataset, device, test_pairs):
    """Quick evaluation on test pairs."""
    model.eval()

    def get_word_emb(word):
        if word not in dataset.loader.lexicon:
            return None
        phonemes = dataset.loader.lexicon[word]
        CLS, SEP, PAD = dataset.phoneme_to_id['<CLS>'], dataset.phoneme_to_id['<SEP>'], dataset.phoneme_to_id['<PAD>']
        input_ids = [CLS] + [dataset.phoneme_to_id[p] for p in phonemes] + [SEP]
        seq_len = len(input_ids)
        padded = input_ids + [PAD] * (dataset.max_length - seq_len)
        mask = [1] * seq_len + [0] * (dataset.max_length - seq_len)

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

    results = {}
    for w1, w2, desc in test_pairs:
        s = sim(w1, w2)
        if s is not None:
            results[f"{w1}-{w2}"] = (s, desc)

    # Compute score (higher rhyme/morph similarity, lower unrelated/anagram)
    rhyme_sim = np.mean([v[0] for k, v in results.items() if v[1] in ['rhyme', 'minimal pair']])
    morph_sim = np.mean([v[0] for k, v in results.items() if v[1] == 'morphology'])
    unrelated_sim = np.mean([v[0] for k, v in results.items() if v[1] == 'unrelated'])
    anagram_sim = results.get('cat-act', (0, ''))[0]

    # Score: maximize (rhyme + morph), minimize (unrelated + anagram)
    score = rhyme_sim + morph_sim - unrelated_sim - anagram_sim

    return {
        'score': score,
        'rhyme_sim': rhyme_sim,
        'morph_sim': morph_sim,
        'unrelated_sim': unrelated_sim,
        'anagram_sim': anagram_sim,
    }


def train_and_evaluate(contra_weight, temperature, num_epochs=10):
    """Train model with given hyperparameters and evaluate."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load data
    loader = EnglishPhonologyLoader()
    dataset = ImprovedPhonologyDataset(loader, mask_prob=0.15)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    # Create model
    model = PhonoLexBERT(
        num_phonemes=len(dataset.phoneme_to_id),
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        max_len=dataset.max_length
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=dataset.phoneme_to_id[dataset.PAD])

    # Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
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

    # Evaluate
    test_pairs = [
        ('cat', 'bat', 'rhyme'),
        ('cat', 'mat', 'rhyme'),
        ('cat', 'act', 'anagram'),
        ('dog', 'fog', 'rhyme'),
        ('make', 'take', 'rhyme'),
        ('run', 'running', 'morphology'),
        ('cat', 'dog', 'unrelated'),
        ('phone', 'clone', 'rhyme'),
        ('cat', 'phone', 'unrelated'),
    ]

    metrics = evaluate_model(model, dataset, device, test_pairs)

    return metrics, model


def main():
    print("="*70)
    print("PHONOLEX-BERT HYPERPARAMETER GRID SEARCH")
    print("="*70)

    # Grid search parameters
    contra_weights = [0.5, 1.0, 2.0, 3.0]
    temperatures = [0.03, 0.05, 0.07, 0.1]
    num_epochs = 8  # Fast iteration

    results = []

    for cw in contra_weights:
        for temp in temperatures:
            print(f"\n{'='*70}")
            print(f"Training: contra_weight={cw}, temperature={temp}")
            print(f"{'='*70}")

            try:
                metrics, model = train_and_evaluate(cw, temp, num_epochs)

                result = {
                    'contra_weight': cw,
                    'temperature': temp,
                    **metrics
                }
                results.append(result)

                print(f"\nResults:")
                print(f"  Score: {metrics['score']:.4f}")
                print(f"  Rhyme sim: {metrics['rhyme_sim']:.4f}")
                print(f"  Morph sim: {metrics['morph_sim']:.4f}")
                print(f"  Unrelated sim: {metrics['unrelated_sim']:.4f}")
                print(f"  Anagram sim: {metrics['anagram_sim']:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # Find best
    print(f"\n{'='*70}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*70}")

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"\n{'Rank':<5} {'CW':<5} {'Temp':<7} {'Score':<8} {'Rhyme':<8} {'Morph':<8} {'Unrel':<8} {'Anagram':<8}")
    print("-"*70)

    for i, r in enumerate(results[:10]):
        print(f"{i+1:<5} {r['contra_weight']:<5.1f} {r['temperature']:<7.2f} "
              f"{r['score']:<8.3f} {r['rhyme_sim']:<8.3f} {r['morph_sim']:<8.3f} "
              f"{r['unrelated_sim']:<8.3f} {r['anagram_sim']:<8.3f}")

    # Save results
    with open('gridsearch_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Best config: contra_weight={results[0]['contra_weight']}, temperature={results[0]['temperature']}")
    print(f"   Score: {results[0]['score']:.4f}")
    print(f"\nResults saved to: gridsearch_results.json")


if __name__ == "__main__":
    main()
