#!/usr/bin/env python3
"""
Training Script for Phoneme Embeddings

Multi-task training loop that combines all data sources.
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Iterator
import json
from tqdm import tqdm

from data_loader import PhonemeEmbeddingDataLoader
from model import create_model, MultiTaskPhonemeEmbedding


class PhonemeEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for phoneme embeddings

    Combines all data sources into unified batches.
    """

    def __init__(
        self,
        data_loader: PhonemeEmbeddingDataLoader,
        context_window: int = 2,
        num_contrastive_pairs: int = 100000
    ):
        self.data_loader = data_loader

        print("\n" + "=" * 70)
        print("LOADING TRAINING DATA")
        print("=" * 70)

        # Load all data into memory
        print("\nLoading context examples...")
        self.context_examples = list(data_loader.get_context_examples(window_size=context_window))

        print("Loading morphology examples...")
        self.morphology_examples = list(data_loader.get_morphology_examples())

        print("Loading contrastive pairs...")
        self.contrastive_pairs = list(data_loader.get_contrastive_pairs(num_pairs=num_contrastive_pairs))

        print("Loading inventories...")
        self.inventories = list(data_loader.get_inventory_examples())

        # Build allomorph vocabulary (for morphology task)
        self.allomorph_to_id = {}
        self.id_to_allomorph = {}
        for ex in self.morphology_examples:
            if ex.allomorph and ex.allomorph not in self.allomorph_to_id:
                idx = len(self.allomorph_to_id)
                self.allomorph_to_id[ex.allomorph] = idx
                self.id_to_allomorph[idx] = ex.allomorph

        print(f"\n✓ Loaded all data:")
        print(f"  Context examples: {len(self.context_examples):,}")
        print(f"  Morphology examples: {len(self.morphology_examples):,}")
        print(f"  Contrastive pairs: {len(self.contrastive_pairs):,}")
        print(f"  Inventories: {len(self.inventories):,}")
        print(f"  Allomorphs: {len(self.allomorph_to_id)}")

    def __len__(self):
        # Use context examples as primary dataset
        return len(self.context_examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example

        Returns a dictionary with data for multiple tasks.
        We'll sample from different data sources for each batch.
        """
        batch = {}

        # 1. Context prediction (primary task)
        context_ex = self.context_examples[idx]
        center_id = self.data_loader.get_phoneme_id(context_ex.center_phoneme)

        if center_id is not None:
            batch['center_id'] = center_id

            # Get context IDs (pad/truncate to fixed length)
            context_ids = []
            for c in context_ex.context_phonemes[:4]:  # Max 4 context
                cid = self.data_loader.get_phoneme_id(c)
                if cid is not None:
                    context_ids.append(cid)

            if context_ids:
                # Pad to length 4
                while len(context_ids) < 4:
                    context_ids.append(context_ids[0])  # Repeat first
                batch['context_ids'] = context_ids[:4]

        # 2. Feature reconstruction (always available)
        if center_id is not None:
            features = self.data_loader.get_phoneme_features(context_ex.center_phoneme)
            if features is not None:
                batch['phoneme_id'] = center_id
                batch['features'] = features

        # 3. Contrastive learning (sample randomly)
        if np.random.random() < 0.5 and self.contrastive_pairs:
            pair_idx = np.random.randint(len(self.contrastive_pairs))
            pair = self.contrastive_pairs[pair_idx]

            p1_id = self.data_loader.get_phoneme_id(pair.phoneme1)
            p2_id = self.data_loader.get_phoneme_id(pair.phoneme2)

            if p1_id is not None and p2_id is not None:
                batch['phoneme1_id'] = p1_id
                batch['phoneme2_id'] = p2_id
                batch['similarity'] = pair.similarity

        # 4. Morphology (sample randomly)
        # Skip for now - needs pronunciation lookup

        # 5. Inventory co-occurrence (sample randomly)
        if np.random.random() < 0.3 and self.inventories:
            inv_idx = np.random.randint(len(self.inventories))
            inventory = self.inventories[inv_idx]

            if len(inventory.phonemes) >= 2:
                # Sample two phonemes from same inventory (positive)
                p1, p2 = np.random.choice(inventory.phonemes, size=2, replace=False)

                p1_id = self.data_loader.get_phoneme_id(p1)
                p2_id = self.data_loader.get_phoneme_id(p2)

                if p1_id is not None and p2_id is not None:
                    batch['inv_phoneme1_id'] = p1_id
                    batch['inv_phoneme2_id'] = p2_id
                    batch['cooccur'] = [1.0]  # Positive example (list for collation)

        return batch


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-sized batches

    Each example may have different tasks available.
    """
    collated = {}

    # Gather all keys
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    for key in all_keys:
        values = [item[key] for item in batch if key in item]

        if not values:
            continue

        # Handle different value types
        if isinstance(values[0], list):
            # Check if list contains floats or ints
            if values[0] and isinstance(values[0][0], (float, np.floating)):
                # List of floats (e.g., cooccur)
                collated[key] = torch.tensor(values, dtype=torch.float32)
            else:
                # List of ints (e.g., context_ids)
                collated[key.replace('_id', '_ids')] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], (int, np.integer)):
            # Single integers
            collated[key.replace('_id', '_ids')] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], (float, np.floating)):
            # Single floats
            collated[key] = torch.tensor(values, dtype=torch.float32)
        elif isinstance(values[0], np.ndarray):
            # Arrays (features)
            collated[key] = torch.tensor(np.stack(values), dtype=torch.float32)

    return collated


def train_epoch(
    model: MultiTaskPhonemeEmbedding,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    task_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Train for one epoch

    Returns:
        metrics: Dictionary of average losses
    """
    model.train()

    total_loss = 0.0
    task_losses = {
        'context': 0.0,
        'morphology': 0.0,
        'contrastive': 0.0,
        'feature': 0.0,
        'inventory': 0.0
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        loss, loss_dict = model.compute_loss(batch, task_weights)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        for task, task_loss in loss_dict.items():
            task_losses[task] += task_loss

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Average losses
    metrics = {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in task_losses.items()}
    }

    return metrics


def train(
    data_dir: str = "data",
    learning_datasets_dir: str = "data/learning_datasets",
    output_dir: str = "models/phoneme_embeddings",
    embedding_dim: int = 32,
    batch_size: int = 128,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "auto"
):
    """
    Main training function

    Args:
        data_dir: Path to data directory
        learning_datasets_dir: Path to learning datasets
        output_dir: Where to save model checkpoints
        embedding_dim: Embedding dimension
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on ('auto', 'cpu', 'cuda', or 'mps')
    """
    print("\n" + "=" * 70)
    print("PHONEME EMBEDDING TRAINING")
    print("=" * 70)

    # Auto-detect best device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"\n✓ Auto-detected device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data_loader = PhonemeEmbeddingDataLoader(
        data_dir=data_dir,
        learning_datasets_dir=learning_datasets_dir
    )

    # Create dataset
    dataset = PhonemeEmbeddingDataset(
        data_loader=data_loader,
        context_window=2,
        num_contrastive_pairs=100000
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Get feature matrix
    feature_matrix = data_loader.get_feature_matrix()

    # Create model
    model = create_model(
        num_phonemes=len(data_loader.phoneme_to_id),
        feature_matrix=feature_matrix,
        embedding_dim=embedding_dim,
        num_allomorphs=10  # Placeholder
    )

    # Move to device
    device = torch.device(device)
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Task weights
    task_weights = {
        'context': 1.0,
        'morphology': 0.0,  # Skip for now (needs pronunciation lookup)
        'contrastive': 0.5,
        'feature': 1.0,
        'inventory': 0.3
    }

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"\nTask weights:")
    for task, weight in task_weights.items():
        print(f"  {task}: {weight}")

    # Training loop
    best_loss = float('inf')
    history = []

    for epoch in range(num_epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 70}")

        metrics = train_epoch(model, dataloader, optimizer, device, task_weights)

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Total loss: {metrics['total']:.4f}")
        for task, loss in metrics.items():
            if task != 'total' and loss > 0:
                print(f"    {task}: {loss:.4f}")

        history.append(metrics)

        # Save best model
        if metrics['total'] < best_loss:
            best_loss = metrics['total']
            checkpoint_path = output_path / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'num_phonemes': len(data_loader.phoneme_to_id),
                    'embedding_dim': embedding_dim,
                    'feature_dim': 38
                }
            }, checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")

    # Save final model
    final_path = output_path / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_phonemes': len(data_loader.phoneme_to_id),
            'embedding_dim': embedding_dim,
            'feature_dim': 38
        }
    }, final_path)

    # Save training history
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Save phoneme vocabulary
    with open(output_path / "phoneme_vocab.json", 'w') as f:
        json.dump({
            'phoneme_to_id': data_loader.phoneme_to_id,
            'id_to_phoneme': data_loader.id_to_phoneme
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n✓ Best loss: {best_loss:.4f}")
    print(f"✓ Models saved to: {output_path}")
    print(f"✓ Vocabulary saved to: {output_path / 'phoneme_vocab.json'}")

    return model, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train phoneme embeddings')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'mps', 'auto'], help='Device (auto = auto-detect)')
    parser.add_argument('--output-dir', type=str, default='models/phoneme_embeddings', help='Output directory')

    args = parser.parse_args()

    # Train
    train(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output_dir
    )
