#!/usr/bin/env python3
"""
Phoneme Embedding Model

Multi-task neural network for learning phoneme embeddings.

Objectives:
1. Context prediction (skip-gram)
2. Morphological pattern prediction
3. Contrastive learning (metric learning)
4. Feature reconstruction (autoencoder)
5. Inventory co-occurrence (cross-linguistic)

Like Word2Vec/BERT but for phonemes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class PhonemeEmbedding(nn.Module):
    """
    Core phoneme embedding model

    Architecture:
    - Input: Phoneme ID (integer)
    - Embedding: Dense vector (32-dim by default)
    - Initialization: From Phoible features (38-dim → 32-dim)

    Or input Phoible features directly (38-dim) and learn projection.
    """

    def __init__(
        self,
        num_phonemes: int,
        embedding_dim: int = 32,
        feature_dim: int = 38,
        init_from_features: bool = True
    ):
        super().__init__()

        self.num_phonemes = num_phonemes
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

        # Phoneme embeddings
        self.embeddings = nn.Embedding(num_phonemes, embedding_dim)

        if init_from_features:
            # Will initialize from Phoible features via set_feature_initialization()
            pass
        else:
            # Random initialization
            nn.init.xavier_uniform_(self.embeddings.weight)

    def set_feature_initialization(self, feature_matrix: np.ndarray):
        """
        Initialize embeddings from Phoible features

        Args:
            feature_matrix: (num_phonemes, 38) array of Phoible features
        """
        # Project 38-dim features to embedding_dim using PCA-like initialization
        feature_tensor = torch.FloatTensor(feature_matrix)

        # SVD for dimensionality reduction
        U, S, V = torch.svd(feature_tensor)
        projected = torch.mm(feature_tensor, V[:, :self.embedding_dim])

        # Normalize
        projected = F.normalize(projected, p=2, dim=1)

        # Set as initial embeddings
        self.embeddings.weight.data.copy_(projected)

        print(f"  ✓ Initialized embeddings from Phoible features")
        print(f"    {feature_matrix.shape[0]} phonemes × {self.embedding_dim} dims")

    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for phoneme IDs

        Args:
            phoneme_ids: (batch_size,) tensor of phoneme IDs

        Returns:
            embeddings: (batch_size, embedding_dim) tensor
        """
        return self.embeddings(phoneme_ids)


class ContextPredictionHead(nn.Module):
    """
    Skip-gram context prediction (like Word2Vec)

    Given center phoneme, predict context phonemes.
    """

    def __init__(self, embedding_dim: int, num_phonemes: int):
        super().__init__()

        self.projection = nn.Linear(embedding_dim, num_phonemes)

    def forward(
        self,
        center_embedding: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict context phonemes

        Args:
            center_embedding: (batch_size, embedding_dim)
            context_ids: (batch_size, num_context) - optional, for loss computation

        Returns:
            logits: (batch_size, num_phonemes) - probability over all phonemes
        """
        logits = self.projection(center_embedding)
        return logits

    def loss(
        self,
        center_embedding: torch.Tensor,
        context_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute skip-gram loss (negative sampling or softmax)

        Args:
            center_embedding: (batch_size, embedding_dim)
            context_ids: (batch_size, num_context)

        Returns:
            loss: scalar
        """
        logits = self.forward(center_embedding)  # (batch, num_phonemes)

        # Cross-entropy loss for each context position
        # Average over context positions
        loss = 0
        for i in range(context_ids.size(1)):
            loss += F.cross_entropy(logits, context_ids[:, i])

        return loss / context_ids.size(1)


class MorphologyPredictionHead(nn.Module):
    """
    Predict morphological allomorph from phonological context

    Given stem-final phoneme embedding, predict which allomorph to use.
    Example: /t/ → /-s/ (cats), /g/ → /-z/ (dogs), /s/ → /-ɪz/ (buses)
    """

    def __init__(self, embedding_dim: int, num_allomorphs: int):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_allomorphs)
        )

    def forward(self, phoneme_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict allomorph

        Args:
            phoneme_embedding: (batch_size, embedding_dim)

        Returns:
            logits: (batch_size, num_allomorphs)
        """
        return self.classifier(phoneme_embedding)

    def loss(
        self,
        phoneme_embedding: torch.Tensor,
        allomorph_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy loss for allomorph prediction

        Args:
            phoneme_embedding: (batch_size, embedding_dim)
            allomorph_ids: (batch_size,)

        Returns:
            loss: scalar
        """
        logits = self.forward(phoneme_embedding)
        return F.cross_entropy(logits, allomorph_ids)


class ContrastiveLearningHead(nn.Module):
    """
    Metric learning: Similar phonemes should have similar embeddings

    Uses cosine similarity + contrastive loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity

        Args:
            embedding1: (batch_size, embedding_dim)
            embedding2: (batch_size, embedding_dim)

        Returns:
            similarity: (batch_size,) - cosine similarity
        """
        # Normalize
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)

        # Cosine similarity
        similarity = torch.sum(embedding1 * embedding2, dim=1)

        return similarity

    def loss(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        target_similarity: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss

        Args:
            embedding1: (batch_size, embedding_dim)
            embedding2: (batch_size, embedding_dim)
            target_similarity: (batch_size,) - ground truth similarity [0, 1]

        Returns:
            loss: scalar (MSE between predicted and target similarity)
        """
        pred_similarity = self.forward(embedding1, embedding2)

        # MSE loss
        return F.mse_loss(pred_similarity, target_similarity)


class FeatureReconstructionHead(nn.Module):
    """
    Reconstruct Phoible features from embedding

    Forces embedding to retain phonological information.
    """

    def __init__(self, embedding_dim: int, feature_dim: int = 38):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.Tanh()  # Features are in [-1, +1]
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct features

        Args:
            embedding: (batch_size, embedding_dim)

        Returns:
            features: (batch_size, feature_dim)
        """
        return self.decoder(embedding)

    def loss(
        self,
        embedding: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE loss for feature reconstruction

        Args:
            embedding: (batch_size, embedding_dim)
            target_features: (batch_size, feature_dim) - ground truth features

        Returns:
            loss: scalar
        """
        pred_features = self.forward(embedding)
        return F.mse_loss(pred_features, target_features)


class InventoryCooccurrenceHead(nn.Module):
    """
    Predict if phonemes co-occur in the same language inventory

    Cross-linguistic regularization: phonemes that appear together
    should have related embeddings.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict co-occurrence probability

        Args:
            embedding1: (batch_size, embedding_dim)
            embedding2: (batch_size, embedding_dim)

        Returns:
            probability: (batch_size, 1) - P(co-occur)
        """
        # Concatenate embeddings
        combined = torch.cat([embedding1, embedding2], dim=1)
        return self.classifier(combined)

    def loss(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss

        Args:
            embedding1: (batch_size, embedding_dim)
            embedding2: (batch_size, embedding_dim)
            target: (batch_size,) or (batch_size, 1) - 1 if co-occur, 0 otherwise

        Returns:
            loss: scalar
        """
        pred = self.forward(embedding1, embedding2)
        # Ensure target has same shape as pred
        if target.dim() == 1:
            target = target.unsqueeze(1)
        return F.binary_cross_entropy(pred, target)


class MultiTaskPhonemeEmbedding(nn.Module):
    """
    Complete multi-task phoneme embedding model

    Combines all objectives:
    1. Context prediction
    2. Morphology prediction
    3. Contrastive learning
    4. Feature reconstruction
    5. Inventory co-occurrence
    """

    def __init__(
        self,
        num_phonemes: int,
        embedding_dim: int = 32,
        feature_dim: int = 38,
        num_allomorphs: int = 10,  # Will be determined from data
        init_from_features: bool = True
    ):
        super().__init__()

        # Core embedding
        self.embedding = PhonemeEmbedding(
            num_phonemes=num_phonemes,
            embedding_dim=embedding_dim,
            feature_dim=feature_dim,
            init_from_features=init_from_features
        )

        # Task heads
        self.context_head = ContextPredictionHead(embedding_dim, num_phonemes)
        self.morphology_head = MorphologyPredictionHead(embedding_dim, num_allomorphs)
        self.contrastive_head = ContrastiveLearningHead()
        self.feature_head = FeatureReconstructionHead(embedding_dim, feature_dim)
        self.inventory_head = InventoryCooccurrenceHead(embedding_dim)

    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings"""
        return self.embedding(phoneme_ids)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        task_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss

        Args:
            batch: Dictionary with task-specific data
            task_weights: Weights for each task (default: equal weights)

        Returns:
            total_loss: Scalar
            loss_dict: Individual losses for logging
        """
        if task_weights is None:
            task_weights = {
                'context': 1.0,
                'morphology': 1.0,
                'contrastive': 1.0,
                'feature': 1.0,
                'inventory': 1.0
            }

        losses = {}

        # Context prediction
        if 'center_ids' in batch and 'context_ids' in batch:
            center_emb = self.embedding(batch['center_ids'])
            losses['context'] = self.context_head.loss(center_emb, batch['context_ids'])

        # Morphology prediction
        if 'stem_final_ids' in batch and 'allomorph_ids' in batch:
            stem_emb = self.embedding(batch['stem_final_ids'])
            losses['morphology'] = self.morphology_head.loss(stem_emb, batch['allomorph_ids'])

        # Contrastive learning
        if 'phoneme1_ids' in batch and 'phoneme2_ids' in batch and 'similarity' in batch:
            emb1 = self.embedding(batch['phoneme1_ids'])
            emb2 = self.embedding(batch['phoneme2_ids'])
            losses['contrastive'] = self.contrastive_head.loss(emb1, emb2, batch['similarity'])

        # Feature reconstruction
        if 'phoneme_ids' in batch and 'features' in batch:
            emb = self.embedding(batch['phoneme_ids'])
            losses['feature'] = self.feature_head.loss(emb, batch['features'])

        # Inventory co-occurrence
        if 'inv_phoneme1_ids' in batch and 'inv_phoneme2_ids' in batch and 'cooccur' in batch:
            emb1 = self.embedding(batch['inv_phoneme1_ids'])
            emb2 = self.embedding(batch['inv_phoneme2_ids'])
            losses['inventory'] = self.inventory_head.loss(emb1, emb2, batch['cooccur'])

        # Weighted sum
        total_loss = sum(task_weights.get(k, 1.0) * v for k, v in losses.items())

        # Convert to floats for logging
        loss_dict = {k: v.item() for k, v in losses.items()}

        return total_loss, loss_dict

    def get_embeddings(self, phoneme_ids: torch.Tensor) -> np.ndarray:
        """
        Get embeddings as numpy array (for evaluation)

        Args:
            phoneme_ids: (batch_size,) tensor

        Returns:
            embeddings: (batch_size, embedding_dim) numpy array
        """
        with torch.no_grad():
            emb = self.forward(phoneme_ids)
            return emb.cpu().numpy()


def create_model(
    num_phonemes: int,
    feature_matrix: Optional[np.ndarray] = None,
    embedding_dim: int = 32,
    num_allomorphs: int = 10
) -> MultiTaskPhonemeEmbedding:
    """
    Factory function to create and initialize model

    Args:
        num_phonemes: Vocabulary size
        feature_matrix: (num_phonemes, 38) Phoible features for initialization
        embedding_dim: Embedding dimension
        num_allomorphs: Number of allomorph classes

    Returns:
        model: Initialized MultiTaskPhonemeEmbedding
    """
    print("\n" + "=" * 70)
    print("CREATING PHONEME EMBEDDING MODEL")
    print("=" * 70)

    model = MultiTaskPhonemeEmbedding(
        num_phonemes=num_phonemes,
        embedding_dim=embedding_dim,
        num_allomorphs=num_allomorphs,
        init_from_features=(feature_matrix is not None)
    )

    if feature_matrix is not None:
        model.embedding.set_feature_initialization(feature_matrix)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  ✓ Model created")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Embedding dimension: {embedding_dim}")

    print("\n" + "=" * 70)

    return model


if __name__ == '__main__':
    # Demo: Create model and test forward pass

    print("Demo: Creating phoneme embedding model\n")

    # Dummy data
    num_phonemes = 100
    feature_matrix = np.random.randn(num_phonemes, 38).astype(np.float32)

    # Create model
    model = create_model(
        num_phonemes=num_phonemes,
        feature_matrix=feature_matrix,
        embedding_dim=32,
        num_allomorphs=5
    )

    # Test forward pass
    print("\nTesting forward pass...")

    batch = {
        'center_ids': torch.randint(0, num_phonemes, (16,)),
        'context_ids': torch.randint(0, num_phonemes, (16, 4)),
        'phoneme1_ids': torch.randint(0, num_phonemes, (16,)),
        'phoneme2_ids': torch.randint(0, num_phonemes, (16,)),
        'similarity': torch.rand(16),
        'phoneme_ids': torch.randint(0, num_phonemes, (16,)),
        'features': torch.randn(16, 38)
    }

    loss, loss_dict = model.compute_loss(batch)

    print(f"\n✓ Forward pass successful")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"\n  Individual losses:")
    for task, task_loss in loss_dict.items():
        print(f"    {task}: {task_loss:.4f}")

    # Test embeddings
    test_ids = torch.tensor([0, 1, 2, 3, 4])
    embeddings = model.get_embeddings(test_ids)
    print(f"\n✓ Embedding extraction successful")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Sample: {embeddings[0][:5]}...")
