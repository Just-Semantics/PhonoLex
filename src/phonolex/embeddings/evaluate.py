#!/usr/bin/env python3
"""
Evaluation Script for Phoneme Embeddings

Evaluates learned embeddings on multiple tasks:
1. Feature reconstruction (how well do embeddings preserve phonological features?)
2. Similarity tasks (do similar phonemes have similar embeddings?)
3. Natural class clustering (do phonemes cluster by natural classes?)
4. Visualization (t-SNE plots)
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from data_loader import PhonemeEmbeddingDataLoader
from model import create_model, MultiTaskPhonemeEmbedding


class PhonemeEmbeddingEvaluator:
    """
    Evaluate phoneme embeddings on various tasks
    """

    def __init__(
        self,
        model_path: str,
        data_dir: str = "data",
        learning_datasets_dir: str = "data/learning_datasets"
    ):
        print("\n" + "=" * 70)
        print("PHONEME EMBEDDING EVALUATION")
        print("=" * 70)

        # Load data
        self.data_loader = PhonemeEmbeddingDataLoader(
            data_dir=data_dir,
            learning_datasets_dir=learning_datasets_dir
        )

        # Load model
        print(f"\nLoading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')

        config = checkpoint['config']
        self.model = create_model(
            num_phonemes=config['num_phonemes'],
            embedding_dim=config['embedding_dim'],
            num_allomorphs=10
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"  ✓ Model loaded")
        print(f"    Phonemes: {config['num_phonemes']}")
        print(f"    Embedding dim: {config['embedding_dim']}")

        # Get all embeddings
        print("\nExtracting embeddings...")
        with torch.no_grad():
            all_ids = torch.arange(len(self.data_loader.phoneme_to_id))
            self.embeddings = self.model.get_embeddings(all_ids)

        print(f"  ✓ Extracted {self.embeddings.shape[0]} embeddings")

    def evaluate_feature_reconstruction(self) -> Dict[str, float]:
        """
        Evaluate how well embeddings preserve phonological features

        Returns:
            metrics: MSE and correlation with ground truth features
        """
        print("\n" + "=" * 70)
        print("1. FEATURE RECONSTRUCTION")
        print("=" * 70)

        # Get ground truth features
        feature_matrix = self.data_loader.get_feature_matrix()

        # Reconstruct features from embeddings
        with torch.no_grad():
            phoneme_ids = torch.arange(len(self.data_loader.phoneme_to_id))
            embeddings = self.model(phoneme_ids)
            reconstructed = self.model.feature_head(embeddings).numpy()

        # Compute metrics
        mse = np.mean((reconstructed - feature_matrix) ** 2)

        # Correlation per feature
        correlations = []
        for i in range(feature_matrix.shape[1]):
            corr, _ = spearmanr(feature_matrix[:, i], reconstructed[:, i])
            if not np.isnan(corr):
                correlations.append(corr)

        mean_corr = np.mean(correlations)

        print(f"\n  MSE: {mse:.4f}")
        print(f"  Mean correlation: {mean_corr:.4f}")

        return {
            'mse': float(mse),
            'mean_correlation': float(mean_corr)
        }

    def evaluate_similarity(self, num_pairs: int = 1000) -> Dict[str, float]:
        """
        Evaluate if similar phonemes have similar embeddings

        Uses ground-truth feature similarity vs embedding similarity.

        Returns:
            metrics: Correlation between feature and embedding similarity
        """
        print("\n" + "=" * 70)
        print("2. SIMILARITY EVALUATION")
        print("=" * 70)

        print(f"\nSampling {num_pairs} phoneme pairs...")

        phonemes = list(self.data_loader.phoneme_to_features.keys())
        feature_sims = []
        embedding_sims = []

        for _ in range(num_pairs):
            # Sample two random phonemes
            p1, p2 = np.random.choice(phonemes, size=2, replace=False)

            # Feature similarity (cosine)
            f1 = self.data_loader.get_phoneme_features(p1)
            f2 = self.data_loader.get_phoneme_features(p2)

            if f1 is not None and f2 is not None:
                feat_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
                feature_sims.append(feat_sim)

                # Embedding similarity
                p1_id = self.data_loader.get_phoneme_id(p1)
                p2_id = self.data_loader.get_phoneme_id(p2)

                e1 = self.embeddings[p1_id]
                e2 = self.embeddings[p2_id]

                emb_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
                embedding_sims.append(emb_sim)

        # Compute correlation
        corr, p_value = spearmanr(feature_sims, embedding_sims)

        print(f"\n  Spearman correlation: {corr:.4f} (p={p_value:.4e})")

        return {
            'spearman_correlation': float(corr),
            'p_value': float(p_value)
        }

    def evaluate_natural_classes(self) -> Dict[str, float]:
        """
        Evaluate if natural classes cluster together

        Natural classes: voiced stops, voiceless fricatives, etc.

        Returns:
            metrics: Clustering quality metrics
        """
        print("\n" + "=" * 70)
        print("3. NATURAL CLASS CLUSTERING")
        print("=" * 70)

        # Define natural classes based on features
        # For simplicity, use binary features: voiced, consonantal, sonorant

        classes = {
            'voiced_consonants': [],
            'voiceless_consonants': [],
            'sonorants': [],
            'obstruents': []
        }

        for phoneme, idx in self.data_loader.phoneme_to_id.items():
            features = self.data_loader.get_phoneme_features(phoneme)
            if features is None:
                continue

            # Assume features are in fixed order (from Phoible)
            # voiced (index 5), consonantal (index 5), sonorant (index 6)
            # This is approximate - would need exact feature indices

            # For now, just report statistics
            pass

        print("\n  Natural class clustering: Not yet implemented")
        print("  (Requires feature index mapping)")

        return {}

    def visualize(self, output_dir: str = "outputs/visualizations"):
        """
        Create t-SNE visualizations of embeddings

        Args:
            output_dir: Where to save plots
        """
        print("\n" + "=" * 70)
        print("4. VISUALIZATION")
        print("=" * 70)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\nRunning t-SNE (this may take a while)...")

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(self.embeddings)

        print("  ✓ t-SNE complete")

        # Plot
        print("\nCreating visualization...")

        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)

        # Annotate a few sample phonemes
        sample_phonemes = ['p', 'b', 't', 'd', 'k', 'g', 's', 'z', 'a', 'i', 'u']
        for p in sample_phonemes:
            if p in self.data_loader.phoneme_to_id:
                idx = self.data_loader.phoneme_to_id[p]
                plt.annotate(
                    p,
                    xy=(embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    fontsize=12,
                    fontweight='bold'
                )

        plt.title("Phoneme Embedding Space (t-SNE)", fontsize=16)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.tight_layout()

        plot_path = output_path / "embeddings_tsne.png"
        plt.savefig(plot_path, dpi=300)
        print(f"  ✓ Saved plot to {plot_path}")

        plt.close()

    def run_all_evaluations(self) -> Dict[str, Dict]:
        """
        Run all evaluation tasks

        Returns:
            results: Dictionary with all metrics
        """
        results = {}

        results['feature_reconstruction'] = self.evaluate_feature_reconstruction()
        results['similarity'] = self.evaluate_similarity(num_pairs=1000)
        results['natural_classes'] = self.evaluate_natural_classes()

        self.visualize()

        return results


def main():
    """
    Run evaluation on trained model
    """
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate phoneme embeddings')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/phoneme_embeddings/best_model.pt',
        help='Path to trained model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Where to save evaluation results'
    )

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first: python train.py")
        return

    # Run evaluation
    evaluator = PhonemeEmbeddingEvaluator(model_path=str(model_path))
    results = evaluator.run_all_evaluations()

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if 'feature_reconstruction' in results:
        print(f"\nFeature Reconstruction:")
        print(f"  MSE: {results['feature_reconstruction']['mse']:.4f}")
        print(f"  Correlation: {results['feature_reconstruction']['mean_correlation']:.4f}")

    if 'similarity' in results:
        print(f"\nSimilarity:")
        print(f"  Spearman ρ: {results['similarity']['spearman_correlation']:.4f}")


if __name__ == '__main__':
    main()
