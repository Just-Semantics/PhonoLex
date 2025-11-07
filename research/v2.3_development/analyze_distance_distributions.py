#!/usr/bin/env python3
"""
Analyze continuous distance distributions for multiple L1 → English language pairs.

Goal: Find natural clustering patterns (Option B) to identify:
- Identical phonemes (d ≈ 0)
- Similar phonemes (valley of confusion)
- New phonemes (tail)

Using data-driven clustering instead of arbitrary thresholds.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Setup paths
project_root = Path(__file__).resolve().parents[2]

# Load data
print("Loading PHOIBLE data...")
with open(project_root / "embeddings/phase2/phoible_all_phonemes_76d.pkl", "rb") as f:
    vectors_76d = pickle.load(f)

with open(project_root / "embeddings/phase2/phoible_language_inventories.json") as f:
    languages = json.load(f)

print(f"✓ Loaded {len(vectors_76d):,} phoneme vectors")
print(f"✓ Loaded {len(languages):,} languages\n")


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return 0 if norm1 == 0 or norm2 == 0 else dot / (norm1 * norm2)


def analyze_l1_to_l2(l1_code, l2_code):
    """
    Analyze phoneme distances from L1 → L2.
    Returns: (distances, identical_count, total_l2_phonemes)
    """
    l1 = languages.get(l1_code)
    l2 = languages.get(l2_code)

    if not l1 or not l2:
        return None, 0, 0

    distances = []
    identical_count = 0

    for l2_phoneme in l2["phonemes"]:
        # Check if identical
        if l2_phoneme in l1["phonemes"]:
            identical_count += 1
            continue

        # Find closest L1 phoneme
        best_distance = float("inf")
        for l1_phoneme in l1["phonemes"]:
            if l2_phoneme not in vectors_76d or l1_phoneme not in vectors_76d:
                continue

            similarity = cosine_similarity(vectors_76d[l2_phoneme], vectors_76d[l1_phoneme])
            distance = 1 - similarity

            if distance < best_distance:
                best_distance = distance

        if best_distance != float("inf"):
            distances.append(best_distance)

    return np.array(distances), identical_count, len(l2["phonemes"])


def find_distribution_gaps(distances, min_gap=0.01):
    """Find gaps in the distance distribution"""
    sorted_dist = np.sort(distances)
    gaps = np.diff(sorted_dist)

    # Find significant gaps
    gap_indices = np.where(gaps > min_gap)[0]

    gap_info = []
    for idx in gap_indices:
        gap_info.append({
            "size": gaps[idx],
            "location": (sorted_dist[idx], sorted_dist[idx + 1]),
            "phonemes_before": idx + 1,
            "phonemes_after": len(distances) - idx - 1,
        })

    # Sort by gap size
    gap_info.sort(key=lambda x: x["size"], reverse=True)

    return gap_info


def fit_gaussian_mixture(distances, n_components=3):
    """Fit Gaussian Mixture Model to find natural clusters"""
    X = distances.reshape(-1, 1)

    # Try different numbers of components
    best_gmm = None
    best_bic = float("inf")
    best_n = n_components

    for n in range(2, min(5, len(distances) // 5)):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n

    # Get cluster assignments
    labels = best_gmm.predict(X)

    # Get cluster means and sort them
    means = best_gmm.means_.flatten()
    sorted_indices = np.argsort(means)

    cluster_info = []
    for i in sorted_indices:
        cluster_mask = labels == i
        cluster_distances = distances[cluster_mask]

        cluster_info.append({
            "mean": means[i],
            "std": np.sqrt(best_gmm.covariances_[i][0, 0]),
            "weight": best_gmm.weights_[i],
            "size": np.sum(cluster_mask),
            "range": (cluster_distances.min(), cluster_distances.max()),
        })

    return cluster_info, best_n, best_bic


def compute_kde_peaks(distances, num_points=1000):
    """Find peaks in kernel density estimation"""
    if len(distances) < 3:
        return []

    # Compute KDE
    kde = stats.gaussian_kde(distances)

    # Evaluate on grid
    x_grid = np.linspace(distances.min(), distances.max(), num_points)
    kde_values = kde(x_grid)

    # Find peaks
    peaks, properties = find_peaks(kde_values, prominence=0.1)

    peak_info = []
    for peak_idx in peaks:
        peak_info.append({
            "location": x_grid[peak_idx],
            "density": kde_values[peak_idx],
            "prominence": properties["prominences"][peaks.tolist().index(peak_idx)],
        })

    return peak_info, x_grid, kde_values


def compute_silhouette_based_difficulty(distances, phoneme_list, l1_code, l2_code):
    """
    Compute difficulty using silhouette-based clustering.

    Key insight:
    - Phonemes IN the "similar" cluster (close to similar-centroid) = HARD
    - Phonemes IN the "new" cluster (far from similar-centroid) = EASY
    - Difficulty based on relative distance from cluster centroids
    """
    if len(distances) < 3:
        return None

    X = distances.reshape(-1, 1)

    # Find optimal number of clusters (usually 2: similar + new)
    best_score = -1
    best_n = 2
    best_kmeans = None

    for n in range(2, min(4, len(distances) // 3)):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        if len(set(labels)) > 1:  # Need at least 2 clusters
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_n = n
                best_kmeans = kmeans

    if best_kmeans is None:
        return None

    labels = best_kmeans.labels_
    centroids = best_kmeans.cluster_centers_.flatten()

    # Sort clusters by centroid (ascending distance)
    cluster_order = np.argsort(centroids)

    # First cluster (lowest centroid) = "similar" = HARD
    # Later clusters (higher centroids) = "new" = EASY
    similar_cluster_idx = cluster_order[0]

    # Compute per-sample silhouette scores
    silhouette_vals = silhouette_samples(X, labels)

    # Assign difficulty based on distance from SIMILAR cluster centroid
    # Key insight from Flege:
    # - Close to similar centroid = HARD (equivalence classification/valley)
    # - Far from similar centroid = EASY (differences discernible/new sounds)
    difficulties = []

    similar_centroid = centroids[similar_cluster_idx]
    max_dist = distances.max()

    for i, (dist, label, silh) in enumerate(zip(distances, labels, silhouette_vals)):
        # Distance from the SIMILAR cluster centroid (the valley)
        dist_from_similar = abs(dist - similar_centroid)

        # Normalize to [0, 1] range
        relative_dist = dist_from_similar / max_dist if max_dist > 0 else 0

        # Difficulty: exponential decay from similar centroid
        # At similar centroid (d=0): difficulty = 5.0 (HARDEST)
        # Far from similar (d=max): difficulty → 1.0 (EASY)
        difficulty = 5.0 * np.exp(-3 * relative_dist)

        # Clamp to [1.0, 5.0] range
        difficulty = min(5.0, max(1.0, difficulty))

        difficulties.append({
            'distance': float(dist),
            'cluster': int(label),
            'cluster_name': 'similar' if label == similar_cluster_idx else f'new_{label}',
            'dist_from_similar_centroid': float(dist_from_similar),
            'silhouette': float(silh),
            'difficulty': float(difficulty),
        })

    return {
        'difficulties': difficulties,
        'n_clusters': best_n,
        'silhouette_score': best_score,
        'centroids': centroids.tolist(),
        'similar_cluster_idx': int(similar_cluster_idx),
    }


def analyze_language_pair(l1_code, l2_code, l1_name, l2_name):
    """Comprehensive analysis of a language pair"""
    print("=" * 80)
    print(f"{l1_name} → {l2_name}")
    print("=" * 80)

    distances, identical_count, total_phonemes = analyze_l1_to_l2(l1_code, l2_code)

    if distances is None or len(distances) == 0:
        print("❌ No data available\n")
        return None

    non_identical_count = len(distances)

    # Get L2 phonemes for tracking
    l2 = languages.get(l2_code)
    l1 = languages.get(l1_code)
    non_identical_phonemes = [p for p in l2['phonemes'] if p not in l1['phonemes']]

    print(f"\n📊 BASIC STATISTICS")
    print(f"   Total L2 phonemes: {total_phonemes}")
    print(f"   Identical (d=0): {identical_count} ({identical_count/total_phonemes*100:.1f}%)")
    print(f"   Non-identical: {non_identical_count} ({non_identical_count/total_phonemes*100:.1f}%)")
    print(f"\n   Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"   Mean: {distances.mean():.4f}")
    print(f"   Median: {np.median(distances):.4f}")
    print(f"   Std dev: {distances.std():.4f}")

    # Percentiles
    print(f"\n   Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"     {p}th: {np.percentile(distances, p):.4f}")

    # Find gaps
    print(f"\n🔍 GAP ANALYSIS")
    gaps = find_distribution_gaps(distances, min_gap=0.015)

    if gaps:
        print(f"   Found {len(gaps)} significant gaps (>0.015):")
        for i, gap in enumerate(gaps[:3], 1):
            print(f"   {i}. Size: {gap['size']:.4f} at [{gap['location'][0]:.4f}, {gap['location'][1]:.4f}]")
            print(f"      Phonemes before: {gap['phonemes_before']}, after: {gap['phonemes_after']}")
    else:
        print("   No significant gaps found")

    # KDE peaks
    print(f"\n📈 KERNEL DENSITY PEAKS")
    peak_info, x_grid, kde_values = compute_kde_peaks(distances)

    if peak_info:
        print(f"   Found {len(peak_info)} peaks:")
        for i, peak in enumerate(peak_info, 1):
            print(f"   {i}. Location: {peak['location']:.4f}, Density: {peak['density']:.2f}")

        if len(peak_info) >= 2:
            print(f"   🔍 BIMODAL distribution detected")
            valley = (peak_info[0]['location'] + peak_info[1]['location']) / 2
            print(f"      Valley between peaks at ≈ {valley:.4f}")
    else:
        print("   Single peak (unimodal)")

    # Gaussian Mixture Model
    print(f"\n🎯 GAUSSIAN MIXTURE MODEL")
    cluster_info, n_clusters, bic = fit_gaussian_mixture(distances)

    print(f"   Optimal clusters: {n_clusters} (BIC: {bic:.1f})")
    for i, cluster in enumerate(cluster_info, 1):
        print(f"   Cluster {i}:")
        print(f"     Mean: {cluster['mean']:.4f} ± {cluster['std']:.4f}")
        print(f"     Size: {cluster['size']} phonemes ({cluster['weight']*100:.1f}%)")
        print(f"     Range: [{cluster['range'][0]:.4f}, {cluster['range'][1]:.4f}]")

    # Suggested thresholds from clustering
    if n_clusters >= 2:
        threshold_1 = (cluster_info[0]['mean'] + cluster_info[1]['mean']) / 2
        print(f"\n   💡 Suggested threshold (Similar/New): {threshold_1:.4f}")

        if n_clusters >= 3:
            threshold_2 = (cluster_info[1]['mean'] + cluster_info[2]['mean']) / 2
            print(f"   💡 Suggested threshold (New/Very New): {threshold_2:.4f}")

    # Silhouette-based difficulty
    print(f"\n🎯 SILHOUETTE-BASED DIFFICULTY")
    silh_result = compute_silhouette_based_difficulty(distances, non_identical_phonemes, l1_code, l2_code)

    if silh_result:
        print(f"   Overall silhouette score: {silh_result['silhouette_score']:.3f}")
        print(f"   Number of clusters: {silh_result['n_clusters']}")
        print(f"   Cluster centroids: {[f'{c:.4f}' for c in silh_result['centroids']]}")
        print(f"   Similar cluster index: {silh_result['similar_cluster_idx']}")

        # Show hardest phonemes (in similar cluster, close to centroid)
        sorted_by_difficulty = sorted(silh_result['difficulties'], key=lambda x: -x['difficulty'])
        print(f"\n   📍 TOP 5 HARDEST PHONEMES (in 'similar' cluster):")
        for i, item in enumerate(sorted_by_difficulty[:5], 1):
            if i <= len(non_identical_phonemes):
                phoneme = non_identical_phonemes[sorted_by_difficulty.index(item)]
                print(f"      {i}. /{phoneme}/ - distance: {item['distance']:.4f}, difficulty: {item['difficulty']:.1f}/5.0")
                print(f"         cluster: {item['cluster_name']}, silhouette: {item['silhouette']:.3f}")

    print()

    return {
        "l1": l1_name,
        "l2": l2_name,
        "distances": distances.tolist(),
        "phonemes": non_identical_phonemes,
        "identical_count": identical_count,
        "total_phonemes": total_phonemes,
        "stats": {
            "min": float(distances.min()),
            "max": float(distances.max()),
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "std": float(distances.std()),
        },
        "gaps": gaps,
        "peaks": peak_info,
        "clusters": cluster_info,
        "n_clusters": n_clusters,
        "silhouette_analysis": silh_result,
        "kde": {
            "x": x_grid.tolist(),
            "y": kde_values.tolist(),
        }
    }


# Analyze multiple language pairs
language_pairs = [
    ("spa", "eng", "Spanish", "English"),
    ("jpn", "eng", "Japanese", "English"),
    ("fra", "eng", "French", "English"),
    ("deu", "eng", "German", "English"),
    ("cmn", "eng", "Mandarin", "English"),
    ("hin", "eng", "Hindi", "English"),
    ("kor", "eng", "Korean", "English"),
    ("rus", "eng", "Russian", "English"),
]

results = []

for l1_code, l2_code, l1_name, l2_name in language_pairs:
    result = analyze_language_pair(l1_code, l2_code, l1_name, l2_name)
    if result:
        results.append(result)

# Save results
output_path = project_root / "research/v2.3_development/distance_distributions.json"

# Convert numpy types to Python types for JSON serialization
def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

results_serializable = convert_to_json_serializable(results)

with open(output_path, "w") as f:
    json.dump(results_serializable, f, indent=2)

print("=" * 80)
print(f"✓ Results saved to {output_path}")
print("=" * 80)

# Summary statistics
print("\n📊 CROSS-LINGUISTIC SUMMARY")
print("=" * 80)

all_means = [r["stats"]["mean"] for r in results]
all_medians = [r["stats"]["median"] for r in results]
all_cluster_counts = [r["n_clusters"] for r in results]

print(f"Mean distance across languages: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
print(f"Median distance across languages: {np.mean(all_medians):.4f} ± {np.std(all_medians):.4f}")
print(f"Typical cluster count: {np.median(all_cluster_counts):.0f}")

# Check for universal patterns
bimodal_count = sum(1 for r in results if len(r["peaks"]) >= 2)
print(f"\nBimodal distributions: {bimodal_count}/{len(results)} ({bimodal_count/len(results)*100:.0f}%)")

# Universal threshold candidates (if they exist)
if all_cluster_counts:
    print("\n💡 THRESHOLD CANDIDATES (data-driven)")

    # Collect all cluster boundaries
    all_boundaries = []
    for r in results:
        clusters = r["clusters"]
        for i in range(len(clusters) - 1):
            boundary = (clusters[i]["mean"] + clusters[i+1]["mean"]) / 2
            all_boundaries.append(boundary)

    if all_boundaries:
        print(f"   All boundaries: {np.mean(all_boundaries):.4f} ± {np.std(all_boundaries):.4f}")
        print(f"   Range: [{min(all_boundaries):.4f}, {max(all_boundaries):.4f}]")

        if len(all_boundaries) >= 2:
            # Cluster the boundaries themselves
            boundary_clusters = KMeans(n_clusters=min(2, len(set(all_boundaries))), random_state=42)
            boundary_array = np.array(all_boundaries).reshape(-1, 1)
            boundary_clusters.fit(boundary_array)

            universal_thresholds = sorted(boundary_clusters.cluster_centers_.flatten())
            print(f"\n   🎯 UNIVERSAL THRESHOLD CANDIDATES:")
            for i, thresh in enumerate(universal_thresholds, 1):
                print(f"      Threshold {i}: {thresh:.4f}")

print("\n✓ Analysis complete!")
