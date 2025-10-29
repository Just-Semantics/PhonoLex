"""
Extract psycholinguistic norms from cognitive graph for PhonoLex integration.

This script loads the cognitive graph from assocnet and extracts word-level features
relevant for phonological treatment word selection:
- Age of Acquisition (AoA)
- Word frequency (SUBTLEX-US)
- Imageability
- Concreteness
- Familiarity
- And more...
"""

import pickle
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_cognitive_graph(graph_path: str = None) -> object:
    """Load the cognitive graph from pickle file."""
    if graph_path is None:
        graph_path = project_root / "data" / "norms" / "cognitive_graph_v7.pkl"

    print(f"Loading cognitive graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # Check what type of object we got
    print(f"Graph type: {type(graph)}")

    # If it's a dict, it might have a 'graph' key with the actual NetworkX graph
    if isinstance(graph, dict):
        if 'graph' in graph:
            graph = graph['graph']
            print(f"Extracted NetworkX graph from dict")
        else:
            # It's just a dict representation, convert to what we need
            print(f"Graph is a dict with keys: {list(graph.keys())[:10]}...")
            return graph

    if hasattr(graph, 'nodes'):
        print(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    return graph


def load_metadata(metadata_path: str = None) -> dict:
    """Load metadata about the graph."""
    if metadata_path is None:
        metadata_path = project_root / "data" / "norms" / "cognitive_graph_v7.metadata.json"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def explore_graph_features(graph) -> pd.DataFrame:
    """
    Explore what features are available in the graph.

    Returns DataFrame with feature name, count, example values.
    """
    feature_counts = {}
    feature_examples = {}

    for word in graph.nodes():
        node_data = graph.nodes[word]

        for feature_name, feature_data in node_data.items():
            if isinstance(feature_data, dict) and 'value' in feature_data:
                # Count how many words have this feature
                if feature_name not in feature_counts:
                    feature_counts[feature_name] = 0
                    feature_examples[feature_name] = []

                feature_counts[feature_name] += 1

                # Save first 3 examples
                if len(feature_examples[feature_name]) < 3:
                    feature_examples[feature_name].append({
                        'word': word,
                        'value': feature_data['value'],
                        'source': feature_data.get('source', 'unknown')
                    })

    # Create summary DataFrame
    summary_data = []
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        summary_data.append({
            'feature': feature,
            'count': count,
            'coverage_pct': f"{count / len(graph.nodes) * 100:.1f}%",
            'example_words': [ex['word'] for ex in feature_examples[feature]],
            'example_values': [ex['value'] for ex in feature_examples[feature]],
            'source': feature_examples[feature][0]['source'] if feature_examples[feature] else 'unknown'
        })

    return pd.DataFrame(summary_data)


def extract_norms_for_phonolex(graph) -> pd.DataFrame:
    """
    Extract all psycholinguistic norms from graph into a DataFrame.

    Columns will be:
    - word: The word (orthographic form)
    - aoa: Age of acquisition (years)
    - frequency: Word frequency (per million)
    - imageability: Imageability rating (1-7)
    - concreteness: Concreteness rating (1-5)
    - familiarity: Familiarity rating (1-7)
    - valence: Emotional valence (1-9)
    - arousal: Emotional arousal (1-9)
    - dominance: Emotional dominance (1-9)
    - ... and more
    """
    rows = []

    for word in graph.nodes():
        node_data = graph.nodes[word]

        # Extract values from nested dict structure
        row = {'word': word}

        for feature_name, feature_data in node_data.items():
            if isinstance(feature_data, dict) and 'value' in feature_data:
                row[feature_name] = feature_data['value']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns: word first, then sorted features
    cols = ['word'] + sorted([c for c in df.columns if c != 'word'])
    df = df[cols]

    return df


def find_words_with_all_features(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """Find words that have all specified features (no NaN values)."""
    subset = df[['word'] + required_features].copy()
    complete_rows = subset.dropna()

    print(f"\nWords with all features {required_features}:")
    print(f"  Total words in graph: {len(df)}")
    print(f"  Words with ALL features: {len(complete_rows)} ({len(complete_rows)/len(df)*100:.1f}%)")

    return complete_rows


def main():
    """Main exploration and extraction pipeline."""

    # Load graph
    graph = load_cognitive_graph()
    metadata = load_metadata()

    print("\n" + "="*80)
    print("COGNITIVE GRAPH METADATA")
    print("="*80)
    print(json.dumps(metadata, indent=2))

    # Explore features
    print("\n" + "="*80)
    print("AVAILABLE FEATURES")
    print("="*80)
    feature_summary = explore_graph_features(graph)
    print(feature_summary.to_string())

    # Extract all norms
    print("\n" + "="*80)
    print("EXTRACTING ALL NORMS")
    print("="*80)
    norms_df = extract_norms_for_phonolex(graph)
    print(f"Extracted norms for {len(norms_df)} words")
    print(f"Features extracted: {list(norms_df.columns)}")

    # Check coverage for key features we need for PhonoLex
    key_features = ['aoa', 'frequency', 'imageability', 'concreteness']
    print("\n" + "="*80)
    print("KEY FEATURES FOR PHONOLEX")
    print("="*80)
    for feature in key_features:
        if feature in norms_df.columns:
            count = norms_df[feature].notna().sum()
            print(f"  {feature:20s}: {count:6d} words ({count/len(norms_df)*100:5.1f}%)")
        else:
            print(f"  {feature:20s}: NOT FOUND")

    # Find words with complete data for treatment word selection
    complete_words = find_words_with_all_features(norms_df, key_features)

    # Save to CSV
    output_path = project_root / "data" / "norms" / "psycholinguistic_norms.csv"
    norms_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved all norms to: {output_path}")

    # Save complete subset
    complete_output_path = project_root / "data" / "norms" / "psycholinguistic_norms_complete.csv"
    complete_words.to_csv(complete_output_path, index=False)
    print(f"✓ Saved complete norms ({len(complete_words)} words) to: {complete_output_path}")

    # Show sample
    print("\n" + "="*80)
    print("SAMPLE WORDS (first 10 with complete data)")
    print("="*80)
    print(complete_words.head(10).to_string())

    return norms_df, complete_words


if __name__ == "__main__":
    norms_df, complete_words = main()
