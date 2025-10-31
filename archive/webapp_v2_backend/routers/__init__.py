"""
FastAPI Routers for PhonoLex API

This package contains all API route modules:
- words: Word queries and filtering
- similarity: Vector similarity search
- graph: Graph operations (minimal pairs, rhymes, neighbors)
- phonemes: Phoneme queries and feature search
"""

from . import words, similarity, graph, phonemes

__all__ = ['words', 'similarity', 'graph', 'phonemes']
