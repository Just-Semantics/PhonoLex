#!/usr/bin/env python
"""
Run the PhonoLex examples.

This script runs the phonological rules example.
"""

import os
import sys
import importlib.util

def run_example(example_path):
    """Run an example script."""
    print(f"Running example: {example_path}")
    print("=" * 60)
    
    # Add the example directory to sys.path
    example_dir = os.path.dirname(example_path)
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)
    
    # Load and run the example module
    module_name = os.path.splitext(os.path.basename(example_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, example_path)
    example_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_module)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Run the phonological rules example
    example_path = os.path.join(project_root, "examples", "phonological_rules.py")
    run_example(example_path) 