#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to create sample data for model comparison testing
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import data module
sys.path.append('/home/joseph/Documents/CHANGE_project_repo')

from data import get_CHANGE_data_for_sentences

def create_sample_dataset():
    """Create a small sample dataset for testing the model comparison tool"""
    
    # Create sample data directory
    sample_dir = 'sample_data'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create some sample educational texts
    sample_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
        "Deep learning uses neural networks with many layers to model complex patterns in data.",
        "Natural language processing enables computers to understand and generate human language.",
        "Transformers have become the dominant architecture for sequence-to-sequence tasks in NLP.",
        "Embedding models convert text into dense vector representations that capture semantic meaning.",
        "Fine-tuning adapts pre-trained models to specific domains or tasks with additional training.",
        "The attention mechanism allows models to focus on relevant parts of input sequences.",
        "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
        "BERT and its variants have achieved state-of-the-art results on many NLP benchmarks.",
        "Large language models require significant computational resources for training and inference."
    ]
    
    # Save sample texts
    sample_file = os.path.join(sample_dir, 'sample_texts.json')
    with open(sample_file, 'w') as f:
        json.dump({
            'texts': sample_texts,
            'description': 'Sample educational texts for model comparison testing'
        }, f, indent=2)
    
    print(f"Created sample dataset with {len(sample_texts)} texts at {sample_file}")
    
    # Also create a simple triplet dataset for testing
    triplets = []
    for i in range(len(sample_texts)):
        for j in range(i+1, len(sample_texts)):
            if abs(i - j) <= 3:  # Create positive pairs from nearby texts
                # Find a negative example (far away text)
                neg_idx = (i + 5) % len(sample_texts)
                triplets.append({
                    'anchor': sample_texts[i],
                    'positive': sample_texts[j],
                    'negative': sample_texts[neg_idx]
                })
    
    triplets_file = os.path.join(sample_dir, 'sample_triplets.json')
    with open(triplets_file, 'w') as f:
        json.dump({
            'triplets': triplets,
            'description': 'Sample triplets for model comparison testing'
        }, f, indent=2)
    
    print(f"Created sample triplet dataset with {len(triplets)} triplets at {triplets_file}")
    
    return sample_texts, triplets

def load_sample_data():
    """Load sample data for testing"""
    sample_file = 'sample_data/sample_texts.json'
    
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            data = json.load(f)
        return data['texts']
    else:
        return create_sample_dataset()[0]

if __name__ == "__main__":
    create_sample_dataset()