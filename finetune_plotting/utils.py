#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for model comparison tool

This module contains helper functions for:
- Data loading and preparation
- Visualization creation
- Report generation
- Common utilities
"""

import os, sys, logging, random
from typing import Dict, List
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
            

# Add parent directory to path to import data.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data import get_CHANGE_data_for_sentences
from data import get_CHANGE_data_by_document

# Configure logging for this module
logger = logging.getLogger(__name__)


def load_dataset_samples(dataset_type: str, sample_size: int) -> List[str]:
    """
    Load sample texts from specified dataset using CHANGE data or AllenAI C4
    
    Args:
        sample_size: Number of text samples to load
        dataset_type: 'our' or 'general' dataset
    
    Returns:
        List of text samples
    
    Raises:
        ValueError: If DATA_STORAGE not set or dataset_type invalid
        RuntimeError: If data loading fails
    """
    data_storage = os.getenv('DATA_STORAGE')
    assert data_storage is not None, "DATA_STORAGE environment variable not set"
    
    if dataset_type == 'our':
        # Load real CHANGE data for our dataset
        logger.info(f"Loading CHANGE education data from {data_storage}")
        dataset = get_CHANGE_data_for_sentences(data_type='education', 
            data_storage=data_storage,
            segmentation_method={"method": "sentence", "chunk_size": 6, "overlap": 0},
            sample_scale=0.00001,  # Small scale for faster loading
        )
        # Extract all unique text samples from triplets
        all_texts = set()
        for split_name in ['train', 'dev', 'test']:
            if split_name in dataset:
                split_data = dataset[split_name]
                for triplet in split_data:
                    all_texts.add(triplet['anchor'])
                    all_texts.add(triplet['positive'])
                    all_texts.add(triplet['negative'])
        texts = list(all_texts)
        
        assert len(texts) > 0, "No text samples found in CHANGE dataset. "
        logger.info(f"Loaded {len(texts)} unique text samples from CHANGE education dataset")
        
        return texts[:sample_size]
            
        
    elif dataset_type == 'general':
        # Load AllenAI C4 dataset for general dataset
        logger.info("Loading AllenAI C4 dataset (English and German subsets)")
        
        # Load English and German subsets in streaming mode
        en_dataset = load_dataset("allenai/c4", "en", streaming=True)
        de_dataset = load_dataset("allenai/c4", "de", streaming=True)
        
        # Sample from both datasets
        samples = []
        for dataset, lang in [(en_dataset['train'], 'en'), (de_dataset['train'], 'de')]:
            # Take samples from each language
            sampled = dataset.take(sample_size // 2)  # Half from each language
            for example in sampled:
                if 'text' in example and len(example['text'].strip()) > 100:
                    samples.append(example['text'].strip())
        
        # Shuffle and return requested number
        random.shuffle(samples)
        logger.info(f"Loaded {len(samples)} samples from AllenAI C4 dataset")
        return samples[:sample_size]
    
    else:
        logger.info("Trying to load datset as a single doc")
        fullpath = os.path.join(data_storage, "Projekt_Change_LLM/Eduscience_data", dataset_type)
        assert os.path.exists(fullpath), f"Data storage path {fullpath} does not exist"
        
        texts = get_CHANGE_data_by_document(fullpath, data_storage, 
                segmentation_method={"method": "sentence", "chunk_size": 6, "overlap": 0},
                max_chunks=sample_size )
        return texts


def create_category_plot(transformed_data: Dict, categories: List[Dict], pca, save_path: str = 'visualizations'):
    """
    Create scatter plot for arbitrary categories with different colors and markers
    
    Args:
        transformed_data: Dictionary containing transformed PCA data for all categories
        categories: List of category definitions with names, colors, and markers
        pca: PCA object for variance information
        save_path: Directory to save the visualization (relative to script directory)
    """
    plt.figure(figsize=(14, 10))
    
    # Plot each category
    for category in categories:
        # Find the data key that matches this category
        data_key = None
        if category['model_type'] == 'base':
            data_key = category['base_key']
        else:
            data_key = category['finetuned_key']
        
        if data_key and data_key in transformed_data:
            data = transformed_data[data_key]
            
            # Use marker if specified, default to circle
            marker = category.get('marker', 'o')
            alpha = category.get('alpha', 0.6)
            size = category.get('size', 80)
            
            plt.scatter(data[:, 0], data[:, 1],
                        color=category['color'],
                        label=category['name'],
                        alpha=alpha,
                        s=size,
                        marker=marker)
    
    # Get PCA variance from PCA object
    pca_variance = pca.explained_variance_ratio_
    
    plt.title('PCA Comparison: Embedding Space', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca_variance[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca_variance[1]*100:.1f}%)', fontsize=12)
    plt.legend(title='Category', fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Get script directory to ensure paths are relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(script_dir, save_path)
    os.makedirs(full_save_path, exist_ok=True)
    # Save visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(full_save_path, f'pca_comparison_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PCA comparison visualization to {filename}")
    return filename
        


def create_categories_for_visualization(self, label_info: List[str]) -> List[Dict]:
    """
    Create category definitions for visualization with proper color scheme
    
    Args:
        label_info: List of label strings
        
    Returns:
        List of category definitions with names and colors
    """
    categories = []
    
    # Define color schemes
    bluish_colors = [
        '#1f77b4', '#6baed6', '#9ecae1', '#c6dbef',  # Blues
        '#21908d', '#4292c6', '#6baed6', '#9ecae1',  # Teals
        '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'   # More blues
    ]
    
    for i, label in enumerate(label_info):
        
        # Assign red to 'general_data' label, cycle through bluish colors for others
        if label == 'general_data':
            color = '#ff7f0e'  # Orange-red
        else:
            color = bluish_colors[i % len(bluish_colors)]
            
        
    # Create category entries for base and fine-tuned models
        categories.append({
            'name': f'{label}/base',
            'color': color,
            'marker': 'o',  # Circle for base model
            'alpha': 0.6,
            'label': label
        })
        categories.append({
            'name': f'{label}/finetuned',
            'color': color,
            'marker': 's',  # Square for fine-tuned model
            'alpha': 1.0,
            'label': label
        })
    return categories
