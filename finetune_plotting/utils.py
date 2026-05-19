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

def load_comparison_data(own_data_size: int = 500, general_data_size: int = 500) -> Dict:
    """
    Load data for 4-category comparison visualization
    
    Args:
        own_data_size: Number of samples from our dataset
        general_data_size: Number of samples from general dataset
    
    Returns:
        Dictionary containing data for all categories
    """
    # Load our dataset samples
    our_data = load_dataset_samples(dataset_type='our', sample_size=own_data_size)
    
    # Load general dataset samples
    general_data = load_dataset_samples(dataset_type='general', sample_size=general_data_size)
    
    # Define categories with labels and colors
    categories = [
        {'name': 'our dataset/base', 'color': 'blue'},
        {'name': 'our dataset/fine-tuned', 'color': 'orange'},
        {'name': 'general dataset/base', 'color': 'green'},
        {'name': 'general dataset/fine-tuned', 'color': 'red'}
    ]
    
    return {
        'our_data': our_data,
        'general_data': general_data,
        'categories': categories
    }


def create_category_plot(transformed_data: Dict, categories: List[Dict], pca=None, save_path: str = 'visualizations'):
    """
    Create 4-category scatter plot with different colors
    
    Args:
        transformed_data: Dictionary containing transformed PCA data for all categories
        categories: List of category definitions with names and colors
        pca: PCA object for variance information
        save_path: Directory to save the visualization (relative to script directory)
    """
    try:
        plt.figure(figsize=(14, 10))
        
        # Plot each category
        for i, category in enumerate(categories):
            if i == 0:  # our dataset/base
                data = transformed_data['our_base']
            elif i == 1:  # our dataset/fine-tuned
                data = transformed_data['our_finetuned']
            elif i == 2:  # general dataset/base
                data = transformed_data['general_base']
            else:  # general dataset/fine-tuned
                data = transformed_data['general_finetuned']
            
            plt.scatter(data[:, 0], data[:, 1],
                       color=category['color'],
                       label=category['name'],
                       alpha=0.6,
                       s=80)
        
        # Get PCA variance from PCA object
        if pca is not None:
            pca_variance = pca.explained_variance_ratio_
        else:
            pca_variance = [0.25, 0.20]  # Fallback values
        
        plt.title('PCA Comparison: Our Dataset vs General Dataset', fontsize=16)
        plt.xlabel(f'Principal Component 1 ({pca_variance[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca_variance[1]*100:.1f}%)', fontsize=12)
        plt.legend(title='Category', fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Get script directory to ensure paths are relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_save_path = os.path.join(script_dir, save_path)
        
        # Create plot directory
        os.makedirs(full_save_path, exist_ok=True)
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(full_save_path, f'pca_comparison_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved 4-category PCA comparison visualization to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error creating category plot: {e}")
        raise
