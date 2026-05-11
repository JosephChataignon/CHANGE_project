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

# Configure logging for this module
logger = logging.getLogger(__name__)


def load_dataset_samples(sample_size: int = 100, dataset_type: str = 'our') -> List[str]:
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
    if dataset_type == 'our':
        # Load real CHANGE data for our dataset
        data_storage = os.getenv('DATA_STORAGE')
        if not data_storage:
            raise ValueError("DATA_STORAGE environment variable not set. ")
        
        try:
            logger.info(f"Loading CHANGE education_sample data from {data_storage}")
            
            # Load the dataset - this returns a DatasetDict with triplets
            dataset = get_CHANGE_data_for_sentences(
                data_type='education_sample', 
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
                        if 'anchor' in triplet:
                            all_texts.add(triplet['anchor'])
                        if 'positive' in triplet:
                            all_texts.add(triplet['positive'])
                        if 'negative' in triplet:
                            all_texts.add(triplet['negative'])
            
            texts = list(all_texts)
            if len(texts) == 0:
                raise RuntimeError("No text samples found in CHANGE dataset. "
                                 "Please check that the data directory contains valid files.")
            
            logger.info(f"Loaded {len(texts)} unique text samples from CHANGE education_sample dataset")
            return texts[:sample_size]
            
        except Exception as e:
            logger.error(f"Error loading CHANGE dataset: {e}")
            raise RuntimeError(f"Cannot load CHANGE dataset: {str(e)}. "
                             "Please ensure DATA_STORAGE points to a valid directory "
                             "with Projekt_Change_LLM/Preprocessed_Eduscience_data/sample_clean subdirectory.")
        
    elif dataset_type == 'general':
        # Load AllenAI C4 dataset for general dataset
        try:
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
            
            if len(samples) < sample_size:
                raise RuntimeError(f"Only collected {len(samples)} valid samples from C4 dataset. ")
            
            # Shuffle and return requested number
            random.shuffle(samples)
            logger.info(f"Loaded {len(samples)} samples from AllenAI C4 dataset")
            return samples[:sample_size]
            
        except Exception as e:
            logger.error(f"Failed to load AllenAI C4 dataset: {e}")
            raise RuntimeError(f"Cannot load general dataset: {str(e)}. ")
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be either 'our' or 'general'.")

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
    our_data = load_dataset_samples(own_data_size, dataset_type='our')
    
    # Load general dataset samples
    general_data = load_dataset_samples(general_data_size, dataset_type='general')
    
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


def save_summary_report(report: Dict, filename: str):
    """
    Save a human-readable summary of the comparison results
    
    Args:
        report: Dictionary containing all analysis results
        filename: Path to save the summary report
    """
    try:
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON REPORT - SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Metadata
            f.write("METADATA:\n")
            f.write(f"Timestamp: {report['metadata']['timestamp']}\n")
            f.write(f"Original Model: {report['metadata']['original_model']}\n")
            f.write(f"Fine-tuned Model: {report['metadata']['fine_tuned_model']}\n")
            f.write(f"Sample Size: {report['metadata']['sample_size']}\n")
            f.write(f"Device: {report['metadata']['device']}\n\n")
            
            # Weight Analysis Summary
            f.write("WEIGHT ANALYSIS SUMMARY:\n")
            weight_analysis = report['weight_analysis']
            if weight_analysis:
                total_layers = len(weight_analysis)
                mean_deltas = [v['mean_abs_delta'] for v in weight_analysis.values()]
                max_deltas = [v['max_abs_delta'] for v in weight_analysis.values()]
                norm_deltas = [v['norm_delta'] for v in weight_analysis.values()]
                rel_changes = [v['relative_change'] for v in weight_analysis.values()]
                
                f.write(f"Total layers analyzed: {total_layers}\n")
                f.write(f"Average mean absolute delta: {np.mean(mean_deltas):.6f}\n")
                f.write(f"Maximum mean absolute delta: {np.max(mean_deltas):.6f}\n")
                f.write(f"Average maximum delta: {np.mean(max_deltas):.6f}\n")
                f.write(f"Maximum delta observed: {np.max(max_deltas):.6f}\n")
                f.write(f"Average norm delta: {np.mean(norm_deltas):.6f}\n")
                f.write(f"Average relative change: {np.mean(rel_changes):.4f}\n\n")
            
            # PCA Comparison Summary
            f.write("PCA COMPARISON SUMMARY:\n")
            pca = report['pca']
            f.write(f"PCA Explained Variance: {pca['pca_variance']}\n")
            f.write(f"Our Dataset Samples: {pca['sample_sizes']['our_dataset']}\n")
            f.write(f"General Dataset Samples: {pca['sample_sizes']['general_dataset']}\n")
            f.write(f"Visualizations saved in 'visualizations/' directory\n\n")
            
            # Interpretation
            f.write("INTERPRETATION:\n")
            sim = report['embedding_similarity']
            mean_sim = sim['mean_cosine_similarity']
            if mean_sim > 0.9:
                f.write("High similarity (>0.9): Fine-tuning preserved most embedding characteristics\n")
            elif mean_sim > 0.7:
                f.write("Moderate similarity (0.7-0.9): Fine-tuning caused noticeable changes\n")
            else:
                f.write("Low similarity (<0.7): Fine-tuning significantly altered embeddings\n")
            
            avg_rel_change = np.mean(rel_changes) if 'rel_changes' in locals() else 0
            if avg_rel_change < 0.01:
                f.write("Small weight changes (<1%): Fine-tuning was conservative\n")
            elif avg_rel_change < 0.05:
                f.write("Moderate weight changes (1-5%): Fine-tuning adapted the model\n")
            else:
                f.write("Large weight changes (>5%): Fine-tuning significantly modified weights\n")
            
            f.write("\nVisualizations created:\n")
            f.write("- 4-category PCA comparison visualization\n")
            f.write("- Weight delta analysis per layer\n\n")
            
            f.write("="*80 + "\n")
            f.write("ANALYSIS COMPLETE\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Summary report saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
        raise