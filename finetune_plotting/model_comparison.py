#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Tool for Fine-tuned Embedding Models

This script compares original and fine-tuned embedding models by:
1. Analyzing weight deltas between models
2. Visualizing embeddings using PCA (4-category comparison)
3. Comparing embedding spaces and similarities
4. Generating comprehensive reports
"""

import os, sys, json, logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_comparison_data, create_category_plot, load_dataset_samples

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelComparator:
    """Class to compare original and fine-tuned embedding models"""
    
    def __init__(self, original_model_path: str, fine_tuned_model_path: str, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            original_model_path: Path to original model (HuggingFace model name or local path)
            fine_tuned_model_path: Path to fine-tuned model (local path)
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.original_model_path = original_model_path
        self.fine_tuned_model_path = fine_tuned_model_path
        self.device = torch.device(device)
        # Placeholders for analysis results
        self.weight_deltas = {}
        self.pca_results = {}
        logger.info(f"Initialized ModelComparator with device: {self.device}")
        
        try:
            self.original_model = SentenceTransformer(self.original_model_path, device=self.device)
            logger.info(f"Original model loaded: {self.original_model_path}")
            self.fine_tuned_model = SentenceTransformer(self.fine_tuned_model_path, device=self.device)
            logger.info(f"Fine-tuned model loaded: {self.fine_tuned_model_path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def compare_weights(self) -> Dict[str, Dict]:
        """
        Compare weights between original and fine-tuned models
        
        Returns:
            Dictionary containing weight comparison metrics for each layer
        """
        weight_deltas = {}
    
        # Get state dicts from both models
        original_state = self.original_model.state_dict()
        fine_tuned_state = self.fine_tuned_model.state_dict()
        
        logger.info(f"Comparing weights across {len(original_state)} layers...")
        
        for key in original_state.keys():
            if key in fine_tuned_state:
                original_weight = original_state[key]
                fine_tuned_weight = fine_tuned_state[key]
                # Only compare tensors (weights, not metadata)
                if isinstance(original_weight, torch.Tensor) and isinstance(fine_tuned_weight, torch.Tensor):
                    if original_weight.dtype in [torch.float32, torch.float16, torch.float64]:
                        delta = fine_tuned_weight - original_weight
                        metrics = {
                            'mean_abs_delta': float(torch.mean(torch.abs(delta)).cpu()),
                            'max_abs_delta': float(torch.max(torch.abs(delta)).cpu()),
                            'std_delta': float(torch.std(delta).cpu()),
                            'mean_original': float(torch.mean(original_weight).cpu()),
                            'mean_fine_tuned': float(torch.mean(fine_tuned_weight).cpu()),
                            'shape': list(original_weight.shape),
                            'dtype': str(original_weight.dtype),
                            'norm_delta': float(torch.norm(delta).cpu()),
                            'relative_change': float((torch.norm(delta) / (torch.norm(original_weight) + 1e-8)).cpu())
                        }
                        weight_deltas[key] = metrics
                        logger.debug(f"Weight delta for {key}: mean_abs={metrics['mean_abs_delta']:.6f}")
        self.weight_deltas = weight_deltas

    def analyze_embedding_similarity(self, texts: List[str], label: str, 
                                    batch_size: int = 32) -> Dict[str, float]:
        """
        Analyze similarity between original and fine-tuned embeddings
        Args:
            texts: List of text strings to analyze
            label: Label for the analysis
            batch_size: Batch size for embedding generation
        
        Returns:
            Dictionary containing similarity metrics
        """
        # Generate embeddings from both models
        original_embeddings = self.original_model.encode(texts, 
                batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        fine_tuned_embeddings = self.fine_tuned_model.encode(texts,
                batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        
        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(original_embeddings, fine_tuned_embeddings)
        similarity_metrics = {
            'label': label,
            'mean_cosine_similarity': float(np.mean(similarities.diagonal())),
            'std_cosine_similarity': float(np.std(similarities.diagonal())),
            'min_cosine_similarity': float(np.min(similarities.diagonal())),
            'max_cosine_similarity': float(np.max(similarities.diagonal())),
            'median_cosine_similarity': float(np.median(similarities.diagonal())),
            'sample_size': len(texts),
            'similarity_distribution': {
                'q25': float(np.percentile(similarities.diagonal(), 25)),
                'q75': float(np.percentile(similarities.diagonal(), 75)),
                'q90': float(np.percentile(similarities.diagonal(), 90))
            }
        }
        logger.info(f"Embedding similarity analysis completed for {label}")
        return similarity_metrics

    def pca_comparison_visualization(self, data: Dict) -> Dict:
        """
        Generate 4-category PCA comparison visualization
        
        Args:
            data: Dictionary containing dataset samples
        """
        logger.info("Generating embeddings for all categories...")
        
        #general_embeddings_base = self.original_model.encode(data_general, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
        #general_embeddings_finetuned = self.fine_tuned_model.encode(data_general, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
        embeddings = {}
        for name,data_samples in data.items():
        
        # Fit PCA on our dataset + base model only
        logger.info("Fitting PCA on our dataset with base model...")
        pca = PCA(n_components=2)
        pca.fit(our_base_embeddings)
        
        # Transform all embeddings
        logger.info("Transforming all embeddings...")
        transformed = {
            'our_base': pca.transform(our_base_embeddings),
            'our_finetuned': pca.transform(our_finetuned_embeddings),
            'general_base': pca.transform(general_base_embeddings),
            'general_finetuned': pca.transform(general_finetuned_embeddings)
        }
        
        # Create 4-category visualization
        create_category_plot(transformed, data['categories'], pca)
        
        # Store results (convert numpy arrays to lists for JSON serialization)
        pca_results = {
            'pca_variance': pca.explained_variance_ratio_.tolist(),
            'categories': data['categories'],
            'sample_sizes': {
                'our_dataset': len(data['our_data']),
                'general_dataset': len(data['general_data'])
            },
            'transformed_data': {
                'our_base': transformed['our_base'].tolist(),
                'our_finetuned': transformed['our_finetuned'].tolist(),
                'general_base': transformed['general_base'].tolist(),
                'general_finetuned': transformed['general_finetuned'].tolist()
            }
        }
        
        self.pca_results = pca_results
        
        logger.info("4-category PCA comparison visualization completed")
        
        return pca_results

    def generate_comparison_report(self, documents_list: List[str]) -> Dict:
        # Get script directory to ensure paths are relative to script location, Create report directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_dir = os.path.join(script_dir, "comparison_reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Weight analysis
        logger.info("Performing weight analysis...")
        self.compare_weights()
        
        # 2. Embedding similarity analysis
        logger.info("Performing embedding similarity analysis...")
        data_ours = load_dataset_samples('our_data')
        data_general = load_dataset_samples('general_data')
        similarity_results_ours = self.analyze_embedding_similarity(texts=data_ours, label='Our Dataset Embeddings Similarity')
        similarity_results_general = self.analyze_embedding_similarity(texts=data_general, label='General Dataset Embeddings Similarity')
        
        # 3. PCA comparison visualization
        logger.info("Performing PCA comparison analysis...")
        data_ours = load_dataset_samples('our_data')
        data_general = load_dataset_samples('general_data')
        docs_data = {}
        for i,doc in enumerate(documents_list):
            doc_data = load_dataset_samples(doc)
            docs_data[f'doc_{i}'] = doc_data
        pca_results_all = self.pca_comparison_visualization({'general_data': data_general, 'our_data': data_ours})
        pca_results_by_doc = self.pca_comparison_visualization(docs_data.update({'general_data': data_general}))
        
        # Compile comprehensive report, save as json
        report = {
            'metadata': {
                'original_model': self.original_model_path,
                'fine_tuned_model': self.fine_tuned_model_path,
                'device': str(self.device)
            },
            'weight_analysis': self.weight_deltas,
            'embedding_similarity': {
                'our_dataset': similarity_results_ours,
                'general_dataset': similarity_results_general
            },
            'pca': {
                'pca_results_all': pca_results_all,
                'pca_results_by_doc': pca_results_by_doc
            }
        }
        report_filename = os.path.join(report_dir, f'comparison_report_{timestamp}.json')
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
            


def main():
    """Main function to run model comparison"""
    load_dotenv()
    original_model = os.getenv("ORIGINAL_MODEL")
    fine_tuned_model = os.getenv("FINETUNED_MODEL")
    assert original_model is not None, "ORIGINAL_MODEL environment variable not set"
    assert fine_tuned_model is not None, "FINETUNED_MODEL environment variable not set"
    
    documents_list = [
        "list_A/2/Competence Oriented Teaching and Learning in Higher Education - Essentials _E-Book_3 Strategies to reduce learning content.txt",
        "list_A/16/Effective Online Teaching _ Tina Stavredes.txt",
        "list_B/springer/7/978-3-531-93068-8_6.txt",
        "list_B/cambridge/1745/defining_teaching_and_learning_professionalism.txt",
        "list_C/brill/55/37616.txt"
    ]
    
    # Initialize comparator
    comparator = ModelComparator(original_model, fine_tuned_model)
    
    # Generate comprehensive report
    comparator.generate_comparison_report(documents_list)
    
    logger.info("Model comparison completed")
    logger.info("Check the 'visualizations/' and 'comparison_reports/' directories for results.")


if __name__ == "__main__":
    main()