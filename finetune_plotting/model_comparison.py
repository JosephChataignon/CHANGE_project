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
from utils import load_comparison_data, create_category_plot, save_summary_report

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
        logger.info(f"Original model: {self.original_model_path}")
        logger.info(f"Fine-tuned model: {self.fine_tuned_model_path}")
    
        try:
            logger.info("Loading original model...")
            self.original_model = SentenceTransformer(self.original_model_path, device=self.device)
            logger.info(f"Original model loaded: {self.original_model_path}")
            
            logger.info("Loading fine-tuned model...")
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
        
        try:
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
                            # Calculate various metrics
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
                            
        except Exception as e:
            logger.error(f"Error comparing weights: {e}")
            raise
        
        self.weight_deltas = weight_deltas

    def analyze_embedding_similarity(self, texts: Optional[List[str]] = None, 
                                    sample_size: int = 500, 
                                    batch_size: int = 32) -> Dict[str, float]:
        """
        Analyze similarity between original and fine-tuned embeddings
        Args:
            texts: Optional list of text strings to analyze
            sample_size: Number of texts to sample if none provided
            batch_size: Batch size for embedding generation
        
        Returns:
            Dictionary containing similarity metrics
        """
        if texts is None:
            # Load sample texts for similarity analysis
            sample_data = load_comparison_data(own_data_size=sample_size, general_data_size=0)
            texts = sample_data['our_data']
        
        try:
            # Generate embeddings from both models
            original_embeddings = self.original_model.encode(texts, 
                                                           batch_size=batch_size, 
                                                           convert_to_numpy=True,
                                                           show_progress_bar=True)
            fine_tuned_embeddings = self.fine_tuned_model.encode(texts,
                                                               batch_size=batch_size, 
                                                               convert_to_numpy=True,
                                                               show_progress_bar=True)
            
            # Calculate pairwise cosine similarities
            similarities = cosine_similarity(original_embeddings, fine_tuned_embeddings)
            similarity_metrics = {
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
            
            logger.info(f"Embedding similarity analysis completed. Mean similarity: {similarity_metrics['mean_cosine_similarity']:.4f}")
            
            return similarity_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing embedding similarity: {e}")
            raise

    def pca_comparison_visualization(self, own_data_size: int = 500, general_data_size: int = 500) -> Dict:
        """
        Generate 4-category PCA comparison visualization
        
        Args:
            own_data_size: Number of samples from our dataset
            general_data_size: Number of samples from general dataset
        
        Returns:
            Dictionary containing PCA results and visualization data
        """
        # Load comparison data
        data = load_comparison_data(own_data_size, general_data_size)
        
        # Generate embeddings for all 4 categories
        logger.info("Generating embeddings for all categories...")
        
        # Our dataset embeddings
        our_base_embeddings = self.original_model.encode(data['our_data'], 
                                                       convert_to_numpy=True, 
                                                       show_progress_bar=True)
        our_finetuned_embeddings = self.fine_tuned_model.encode(data['our_data'], 
                                                              convert_to_numpy=True, 
                                                              show_progress_bar=True)
        
        # General dataset embeddings
        general_base_embeddings = self.original_model.encode(data['general_data'], 
                                                           convert_to_numpy=True, 
                                                           show_progress_bar=True)
        general_finetuned_embeddings = self.fine_tuned_model.encode(data['general_data'], 
                                                                    convert_to_numpy=True, 
                                                                    show_progress_bar=True)
        
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

    def generate_comparison_report(self, sample_size: int = 50) -> Dict:
        """
        Args:
            sample_size: Number of text samples to use for analysis        
        Returns:
            Dictionary containing all comparison results
        """
        try:
            # Create report directory
            report_dir = "comparison_reports"
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
            # 1. Weight analysis
            logger.info("Performing weight analysis...")
            self.compare_weights()
            
            # 2. Embedding similarity analysis
            logger.info("Performing embedding similarity analysis...")
            similarity_results = self.analyze_embedding_similarity(sample_size=sample_size)
            
            # 3. PCA comparison visualization
            logger.info("Performing PCA comparison analysis...")
            pca_results = self.pca_comparison_visualization(own_data_size=sample_size, 
                                                           general_data_size=sample_size)
            
            # Compile comprehensive report, save as json
            report = {
                'metadata': {
                    'timestamp': timestamp,
                    'original_model': self.original_model_path,
                    'fine_tuned_model': self.fine_tuned_model_path,
                    'sample_size': sample_size,
                    'device': str(self.device)
                },
                'weight_analysis': self.weight_results,
                'embedding_similarity': similarity_results,
                'pca': pca_results
            }
            report_filename = os.path.join(report_dir, f'comparison_report_{timestamp}.json')
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Comparison report saved to {report_filename}")
            summary_filename = os.path.join(report_dir, f'comparison_summary_{timestamp}.txt')
            save_summary_report(report, summary_filename)
            return report
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            raise


def main():
    """Main function to run model comparison"""
    load_dotenv()
    original_model = os.getenv("ORIGINAL_MODEL")
    fine_tuned_model = os.getenv("FINETUNED_MODEL")
    
    # Initialize comparator
    comparator = ModelComparator(original_model, fine_tuned_model)
    
    # Generate comprehensive report
    report = comparator.generate_comparison_report(sample_size=30)
    
    logger.info("Model comparison completed")
    logger.info("Check the 'visualizations/' and 'comparison_reports/' directories for results.")
        


if __name__ == "__main__":
    main()