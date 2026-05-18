#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Tool for Fine-tuned Embedding Models

This script compares original and fine-tuned embedding models by:
1. Analyzing weight deltas between models
2. Visualizing embeddings using PCA
3. Comparing embedding spaces and similarities
4. Generating comprehensive reports
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
from pathlib import Path

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
        Initialize the model comparator
        
        Args:
            original_model_path: Path to original model (HuggingFace model name or local path)
            fine_tuned_model_path: Path to fine-tuned model (local path)
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.original_model_path = original_model_path
        self.fine_tuned_model_path = fine_tuned_model_path
        self.device = torch.device(device)
        
        # Placeholders for models
        self.original_model = None
        self.fine_tuned_model = None
        
        # Placeholders for analysis results
        self.weight_deltas = {}
        self.embedding_comparison = {}
        self.pca_results = {}
        
        logger.info(f"Initialized ModelComparator with device: {self.device}")
        logger.info(f"Original model: {original_model_path}")
        logger.info(f"Fine-tuned model: {fine_tuned_model_path}")
    
    def load_models(self):
        """Load both original and fine-tuned models"""
        try:
            logger.info("Loading original model...")
            self.original_model = SentenceTransformer(self.original_model_path, device=self.device)
            logger.info(f"Original model loaded: {self.original_model_path}")
            
            logger.info("Loading fine-tuned model...")
            self.fine_tuned_model = SentenceTransformer(self.fine_tuned_model_path, device=self.device)
            logger.info(f"Fine-tuned model loaded: {self.fine_tuned_model_path}")
            
            # Ensure both models use the same max sequence length
            if hasattr(self.original_model, 'max_seq_length'):
                self.fine_tuned_model.max_seq_length = self.original_model.max_seq_length
                self.fine_tuned_model.tokenizer.model_max_length = self.original_model.max_seq_length
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def compare_weights(self) -> Dict[str, Dict]:
        """
        Compare weights between original and fine-tuned models
        
        Returns:
            Dictionary containing weight comparison metrics for each layer
        """
        if not self.original_model or not self.fine_tuned_model:
            logger.error("Models not loaded. Call load_models() first.")
            raise ValueError("Models not loaded")
        
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
                                'dtype': str(original_weight.dtype)
                            }
                            
                            weight_deltas[key] = metrics
                            logger.debug(f"Weight delta for {key}: mean_abs={metrics['mean_abs_delta']:.6f}")
        
        except Exception as e:
            logger.error(f"Error comparing weights: {e}")
            raise
        
        self.weight_deltas = weight_deltas
        return weight_deltas
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings from both models for comparison
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
        
        Returns:
            Tuple of (original_embeddings, fine_tuned_embeddings) as numpy arrays
        """
        if not self.original_model or not self.fine_tuned_model:
            logger.error("Models not loaded. Call load_models() first.")
            raise ValueError("Models not loaded")
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            
            # Generate embeddings from original model
            original_embeddings = self.original_model.encode(texts, 
                                                           batch_size=batch_size, 
                                                           convert_to_numpy=True,
                                                           show_progress_bar=True)
            
            # Generate embeddings from fine-tuned model
            fine_tuned_embeddings = self.fine_tuned_model.encode(texts,
                                                               batch_size=batch_size,
                                                               convert_to_numpy=True,
                                                               show_progress_bar=True)
            
            logger.info(f"Embeddings generated. Shape: {original_embeddings.shape}")
            return original_embeddings, fine_tuned_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def analyze_embedding_similarity(self, texts: List[str], 
                                    batch_size: int = 32) -> Dict[str, float]:
        """
        Analyze similarity between original and fine-tuned embeddings
        
        Args:
            texts: List of text strings to analyze
            batch_size: Batch size for embedding generation
        
        Returns:
            Dictionary containing similarity metrics
        """
        original_embeddings, fine_tuned_embeddings = self.generate_embeddings(texts, batch_size)
        
        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(original_embeddings, fine_tuned_embeddings)
        
        # Calculate metrics
        similarity_metrics = {
            'mean_cosine_similarity': float(np.mean(similarities.diagonal())),
            'std_cosine_similarity': float(np.std(similarities.diagonal())),
            'min_cosine_similarity': float(np.min(similarities.diagonal())),
            'max_cosine_similarity': float(np.max(similarities.diagonal())),
            'median_cosine_similarity': float(np.median(similarities.diagonal())),
            'sample_size': len(texts)
        }
        
        self.embedding_comparison['similarity_metrics'] = similarity_metrics
        
        # Store the similarity matrix
        self.embedding_comparison['similarity_matrix'] = similarities.tolist()
        
        logger.info(f"Embedding similarity analysis completed. Mean similarity: {similarity_metrics['mean_cosine_similarity']:.4f}")
        
        return similarity_metrics
    
    def pca_visualization(self, texts: List[str], labels: Optional[List[str]] = None,
                         n_components: int = 2, batch_size: int = 32) -> Dict:
        """
        Perform PCA visualization of embeddings
        
        Args:
            texts: List of text strings to visualize
            labels: Optional list of labels for each text
            n_components: Number of PCA components (2 or 3)
            batch_size: Batch size for embedding generation
        
        Returns:
            Dictionary containing PCA results and visualization data
        """
        original_embeddings, fine_tuned_embeddings = self.generate_embeddings(texts, batch_size)
        
        # Perform PCA on both sets of embeddings
        pca = PCA(n_components=n_components)
        
        original_pca = pca.fit_transform(original_embeddings)
        fine_tuned_pca = pca.transform(fine_tuned_embeddings)
        
        # Store results
        pca_results = {
            'original_pca': original_pca.tolist(),
            'fine_tuned_pca': fine_tuned_pca.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'texts': texts,
            'labels': labels if labels else [f"text_{i}" for i in range(len(texts))]
        }
        
        self.pca_results = pca_results
        
        # Create visualization
        self._create_pca_plot(pca_results, n_components)
        
        logger.info(f"PCA visualization completed with {n_components} components")
        
        return pca_results
    
    def _create_pca_plot(self, pca_results: Dict, n_components: int):
        """Create and save PCA visualization plots"""
        try:
            # Create DataFrame for plotting
            df_original = pd.DataFrame({
                'PC1': [x[0] for x in pca_results['original_pca']],
                'PC2': [x[1] for x in pca_results['original_pca']] if n_components >= 2 else [0] * len(pca_results['original_pca']),
                'PC3': [x[2] for x in pca_results['original_pca']] if n_components >= 3 else [0] * len(pca_results['original_pca']),
                'Model': ['Original'] * len(pca_results['original_pca']),
                'Text': pca_results['labels']
            })
            
            df_fine_tuned = pd.DataFrame({
                'PC1': [x[0] for x in pca_results['fine_tuned_pca']],
                'PC2': [x[1] for x in pca_results['fine_tuned_pca']] if n_components >= 2 else [0] * len(pca_results['fine_tuned_pca']),
                'PC3': [x[2] for x in pca_results['fine_tuned_pca']] if n_components >= 3 else [0] * len(pca_results['fine_tuned_pca']),
                'Model': ['Fine-tuned'] * len(pca_results['fine_tuned_pca']),
                'Text': pca_results['labels']
            })
            
            df_combined = pd.concat([df_original, df_fine_tuned], ignore_index=True)
            
            # Create plot directory
            os.makedirs('visualizations', exist_ok=True)
            
            if n_components == 2:
                # 2D plot
                plt.figure(figsize=(12, 8))
                sns.scatterplot(data=df_combined, x='PC1', y='PC2', hue='Model', 
                               style='Model', palette=['blue', 'orange'], s=100)
                
                # Add arrows to show movement
                for i, text in enumerate(pca_results['labels']):
                    orig = pca_results['original_pca'][i]
                    fine = pca_results['fine_tuned_pca'][i]
                    plt.arrow(orig[0], orig[1], fine[0] - orig[0], fine[1] - orig[1], 
                             color='gray', alpha=0.3, head_width=0.5, width=0.1)
                    plt.text(orig[0], orig[1], str(i), fontsize=8, ha='right')
                    plt.text(fine[0], fine[1], str(i), fontsize=8, ha='left')
                
                plt.title('PCA Visualization: Original vs Fine-tuned Model Embeddings (2D)')
                plt.xlabel(f'Principal Component 1 ({pca_results["explained_variance_ratio"][0]*100:.1f}%)')
                plt.ylabel(f'Principal Component 2 ({pca_results["explained_variance_ratio"][1]*100:.1f}%)')
                plt.legend(title='Model Type')
                plt.grid(True, alpha=0.3)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'visualizations/pca_2d_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved 2D PCA visualization to {filename}")
                
                # Also create interactive plotly version
                self._create_interactive_pca_plot(pca_results, 2)
                
            elif n_components == 3:
                # 3D plot
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                orig_points = df_original[['PC1', 'PC2', 'PC3']].values
                fine_points = df_fine_tuned[['PC1', 'PC2', 'PC3']].values
                
                ax.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2], 
                          c='blue', label='Original', alpha=0.7, s=50)
                ax.scatter(fine_points[:, 0], fine_points[:, 1], fine_points[:, 2], 
                          c='orange', label='Fine-tuned', alpha=0.7, s=50)
                
                # Add connecting lines
                for i in range(len(orig_points)):
                    ax.plot([orig_points[i, 0], fine_points[i, 0]], 
                           [orig_points[i, 1], fine_points[i, 1]], 
                           [orig_points[i, 2], fine_points[i, 2]], 
                           c='gray', alpha=0.2, linestyle='--')
                
                ax.set_title('PCA Visualization: Original vs Fine-tuned Model Embeddings (3D)')
                ax.set_xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca_results["explained_variance_ratio"][1]*100:.1f}%)')
                ax.set_zlabel(f'PC3 ({pca_results["explained_variance_ratio"][2]*100:.1f}%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'visualizations/pca_3d_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved 3D PCA visualization to {filename}")
                
                # Also create interactive plotly version
                self._create_interactive_pca_plot(pca_results, 3)
                
        except Exception as e:
            logger.error(f"Error creating PCA plot: {e}")
            raise
    
    def _create_interactive_pca_plot(self, pca_results: Dict, n_components: int):
        """Create interactive Plotly visualization"""
        try:
            if n_components == 2:
                df = pd.DataFrame({
                    'PC1': [x[0] for x in pca_results['original_pca']] + [x[0] for x in pca_results['fine_tuned_pca']],
                    'PC2': [x[1] for x in pca_results['original_pca']] + [x[1] for x in pca_results['fine_tuned_pca']],
                    'Model': ['Original'] * len(pca_results['original_pca']) + ['Fine-tuned'] * len(pca_results['fine_tuned_pca']),
                    'Text': pca_results['labels'] * 2,
                    'Index': list(range(len(pca_results['labels']))) * 2
                })
                
                fig = px.scatter(df, x='PC1', y='PC2', color='Model', 
                                hover_data=['Text', 'Index'],
                                title='Interactive PCA: Original vs Fine-tuned Embeddings')
                
                # Add arrows
                for i in range(len(pca_results['labels'])):
                    fig.add_annotation(
                        x=pca_results['original_pca'][i][0],
                        y=pca_results['original_pca'][i][1],
                        xref="x", yref="y",
                        text=str(i),
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40
                    )
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'visualizations/pca_interactive_2d_{timestamp}.html'
                fig.write_html(filename)
                logger.info(f"Saved interactive 2D PCA visualization to {filename}")
                
            elif n_components == 3:
                df = pd.DataFrame({
                    'PC1': [x[0] for x in pca_results['original_pca']] + [x[0] for x in pca_results['fine_tuned_pca']],
                    'PC2': [x[1] for x in pca_results['original_pca']] + [x[1] for x in pca_results['fine_tuned_pca']],
                    'PC3': [x[2] for x in pca_results['original_pca']] + [x[2] for x in pca_results['fine_tuned_pca']],
                    'Model': ['Original'] * len(pca_results['original_pca']) + ['Fine-tuned'] * len(pca_results['fine_tuned_pca']),
                    'Text': pca_results['labels'] * 2,
                    'Index': list(range(len(pca_results['labels']))) * 2
                })
                
                fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Model',
                                   hover_data=['Text', 'Index'],
                                   title='Interactive 3D PCA: Original vs Fine-tuned Embeddings')
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'visualizations/pca_interactive_3d_{timestamp}.html'
                fig.write_html(filename)
                logger.info(f"Saved interactive 3D PCA visualization to {filename}")
                
        except Exception as e:
            logger.error(f"Error creating interactive plot: {e}")
            raise
    
    def tsne_visualization(self, texts: List[str], labels: Optional[List[str]] = None,
                          perplexity: int = 30, batch_size: int = 32) -> Dict:
        """
        Perform t-SNE visualization of embeddings
        
        Args:
            texts: List of text strings to visualize
            labels: Optional list of labels for each text
            perplexity: t-SNE perplexity parameter
            batch_size: Batch size for embedding generation
        
        Returns:
            Dictionary containing t-SNE results
        """
        original_embeddings, fine_tuned_embeddings = self.generate_embeddings(texts, batch_size)
        
        # Perform t-SNE on both sets of embeddings
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        
        original_tsne = tsne.fit_transform(original_embeddings)
        fine_tuned_tsne = tsne.fit_transform(fine_tuned_embeddings)
        
        # Store results
        tsne_results = {
            'original_tsne': original_tsne.tolist(),
            'fine_tuned_tsne': fine_tuned_tsne.tolist(),
            'texts': texts,
            'labels': labels if labels else [f"text_{i}" for i in range(len(texts))]
        }
        
        # Create visualization
        self._create_tsne_plot(tsne_results)
        
        logger.info("t-SNE visualization completed")
        
        return tsne_results
    
    def generate_comparison_report(self, sample_texts: List[str], 
                                  report_dir: str = 'comparison_reports') -> Dict:
        """
        Generate comprehensive comparison report
        
        Args:
            sample_texts: List of sample texts for embedding analysis
            report_dir: Directory to save the report
        
        Returns:
            Dictionary containing all comparison results
        """
        try:
            # Create report directory
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Perform all analyses
            logger.info("Starting comprehensive model comparison...")
            
            # 1. Weight analysis
            logger.info("Performing weight analysis...")
            weight_results = self.compare_weights()
            
            # 2. Embedding similarity analysis
            logger.info("Performing embedding similarity analysis...")
            similarity_results = self.analyze_embedding_similarity(sample_texts)
            
            # 3. PCA visualization (2D)
            logger.info("Performing PCA analysis (2D)...")
            pca_2d_results = self.pca_visualization(sample_texts, n_components=2)
            
            # Compile comprehensive report
            report = {
                'metadata': {
                    'timestamp': timestamp,
                    'original_model': self.original_model_path,
                    'fine_tuned_model': self.fine_tuned_model_path,
                    'sample_size': len(sample_texts),
                    'device': str(self.device)
                },
                'weight_analysis': weight_results,
                'embedding_similarity': similarity_results,
                'pca_2d': pca_2d_results,
            }
            
            # Save report as JSON
            report_filename = os.path.join(report_dir, f'comparison_report_{timestamp}.json')
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Comparison report saved to {report_filename}")
            
            # Save summary statistics
            summary_filename = os.path.join(report_dir, f'comparison_summary_{timestamp}.txt')
            self._save_summary_report(report, summary_filename)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            raise
    
    def _save_summary_report(self, report: Dict, filename: str):
        """Save a human-readable summary of the comparison results"""
        try:
            with open(filename, 'w') as f:
                f.write("="*80 + "\n")
                f.write("MODEL COMPARISON REPORT\n")
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
                if report['weight_analysis']:
                    total_layers = len(report['weight_analysis'])
                    mean_deltas = [v['mean_abs_delta'] for v in report['weight_analysis'].values()]
                    max_deltas = [v['max_abs_delta'] for v in report['weight_analysis'].values()]
                    
                    f.write(f"Total layers analyzed: {total_layers}\n")
                    f.write(f"Average mean absolute delta: {np.mean(mean_deltas):.6f}\n")
                    f.write(f"Maximum mean absolute delta: {np.max(mean_deltas):.6f}\n")
                    f.write(f"Average maximum delta: {np.mean(max_deltas):.6f}\n")
                    f.write(f"Maximum delta observed: {np.max(max_deltas):.6f}\n\n")
                
                # Embedding Similarity Summary
                f.write("EMBEDDING SIMILARITY SUMMARY:\n")
                sim = report['embedding_similarity']
                f.write(f"Mean Cosine Similarity: {sim['mean_cosine_similarity']:.4f}\n")
                f.write(f"Median Cosine Similarity: {sim['median_cosine_similarity']:.4f}\n")
                f.write(f"Std Cosine Similarity: {sim['std_cosine_similarity']:.4f}\n")
                f.write(f"Min Cosine Similarity: {sim['min_cosine_similarity']:.4f}\n")
                f.write(f"Max Cosine Similarity: {sim['max_cosine_similarity']:.4f}\n\n")
                
                # PCA Summary
                f.write("PCA ANALYSIS SUMMARY:\n")
                pca_2d = report['pca_2d']
                f.write(f"2D PCA Explained Variance: {pca_2d['explained_variance_ratio']}\n")
                f.write(f"Visualizations saved in 'visualizations/' directory\n\n")
                
                f.write("="*80 + "\n")
                f.write("ANALYSIS COMPLETE\n")
                f.write("="*80 + "\n")
                
            logger.info(f"Summary report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
            raise

def main():
    """Main function to run model comparison"""
    # Example usage - this will be replaced with actual model paths
    original_model = "sentence-transformers/all-mpnet-base-v2"  # Placeholder
    fine_tuned_model = "/path/to/fine/tuned/model"  # Placeholder - will be provided later
    
    # Sample texts for analysis
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
        "Natural language processing helps computers understand human language",
        "Deep learning models require large amounts of training data",
        "Transformers have revolutionized sequence-to-sequence tasks",
        "Embedding models convert text to dense vector representations",
        "Fine-tuning adapts pre-trained models to specific domains",
        "The university research project focuses on educational applications",
        "Students learn best through interactive and engaging content",
        "Technology enhances learning outcomes in modern education"
    ]
    
    try:
        # Initialize comparator
        comparator = ModelComparator(original_model, fine_tuned_model)
        
        # Load models
        comparator.load_models()
        
        # Generate comprehensive report
        report = comparator.generate_comparison_report(sample_texts)
        
        logger.info("Model comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()