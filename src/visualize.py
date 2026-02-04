"""
Visualization module for PIC Assembly-to-C Decompiler
Generates performance comparison plots and charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from pathlib import Path


class PerformanceVisualizer:
    """Create visualization plots for model performance"""
    
    def __init__(self, style: str = "whitegrid", figsize: tuple = (18, 10)):
        """
        Initialize visualizer
        
        Args:
            style: Seaborn style
            figsize: Default figure size
        """
        sns.set_style(style)
        self.figsize = figsize
        self.colors = {
            'base': '#e74c3c',
            'fine_tuned': '#27ae60',
            'improvement': '#3498db'
        }
    
    def plot_comparison(self,
                       base_metrics: Dict[str, List[float]],
                       ft_metrics: Dict[str, List[float]],
                       save_path: str = "performance_comparison.png",
                       dpi: int = 300):
        """
        Create comprehensive 6-subplot comparison visualization
        
        Args:
            base_metrics: Base model metrics dictionary
            ft_metrics: Fine-tuned model metrics dictionary
            save_path: Output file path
            dpi: Image resolution
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('PIC Assembly-to-C Decompiler: Performance Comparison\nBase Model vs Fine-Tuned Model',
                    fontsize=16, fontweight='bold', y=1.00)
        
        # Plot 1: Bar Chart - Average Metrics Comparison
        self._plot_bar_comparison(axes[0, 0], base_metrics, ft_metrics)
        
        # Plot 2: Improvement Heatmap
        self._plot_improvement_heatmap(axes[0, 1], base_metrics, ft_metrics)
        
        # Plot 3: Code Similarity Distribution
        self._plot_distribution(axes[0, 2], base_metrics, ft_metrics)
        
        # Plot 4: Per-Example Comparison
        self._plot_per_example(axes[1, 0], base_metrics, ft_metrics)
        
        # Plot 5: Token Accuracy Scatter
        self._plot_accuracy_scatter(axes[1, 1], base_metrics, ft_metrics)
        
        # Plot 6: Radar Chart
        self._plot_radar_chart(axes[1, 2], base_metrics, ft_metrics)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Performance plots saved to: {save_path}")
        plt.close()
    
    def _plot_bar_comparison(self, ax, base_metrics, ft_metrics):
        """Plot bar chart comparing average metrics"""
        metric_names = [name.replace('_', '\n').title() for name in base_metrics.keys()]
        base_avgs = [np.mean(base_metrics[key]) * 100 for key in base_metrics.keys()]
        ft_avgs = [np.mean(ft_metrics[key]) * 100 for key in ft_metrics.keys()]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_avgs, width, label='Base Model', 
                      color=self.colors['base'], alpha=0.8)
        bars2 = ax.bar(x + width/2, ft_avgs, width, label='Fine-Tuned', 
                      color=self.colors['fine_tuned'], alpha=0.8)
        
        ax.set_ylabel('Score (%)', fontweight='bold')
        ax.set_title('Average Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    def _plot_improvement_heatmap(self, ax, base_metrics, ft_metrics):
        """Plot heatmap showing improvements"""
        metric_names = [name.replace('_', '\n').title() for name in base_metrics.keys()]
        base_avgs = [np.mean(base_metrics[key]) * 100 for key in base_metrics.keys()]
        ft_avgs = [np.mean(ft_metrics[key]) * 100 for key in ft_metrics.keys()]
        
        improvements = [[ft_avgs[i] - base_avgs[i] for i in range(len(base_avgs))]]
        
        im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=50)
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, fontsize=9, rotation=0)
        ax.set_yticks([0])
        ax.set_yticklabels(['Improvement'])
        ax.set_title('Performance Improvement (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(metric_names)):
            ax.text(i, 0, f'{improvements[0][i]:+.1f}%',
                   ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Improvement (%)')
    
    def _plot_distribution(self, ax, base_metrics, ft_metrics):
        """Plot boxplot showing distribution"""
        ax.boxplot([base_metrics['code_similarity'], ft_metrics['code_similarity']],
                  labels=['Base Model', 'Fine-Tuned'],
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel('Similarity Score', fontweight='bold')
        ax.set_title('Code Similarity Distribution', fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_per_example(self, ax, base_metrics, ft_metrics):
        """Plot line chart showing per-example performance"""
        examples = list(range(1, len(base_metrics['code_similarity']) + 1))
        ax.plot(examples, [s * 100 for s in base_metrics['code_similarity']],
               marker='o', label='Base Model', linewidth=2, markersize=8, 
               color=self.colors['base'])
        ax.plot(examples, [s * 100 for s in ft_metrics['code_similarity']],
               marker='s', label='Fine-Tuned', linewidth=2, markersize=8, 
               color=self.colors['fine_tuned'])
        ax.set_xlabel('Example Number', fontweight='bold')
        ax.set_ylabel('Code Similarity (%)', fontweight='bold')
        ax.set_title('Per-Example Code Similarity', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(examples)
    
    def _plot_accuracy_scatter(self, ax, base_metrics, ft_metrics):
        """Plot scatter showing token accuracy correlation"""
        ax.scatter(base_metrics['token_accuracy'], ft_metrics['token_accuracy'],
                  s=200, alpha=0.6, c=range(len(base_metrics['token_accuracy'])), 
                  cmap='viridis')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equal Performance')
        ax.set_xlabel('Base Model Token Accuracy', fontweight='bold')
        ax.set_ylabel('Fine-Tuned Token Accuracy', fontweight='bold')
        ax.set_title('Token Accuracy: Base vs Fine-Tuned', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Annotate points
        for i, (x, y) in enumerate(zip(base_metrics['token_accuracy'], 
                                       ft_metrics['token_accuracy'])):
            ax.annotate(f'Ex{i+1}', (x, y), fontsize=8, ha='right')
    
    def _plot_radar_chart(self, ax, base_metrics, ft_metrics):
        """Plot radar chart showing overall performance profile"""
        categories = [name.replace('_', '\n').title() for name in base_metrics.keys()]
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        base_avgs = [np.mean(base_metrics[key]) * 100 for key in base_metrics.keys()]
        ft_avgs = [np.mean(ft_metrics[key]) * 100 for key in ft_metrics.keys()]
        
        base_values = base_avgs + [base_avgs[0]]
        ft_values = ft_avgs + [ft_avgs[0]]
        angles += angles[:1]
        
        # Convert to polar plot
        ax.remove()
        ax = plt.subplot(2, 3, 6, projection='polar')
        
        ax.plot(angles, base_values, 'o-', linewidth=2, label='Base Model', 
               color=self.colors['base'])
        ax.fill(angles, base_values, alpha=0.15, color=self.colors['base'])
        ax.plot(angles, ft_values, 'o-', linewidth=2, label='Fine-Tuned', 
               color=self.colors['fine_tuned'])
        ax.fill(angles, ft_values, alpha=0.15, color=self.colors['fine_tuned'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title('Overall Performance Profile', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    def plot_training_curves(self,
                            train_losses: List[float],
                            val_losses: List[float] = None,
                            save_path: str = "training_curves.png",
                            dpi: int = 300):
        """
        Plot training curves
        
        Args:
            train_losses: Training loss values
            val_losses: Optional validation loss values
            save_path: Output file path
            dpi: Image resolution
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = list(range(1, len(train_losses) + 1))
        ax.plot(steps, train_losses, label='Training Loss', 
               linewidth=2, color=self.colors['base'])
        
        if val_losses:
            val_steps = list(range(1, len(val_losses) + 1))
            ax.plot(val_steps, val_losses, label='Validation Loss', 
                   linewidth=2, color=self.colors['fine_tuned'])
        
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Training Progress', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Training curves saved to: {save_path}")
        plt.close()
    
    def plot_metric_breakdown(self,
                             metrics: Dict[str, List[float]],
                             model_name: str = "Model",
                             save_path: str = "metric_breakdown.png",
                             dpi: int = 300):
        """
        Plot detailed breakdown of each metric
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
            save_path: Output file path
            dpi: Image resolution
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'{model_name} - Metric Breakdown', 
                    fontsize=14, fontweight='bold')
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            
            # Convert to percentages
            values_pct = [v * 100 for v in values]
            
            # Plot histogram
            ax.hist(values_pct, bins=20, alpha=0.7, color=self.colors['fine_tuned'])
            ax.axvline(np.mean(values_pct), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(values_pct):.1f}%')
            
            ax.set_xlabel('Score (%)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Metric breakdown saved to: {save_path}")
        plt.close()


def visualize_from_file(comparison_file: str, 
                       output_dir: str = "visualizations"):
    """
    Create visualizations from saved comparison data
    
    Args:
        comparison_file: Path to comparison JSON file
        output_dir: Output directory for plots
    """
    import json
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = PerformanceVisualizer()
    
    # Extract metrics
    base_metrics = data.get('base_metrics', {})
    ft_metrics = data.get('ft_metrics', {})
    
    # Create comparison plot
    visualizer.plot_comparison(
        base_metrics, 
        ft_metrics,
        save_path=str(output_path / "performance_comparison.png")
    )
    
    # Create individual metric breakdowns
    visualizer.plot_metric_breakdown(
        ft_metrics,
        model_name="Fine-Tuned Model",
        save_path=str(output_path / "ft_metric_breakdown.png")
    )
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization module...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 5
    
    base_metrics = {
        'token_accuracy': list(np.random.uniform(0.5, 0.7, n_samples)),
        'code_similarity': list(np.random.uniform(0.6, 0.8, n_samples)),
        'function_accuracy': list(np.random.uniform(0.5, 0.75, n_samples)),
        'keyword_accuracy': list(np.random.uniform(0.6, 0.8, n_samples)),
    }
    
    ft_metrics = {
        'token_accuracy': list(np.random.uniform(0.7, 0.9, n_samples)),
        'code_similarity': list(np.random.uniform(0.75, 0.95, n_samples)),
        'function_accuracy': list(np.random.uniform(0.8, 1.0, n_samples)),
        'keyword_accuracy': list(np.random.uniform(0.8, 0.95, n_samples)),
    }
    
    visualizer = PerformanceVisualizer()
    visualizer.plot_comparison(base_metrics, ft_metrics, "test_comparison.png")
    
    print("\n✓ Visualization module ready")
