import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Tuple, Dict, List
import warnings

console = Console()

class DBSCANOptimizer:
    """
    Optimize DBSCAN parameters for better anomaly detection.
    Uses knee/elbow method for eps and validates with multiple metrics.
    """
    
    def __init__(self, 
                 target_anomaly_rate: float = 0.05,
                 min_clusters: int = 2,
                 max_clusters: int = 20,
                 random_state: int = 42):
        """
        Initialize DBSCAN optimizer.
        
        Parameters:
        -----------
        target_anomaly_rate : float
            Target percentage of anomalies to detect (0.05 = 5%)
        min_clusters : int
            Minimum number of clusters desired
        max_clusters : int
            Maximum number of clusters desired
        random_state : int
            Random seed for reproducibility
        """
        self.target_anomaly_rate = target_anomaly_rate
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.best_params = None
        self.optimization_results = []
        
    def find_optimal_eps(self, X: pd.DataFrame, k: int = 5) -> float:
        """
        Find optimal eps using k-distance graph and knee method.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data for clustering
        k : int
            Number of neighbors to consider (usually min_samples - 1)
            
        Returns:
        --------
        optimal_eps : float
            Suggested eps value
        """
        console.print(f"[cyan]Finding optimal eps using {k}-distance graph...[/cyan]")
        
        # Calculate k-distance for each point
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        
        # Sort distances to k-th neighbor
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # Find knee/elbow point
        # Use simple method: maximum curvature point
        x = np.arange(len(k_distances))
        y = k_distances
        
        # Calculate second derivative to find maximum curvature
        if len(y) > 10:
            # Smooth the curve first
            window_size = min(len(y) // 20, 50)
            if window_size > 3:
                smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='same')
            else:
                smoothed_y = y
            
            # Calculate differences
            first_diff = np.diff(smoothed_y)
            second_diff = np.diff(first_diff)
            
            # Find point of maximum change (knee)
            if len(second_diff) > 0:
                knee_idx = np.argmax(second_diff) + 1
                optimal_eps = k_distances[knee_idx]
            else:
                # Fallback: use median of upper quartile
                optimal_eps = np.percentile(k_distances, 75)
        else:
            # Very small dataset
            optimal_eps = np.median(k_distances)
        
        console.print(f"[green]✓ Suggested eps from k-distance analysis: {optimal_eps:.3f}[/green]")
        return optimal_eps
    
    def evaluate_dbscan_params(self, X: pd.DataFrame, eps: float, min_samples: int) -> Dict:
        """
        Evaluate DBSCAN with given parameters.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            anomaly_rate = n_noise / len(labels) if len(labels) > 0 else 0
            
            # Calculate silhouette score if possible
            silhouette = None
            if n_clusters > 1 and n_noise < len(labels):
                try:
                    # Only use non-noise points for silhouette calculation
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                except:
                    silhouette = None
            
            # Calculate cluster size statistics
            if n_clusters > 0:
                cluster_sizes = []
                for cluster_id in set(labels):
                    if cluster_id != -1:  # Exclude noise
                        cluster_sizes.append(np.sum(labels == cluster_id))
                
                avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
                min_cluster_size = np.min(cluster_sizes) if cluster_sizes else 0
                max_cluster_size = np.max(cluster_sizes) if cluster_sizes else 0
            else:
                avg_cluster_size = min_cluster_size = max_cluster_size = 0
            
            # Custom scoring function
            score = self._calculate_dbscan_score(
                n_clusters, anomaly_rate, silhouette, avg_cluster_size, len(X)
            )
            
            return {
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'anomaly_rate': anomaly_rate,
                'silhouette': silhouette,
                'avg_cluster_size': avg_cluster_size,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size,
                'score': score,
                'labels': labels
            }
            
        except Exception as e:
            return {
                'eps': eps,
                'min_samples': min_samples,
                'score': -1,
                'error': str(e)
            }
    
    def _calculate_dbscan_score(self, n_clusters: int, anomaly_rate: float, 
                               silhouette: float, avg_cluster_size: float, 
                               total_points: int) -> float:
        """
        Calculate custom score for DBSCAN parameter combination.
        Higher score is better.
        """
        score = 0
        
        # Penalty for too few or too many clusters
        if self.min_clusters <= n_clusters <= self.max_clusters:
            cluster_score = 1.0
        elif n_clusters < self.min_clusters:
            cluster_score = 0.5 * (n_clusters / self.min_clusters)
        else:  # n_clusters > self.max_clusters
            cluster_score = 0.5 * (self.max_clusters / n_clusters)
        
        score += 0.3 * cluster_score
        
        # Anomaly rate score (closer to target is better)
        anomaly_diff = abs(anomaly_rate - self.target_anomaly_rate)
        anomaly_score = max(0, 1 - (anomaly_diff / self.target_anomaly_rate))
        score += 0.4 * anomaly_score
        
        # Silhouette score (if available)
        if silhouette is not None:
            silhouette_score = max(0, silhouette)  # Silhouette can be negative
            score += 0.2 * silhouette_score
        
        # Cluster size balance (prefer reasonable cluster sizes)
        if avg_cluster_size > 0:
            ideal_cluster_size = total_points / max(1, n_clusters) * 0.8  # 80% of ideal
            size_ratio = min(avg_cluster_size / ideal_cluster_size, 
                           ideal_cluster_size / avg_cluster_size)
            score += 0.1 * size_ratio
        
        return score
    
    def optimize_parameters(self, X: pd.DataFrame, 
                          eps_range: Tuple[float, float] = None,
                          min_samples_range: Tuple[int, int] = None,
                          n_trials: int = 20) -> Dict:
        """
        Optimize DBSCAN parameters using grid search with intelligent sampling.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data for clustering
        eps_range : tuple, optional
            Range of eps values to try (min_eps, max_eps)
        min_samples_range : tuple, optional
            Range of min_samples to try (min_val, max_val)
        n_trials : int
            Number of parameter combinations to try
            
        Returns:
        --------
        best_params : dict
            Best parameters found
        """
        console.print(f"[cyan]Optimizing DBSCAN parameters for anomaly detection...[/cyan]")
        
        # Determine parameter ranges
        if eps_range is None:
            # Use k-distance method to suggest eps range
            suggested_eps = self.find_optimal_eps(X, k=5)
            eps_min = suggested_eps * 0.3
            eps_max = suggested_eps * 2.0
        else:
            eps_min, eps_max = eps_range
        
        if min_samples_range is None:
            # Rule of thumb: min_samples = 2 * dimensions, but cap it reasonably
            min_dim = max(2, min(X.shape[1], 10))
            max_dim = min(X.shape[1] * 2, 20)
            min_samples_range = (3, max_dim)
        
        min_samples_min, min_samples_max = min_samples_range
        
        console.print(f"[cyan]Parameter ranges: eps=[{eps_min:.3f}, {eps_max:.3f}], min_samples=[{min_samples_min}, {min_samples_max}][/cyan]")
        
        # Generate parameter combinations
        eps_values = np.linspace(eps_min, eps_max, max(10, n_trials // 3))
        min_samples_values = np.arange(min_samples_min, min_samples_max + 1)
        
        # Create parameter grid
        param_combinations = []
        for eps in eps_values:
            for min_samples in min_samples_values:
                param_combinations.append((eps, min_samples))
        
        # Limit to n_trials
        if len(param_combinations) > n_trials:
            # Sample intelligently: include boundary values and random middle values
            indices = np.linspace(0, len(param_combinations)-1, n_trials, dtype=int)
            param_combinations = [param_combinations[i] for i in indices]
        
        # Evaluate parameters
        self.optimization_results = []
        best_score = -1
        best_params = None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            
            task = progress.add_task(description="Evaluating parameter combinations...", total=len(param_combinations))
            
            for eps, min_samples in param_combinations:
                result = self.evaluate_dbscan_params(X, eps, int(min_samples))
                self.optimization_results.append(result)
                
                if result['score'] > best_score:
                    best_score = result['score']
                    best_params = result
                
                progress.advance(task)
        
        self.best_params = best_params
        
        console.print(f"[green]✓ Optimization completed. Best score: {best_score:.3f}[/green]")
        console.print(f"[green]✓ Best parameters: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}[/green]")
        console.print(f"[green]✓ Expected results: {best_params['n_clusters']} clusters, {best_params['anomaly_rate']*100:.1f}% anomalies[/green]")
        
        return best_params
    
    def plot_optimization_results(self, save_path: str = None):
        """Plot the optimization results to visualize parameter effects."""
        if not self.optimization_results:
            console.print("[yellow]No optimization results to plot[/yellow]")
            return
        
        results_df = pd.DataFrame(self.optimization_results)
        
        # Filter out failed results
        results_df = results_df[results_df['score'] >= 0]
        
        if results_df.empty:
            console.print("[yellow]No successful parameter combinations to plot[/yellow]")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Score vs eps
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(results_df['eps'], results_df['score'], 
                              c=results_df['min_samples'], cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Eps')
        ax1.set_ylabel('Score')
        ax1.set_title('DBSCAN Score vs Eps')
        plt.colorbar(scatter1, ax=ax1, label='Min Samples')
        
        # Mark best parameters
        if self.best_params:
            ax1.scatter(self.best_params['eps'], self.best_params['score'], 
                       color='red', s=100, marker='*', label='Best')
            ax1.legend()
        
        # 2. Number of clusters vs parameters
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(results_df['eps'], results_df['n_clusters'], 
                              c=results_df['min_samples'], cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Eps')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Number of Clusters vs Eps')
        plt.colorbar(scatter2, ax=ax2, label='Min Samples')
        
        # 3. Anomaly rate vs parameters
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(results_df['eps'], results_df['anomaly_rate'], 
                              c=results_df['min_samples'], cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Eps')
        ax3.set_ylabel('Anomaly Rate')
        ax3.set_title('Anomaly Rate vs Eps')
        ax3.axhline(y=self.target_anomaly_rate, color='red', linestyle='--', 
                   label=f'Target ({self.target_anomaly_rate*100:.1f}%)')
        ax3.legend()
        plt.colorbar(scatter3, ax=ax3, label='Min Samples')
        
        # 4. Parameter combinations heatmap
        ax4 = axes[1, 1]
        
        # Create grid for heatmap
        eps_unique = sorted(results_df['eps'].unique())
        min_samples_unique = sorted(results_df['min_samples'].unique())
        
        if len(eps_unique) > 1 and len(min_samples_unique) > 1:
            score_grid = np.full((len(min_samples_unique), len(eps_unique)), np.nan)
            
            for i, ms in enumerate(min_samples_unique):
                for j, eps in enumerate(eps_unique):
                    mask = (results_df['eps'] == eps) & (results_df['min_samples'] == ms)
                    if mask.any():
                        score_grid[i, j] = results_df[mask]['score'].iloc[0]
            
            im = ax4.imshow(score_grid, cmap='viridis', aspect='auto')
            ax4.set_xticks(range(len(eps_unique)))
            ax4.set_xticklabels([f'{eps:.2f}' for eps in eps_unique], rotation=45)
            ax4.set_yticks(range(len(min_samples_unique)))
            ax4.set_yticklabels(min_samples_unique)
            ax4.set_xlabel('Eps')
            ax4.set_ylabel('Min Samples')
            ax4.set_title('Parameter Combination Scores')
            plt.colorbar(im, ax=ax4, label='Score')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor heatmap', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Parameter Combination Scores')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓ Optimization results plot saved to [bold]{save_path}[/bold][/green]")
        
        plt.show()
    
    def get_alternative_parameters(self, top_n: int = 3) -> List[Dict]:
        """Get top N alternative parameter combinations."""
        if not self.optimization_results:
            return []
        
        # Sort by score and return top alternatives
        sorted_results = sorted(self.optimization_results, 
                              key=lambda x: x['score'], reverse=True)
        
        # Filter out failed results and return top N
        good_results = [r for r in sorted_results if r['score'] >= 0]
        return good_results[:top_n]