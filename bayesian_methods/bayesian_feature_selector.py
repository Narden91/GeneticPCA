import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from rich.console import Console
import matplotlib.pyplot as plt
import networkx as nx
from rich.progress import Progress, SpinnerColumn, TextColumn
import warnings
from typing import List, Dict, Set, Tuple, Optional
import seaborn as sns
from scipy.stats import chi2_contingency
from itertools import combinations
import time

console = Console()

class OptimizedBayesianFeatureSelector:
    """
    Optimized Bayesian Feature Selector for high-dimensional data.
    Uses pre-filtering, correlation analysis, and faster structure learning.
    """
    
    def __init__(self, 
                 max_parents: int = 2,
                 scoring_method: str = "bic",
                 target_variable: Optional[str] = None,
                 max_features_for_bn: int = 50,
                 prefilter_method: str = "correlation",
                 prefilter_threshold: float = 0.1,
                 max_iter: int = 50,
                 random_state: int = 42):
        """
        Initialize the Optimized Bayesian Feature Selector.
        
        Parameters:
        -----------
        max_parents : int
            Maximum number of parents per node (reduced from 3 to 2 for speed)
        scoring_method : str
            Scoring method for structure learning ('bic' or 'k2')
        target_variable : str, optional
            If specified, focus feature selection around this target variable
        max_features_for_bn : int
            Maximum number of features to use for Bayesian Network (pre-filtering)
        prefilter_method : str
            Method for pre-filtering: 'correlation', 'mutual_info', 'variance'
        prefilter_threshold : float
            Threshold for pre-filtering (correlation/MI threshold or variance percentile)
        max_iter : int
            Maximum iterations for Hill Climbing (reduced from 200)
        random_state : int
            Random seed for reproducibility
        """
        self.max_parents = max_parents
        self.scoring_method = scoring_method.lower()
        self.target_variable = target_variable
        self.max_features_for_bn = max_features_for_bn
        self.prefilter_method = prefilter_method
        self.prefilter_threshold = prefilter_threshold
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
        self.discrete_data = None
        self.bins = {}
        self.selected_features = []
        self.markov_blankets = {}
        self.feature_importance_scores = {}
        self.prefiltered_features = []
        self.prefilter_scores = {}
        
    def _prefilter_features(self, X: pd.DataFrame) -> List[str]:
        """Pre-filter features to reduce dimensionality before BN learning"""
        console.print(f"[cyan]Pre-filtering {X.shape[1]} features to top {self.max_features_for_bn}...[/cyan]")
        
        if X.shape[1] <= self.max_features_for_bn:
            console.print(f"[green]No pre-filtering needed ({X.shape[1]} <= {self.max_features_for_bn})[/green]")
            return list(X.columns)
        
        feature_scores = {}
        
        if self.prefilter_method == "variance":
            # Use variance-based filtering
            variances = X.var()
            threshold = variances.quantile(1 - (self.max_features_for_bn / len(X.columns)))
            selected_features = variances[variances >= threshold].index.tolist()
            feature_scores = variances.to_dict()
            
        elif self.prefilter_method == "correlation":
            # Use maximum correlation with other features
            corr_matrix = X.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)  # Remove self-correlation
            
            max_correlations = corr_matrix.max(axis=1)
            feature_scores = max_correlations.to_dict()
            
            # Select top features by maximum correlation
            selected_features = max_correlations.nlargest(self.max_features_for_bn).index.tolist()
            
        elif self.prefilter_method == "pairwise_mi":
            # Use average mutual information with other features
            console.print("[cyan]Computing pairwise mutual information (this may take a moment)...[/cyan]")
            
            # Sample subset for faster computation if dataset is large
            if len(X) > 1000:
                sample_indices = np.random.choice(len(X), size=1000, replace=False)
                X_sample = X.iloc[sample_indices]
            else:
                X_sample = X
            
            mi_scores = {}
            features = list(X.columns)
            
            # Compute average MI for each feature with a sample of others
            n_comparisons = min(20, len(features))  # Limit comparisons for speed
            
            for i, feat1 in enumerate(features):
                if i % 50 == 0:
                    console.print(f"[cyan]Processing feature {i+1}/{len(features)}...[/cyan]")
                
                mi_values = []
                # Compare with random sample of other features
                other_features = [f for f in features if f != feat1]
                comparison_features = np.random.choice(other_features, 
                                                     size=min(n_comparisons, len(other_features)), 
                                                     replace=False)
                
                for feat2 in comparison_features:
                    try:
                        # Discretize for MI calculation
                        x1_disc = pd.cut(X_sample[feat1], bins=5, labels=False, duplicates='drop')
                        x2_disc = pd.cut(X_sample[feat2], bins=5, labels=False, duplicates='drop')
                        
                        # Remove NaN values
                        mask = ~(pd.isna(x1_disc) | pd.isna(x2_disc))
                        if mask.sum() < 10:  # Need enough data points
                            continue
                            
                        x1_clean = x1_disc[mask]
                        x2_clean = x2_disc[mask]
                        
                        # Create contingency table and compute MI
                        contingency = pd.crosstab(x1_clean, x2_clean)
                        if contingency.size > 1:
                            chi2, p_value, _, _ = chi2_contingency(contingency)
                            mi = chi2 / len(x1_clean)  # Normalized MI approximation
                            mi_values.append(mi)
                    except:
                        continue
                
                mi_scores[feat1] = np.mean(mi_values) if mi_values else 0
            
            feature_scores = mi_scores
            selected_features = sorted(mi_scores.keys(), key=lambda x: mi_scores[x], reverse=True)[:self.max_features_for_bn]
            
        else:
            # Fallback to variance
            console.print(f"[yellow]Unknown prefilter method '{self.prefilter_method}', using variance[/yellow]")
            variances = X.var()
            threshold = variances.quantile(1 - (self.max_features_for_bn / len(X.columns)))
            selected_features = variances[variances >= threshold].index.tolist()
            feature_scores = variances.to_dict()
        
        self.prefilter_scores = feature_scores
        console.print(f"[green]✓ Pre-filtered to {len(selected_features)} features using {self.prefilter_method}[/green]")
        
        return selected_features
    
    def _prepare_data_for_bn(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert continuous data to discrete for Bayesian Network learning (optimized)"""
        df_discrete = pd.DataFrame(index=X.index)
        self.bins = {}
        
        n_samples = len(X)
        skipped_cols = []
        categorical_cols = []
        discretized_cols = []
        
        for col in X.columns:
            unique_values = X[col].nunique()
            if unique_values <= 1:
                skipped_cols.append(col)
                continue
            
            # More aggressive binning for speed
            if unique_values <= 3:
                df_discrete[col] = X[col].astype('category')
                categorical_cols.append(col)
            else:
                # Reduced number of bins for faster processing
                n_bins = min(5, unique_values)  # Reduced from 8 to 5
                n_bins = max(3, n_bins)
                
                try:
                    # Use simpler equal-width binning by default for speed
                    df_discrete[col], self.bins[col] = pd.cut(
                        X[col], 
                        bins=n_bins, 
                        labels=[f"{col}_bin{i}" for i in range(n_bins)],
                        retbins=True,
                        include_lowest=True,
                        duplicates='drop'
                    )
                except:
                    # Fallback to quantile if cut fails
                    try:
                        df_discrete[col], self.bins[col] = pd.qcut(
                            X[col], 
                            q=n_bins, 
                            labels=[f"{col}_bin{i}" for i in range(n_bins)],
                            retbins=True, 
                            duplicates='drop'
                        )
                    except:
                        # Skip problematic columns
                        skipped_cols.append(col)
                        continue
                
                discretized_cols.append(col)
        
        # Summary output
        if skipped_cols:
            console.print(f"[yellow]Skipped {len(skipped_cols)} problematic columns[/yellow]")
        if categorical_cols:
            console.print(f"[cyan]Treated {len(categorical_cols)} columns as categorical[/cyan]")
        if discretized_cols:
            console.print(f"[cyan]Discretized {len(discretized_cols)} continuous columns[/cyan]")
        
        if df_discrete.empty:
            raise ValueError("No suitable columns for discretization")
        
        self.discrete_data = df_discrete
        return df_discrete
    
    def _learn_network_structure_fast(self, X_discrete: pd.DataFrame) -> BayesianNetwork:
        """Fast network structure learning with optimizations"""
        console.print(f"[cyan]Learning network structure ({X_discrete.shape[1]} features, max_iter={self.max_iter})...[/cyan]")
        
        model = None
        best_score = -np.inf
        
        # Try only the primary scoring method for speed
        scoring_methods = [(self.scoring_method.upper(), BicScore if self.scoring_method == "bic" else K2Score)]
        
        for method_name, scoring_class in scoring_methods:
            try:
                start_time = time.time()
                
                hc = HillClimbSearch(X_discrete)
                dag = hc.estimate(
                    scoring_method=scoring_class(X_discrete),
                    max_indegree=self.max_parents,
                    max_iter=self.max_iter,  # Reduced iterations
                    epsilon=1e-3,  # Less strict convergence
                    show_progress=True  # Show progress bar
                )
                
                elapsed = time.time() - start_time
                console.print(f"[green]{method_name} completed in {elapsed:.1f} seconds[/green]")
                
                # Convert DAG to BayesianNetwork
                model = BayesianNetwork(dag.edges())
                
                # Calculate score
                scorer = scoring_class(X_discrete)
                score = scorer.score(model)
                best_score = score
                
                console.print(f"[green]{method_name} network score: {score:.2f}[/green]")
                break  # Use first successful method
                
            except Exception as e:
                console.print(f"[yellow]{method_name} structure learning failed: {str(e)[:100]}[/yellow]")
        
        # If no model found, create a fast simple structure
        if model is None or not model.edges():
            console.print("[yellow]Creating fast tree structure...[/yellow]")
            model = self._create_fast_tree_structure(X_discrete)
        
        return model
    
    def _create_fast_tree_structure(self, X_discrete: pd.DataFrame) -> BayesianNetwork:
        """Create a fast, simple tree structure"""
        variables = list(X_discrete.columns)
        n_vars = len(variables)
        
        if n_vars < 2:
            return BayesianNetwork()
        
        edges = []
        
        # Simple star topology for speed
        if self.target_variable and self.target_variable in variables:
            root = self.target_variable
        else:
            # Use first variable as root
            root = variables[0]
        
        other_vars = [v for v in variables if v != root]
        
        # Connect root to limited number of children
        max_connections = min(self.max_parents * 3, len(other_vars))
        for var in other_vars[:max_connections]:
            edges.append((root, var))
        
        console.print(f"[green]Created fast tree structure with {len(edges)} edges[/green]")
        return BayesianNetwork(edges)
    
    def _calculate_markov_blankets_fast(self) -> Dict[str, Set[str]]:
        """Fast Markov Blanket calculation"""
        markov_blankets = {}
        
        for node in self.model.nodes():
            try:
                # Simple MB approximation for speed
                parents = set(self.model.get_parents(node))
                children = set(self.model.get_children(node))
                
                # Parents of children (spouses)
                spouses = set()
                for child in children:
                    spouses.update(self.model.get_parents(child))
                spouses.discard(node)
                
                mb = parents.union(children).union(spouses)
                markov_blankets[node] = mb
                
            except:
                markov_blankets[node] = set()
        
        return markov_blankets
    
    def _calculate_feature_importance_fast(self) -> Dict[str, float]:
        """Fast feature importance calculation"""
        importance_scores = {}
        
        for node in self.model.nodes():
            # Simplified importance based on connectivity and pre-filter scores
            degree = len(self.model.get_parents(node)) + len(self.model.get_children(node))
            mb_size = len(self.markov_blankets.get(node, set()))
            
            # Include pre-filter score if available
            prefilter_score = self.prefilter_scores.get(node, 0)
            prefilter_score_norm = prefilter_score / max(self.prefilter_scores.values()) if self.prefilter_scores else 0
            
            # Simplified importance calculation
            importance = (
                0.4 * (degree / max(1, len(self.model.nodes()))) +
                0.3 * (mb_size / max(1, len(self.model.nodes()))) +
                0.3 * prefilter_score_norm
            )
            
            importance_scores[node] = importance
        
        return importance_scores
    
    def fit(self, X: pd.DataFrame, target_variable: Optional[str] = None) -> 'OptimizedBayesianFeatureSelector':
        """Fit the optimized selector with speed improvements"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self.feature_names = list(X.columns)
        
        if target_variable is not None:
            self.target_variable = target_variable
        
        console.print(f"[cyan]Fitting Optimized Bayesian Feature Selector on {X.shape[1]} features...[/cyan]")
        
        start_time = time.time()
        
        # Step 1: Pre-filter features
        self.prefiltered_features = self._prefilter_features(X)
        X_filtered = X[self.prefiltered_features]
        
        # Step 2: Discretize filtered features
        console.print(f"[cyan]Discretizing {len(self.prefiltered_features)} pre-filtered features...[/cyan]")
        X_discrete = self._prepare_data_for_bn(X_filtered)
        
        # Step 3: Learn structure with optimizations
        self.model = self._learn_network_structure_fast(X_discrete)
        
        # Step 4: Skip parameter learning for speed (not needed for structure analysis)
        console.print(f"[cyan]Skipping parameter learning for speed optimization[/cyan]")
        
        # Step 5: Calculate Markov Blankets
        self.markov_blankets = self._calculate_markov_blankets_fast()
        
        # Step 6: Calculate feature importance
        self.feature_importance_scores = self._calculate_feature_importance_fast()
        
        elapsed = time.time() - start_time
        console.print(f"[green]✓ Optimized fitting completed in {elapsed:.1f} seconds[/green]")
        console.print(f"[green]✓ Network: {len(self.model.nodes())} nodes, {len(self.model.edges())} edges[/green]")
        
        return self
    
    def select_features_by_markov_blanket(self, 
                                        target_variable: Optional[str] = None,
                                        min_features: int = 10,
                                        max_features: Optional[int] = None) -> List[str]:
        """Select features with adjusted defaults for high-dimensional data"""
        if target_variable is None:
            target_variable = self.target_variable
        
        selected = set()
        
        # Strategy 1: Target variable's Markov Blanket
        if target_variable and target_variable in self.markov_blankets:
            mb = self.markov_blankets[target_variable]
            selected.update(mb)
            if mb:
                console.print(f"[cyan]Added {len(mb)} features from Markov Blanket of '{target_variable}'[/cyan]")
        
        # Strategy 2: Top features by importance
        features_by_importance = sorted(
            self.feature_importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add top features
        target_count = max_features or min(len(self.prefiltered_features), min_features * 2)
        for feature, importance in features_by_importance:
            if len(selected) >= target_count:
                break
            selected.add(feature)
        
        # Ensure minimum features
        if len(selected) < min_features:
            remaining = [f for f in self.prefiltered_features if f not in selected]
            additional_needed = min_features - len(selected)
            selected.update(remaining[:additional_needed])
        
        self.selected_features = list(selected)
        
        console.print(f"[green]✓ Selected {len(self.selected_features)} features from {len(self.prefiltered_features)} pre-filtered[/green]")
        return self.selected_features
    
    def get_feature_ranking(self) -> pd.DataFrame:
        """Get feature ranking with pre-filter information"""
        ranking_data = []
        
        # Include all original features, marking which were pre-filtered
        for feature in self.feature_names:
            was_prefiltered = feature in self.prefiltered_features
            mb_size = len(self.markov_blankets.get(feature, set())) if was_prefiltered else 0
            importance = self.feature_importance_scores.get(feature, 0) if was_prefiltered else 0
            prefilter_score = self.prefilter_scores.get(feature, 0)
            is_selected = feature in self.selected_features
            
            ranking_data.append({
                'feature': feature,
                'prefilter_score': prefilter_score,
                'was_prefiltered': was_prefiltered,
                'importance_score': importance,
                'markov_blanket_size': mb_size,
                'is_selected': is_selected
            })
        
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking = df_ranking.sort_values(['was_prefiltered', 'importance_score'], ascending=[False, False])
        
        return df_ranking
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features"""
        if not self.selected_features:
            console.print("[yellow]No features selected. Using top pre-filtered features.[/yellow]")
            features_to_use = self.prefiltered_features[:20]  # Fallback
        else:
            features_to_use = self.selected_features
        
        available_features = [f for f in features_to_use if f in X.columns]
        
        if not available_features:
            console.print("[yellow]No selected features found in data. Returning original data.[/yellow]")
            return X
        
        return X[available_features]
    
    def visualize_network(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Visualize the network (simplified for large networks)"""
        if self.model is None or not self.model.nodes():
            console.print("[yellow]No network to visualize[/yellow]")
            return
        
        console.print(f"[cyan]Visualizing network with {len(self.model.nodes())} nodes...[/cyan]")
        
        # For large networks, show only most important nodes
        if len(self.model.nodes()) > 30:
            top_features = sorted(self.feature_importance_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:30]
            nodes_to_show = [f[0] for f in top_features]
            
            # Create subgraph
            G_full = nx.DiGraph()
            G_full.add_edges_from(self.model.edges())
            G = G_full.subgraph(nodes_to_show).copy()
            
            console.print(f"[cyan]Showing top 30 nodes (out of {len(self.model.nodes())})[/cyan]")
        else:
            G = nx.DiGraph()
            G.add_edges_from(self.model.edges())
        
        plt.figure(figsize=figsize)
        
        # Use faster layout for large graphs
        if len(G.nodes()) > 20:
            pos = nx.spring_layout(G, k=1, iterations=20, seed=self.random_state)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=self.random_state)
        
        # Color nodes by selection status
        node_colors = ['lightgreen' if node in self.selected_features 
                      else 'lightblue' for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=15, width=1)
        
        # Add labels for important nodes only
        important_nodes = [n for n in G.nodes() if n in self.selected_features]
        labels = {n: n for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Optimized Bayesian Network\n({len(self.selected_features)} selected features)", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓ Network visualization saved to [bold]{save_path}[/bold][/green]")
        
        plt.show()
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     target_variable: Optional[str] = None,
                     min_features: int = 10,
                     max_features: Optional[int] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, target_variable)
        self.select_features_by_markov_blanket(target_variable, min_features, max_features)
        return self.transform(X)