import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from rich.console import Console
import matplotlib.pyplot as plt
import networkx as nx
from rich.progress import Progress, SpinnerColumn, TextColumn
import warnings

console = Console()

class BayesianAnomalyDetector:
    """
    Anomaly detection using Bayesian networks or kernel density estimation
    depending on data dimensionality and complexity
    """
    
    def __init__(self, threshold_percentile=95, method="auto", contamination=0.05, random_state=42):
        """
        Initialize the Bayesian anomaly detector.
        
        Parameters:
        -----------
        threshold_percentile : int or float
            Percentile for anomaly threshold (default 95 means 5% anomalies)
        method : str
            Method to use: 'bayesian_network', 'kde', or 'auto' (default)
            'auto' will select the appropriate method based on data characteristics
        contamination : float
            Expected proportion of anomalies in the data (0.05 = 5%)
        random_state : int
            Random seed for reproducibility
        """
        self.threshold_percentile = threshold_percentile
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.inference = None
        self.kde = None
        self.selected_method = None
        self.feature_names = None
        self.thresholds = None
        self.discrete_data = None
        self.bins = {}
        
    def _prepare_data_for_bn(self, X):
        """Convert continuous data to discrete for Bayesian Network with better sensitivity"""
        # Convert data to discrete bins if it's continuous
        df_discrete = pd.DataFrame(index=X.index)
        self.bins = {}
        
        # Calculate optimal number of bins based on data characteristics
        n_samples = len(X)
        
        for col in X.columns:
            # Skip columns with too few unique values
            unique_values = X[col].nunique()
            if unique_values <= 1:
                console.print(f"[yellow]Skipping column '{col}' - only {unique_values} unique value(s)[/yellow]")
                continue
            
            # Adaptive binning based on data distribution
            if unique_values <= 5:
                # Already discrete-like
                df_discrete[col] = X[col].astype('category')
                console.print(f"[cyan]Column '{col}' treated as categorical ({unique_values} unique values)[/cyan]")
            else:
                # Use Sturges' rule for bin count, but cap it
                n_bins = min(int(np.ceil(np.log2(n_samples) + 1)), 10, unique_values)
                n_bins = max(3, n_bins)  # At least 3 bins
                
                try:
                    # Try quantile-based binning first for better distribution
                    df_discrete[col], self.bins[col] = pd.qcut(
                        X[col], 
                        q=n_bins, 
                        labels=[f"{col}_bin{i}" for i in range(n_bins)],
                        retbins=True, 
                        duplicates='drop'
                    )
                except ValueError:
                    # Fall back to equal-width binning
                    df_discrete[col], self.bins[col] = pd.cut(
                        X[col], 
                        bins=n_bins, 
                        labels=[f"{col}_bin{i}" for i in range(n_bins)],
                        retbins=True,
                        include_lowest=True
                    )
                
                console.print(f"[cyan]Column '{col}' discretized into {len(df_discrete[col].cat.categories)} bins[/cyan]")
        
        # Handle empty dataframe
        if df_discrete.empty:
            raise ValueError("No suitable columns for discretization")
        
        # Store for later use
        self.discrete_data = df_discrete
        return df_discrete
        
    def _create_initial_network_structure(self, X_discrete, max_parents=3):
        """Create an initial network structure using domain knowledge or heuristics"""
        variables = list(X_discrete.columns)
        n_vars = len(variables)
        
        if n_vars < 2:
            raise ValueError("Need at least 2 variables for Bayesian network")
        
        edges = []
        parent_count = {var: 0 for var in variables}
        
        # Strategy 1: Create a tree structure based on correlation
        if n_vars <= 10:
            # Calculate mutual information or correlation between variables
            correlation_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)
            
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        # Use cramers V for categorical association
                        contingency = pd.crosstab(X_discrete[var1], X_discrete[var2])
                        chi2 = np.sum((contingency - contingency.sum(1).values[:, None] * contingency.sum(0).values / contingency.sum().sum()) ** 2 / 
                                     (contingency.sum(1).values[:, None] * contingency.sum(0).values / contingency.sum().sum()))
                        correlation_matrix.loc[var1, var2] = chi2 / (len(X_discrete) * min(len(X_discrete[var1].cat.categories) - 1, 
                                                                                          len(X_discrete[var2].cat.categories) - 1))
            
            # Create maximum spanning tree with parent limit
            visited = set()
            edges_weights = []
            
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables[i+1:], i+1):
                    weight = correlation_matrix.loc[var1, var2]
                    if not np.isnan(weight):
                        edges_weights.append((weight, var1, var2))
            
            edges_weights.sort(reverse=True)
            
            # Modified Kruskal's algorithm with parent limit
            parent = {v: v for v in variables}
            
            def find(v):
                if parent[v] != v:
                    parent[v] = find(parent[v])
                return parent[v]
            
            def union(v1, v2):
                root1, root2 = find(v1), find(v2)
                if root1 != root2:
                    parent[root1] = root2
                    return True
                return False
            
            for weight, var1, var2 in edges_weights:
                # Check parent constraints before adding edge
                if parent_count[var2] < max_parents and union(var1, var2):
                    edges.append((var1, var2))
                    parent_count[var2] += 1
                    if len(edges) == n_vars - 1:
                        break
                elif parent_count[var1] < max_parents and union(var2, var1):
                    edges.append((var2, var1))
                    parent_count[var1] += 1
                    if len(edges) == n_vars - 1:
                        break
        
        # Strategy 2: For larger networks, use a simpler structure with parent limit
        else:
            # Create a layered structure
            layer_size = int(np.sqrt(n_vars))
            for i in range(n_vars):
                # Connect to up to max_parents previous nodes
                for j in range(max(0, i - max_parents), i):
                    if parent_count[variables[i]] < max_parents:
                        edges.append((variables[j], variables[i]))
                        parent_count[variables[i]] += 1
        
        return edges
    
    def fit(self, X):
        """
        Fit a Bayesian network or KDE to the data
        
        Parameters:
        -----------
        X : pandas DataFrame
            Training data (normal instances)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        self.feature_names = X.columns
        n_samples, n_features = X.shape
        
        # Automatically select method based on data characteristics
        if self.method == "auto":
            if n_features <= 15 and n_samples >= 100:
                self.selected_method = "bayesian_network"
            else:
                self.selected_method = "kde"
            console.print(f"[cyan]Auto-selected method: {self.selected_method}[/cyan]")
        else:
            self.selected_method = self.method
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            if self.selected_method == "bayesian_network":
                task = progress.add_task(description="Learning Bayesian network structure...", total=None)
                self._fit_bayesian_network(X)
                progress.remove_task(task)
            else:
                task = progress.add_task(description="Fitting kernel density estimation...", total=None)
                self._fit_kde(X)
                progress.remove_task(task)
                
        return self
        
    def _fit_bayesian_network(self, X):
        """Fit a Bayesian network to the data with improved structure learning"""
        try:
            # Prepare discrete data
            X_discrete = self._prepare_data_for_bn(X)
            
            console.print("[cyan]Learning Bayesian network structure...[/cyan]")
            
            # Try multiple scoring methods
            model = None
            best_score = -np.inf
            
            # Method 1: Hill Climbing with BIC score
            try:
                hc_bic = HillClimbSearch(X_discrete)
                # Estimate returns a DAG, we need to convert to BayesianNetwork
                dag_bic = hc_bic.estimate(
                    scoring_method=BicScore(X_discrete),
                    max_indegree=3,  # Maximum 3 parents per node
                    max_iter=100,    # Reduced iterations for faster convergence
                    epsilon=1e-4
                )
                
                # Convert DAG to BayesianNetwork
                model_bic = BayesianNetwork(dag_bic.edges())
                
                # Calculate BIC score
                bic = BicScore(X_discrete)
                score_bic = bic.score(model_bic)
                
                if score_bic > best_score:
                    best_score = score_bic
                    model = model_bic
                    console.print(f"[green]Hill Climbing with BIC found network with score: {score_bic:.2f}[/green]")
            except Exception as e:
                console.print(f"[yellow]Hill Climbing with BIC failed: {e}[/yellow]")
            
            # Method 2: Hill Climbing with K2 score
            try:
                hc_k2 = HillClimbSearch(X_discrete)
                # Estimate returns a DAG, we need to convert to BayesianNetwork
                dag_k2 = hc_k2.estimate(
                    scoring_method=K2Score(X_discrete),
                    max_indegree=3,  # Maximum 3 parents per node
                    max_iter=100     # Reduced iterations for faster convergence
                )
                
                # Convert DAG to BayesianNetwork
                model_k2 = BayesianNetwork(dag_k2.edges())
                
                # Calculate K2 score
                k2 = K2Score(X_discrete)
                score_k2 = k2.score(model_k2)
                
                if score_k2 > best_score:
                    best_score = score_k2
                    model = model_k2
                    console.print(f"[green]Hill Climbing with K2 found network with score: {score_k2:.2f}[/green]")
            except Exception as e:
                console.print(f"[yellow]Hill Climbing with K2 failed: {e}[/yellow]")
            
            # If no edges found or model is None, create initial structure
            if model is None or not model.edges():
                console.print("[yellow]No structure found by search algorithms. Creating initial structure...[/yellow]")
                initial_edges = self._create_initial_network_structure(X_discrete, max_parents=3)
                model = BayesianNetwork(initial_edges)
                console.print(f"[green]Created initial network with {len(initial_edges)} edges[/green]")
            
            self.model = model
            
            # Fit parameters (CPDs)
            console.print("[cyan]Learning conditional probability distributions...[/cyan]")
            # Use MaximumLikelihoodEstimator to fit the model
            mle = MaximumLikelihoodEstimator(self.model, X_discrete)
            
            # Estimate CPDs for all nodes
            for node in self.model.nodes():
                try:
                    cpd = mle.estimate_cpd(node)
                    if cpd is not None:
                        self.model.add_cpds(cpd)
                    else:
                        console.print(f"[yellow]Warning: Could not estimate CPD for node {node}[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Error estimating CPD for node {node}: {e}[/yellow]")
            
            # Check if the model is valid
            try:
                if self.model.check_model():
                    console.print("[green]✓ Model validation passed[/green]")
                else:
                    console.print("[yellow]Warning: Model validation failed, but continuing...[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not validate model: {e}[/yellow]")
            
            # Verify CPDs are properly set
            for node in self.model.nodes():
                cpd = self.model.get_cpds(node)
                if cpd is None:
                    console.print(f"[red]Warning: No CPD for node {node}[/red]")
            
            # Setup inference engine
            self.inference = VariableElimination(self.model)
            
            # Calculate log-likelihoods for threshold
            log_likelihoods = self._calculate_log_likelihood(X)
            self.threshold = np.percentile(log_likelihoods, 100 - self.threshold_percentile)
            
            console.print(f"[green]✓ Bayesian network trained with {len(self.model.edges())} edges[/green]")
            console.print(f"[green]✓ Anomaly threshold: {self.threshold:.4f}[/green]")
            
        except Exception as e:
            console.print(f"[bold red]Error fitting Bayesian network: {e}[/bold red]")
            raise
    
    def _fit_kde(self, X):
        """Fit a kernel density estimator to the data"""
        try:
            X_np = X.values
            
            # Calculate optimal bandwidth using Scott's rule
            n_samples, n_dims = X_np.shape
            bandwidth = n_samples ** (-1. / (n_dims + 4))
            
            # Create and fit KDE model
            try:
                self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            except TypeError:
                self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
                
            self.kde.fit(X_np)
            
            # Calculate log-density scores to determine threshold
            log_density = self.kde.score_samples(X_np)
            self.threshold = np.percentile(log_density, 100 - self.threshold_percentile)
            
            console.print(f"[green]✓ Kernel density model trained. Anomaly threshold: {self.threshold:.4f}[/green]")
            
        except Exception as e:
            console.print(f"[bold red]Error fitting KDE model: {e}[/bold red]")
            raise
    
    def visualize_network(self, save_path=None, figsize=(12, 8)):
        """
        Visualize the Bayesian network structure
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        figsize : tuple
            Figure size for the plot
        """
        if self.selected_method != "bayesian_network" or self.model is None:
            console.print("[yellow]No Bayesian network to visualize[/yellow]")
            return
        
        # Create a directed graph
        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())
        
        plt.figure(figsize=figsize)
        
        # Calculate layout
        if len(self.model.nodes()) <= 10:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, 
                              arrowstyle='->', width=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title("Bayesian Network Structure", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓ Network visualization saved to [bold]{save_path}[/bold][/green]")
        
        plt.show()
        
        # Print network statistics
        console.print(f"\n[bold]Network Statistics:[/bold]")
        console.print(f"  Nodes: {len(self.model.nodes())}")
        console.print(f"  Edges: {len(self.model.edges())}")
        console.print(f"  Average degree: {2 * len(self.model.edges()) / len(self.model.nodes()):.2f}")
        
        # Print Markov blanket for each node
        console.print(f"\n[bold]Markov Blankets:[/bold]")
        for node in self.model.nodes():
            mb = self.model.get_markov_blanket(node)
            console.print(f"  {node}: {list(mb)}")
    
    def _calculate_log_likelihood(self, X):
        """Calculate log-likelihood for data using Bayesian network"""
        if self.selected_method != "bayesian_network" or self.model is None:
            return None
            
        X_discrete = self._prepare_data_for_bn(X)
        log_likelihoods = []
        
        console.print("[cyan]Calculating log-likelihoods...[/cyan]")
        
        for idx, row in X_discrete.iterrows():
            try:
                log_prob = 0.0
                
                # Calculate joint log probability
                for node in self.model.nodes():
                    if pd.isna(row[node]):
                        continue
                    
                    # Get CPD for this node
                    cpd = self.model.get_cpds(node)
                    if cpd is None:
                        continue
                    
                    # Get parent values
                    parents = self.model.get_parents(node)
                    parent_values = []
                    
                    for parent in cpd.variables[1:]:  # Skip the node itself
                        if parent in row and not pd.isna(row[parent]):
                            parent_values.append(row[parent])
                        else:
                            parent_values.append(cpd.state_names[parent][0])  # Default value
                    
                    # Get probability
                    try:
                        prob = cpd.get_value(**{node: row[node], **dict(zip(cpd.variables[1:], parent_values))})
                        log_prob += np.log(max(prob, 1e-10))
                    except:
                        log_prob += np.log(0.01)  # Default low probability
                
                log_likelihoods.append(log_prob)
                
            except Exception:
                log_likelihoods.append(-100.0)  # Default low likelihood for errors
        
        return np.array(log_likelihoods)
    
    def predict_proba(self, X):
        """
        Calculate anomaly probability scores
        
        Parameters:
        -----------
        X : pandas DataFrame
            Data to score
            
        Returns:
        --------
        anomaly_scores : numpy array
            Anomaly scores where higher values indicate more anomalous points
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        if self.selected_method == "bayesian_network" and self.model is not None:
            # For Bayesian network, calculate negative log-likelihood
            log_likelihoods = self._calculate_log_likelihood(X)
            # Convert log-likelihood to anomaly score (higher = more anomalous)
            anomaly_scores = -log_likelihoods
        elif self.kde is not None:
            # For KDE, use negative log-density
            X_np = X.values
            anomaly_scores = -self.kde.score_samples(X_np)
        else:
            # Fallback
            anomaly_scores = np.random.rand(len(X))
        
        # Normalize scores to [0, 1]
        if len(anomaly_scores) > 0:
            min_score = np.nanmin(anomaly_scores)
            max_score = np.nanmax(anomaly_scores)
            if max_score > min_score:
                scaled_scores = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                scaled_scores = np.zeros_like(anomaly_scores)
            
            scaled_scores = np.nan_to_num(scaled_scores, nan=0.5)
            return scaled_scores
        else:
            return np.array([])
    
    def predict(self, X):
        """Predict if points are anomalies"""
        anomaly_scores = self.predict_proba(X)
        
        # Determine threshold based on scores
        threshold = np.percentile(anomaly_scores, self.threshold_percentile)
        labels = np.where(anomaly_scores > threshold, -1, 1)
        
        # Print detection summary
        anomaly_count = np.sum(labels == -1)
        console.print(f"[cyan]Bayesian detector found {anomaly_count} anomalies ({anomaly_count/len(labels)*100:.2f}%)[/cyan]")
        
        return labels
    
    def plot_scores(self, X, title="Anomaly Scores Distribution", save_path=None):
        """Plot the distribution of anomaly scores"""
        anomaly_scores = self.predict_proba(X)
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        
        # Create bins for histogram
        bins = np.linspace(0, 1, 50)
        
        # Plot histogram
        n, bins, patches = plt.hist(anomaly_scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Color anomalous bins differently
        threshold = np.percentile(anomaly_scores, self.threshold_percentile)
        for i, patch in enumerate(patches):
            if bins[i] >= threshold:
                patch.set_facecolor('salmon')
        
        plt.axvline(x=threshold, color='r', linestyle='--', linewidth=2,
                    label=f'{self.threshold_percentile}% Threshold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Anomaly Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with statistics
        anomaly_count = np.sum(predictions == -1)
        total_count = len(predictions)
        textstr = f'Total Points: {total_count}\nAnomalies: {anomaly_count} ({anomaly_count/total_count*100:.1f}%)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓ Anomaly score plot saved to [bold]{save_path}[/bold][/green]")
        
        plt.show()
        
        return anomaly_scores, predictions
    
    def compare_with_dbscan(self, X, dbscan_labels, save_path=None):
        """Compare Bayesian anomaly detection with DBSCAN results"""
        bayesian_labels = self.predict(X)
        bayesian_scores = self.predict_proba(X)
        
        # Convert to binary anomaly indicators
        dbscan_anomalies = (dbscan_labels == -1).astype(int)
        bayesian_anomalies = (bayesian_labels == -1).astype(int)
        
        # Calculate agreement metrics
        total_points = len(X)
        agreement_count = np.sum(dbscan_anomalies == bayesian_anomalies)
        agreement_percentage = (agreement_count / total_points) * 100
        
        # Confusion matrix elements
        both_anomaly = np.sum((dbscan_anomalies == 1) & (bayesian_anomalies == 1))
        only_dbscan = np.sum((dbscan_anomalies == 1) & (bayesian_anomalies == 0))
        only_bayesian = np.sum((dbscan_anomalies == 0) & (bayesian_anomalies == 1))
        both_normal = np.sum((dbscan_anomalies == 0) & (bayesian_anomalies == 0))
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: 2D visualization if possible
        if X.shape[1] >= 2:
            ax1 = axes[0]
            
            # Use first two dimensions
            x_col, y_col = 0, 1
            
            # Create color map
            colors = np.zeros(total_points, dtype=int)
            colors[(dbscan_anomalies == 0) & (bayesian_anomalies == 0)] = 0  # Both normal (green)
            colors[(dbscan_anomalies == 1) & (bayesian_anomalies == 1)] = 1  # Both anomaly (red)
            colors[(dbscan_anomalies == 1) & (bayesian_anomalies == 0)] = 2  # Only DBSCAN (blue)
            colors[(dbscan_anomalies == 0) & (bayesian_anomalies == 1)] = 3  # Only Bayesian (purple)
            
            scatter = ax1.scatter(X.iloc[:, x_col], X.iloc[:, y_col], c=colors, 
                                 cmap='viridis', alpha=0.7, s=50, edgecolors='none')
            
            ax1.set_xlabel(f"Feature {x_col}")
            ax1.set_ylabel(f"Feature {y_col}")
            ax1.set_title("DBSCAN vs Bayesian Anomaly Detection")
            ax1.grid(True, alpha=0.3)
            
            # Custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#440154', 
                      markersize=10, label='Both Normal'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#31688e', 
                      markersize=10, label='Both Anomaly'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#35b779', 
                      markersize=10, label='Only DBSCAN'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#fde725', 
                      markersize=10, label='Only Bayesian')
            ]
            ax1.legend(handles=legend_elements, loc='best')
        
        # Right plot: Confusion matrix style visualization
        ax2 = axes[1]
        
        # Create confusion matrix
        confusion_matrix = np.array([[both_normal, only_bayesian],
                                    [only_dbscan, both_anomaly]])
        
        im = ax2.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2)
        
        # Set ticks and labels
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Normal (Bayesian)', 'Anomaly (Bayesian)'])
        ax2.set_yticklabels(['Normal (DBSCAN)', 'Anomaly (DBSCAN)'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax2.text(j, i, f'{confusion_matrix[i, j]}\n({confusion_matrix[i, j]/total_points*100:.1f}%)',
                               ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max()/2 else "white")
        
        ax2.set_title("Method Agreement Matrix")
        
        # Add overall statistics
        fig.suptitle(f"Anomaly Detection Comparison - Agreement: {agreement_percentage:.1f}%", 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓ Comparison plot saved to [bold]{save_path}[/bold][/green]")
            
        plt.show()
        
        # Print detailed summary
        console.print("\n[bold]Anomaly Detection Method Comparison:[/bold]")
        console.print(f"Agreement between methods: [cyan]{agreement_percentage:.1f}%[/cyan]")
        console.print(f"Points identified as normal by both: [green]{both_normal} ({both_normal/total_points*100:.1f}%)[/green]")
        console.print(f"Points identified as anomalies by both: [red]{both_anomaly} ({both_anomaly/total_points*100:.1f}%)[/red]")
        console.print(f"Points identified as anomalies only by DBSCAN: [blue]{only_dbscan} ({only_dbscan/total_points*100:.1f}%)[/blue]")
        console.print(f"Points identified as anomalies only by Bayesian: [magenta]{only_bayesian} ({only_bayesian/total_points*100:.1f}%)[/magenta]")
        
        # Calculate and print additional metrics
        if both_anomaly + only_dbscan > 0:
            precision_vs_dbscan = both_anomaly / (both_anomaly + only_bayesian)
            recall_vs_dbscan = both_anomaly / (both_anomaly + only_dbscan)
            f1_vs_dbscan = 2 * (precision_vs_dbscan * recall_vs_dbscan) / (precision_vs_dbscan + recall_vs_dbscan) if (precision_vs_dbscan + recall_vs_dbscan) > 0 else 0
            
            console.print(f"\n[bold]Performance Metrics (treating DBSCAN as ground truth):[/bold]")
            console.print(f"Precision: [cyan]{precision_vs_dbscan:.3f}[/cyan]")
            console.print(f"Recall: [cyan]{recall_vs_dbscan:.3f}[/cyan]")
            console.print(f"F1-Score: [cyan]{f1_vs_dbscan:.3f}[/cyan]")
        
        return {
            'agreement_percentage': agreement_percentage,
            'both_normal': both_normal,
            'both_anomaly': both_anomaly,
            'only_dbscan': only_dbscan,
            'only_bayesian': only_bayesian
        }