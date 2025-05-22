import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from rich.console import Console
import matplotlib.pyplot as plt
from rich.progress import Progress, SpinnerColumn, TextColumn

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
        
    def _prepare_data_for_bn(self, X):
        """Convert continuous data to discrete for Bayesian Network with better sensitivity"""
        # Convert data to discrete bins if it's continuous
        df_discrete = pd.DataFrame(index=X.index)
        self.bins = {}
        
        # First, calculate how many bins to use based on data size
        n_samples = len(X)
        # Aim for approximately sqrt(n) bins but cap it between 3 and 8
        optimal_bins = max(3, min(8, int(np.sqrt(n_samples/10))))
        
        console.print(f"[cyan]Discretizing data into {optimal_bins} bins per feature...[/cyan]")
        
        for col in X.columns:
            # Skip columns with too few unique values
            unique_values = X[col].nunique()
            if unique_values <= 1:
                continue
                
            # For better anomaly detection, use more bins for features with more variation
            bins = min(optimal_bins, unique_values)
            
            # Create labels for the bins
            labels = [f"{col}_{i}" for i in range(bins)]
            
            try:
                # Use qcut for equal-frequency binning, handle potential errors
                df_discrete[col], self.bins[col] = pd.qcut(X[col], q=bins, labels=labels, 
                                                        retbins=True, duplicates='drop')
            except ValueError:
                # If qcut fails (e.g. too many identical values), use cut
                df_discrete[col], self.bins[col] = pd.cut(X[col], bins=bins, labels=labels,
                                                        retbins=True, duplicates='drop')
        
        # Handle empty dataframe (all columns had too few unique values)
        if df_discrete.empty:
            raise ValueError("No suitable columns for discretization")
            
        return df_discrete
        
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
        
        # Automatically select method based on data characteristics if method='auto'
        if self.method == "auto":
            if n_features <= 10 and n_samples >= 200:
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
                progress.add_task(description="Learning Bayesian network structure...", total=None)
                self._fit_bayesian_network(X)
            else:  # Default to KDE for high-dimensional data
                progress.add_task(description="Fitting kernel density estimation...", total=None)
                self._fit_kde(X)
                
        return self
        
    def _fit_bayesian_network(self, X):
        """Fit a Bayesian network to the data"""
        try:
            # For Bayesian Network, we need discrete data
            X_discrete = self._prepare_data_for_bn(X)
            
            # Learn network structure
            console.print("[cyan]Learning Bayesian network structure...[/cyan]")
            hc = HillClimbSearch(X_discrete)
            model = hc.estimate(scoring_method=BicScore(X_discrete))
            
            if not model.edges():
                console.print("[yellow]Warning: No edges found in Bayesian network, using naive Bayes structure[/yellow]")
                # Create a naive Bayes structure (first variable is "root")
                edges = []
                if len(X_discrete.columns) >= 2:
                    root = X_discrete.columns[0]
                    for col in X_discrete.columns[1:]:
                        edges.append((root, col))
                    model = BayesianNetwork(edges)
            
            self.model = model
            console.print(f"[green]✓ Bayesian network structure learned with {len(model.edges())} edges[/green]")
            
            # Correctly fit parameters (CPTs)
            console.print("[cyan]Estimating conditional probability tables...[/cyan]")
            estimator = MaximumLikelihoodEstimator(self.model, X_discrete)
            self.model.cpds = estimator.get_parameters()
            
            # Setup inference engine
            self.inference = VariableElimination(self.model)
            
            # Calculate log-likelihoods for training data to determine thresholds
            log_likelihoods = self._calculate_log_likelihood(X)
            self.threshold = np.percentile(log_likelihoods, 100 - self.threshold_percentile)
            
            console.print(f"[green]✓ Bayesian network model trained. Anomaly threshold: {self.threshold:.4f}[/green]")
            
        except Exception as e:
            console.print(f"[bold red]Error fitting Bayesian network: {e}[/bold red]")
            # Only fall back to KDE if auto-selection was chosen
            if self.method == "auto":
                console.print("[yellow]Falling back to kernel density estimation...[/yellow]")
                self.selected_method = "kde"
                self._fit_kde(X)
            else:
                # If specific method was requested, use simpler BN or report failure
                console.print("[yellow]Attempting with simpler Bayesian structure...[/yellow]")
                try:
                    self._fit_simple_bayesian_network(X)
                except Exception as e2:
                    console.print(f"[bold red]Error fitting simple Bayesian network: {e2}[/bold red]")
                    # Set fallback method
                    self.selected_method = "fallback"
                    self.model = None
    
    def _fit_simple_bayesian_network(self, X):
        """Fit a simpler Bayesian network when the hill-climbing approach fails"""
        # For Bayesian Network, we need discrete data
        X_discrete = self._prepare_data_for_bn(X)
        
        # Create a simple naive Bayes network
        if len(X_discrete.columns) < 2:
            raise ValueError("Need at least 2 columns for a Bayesian network")
            
        # Choose first column as root
        root = X_discrete.columns[0]
        edges = [(root, col) for col in X_discrete.columns[1:]]
        
        # Create the model
        self.model = BayesianNetwork(edges)
        console.print(f"[yellow]Created simple naive Bayes network with {len(edges)} edges[/yellow]")
        
        # Fit parameters
        estimator = MaximumLikelihoodEstimator(self.model, X_discrete)
        self.model.cpds = estimator.get_parameters()
        
        # Setup inference
        self.inference = VariableElimination(self.model)
        
        # Calculate log-likelihoods and threshold
        log_likelihoods = self._calculate_log_likelihood(X)
        self.threshold = np.percentile(log_likelihoods, 100 - self.threshold_percentile)
        
        console.print(f"[green]✓ Simple Bayesian network trained. Anomaly threshold: {self.threshold:.4f}[/green]")
    
    def _fit_kde(self, X):
        """Fit a kernel density estimator to the data"""
        try:
            # For KDE, use standardized continuous data
            X_np = X.values
            
            # Calculate optimal bandwidth using Scott's rule
            n_samples, n_dims = X_np.shape
            bandwidth = n_samples ** (-1. / (n_dims + 4))
            
            # Create and fit KDE model - Compatible with older scikit-learn versions
            try:
                # Try with random_state parameter (newer scikit-learn)
                self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, random_state=self.random_state)
            except TypeError:
                # Fall back to version without random_state (older scikit-learn)
                console.print("[yellow]Using older scikit-learn KDE implementation[/yellow]")
                self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
                
            self.kde.fit(X_np)
            
            # Calculate log-density scores to determine threshold
            log_density = self.kde.score_samples(X_np)
            self.threshold = np.percentile(log_density, 100 - self.threshold_percentile)
            
            console.print(f"[green]✓ Kernel density model trained. Anomaly threshold: {self.threshold:.4f}[/green]")
            
        except Exception as e:
            console.print(f"[bold red]Error fitting KDE model: {e}[/bold red]")
            # Return simple model as fallback
            self.kde = None
            self.selected_method = "fallback"
            return None
    
    def _calculate_log_likelihood(self, X):
        """Calculate log-likelihood for data using Bayesian network"""
        if self.selected_method != "bayesian_network" or self.model is None:
            return None
            
        X_discrete = self._prepare_data_for_bn(X)
        log_likelihoods = []
        
        # Instead of calculating per instance, calculate joint probabilities for all variables
        # This is more efficient and gives better results
        console.print("[cyan]Calculating log-likelihoods for all data points...[/cyan]")
        
        # Get all variables in the model
        all_vars = self.model.nodes()
        
        for _, instance in X_discrete.iterrows():
            # Filter valid evidence (non-missing values)
            evidence = {col: val for col, val in instance.items() if pd.notna(val)}
            if not evidence:
                log_likelihoods.append(np.nan)
                continue
                
            try:
                # Calculate joint log probability for this instance
                log_prob = 0.0
                
                # For each variable, calculate P(var | parents(var))
                for var in all_vars:
                    if var not in evidence:
                        continue
                        
                    # Get parents of this variable
                    parents = self.model.get_parents(var)
                    
                    # Get evidence for parents
                    parent_evidence = {p: evidence[p] for p in parents if p in evidence}
                    
                    # Get the CPD for this variable
                    cpd = self.model.get_cpds(var)
                    
                    # Calculate probability of this value given its parents
                    prob = 0.1  # Default fallback
                    
                    try:
                        # Get index of current value
                        var_val = evidence[var]
                        var_idx = cpd.state_names[var].index(var_val)
                        
                        # Get indices of parent values
                        if parent_evidence:
                            parent_idxs = [cpd.state_names[p].index(parent_evidence[p]) 
                                        for p in parents if p in parent_evidence]
                            # Get probability from CPD table using these indices
                            if len(parent_idxs) == len(parents):
                                prob_idx = tuple([var_idx] + parent_idxs)
                                prob = max(cpd.values[prob_idx], 0.01)  # Ensure non-zero prob
                            else:
                                # If missing some parents, use average probability
                                prob = np.mean(cpd.values[var_idx])
                        else:
                            # If no parents or missing parent values, use marginal
                            prob = np.mean(cpd.values[var_idx])
                            
                    except (ValueError, IndexError, KeyError):
                        # If lookup fails, use small probability
                        prob = 0.05
                    
                    # Add log probability
                    log_prob += np.log(max(prob, 1e-10))
                    
                log_likelihoods.append(log_prob)
                
            except Exception as e:
                # If inference fails for this instance, use a default low likelihood
                log_likelihoods.append(-100.0)
        
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
        try:
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
            elif self.selected_method == "fallback":
                # Simple fallback: use Euclidean distance from mean
                X_np = X.values
                mean = np.mean(X_np, axis=0)
                anomaly_scores = np.sqrt(np.sum((X_np - mean)**2, axis=1))
            else:
                console.print("[bold red]Error: No trained model available[/bold red]")
                # Return random scores as last resort
                anomaly_scores = np.random.rand(len(X))
            
            # Min-max scale scores to [0,1] for easier interpretation
            if len(anomaly_scores) > 0:  # Check to avoid empty array errors
                min_score = np.nanmin(anomaly_scores)
                max_score = np.nanmax(anomaly_scores)
                if max_score > min_score:  # Check to avoid division by zero
                    scaled_scores = (anomaly_scores - min_score) / (max_score - min_score)
                else:
                    scaled_scores = np.zeros_like(anomaly_scores)
                
                # Replace any NaNs with 0.5 (neutral anomaly score)
                scaled_scores = np.nan_to_num(scaled_scores, nan=0.5)
                return scaled_scores
            else:
                return np.array([])
        except Exception as e:
            console.print(f"[bold red]Error calculating anomaly scores: {e}[/bold red]")
            # Return neutral scores as fallback
            return np.ones(len(X)) * 0.5
    
    def predict(self, X):
        """Predict if points are anomalies"""
        try:
            anomaly_scores = self.predict_proba(X)
            
            if self.selected_method == "bayesian_network" and self.model is not None:
                # FIXED: For Bayesian network, higher score = more anomalous
                # The issue was comparing with -self.threshold instead of the actual threshold
                threshold = np.percentile(anomaly_scores, self.threshold_percentile)
                labels = np.where(anomaly_scores > threshold, -1, 1)
                
            elif self.kde is not None:
                # For KDE, higher score (lower density) = more anomalous
                threshold = np.percentile(anomaly_scores, self.threshold_percentile)
                labels = np.where(anomaly_scores > threshold, -1, 1)
                
            elif self.selected_method == "fallback":
                # Simple fallback method
                threshold = np.percentile(anomaly_scores, self.threshold_percentile)
                labels = np.where(anomaly_scores > threshold, -1, 1)
                
            else:
                # If no model is available, return all normal
                labels = np.ones(len(X))
                
            # Print how many anomalies were detected
            anomaly_count = np.sum(labels == -1)
            console.print(f"[cyan]Bayesian detector found {anomaly_count} anomalies ({anomaly_count/len(labels)*100:.2f}%)[/cyan]")
            
            return labels
        except Exception as e:
            console.print(f"[bold red]Error in anomaly prediction: {e}[/bold red]")
            # In case of any error, return all normal points
            return np.ones(len(X))
    
    def plot_scores(self, X, title="Anomaly Scores Distribution", save_path=None):
        """Plot the distribution of anomaly scores"""
        anomaly_scores = self.predict_proba(X)
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_scores, bins=50, alpha=0.7, color='skyblue')
        plt.axvline(x=np.percentile(anomaly_scores, self.threshold_percentile), 
                    color='r', linestyle='--', label=f'{self.threshold_percentile}% Threshold')
        plt.title(title)
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓ Anomaly score plot saved to [bold]{save_path}[/bold][/green]")
        
        plt.tight_layout()
        plt.show()
        
        return anomaly_scores, predictions
    
    def compare_with_dbscan(self, X, dbscan_labels, save_path=None):
        """Compare Bayesian anomaly detection with DBSCAN results"""
        bayesian_labels = self.predict(X)
        bayesian_scores = self.predict_proba(X)
        
        # Convert to binary anomaly indicators (1 for anomaly, 0 for normal)
        dbscan_anomalies = (dbscan_labels == -1).astype(int)
        bayesian_anomalies = (bayesian_labels == -1).astype(int)
        
        # Calculate agreement metrics
        total_points = len(X)
        agreement_count = np.sum(dbscan_anomalies == bayesian_anomalies)
        agreement_percentage = (agreement_count / total_points) * 100
        
        # Points identified as anomalies by both methods
        both_anomaly = np.sum((dbscan_anomalies == 1) & (bayesian_anomalies == 1))
        
        # Points identified as anomalies by only one method
        only_dbscan = np.sum((dbscan_anomalies == 1) & (bayesian_anomalies == 0))
        only_bayesian = np.sum((dbscan_anomalies == 0) & (bayesian_anomalies == 1))
        
        # Points identified as normal by both methods
        both_normal = np.sum((dbscan_anomalies == 0) & (bayesian_anomalies == 0))
        
        # Create a plot comparing the methods
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of points, colored by agreement/disagreement
        if X.shape[1] >= 2:  # If data has at least 2 dimensions
            # Use the first two dimensions for visualization
            x_col, y_col = 0, 1
            
            # Create a categorical color map for the four scenarios
            colors = np.zeros(total_points, dtype=int)
            colors[(dbscan_anomalies == 0) & (bayesian_anomalies == 0)] = 0  # Both normal (green)
            colors[(dbscan_anomalies == 1) & (bayesian_anomalies == 1)] = 1  # Both anomaly (red)
            colors[(dbscan_anomalies == 1) & (bayesian_anomalies == 0)] = 2  # Only DBSCAN (blue)
            colors[(dbscan_anomalies == 0) & (bayesian_anomalies == 1)] = 3  # Only Bayesian (purple)
            
            plt.scatter(X.iloc[:, x_col], X.iloc[:, y_col], c=colors, cmap='viridis', 
                       alpha=0.7, s=50, edgecolors='none')
            
            plt.xlabel(f"Feature {x_col}")
            plt.ylabel(f"Feature {y_col}")
            plt.title("Comparison of DBSCAN and Bayesian Anomaly Detection")
            
            # Custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Both Normal'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Both Anomaly'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Only DBSCAN'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                      markersize=10, label='Only Bayesian')
            ]
            plt.legend(handles=legend_elements, loc='best')
            
        # Add a text box with statistics
        stats_text = (
            f"Total Points: {total_points}\n"
            f"Agreement: {agreement_percentage:.1f}%\n"
            f"Both Normal: {both_normal} ({both_normal/total_points*100:.1f}%)\n"
            f"Both Anomaly: {both_anomaly} ({both_anomaly/total_points*100:.1f}%)\n"
            f"Only DBSCAN: {only_dbscan} ({only_dbscan/total_points*100:.1f}%)\n"
            f"Only Bayesian: {only_bayesian} ({only_bayesian/total_points*100:.1f}%)"
        )
        
        plt.figtext(0.92, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.8), 
                  fontsize=10, ha='right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]✓ Comparison plot saved to [bold]{save_path}[/bold][/green]")
            
        plt.show()
        
        # Print summary
        console.print("\n[bold]Anomaly Detection Method Comparison:[/bold]")
        console.print(f"Agreement between methods: [cyan]{agreement_percentage:.1f}%[/cyan]")
        console.print(f"Points identified as normal by both: [green]{both_normal} ({both_normal/total_points*100:.1f}%)[/green]")
        console.print(f"Points identified as anomalies by both: [red]{both_anomaly} ({both_anomaly/total_points*100:.1f}%)[/red]")
        console.print(f"Points identified as anomalies only by DBSCAN: [blue]{only_dbscan} ({only_dbscan/total_points*100:.1f}%)[/blue]")
        console.print(f"Points identified as anomalies only by Bayesian: [magenta]{only_bayesian} ({only_bayesian/total_points*100:.1f}%)[/magenta]")
        
        return {
            'agreement_percentage': agreement_percentage,
            'both_normal': both_normal,
            'both_anomaly': both_anomaly,
            'only_dbscan': only_dbscan,
            'only_bayesian': only_bayesian
        }