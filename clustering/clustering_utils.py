import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from rich.console import Console

console = Console()

def apply_pca(data: pd.DataFrame, n_components=None, random_state=42, verbose: int = 0):
    """
    Applies PCA to the data.
    n_components:
        - int: number of components to keep.
        - float (0 to 1): percentage of variance to keep.
        - None: keeps all components (useful for checking variance ratio).
    Returns PCA-transformed data as a DataFrame, and the PCA object.
    """
    if data is None or data.empty:
        console.print("[yellow]⚠️ PCA skipped: Input data is None or empty.[/yellow]")
        return pd.DataFrame(), None

    if n_components is None or n_components <= 0: # 0 or None means no PCA from config
        console.print("[cyan]PCA not applied (n_components is 0 or None).[/cyan]")
        return data, None

    original_features = data.shape[1]
    if isinstance(n_components, float) and 0 < n_components < 1:
        console.print(f"[cyan]Applying PCA to retain {n_components*100:.2f}% of variance...[/cyan]")
        pca = PCA(n_components=n_components, random_state=random_state)
    elif isinstance(n_components, int) and n_components > 0:
        if n_components >= original_features:
            console.print(f"[yellow]⚠️ Requested n_components ({n_components}) >= original features ({original_features}). PCA will not reduce dimensionality significantly. Applying with n_components={original_features-1 if original_features >1 else 1}.[/yellow]")
            n_components = max(1, original_features -1 if original_features > 1 else 1) # Ensure at least 1 component and reduction
        console.print(f"[cyan]Applying PCA with n_components = {n_components}...[/cyan]")
        pca = PCA(n_components=n_components, random_state=random_state)
    else:
        console.print(f"[yellow]⚠️ Invalid n_components for PCA: {n_components}. PCA not applied.[/yellow]")
        return data, None

    try:
        data_pca_array = pca.fit_transform(data)
        # Create column names for PCA components
        pca_cols = [f"PC{i+1}" for i in range(data_pca_array.shape[1])]
        data_pca_df = pd.DataFrame(data_pca_array, columns=pca_cols, index=data.index)
        
        if verbose > 0:
            console.print(f"[green]✓ PCA applied. Original features: {original_features}, PCA components: {pca.n_components_}[/green]")
            explained_variance_ratio = pca.explained_variance_ratio_
            console.print(f"[cyan]Explained variance per component: {explained_variance_ratio}[/cyan]")
            console.print(f"[cyan]Total explained variance: {np.sum(explained_variance_ratio)*100:.2f}%[/cyan]")
        return data_pca_df, pca
    except Exception as e:
        console.print(f"[bold red]ERROR applying PCA: {e}[/bold red]")
        return data, None


def perform_dbscan(data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
    """
    Performs DBSCAN clustering.
    Returns cluster labels.
    """
    if data is None or data.empty:
        console.print("[yellow]⚠️ DBSCAN skipped: Input data is None or empty.[/yellow]")
        return np.array([])
        
    if data.shape[0] < min_samples :
        console.print(f"[bold red]ERROR: DBSCAN min_samples ({min_samples}) is greater than the number of data points ({data.shape[0]}). Cannot perform DBSCAN.[/bold red]")
        return np.array([-1] * data.shape[0]) # All noise

    console.print(f"[cyan]Performing DBSCAN with eps={eps}, min_samples={min_samples}...[/cyan]")
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        console.print(f"[green]✓ DBSCAN completed. Found {n_clusters} clusters and {n_noise} noise points.[/green]")
        return labels
    except Exception as e:
        console.print(f"[bold red]ERROR during DBSCAN: {e}[/bold red]")
        return np.array([-1] * data.shape[0]) # All noise


def evaluate_clustering(data: pd.DataFrame, labels: np.ndarray, verbose: int = 0):
    """
    Evaluates clustering performance.
    Returns a dictionary of metrics.
    """
    results = {}
    if data is None or data.empty or labels is None or len(labels) == 0 or len(labels) != data.shape[0]:
        console.print("[yellow]⚠️ Clustering evaluation skipped: Invalid data or labels.[/yellow]")
        return results

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    results['n_clusters'] = n_clusters
    results['n_noise_points'] = np.sum(labels == -1)
    results['percentage_noise'] = (results['n_noise_points'] / len(labels)) * 100 if len(labels) > 0 else 0

    if n_clusters < 2: # Silhouette and Davies-Bouldin require at least 2 clusters
        console.print(f"[yellow]Found {n_clusters} cluster(s). Silhouette Score and Davies-Bouldin Index require at least 2 clusters (excluding noise).[/yellow]")
        results['silhouette_score'] = None
        results['davies_bouldin_score'] = None
    else:
        try:
            silhouette = silhouette_score(data, labels)
            results['silhouette_score'] = silhouette
            if verbose > 0:
                console.print(f"Silhouette Score: [bold cyan]{silhouette:.4f}[/bold cyan]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not calculate Silhouette Score: {e}[/yellow]")
            results['silhouette_score'] = None
        
        try:
            db_score = davies_bouldin_score(data, labels)
            results['davies_bouldin_score'] = db_score
            if verbose > 0:
                console.print(f"Davies-Bouldin Score: [bold cyan]{db_score:.4f}[/bold cyan] (lower is better)")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not calculate Davies-Bouldin Score: {e}[/yellow]")
            results['davies_bouldin_score'] = None
            
    if verbose > 0:
        console.print(f"Number of clusters found: [bold cyan]{results['n_clusters']}[/bold cyan]")
        console.print(f"Number of noise points: [bold cyan]{results['n_noise_points']} ({results['percentage_noise']:.2f}%)[/bold cyan]")
    return results