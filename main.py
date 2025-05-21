import time
import pandas as pd
import yaml
import os
import warnings
from rich.console import Console
from rich.panel import Panel
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
import matplotlib.pyplot as plt
from rich import print as rprint 

# Import modules
from data_loader import load_all_csvs_from_folder, load_specific_csv_from_folder
from preprocessor import preprocess_data_unsupervised
from clustering import apply_pca, perform_dbscan, evaluate_clustering
from genetic_algorithm import GeneticPCAOptimizer 

# Create a console instance
console = Console()

def plot_clusters_2d(data_for_plot: pd.DataFrame, labels: pd.Series, title: str, save_path: str = None):
    """
    Plots clusters in 2D using PCA for dimensionality reduction.
    Assumes data_for_plot is the data used for clustering (e.g., X_scaled or X_pca).
    """
    if data_for_plot is None or data_for_plot.empty or labels.empty:
        console.print("[yellow]⚠️ Plotting skipped: No data or labels to plot.[/yellow]")
        return

    # Reduce to 2D using PCA for visualization
    # This PCA is *only* for visualization if the original clustering was done on higher-D data
    n_plot_components = min(2, data_for_plot.shape[1])
    if n_plot_components < 2 and data_for_plot.shape[1] == 1: # If data is 1D, plot as is
        console.print("[cyan]Data is 1D. Plotting 1D scatter with jitter.[/cyan]")
        plt.figure(figsize=(10, 6))
        # Add jitter for better visibility of 1D points
        jitter = np.random.normal(0, 0.02, size=data_for_plot.shape[0])
        plt.scatter(data_for_plot.iloc[:, 0], jitter, c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.yticks([]) # No meaningful y-axis for jitter
        plt.xlabel(data_for_plot.columns[0])

    elif n_plot_components < 2 :
        console.print(f"[yellow]⚠️ Cannot create 2D plot. Data has {data_for_plot.shape[1]} features. Skipping plot.[/yellow]")
        return
    else:
        pca_plot = SklearnPCA(n_components=2, random_state=42)
        data_2d = pca_plot.fit_transform(data_for_plot)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.xlabel("Principal Component 1 (for visualization)")
        plt.ylabel("Principal Component 2 (for visualization)")
    
    plt.title(title)
    
    # Create a legend for clusters
    unique_labels = sorted(labels.unique())
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}' if l != -1 else 'Noise',
                          markerfacecolor=scatter.cmap(scatter.norm(l)), markersize=10) for l in unique_labels]
    plt.legend(handles=handles, title="Clusters")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]✓ Cluster plot saved to [bold]{save_path}[/bold][/green]")
    plt.show()


def main():
    console.print(Panel.fit("[bold blue]Starting Unsupervised Clustering Pipeline (PCA + DBSCAN)...[/bold blue]",
                           border_style="blue"))

    # ... (rest of your config loading and data loading code remains the same) ...
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        console.print(f"[green]✓ Configuration loaded from [bold]{config_path}[/bold][/green]")
    except FileNotFoundError:
        console.print(f"[bold red]ERROR:[/bold red] Configuration file '[bold]{config_path}[/bold]' not found.", style="red")
        exit(1)
    # ... (and so on, up to the point after clustering_metrics)

    verbose_level = config.get('settings', {}).get('verbose', 0)

    # 1. Load Data
    data_loader_config = config.get('data_loader', {})
    data_folder = data_loader_config.get('data_folder_path', 'data')
    specific_file = data_loader_config.get('specific_file')

    console.print(Panel(f"[yellow]📂 Loading data from: [bold]{data_folder}[/bold][/yellow]",
                       border_style="yellow"))

    dataframe = None
    if specific_file:
        console.print(f"[yellow]Specific file: [bold]{specific_file}[/bold][/yellow]")
        dataframe = load_specific_csv_from_folder(data_folder, specific_file)
    else:
        console.print(f"[yellow]Loading all CSV files from folder[/yellow]")
        dataframe = load_all_csvs_from_folder(data_folder)

    if dataframe is None or dataframe.empty:
        console.print(Panel(f"[bold red]Pipeline aborted: No data loaded from '{data_folder}'.[/bold red]",
                            border_style="red"))
        exit(1)

    console.print(f"[green]✓ Data loaded: [/green][cyan]{len(dataframe)} rows, {len(dataframe.columns)} columns[/cyan]")
    if verbose_level > 1:
        console.print("[cyan]Initial data head:[/cyan]")
        rprint(dataframe.head(3))

    # 2. Preprocess Data (for unsupervised)
    console.print(Panel("[bold magenta]--- Starting Preprocessing ---[/bold magenta]", border_style="magenta"))
    scaler_type = config.get('preprocessing', {}).get('scaler', 'StandardScaler')

    with console.status("[bold magenta]Preprocessing data...[/bold magenta]", spinner="dots"):
        X_scaled = preprocess_data_unsupervised(dataframe, scaler_type=scaler_type, verbose=verbose_level)

    if X_scaled is None or X_scaled.empty:
        console.print(Panel("[bold red]Pipeline aborted: Preprocessing failed or resulted in empty dataset.[/bold red]", border_style="red"))
        exit(1)
    console.print("[green]✓ Preprocessing completed.[/green]")
    if verbose_level > 1 and not X_scaled.empty:
        console.print("[cyan]Scaled data head:[/cyan]")
        rprint(X_scaled.head(3))


    # 3. PCA (Potentially optimized by GA)
    console.print(Panel("[bold yellow]--- PCA Optimization & Application ---[/bold yellow]", border_style="yellow"))
    run_ga_for_pca = config.get('settings', {}).get('run_ga_for_pca', False)
    pca_config = config.get('pca', {})

    X_for_clustering = X_scaled.copy() # Start with scaled data
    selected_n_components = 0 # Default to no PCA
    pca_applied_for_clustering = False # Flag to track if PCA was used *before* clustering

    if X_scaled.shape[1] <= 1:
        console.print("[yellow]PCA skipped: Data has 1 or fewer features after preprocessing.[/yellow]")
        run_ga_for_pca = False
    
    if run_ga_for_pca and X_scaled.shape[1] > 1:
        ga_config = config.get('genetic_pca_optimizer', {}).get('params', {})
        dbscan_params_for_ga = config.get('clustering', {}).get('dbscan', {})

        console.print("[cyan]Using Genetic Algorithm to optimize PCA components...[/cyan]")
        with console.status("[bold cyan]Running genetic algorithm for PCA...[/bold cyan]", spinner="dots"):
            ga_optimizer = GeneticPCAOptimizer(
                X_scaled=X_scaled,
                dbscan_eps=dbscan_params_for_ga.get('eps', 0.5),
                dbscan_min_samples=dbscan_params_for_ga.get('min_samples', 5),
                population_size=ga_config.get('population_size', 20),
                generations=ga_config.get('generations', 10),
                # ... (other GA params from your code)
                crossover_prob=ga_config.get('crossover_prob', 0.7),
                mutation_prob=ga_config.get('mutation_prob', 0.2),
                silhouette_weight=ga_config.get('silhouette_weight', 0.8),
                components_penalty_weight=ga_config.get('components_penalty_weight', 0.2),
                min_pca_components_ratio=ga_config.get('min_pca_components_ratio', 0.0),
                max_pca_components_ratio=ga_config.get('max_pca_components_ratio', 0.5),
                random_state=42
            )
            selected_n_components = ga_optimizer.run()
            console.print(f"[green]✓ GA selected [bold]{selected_n_components}[/bold] PCA components.[/green]")
        
        if verbose_level > 0 and ga_optimizer.generations > 1:
            ga_optimizer.plot_fitness_history(save_path="models/ga_pca_fitness_plot.png")
            
    elif X_scaled.shape[1] > 1 :
        manual_pca_components = pca_config.get('manual_pca_components', 0)
        if manual_pca_components > 0 :
            selected_n_components = manual_pca_components
            console.print(f"[cyan]Using manual PCA setting: {selected_n_components} components.[/cyan]")
        else:
            console.print("[cyan]No PCA will be applied (manual_pca_components is 0 or GA is off).[/cyan]")
            selected_n_components = 0
    
    if selected_n_components > 0 and X_scaled.shape[1] > 1:
        # apply_pca is from clustering_utils
        X_for_clustering, pca_model = apply_pca(X_scaled, n_components=selected_n_components, verbose=verbose_level)
        if X_for_clustering.empty or pca_model is None:
             console.print("[yellow]PCA application resulted in empty data or failed. Using scaled data without PCA.[/yellow]")
             X_for_clustering = X_scaled.copy()
        elif pca_model:
             console.print(f"[green]✓ PCA applied. Data shape for clustering: {X_for_clustering.shape}[/green]")
             pca_applied_for_clustering = True # PCA was used for clustering
    else:
        console.print("[cyan]Skipping PCA application step. Using preprocessed data directly for clustering.[/cyan]")
        X_for_clustering = X_scaled.copy()

    if X_for_clustering.empty:
        console.print(Panel("[bold red]Pipeline aborted: Data for clustering is empty after PCA step.[/bold red]", border_style="red"))
        exit(1)

    # 4. Perform DBSCAN Clustering
    console.print(Panel("[bold blue]--- DBSCAN Clustering ---[/bold blue]", border_style="blue"))
    dbscan_config = config.get('clustering', {}).get('dbscan', {})
    dbscan_eps = dbscan_config.get('eps', 0.5)
    dbscan_min_samples = dbscan_config.get('min_samples', 5)
    
    cluster_labels_array = perform_dbscan(X_for_clustering, eps=dbscan_eps, min_samples=dbscan_min_samples)
    cluster_labels = pd.Series(cluster_labels_array, index=X_for_clustering.index)


    if len(cluster_labels) == 0 :
         console.print(Panel("[bold red]Pipeline aborted: DBSCAN failed to produce labels.[/bold red]", border_style="red"))
         exit(1)
         
    # 5. Evaluate Clustering
    console.print(Panel("[bold green]--- Clustering Evaluation ---[/bold green]", border_style="green"))
    clustering_metrics = evaluate_clustering(X_for_clustering, cluster_labels.to_numpy(), verbose=verbose_level) # Pass numpy array
    
    rprint("\n[bold]Clustering Metrics:[/bold]")
    for metric, value in clustering_metrics.items():
        if isinstance(value, float):
            rprint(f"  {metric.replace('_', ' ').title()}: [cyan]{value:.4f}[/cyan]")
        else:
            rprint(f"  {metric.replace('_', ' ').title()}: [cyan]{value}[/cyan]")

    # 6. Plot Clusters (New Step)
    console.print(Panel("[bold yellow]--- Visualizing Clusters ---[/bold yellow]", border_style="yellow"))
    plot_title = f"DBSCAN Clusters (eps={dbscan_eps}, min_samples={dbscan_min_samples})"
    if pca_applied_for_clustering:
        plot_title += f"\nData Clustered on {X_for_clustering.shape[1]} PCA Components"
    else:
        plot_title += f"\nData Clustered on Original {X_for_clustering.shape[1]} Scaled Features"
    
    # X_for_clustering is the data that DBSCAN actually saw (either scaled or PCA-reduced)
    plot_clusters_2d(X_for_clustering, cluster_labels, 
                     title=plot_title, 
                     save_path="models/dbscan_cluster_plot.png")


    # Save results
    if len(dataframe.index.intersection(cluster_labels.index)) == len(cluster_labels):
        # Align original dataframe with results from scaled/clustered data
        # X_scaled.index contains the correct indices after potential NaN drops
        # cluster_labels should ideally share this index if X_for_clustering was derived from X_scaled
        
        output_df = dataframe.loc[cluster_labels.index].copy() # Use .loc with the index from cluster_labels
        output_df['cluster_label'] = cluster_labels.values # Assign values to avoid index issues
        output_file = "results_with_clusters.csv"
        try:
            output_df.to_csv(output_file, index=True) # Save original index if useful
            console.print(f"\n[green]✓ Results with cluster labels saved to [bold]{output_file}[/bold][/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not save results with cluster labels: {e}[/yellow]")
    else:
        console.print(f"[yellow]⚠️ Index mismatch: Could not reliably align original dataframe with cluster labels for saving.[/yellow]")
        console.print(f"   Original df index len: {len(dataframe.index)}, Unique: {dataframe.index.nunique()}")
        console.print(f"   Cluster labels index len: {len(cluster_labels.index)}, Unique: {cluster_labels.index.nunique()}")
        console.print(f"   Intersection len: {len(dataframe.index.intersection(cluster_labels.index))}")


    console.print(Panel("[bold blue]Unsupervised clustering pipeline completed.[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    warnings.filterwarnings("ignore", category=FutureWarning)
    if not os.path.exists('models'):
        os.makedirs('models')
        console.print("[cyan]Created 'models' directory.[/cyan]")
    
    if not os.path.exists('config.yaml'):
        console.print("[bold red]ERROR:[/bold red] Configuration file 'config.yaml' does not exist.", style="red")
        exit(1)
    
    main()