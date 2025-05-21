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
from data_loader import load_all_csvs_from_folder, load_specific_csv_from_folder
from preprocessor import preprocess_data_unsupervised
from clustering import apply_pca, perform_dbscan, evaluate_clustering
from genetic_algorithm import GeneticPCAOptimizer 

console = Console()


def plot_clusters_2d(data_for_plot: pd.DataFrame, labels: pd.Series, title: str, save_path: str = None, highlight_anomalies=True):
    """
    Plots clusters in 2D using PCA for dimensionality reduction.
    Highlights anomaly points (label -1) if requested.
    """
    if data_for_plot is None or data_for_plot.empty or labels.empty:
        console.print("[yellow]⚠️ Plotting skipped: No data or labels to plot.[/yellow]")
        return

    n_plot_components = min(2, data_for_plot.shape[1])
    if n_plot_components < 2 and data_for_plot.shape[1] == 1:
        console.print("[cyan]Data is 1D. Plotting 1D scatter with jitter.[/cyan]")
        plt.figure(figsize=(10, 6))
        jitter = np.random.normal(0, 0.02, size=data_for_plot.shape[0])
        normal_mask = labels != -1
        anomaly_mask = labels == -1

        plt.scatter(data_for_plot.iloc[normal_mask.values, 0], jitter[normal_mask.values],
                    c=labels[normal_mask], cmap='viridis', s=50, alpha=0.7, label='Normal Clusters')
        if highlight_anomalies and anomaly_mask.any():
            plt.scatter(data_for_plot.iloc[anomaly_mask.values, 0], jitter[anomaly_mask.values],
                        c='red', marker='x', s=100, label='Anomalies (Noise)') # Removed edgecolors

        plt.yticks([])
        plt.xlabel(data_for_plot.columns[0])

    elif n_plot_components < 2:
        console.print(f"[yellow]⚠️ Cannot create 2D plot. Data has {data_for_plot.shape[1]} features. Skipping plot.[/yellow]")
        return
    else:
        pca_plot = SklearnPCA(n_components=2, random_state=42)
        data_2d = pca_plot.fit_transform(data_for_plot)

        plt.figure(figsize=(12, 8))
        normal_mask = labels != -1
        anomaly_mask = labels == -1

        scatter = plt.scatter(data_2d[normal_mask.values, 0], data_2d[normal_mask.values, 1],
                              c=labels[normal_mask], cmap='viridis', s=50, alpha=0.6, label='Normal Clusters')

        if highlight_anomalies and anomaly_mask.any():
            plt.scatter(data_2d[anomaly_mask.values, 0], data_2d[anomaly_mask.values, 1],
                        c='red', marker='x', s=100, label='Anomalies (Noise)') # Removed edgecolors and linewidths for 'x'

        plt.xlabel("Principal Component 1 (for visualization)")
        plt.ylabel("Principal Component 2 (for visualization)")

    plt.title(title)

    handles = []
    # Use a mask to get unique labels from normal points for colormap consistency
    unique_normal_labels = sorted(labels[normal_mask].unique())

    # Create legend entries for normal clusters
    for l in unique_normal_labels:
        if l != -1: # Should always be true due to normal_mask, but good check
            # Get color from the scatter object's colormap and normalization
            color_val = scatter.cmap(scatter.norm(l))
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}',
                                      markerfacecolor=color_val, markersize=10))

    # Create legend entry for anomalies if they exist and are highlighted
    if highlight_anomalies and anomaly_mask.any():
        handles.append(plt.Line2D([0], [0], marker='x', color='w', label='Anomalies (Noise)',
                                  markerfacecolor='red', markersize=10)) # For 'x', markerfacecolor sets the line color

    plt.legend(handles=handles, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]✓ Cluster (Anomaly) plot saved to [bold]{save_path}[/bold][/green]")
    plt.show()


def evaluate_combined_score(silhouette, davies_bouldin, formula="S - DB"):
    """
    Evaluates a combined score from Silhouette and Davies-Bouldin.
    Higher is better.
    """
    S = silhouette if silhouette is not None else -1.0 # Penalize if no silhouette
    DB = davies_bouldin if davies_bouldin is not None else float('inf') # Penalize if no DB

    # Ensure DB is not zero for division, and handle cases where it might be very large
    if "DB" in formula and DB == float('inf'): # If DB is inf, some formulas will be problematic
        return -float('inf') # Very bad score

    if formula == "S - DB":
        return S - (DB / 10 if DB > 1 else DB) # Try to scale DB down if it's large to not overly dominate S
    elif formula == "S / (1 + DB)":
        if DB == float('inf'): return -1.0 # Silhouette / infinity -> 0, but -1 is worse for no clusters
        return S / (1 + max(0, DB)) # Ensure 1+DB is not zero
    # Add more complex formulas with normalization if needed
    else: # Default to Silhouette if formula is unknown
        console.print(f"[yellow]Unknown combined_score_formula: '{formula}'. Defaulting to Silhouette score.[/yellow]")
        return S




def main():
    console.print(Panel.fit("[bold blue]Starting Unsupervised Clustering Pipeline (PCA + DBSCAN)...[/bold blue]",
                           border_style="blue"))

    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        console.print(f"[green]✓ Configuration loaded from [bold]{config_path}[/bold][/green]")
    except FileNotFoundError:
        console.print(f"[bold red]ERROR:[/bold red] Configuration file '[bold]{config_path}[/bold]' not found.", style="red")
        exit(1)

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

    # 2. Preprocess Data
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
        X_for_clustering, pca_model = apply_pca(X_scaled, n_components=selected_n_components, verbose=verbose_level)
        if X_for_clustering.empty or pca_model is None:
             console.print("[yellow]PCA application resulted in empty data or failed. Using scaled data without PCA.[/yellow]")
             X_for_clustering = X_scaled.copy()
        elif pca_model:
             console.print(f"[green]✓ PCA applied. Data shape for clustering: {X_for_clustering.shape}[/green]")
             pca_applied_for_clustering = True 
    else:
        console.print("[cyan]Skipping PCA application step. Using preprocessed data directly for clustering.[/cyan]")
        X_for_clustering = X_scaled.copy()

    if X_for_clustering.empty:
        console.print(Panel("[bold red]Pipeline aborted: Data for clustering is empty after PCA step.[/bold red]", border_style="red"))
        exit(1)

    # 4. Perform DBSCAN Clustering
    console.print(Panel("[bold blue]--- DBSCAN Clustering for Anomaly Detection ---[/bold blue]", border_style="blue"))
    dbscan_config = config.get('clustering', {}).get('dbscan', {})
    dbscan_eps = dbscan_config.get('eps', 0.5)
    dbscan_min_samples = dbscan_config.get('min_samples', 5)
    
    console.print(f"[cyan]Identifying normal behavior clusters and anomalies (noise points) using DBSCAN...[/cyan]")
    cluster_labels_array = perform_dbscan(X_for_clustering, eps=dbscan_eps, min_samples=dbscan_min_samples, verbose=verbose_level)
    cluster_labels = pd.Series(cluster_labels_array, index=X_for_clustering.index, name="dbscan_label")


    if len(cluster_labels) == 0 :
         console.print(Panel("[bold red]Pipeline aborted: DBSCAN failed to produce labels.[/bold red]", border_style="red"))
         exit(1)
         
    # 5. Evaluate Clustering (and report anomalies)
    console.print(Panel("[bold green]--- Clustering & Anomaly Detection Evaluation ---[/bold green]", border_style="green"))
    clustering_metrics = evaluate_clustering(X_for_clustering, cluster_labels.to_numpy(), verbose=verbose_level)
    
    num_anomalies = clustering_metrics.get('n_noise_points', 0)
    percentage_anomalies = clustering_metrics.get('percentage_noise', 0.0)

    rprint("\n[bold]Normal Behavior Clustering Metrics:[/bold]")
    for metric, value in clustering_metrics.items():
        if metric not in ['n_noise_points', 'percentage_noise']: 
            if isinstance(value, float):
                rprint(f"  {metric.replace('_', ' ').title()}: [cyan]{value:.4f}[/cyan]")
            else:
                rprint(f"  {metric.replace('_', ' ').title()}: [cyan]{value}[/cyan]")
    
    console.print(f"\n[bold red]Anomaly Detection Results:[/bold red]")
    console.print(f"  Number of Potential Anomalies (Noise Points): [bold cyan]{num_anomalies}[/bold cyan]")
    console.print(f"  Percentage of Data Flagged as Anomalies: [bold cyan]{percentage_anomalies:.2f}%[/bold cyan]")


    # 6. Plot Clusters (highlighting anomalies)
    console.print(Panel("[bold yellow]--- Visualizing Clusters & Anomalies ---[/bold yellow]", border_style="yellow"))
    plot_title = f"DBSCAN Clusters & Anomalies (eps={dbscan_eps}, min_samples={dbscan_min_samples})"
    if pca_applied_for_clustering: # pca_applied_for_clustering flag from earlier
        plot_title += f"\nData Clustered on {X_for_clustering.shape[1]} PCA Components"
    else:
        plot_title += f"\nData Clustered on Original {X_for_clustering.shape[1]} Scaled Features"
    
    plot_clusters_2d(X_for_clustering, cluster_labels, 
                     title=plot_title, 
                     save_path="models/dbscan_anomaly_plot.png",
                     highlight_anomalies=True) # Ensure anomalies are highlighted

    # 7. Save results (including anomaly flags)
    console.print(Panel("[bold blue]--- Saving Results ---[/bold blue]", border_style="blue"))
    
    # Create an 'is_anomaly' column
    anomaly_series = (cluster_labels == -1).rename('is_anomaly')

    if len(dataframe.index.intersection(cluster_labels.index)) == len(cluster_labels):
        output_df = dataframe.loc[cluster_labels.index].copy()
        output_df['cluster_label'] = cluster_labels.values # DBSCAN's cluster label
        output_df['is_anomaly'] = anomaly_series.loc[output_df.index].values # Add anomaly flag
        
        output_file = "results_with_anomalies.csv"
        try:
            # If original dataframe had a time index, preserve it if possible
            # Assuming 'dataframe' might have a meaningful index
            output_df.to_csv(output_file, index=True)
            console.print(f"\n[green]✓ Results with cluster labels and anomaly flags saved to [bold]{output_file}[/bold][/green]")
            
            anomalous_data = output_df[output_df['is_anomaly'] == True]
            if not anomalous_data.empty:
                console.print(f"\n[cyan]Sample of detected anomalous data points (first 5):[/cyan]")
                rprint(anomalous_data.head())
            else:
                console.print("[cyan]No anomalies detected with current parameters.[/cyan]")

        except Exception as e:
            console.print(f"[yellow]⚠️ Could not save results: {e}[/yellow]")
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