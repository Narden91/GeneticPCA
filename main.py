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
import time

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
                        c='red', marker='x', s=100, label='Anomalies (Noise)')

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
                        c='red', marker='x', s=100, label='Anomalies (Noise)')

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
                                  markerfacecolor='red', markersize=10))

    plt.legend(handles=handles, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]✓ Cluster plot saved to [bold]{save_path}[/bold][/green]")
    plt.show()


def main():
    console.print(Panel.fit("[bold blue]Optimized Unsupervised Clustering Pipeline\n(Bayesian Feature Selection + PCA + DBSCAN)[/bold blue]",
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
    console.print(Panel("[bold magenta]--- Data Preprocessing ---[/bold magenta]", border_style="magenta"))
    scaler_type = config.get('preprocessing', {}).get('scaler', 'StandardScaler')

    with console.status("[bold magenta]Preprocessing data...[/bold magenta]", spinner="dots"):
        X_scaled = preprocess_data_unsupervised(dataframe, scaler_type=scaler_type, verbose=verbose_level)

    if X_scaled is None or X_scaled.empty:
        console.print(Panel("[bold red]Pipeline aborted: Preprocessing failed or resulted in empty dataset.[/bold red]", border_style="red"))
        exit(1)
    console.print(f"[green]✓ Preprocessing completed: {X_scaled.shape}[/green]")
    
    if verbose_level > 1 and not X_scaled.empty:
        console.print("[cyan]Scaled data head:[/cyan]")
        rprint(X_scaled.head(3))

    # 3. Bayesian Feature Selection
    console.print(Panel("[bold purple]--- Bayesian Feature Selection ---[/bold purple]", border_style="purple"))
    
    bayesian_fs_config = config.get('bayesian_feature_selection', {})
    use_bayesian_fs = bayesian_fs_config.get('enabled', False)
    
    X_for_analysis = X_scaled.copy()  # Data that will be used for further analysis
    selected_features_list = None
    feature_selector = None
    
    if use_bayesian_fs and X_scaled.shape[1] > 2:
        try:
            # Use optimized version for high-dimensional data
            from bayesian_methods.bayesian_feature_selector import OptimizedBayesianFeatureSelector
            
            # Get configuration parameters
            max_parents = bayesian_fs_config.get('max_parents', 2)
            scoring_method = bayesian_fs_config.get('scoring_method', 'bic')
            target_variable = bayesian_fs_config.get('target_variable', None)
            min_features = bayesian_fs_config.get('min_features', 15)
            max_features = bayesian_fs_config.get('max_features', None)
            visualize_network = bayesian_fs_config.get('visualize_network', True)
            save_feature_ranking = bayesian_fs_config.get('save_feature_ranking', True)
            
            # Optimization parameters
            optimization_config = bayesian_fs_config.get('optimization', {})
            
            console.print(f"[cyan]Initializing Optimized Bayesian Feature Selector[/cyan]")
            console.print(f"[cyan]Parameters: max_parents={max_parents}, method={scoring_method}, max_features_for_bn={optimization_config.get('max_features_for_bn', 50)}[/cyan]")
            
            # Create optimized feature selector
            feature_selector = OptimizedBayesianFeatureSelector(
                max_parents=max_parents,
                scoring_method=scoring_method,
                target_variable=target_variable,
                max_features_for_bn=optimization_config.get('max_features_for_bn', 50),
                prefilter_method=optimization_config.get('prefilter_method', 'correlation'),
                prefilter_threshold=optimization_config.get('prefilter_threshold', 0.1),
                max_iter=optimization_config.get('max_iter', 30),
                random_state=42
            )
            
            # Fit and select features
            start_time = time.time()
            feature_selector.fit(X_scaled, target_variable)
            
            # Select features using Markov Blanket analysis
            selected_features_list = feature_selector.select_features_by_markov_blanket(
                target_variable=target_variable,
                min_features=min_features,
                max_features=max_features
            )
            
            elapsed_time = time.time() - start_time
            console.print(f"[green]✓ Feature selection completed in {elapsed_time:.1f} seconds[/green]")
            console.print(f"[green]✓ Selected {len(selected_features_list)} features[/green]")
            
            if verbose_level > 0:
                console.print(f"[cyan]Selected features: {selected_features_list[:10]}{'...' if len(selected_features_list) > 10 else ''}[/cyan]")
            
            # Apply feature selection
            X_for_analysis = feature_selector.transform(X_scaled)
            console.print(f"[green]✓ Data shape after feature selection: {X_for_analysis.shape}[/green]")
            
            # Visualize the Bayesian network if requested
            if visualize_network:
                console.print("\n[bold purple]Visualizing Bayesian Network Structure:[/bold purple]")
                feature_selector.visualize_network(
                    save_path="models/bayesian_feature_network.png"
                )
            
            # Save detailed feature ranking if requested
            if save_feature_ranking:
                try:
                    ranking_df = feature_selector.get_feature_ranking()
                    ranking_df.to_csv("models/bayesian_feature_ranking.csv", index=False)
                    console.print(f"[green]✓ Feature ranking saved to models/bayesian_feature_ranking.csv[/green]")
                    
                    if verbose_level > 0:
                        console.print("\n[cyan]Top 10 Features by Importance:[/cyan]")
                        display_cols = ['feature', 'prefilter_score', 'was_prefiltered', 'importance_score', 'is_selected']
                        rprint(ranking_df.head(10)[display_cols])
                        
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not save feature ranking: {e}[/yellow]")
            
        except ImportError as e:
            console.print(f"[yellow]⚠️ Could not load Bayesian feature selector: {e}[/yellow]")
            console.print("[yellow]Make sure pgmpy is installed. Continuing without feature selection.[/yellow]")
            use_bayesian_fs = False
        except Exception as e:
            console.print(f"[bold red]ERROR in Bayesian feature selection: {str(e)[:200]}[/bold red]")
            console.print("[yellow]Continuing without feature selection.[/yellow]")
            use_bayesian_fs = False
    
    elif use_bayesian_fs and X_scaled.shape[1] <= 2:
        console.print("[yellow]⚠️ Bayesian feature selection skipped: Too few features (need > 2)[/yellow]")
        use_bayesian_fs = False
    else:
        console.print("[cyan]Bayesian feature selection disabled in configuration.[/cyan]")

    # 4. PCA Optimization & Application
    console.print(Panel("[bold yellow]--- PCA Optimization & Application ---[/bold yellow]", border_style="yellow"))
    run_ga_for_pca = config.get('settings', {}).get('run_ga_for_pca', False)
    pca_config = config.get('pca', {})

    X_for_clustering = X_for_analysis.copy()  # Start with feature-selected data
    selected_n_components = 0  # Default to no PCA
    pca_applied_for_clustering = False  # Flag to track if PCA was used

    if X_for_analysis.shape[1] <= 1:
        console.print("[yellow]PCA skipped: Data has 1 or fewer features after feature selection.[/yellow]")
        run_ga_for_pca = False
    
    if run_ga_for_pca and X_for_analysis.shape[1] > 1:
        ga_config = config.get('genetic_pca_optimizer', {}).get('params', {})
        dbscan_params_for_ga = config.get('clustering', {}).get('dbscan', {})

        console.print("[cyan]Using Genetic Algorithm to optimize PCA components on selected features...[/cyan]")
        with console.status("[bold cyan]Running genetic algorithm for PCA...[/bold cyan]", spinner="dots"):
            ga_optimizer = GeneticPCAOptimizer(
                X_scaled=X_for_analysis,
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
            
    elif X_for_analysis.shape[1] > 1:
        manual_pca_components = pca_config.get('manual_pca_components', 0)
        if manual_pca_components > 0:
            selected_n_components = manual_pca_components
            console.print(f"[cyan]Using manual PCA setting: {selected_n_components} components.[/cyan]")
        else:
            console.print("[cyan]No PCA will be applied (manual_pca_components is 0 or GA is off).[/cyan]")
            selected_n_components = 0
    
    if selected_n_components > 0 and X_for_analysis.shape[1] > 1:
        X_for_clustering, pca_model = apply_pca(X_for_analysis, n_components=selected_n_components, verbose=verbose_level)
        if X_for_clustering.empty or pca_model is None:
             console.print("[yellow]PCA application failed. Using feature-selected data without PCA.[/yellow]")
             X_for_clustering = X_for_analysis.copy()
        elif pca_model:
             console.print(f"[green]✓ PCA applied. Data shape for clustering: {X_for_clustering.shape}[/green]")
             pca_applied_for_clustering = True 
    else:
        console.print("[cyan]Skipping PCA. Using feature-selected data directly for clustering.[/cyan]")
        X_for_clustering = X_for_analysis.copy()

    if X_for_clustering.empty:
        console.print(Panel("[bold red]Pipeline aborted: Data for clustering is empty after PCA step.[/bold red]", border_style="red"))
        exit(1)

    # 5. DBSCAN Clustering for Anomaly Detection with Parameter Optimization
    console.print(Panel("[bold blue]--- DBSCAN Clustering & Anomaly Detection ---[/bold blue]", border_style="blue"))
    dbscan_config = config.get('clustering', {}).get('dbscan', {})
    dbscan_eps = dbscan_config.get('eps', 0.5)
    dbscan_min_samples = dbscan_config.get('min_samples', 5)
    auto_optimize = dbscan_config.get('auto_optimize', False)
    
    # Optimize DBSCAN parameters if requested
    if auto_optimize:
        console.print(f"[cyan]🔍 Auto-optimizing DBSCAN parameters for better anomaly detection...[/cyan]")
        
        try:
            # Import the DBSCAN optimizer (assuming it's in the same directory)
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from dbscan_optimizer import DBSCANOptimizer
            
            optimization_config = dbscan_config.get('optimization', {})
            
            optimizer = DBSCANOptimizer(
                target_anomaly_rate=optimization_config.get('target_anomaly_rate', 0.05),
                min_clusters=optimization_config.get('min_clusters', 2),
                max_clusters=optimization_config.get('max_clusters', 15),
                random_state=42
            )
            
            # Optimize parameters
            start_opt_time = time.time()
            best_params = optimizer.optimize_parameters(
                X_for_clustering,
                n_trials=optimization_config.get('n_trials', 25)
            )
            opt_time = time.time() - start_opt_time
            
            # Update parameters with optimized values
            dbscan_eps = best_params['eps']
            dbscan_min_samples = best_params['min_samples']
            
            console.print(f"[green]✓ Parameter optimization completed in {opt_time:.1f} seconds[/green]")
            console.print(f"[green]✓ Optimized parameters: eps={dbscan_eps:.3f}, min_samples={dbscan_min_samples}[/green]")
            
            # Plot optimization results if requested
            if optimization_config.get('plot_results', True):
                optimizer.plot_optimization_results(save_path="models/dbscan_optimization_results.png")
            
            # Show alternative parameter suggestions
            alternatives = optimizer.get_alternative_parameters(top_n=3)
            if len(alternatives) > 1:
                console.print("\n[cyan]📊 Top 3 parameter alternatives:[/cyan]")
                for i, alt in enumerate(alternatives[:3], 1):
                    console.print(f"  {i}. eps={alt['eps']:.3f}, min_samples={alt['min_samples']}, "
                                f"clusters={alt['n_clusters']}, anomalies={alt['anomaly_rate']*100:.1f}%")
        
        except ImportError:
            console.print("[yellow]⚠️ DBSCAN optimizer not available. Using manual parameters.[/yellow]")
            auto_optimize = False
        except Exception as e:
            console.print(f"[yellow]⚠️ Parameter optimization failed: {str(e)[:100]}. Using manual parameters.[/yellow]")
            auto_optimize = False
    else:
        console.print(f"[cyan]Using manual DBSCAN parameters[/cyan]")
    
    console.print(f"[cyan]Performing DBSCAN clustering for anomaly detection...[/cyan]")
    console.print(f"[cyan]Final parameters: eps={dbscan_eps:.3f}, min_samples={dbscan_min_samples}[/cyan]")
    
    cluster_labels_array = perform_dbscan(X_for_clustering, eps=dbscan_eps, min_samples=dbscan_min_samples, verbose=verbose_level)
    cluster_labels = pd.Series(cluster_labels_array, index=X_for_clustering.index, name="dbscan_label")

    if len(cluster_labels) == 0:
         console.print(Panel("[bold red]Pipeline aborted: DBSCAN failed to produce labels.[/bold red]", border_style="red"))
         exit(1)
         
    # 6. Evaluate Clustering & Report Anomalies
    console.print(Panel("[bold green]--- Clustering Evaluation & Anomaly Results ---[/bold green]", border_style="green"))
    clustering_metrics = evaluate_clustering(X_for_clustering, cluster_labels.to_numpy(), verbose=verbose_level)
    
    num_anomalies = clustering_metrics.get('n_noise_points', 0)
    percentage_anomalies = clustering_metrics.get('percentage_noise', 0.0)
    n_clusters = clustering_metrics.get('n_clusters', 0)

    # Display clustering performance
    rprint("\n[bold]Clustering Performance Metrics:[/bold]")
    for metric, value in clustering_metrics.items():
        if metric not in ['n_noise_points', 'percentage_noise']: 
            if isinstance(value, float) and value is not None:
                rprint(f"  {metric.replace('_', ' ').title()}: [cyan]{value:.4f}[/cyan]")
            else:
                rprint(f"  {metric.replace('_', ' ').title()}: [cyan]{value}[/cyan]")
    
    # Display anomaly detection results
    console.print(f"\n[bold red]🚨 Anomaly Detection Results:[/bold red]")
    console.print(f"  Total Data Points: [bold cyan]{len(cluster_labels)}[/bold cyan]")
    console.print(f"  Normal Clusters Found: [bold green]{n_clusters}[/bold green]")
    console.print(f"  Anomalies Detected: [bold red]{num_anomalies}[/bold red]")
    console.print(f"  Anomaly Rate: [bold red]{percentage_anomalies:.2f}%[/bold red]")
    
    # Quality assessment
    if percentage_anomalies > 15:
        console.print("  [yellow]⚠️ High anomaly rate - consider adjusting DBSCAN parameters[/yellow]")
    elif percentage_anomalies < 1:
        console.print("  [yellow]⚠️ Very low anomaly rate - parameters might be too strict[/yellow]")
    else:
        console.print("  [green]✓ Reasonable anomaly detection rate[/green]")

    # 7. Visualize Clusters & Anomalies
    console.print(Panel("[bold yellow]--- Visualizing Results ---[/bold yellow]", border_style="yellow"))
    
    # Create descriptive plot title
    plot_title = f"DBSCAN Anomaly Detection Results\neps={dbscan_eps}, min_samples={dbscan_min_samples}"
    
    if use_bayesian_fs and selected_features_list:
        plot_title += f"\nBayesian Feature Selection: {len(selected_features_list)} features"
    
    if pca_applied_for_clustering:
        plot_title += f"\nPCA Applied: {X_for_clustering.shape[1]} components"
    else:
        plot_title += f"\nOriginal Features: {X_for_clustering.shape[1]} dimensions"
    
    plot_clusters_2d(X_for_clustering, cluster_labels, 
                     title=plot_title, 
                     save_path="models/anomaly_detection_results.png",
                     highlight_anomalies=True)

    # 8. Save Comprehensive Results
    console.print(Panel("[bold blue]--- Saving Results ---[/bold blue]", border_style="blue"))
    
    # Create anomaly flag
    anomaly_series = (cluster_labels == -1).rename('is_anomaly')

    if len(dataframe.index.intersection(cluster_labels.index)) == len(cluster_labels):
        output_df = dataframe.loc[cluster_labels.index].copy()
        output_df['cluster_label'] = cluster_labels.values
        output_df['is_anomaly'] = anomaly_series.loc[output_df.index].values
        
        # Add feature selection information
        if use_bayesian_fs and selected_features_list:
            output_df['selected_features_used'] = ', '.join(selected_features_list)
            output_df['num_features_selected'] = len(selected_features_list)
            output_df['original_num_features'] = X_scaled.shape[1]
            output_df['feature_reduction_ratio'] = f"{(1 - len(selected_features_list)/X_scaled.shape[1])*100:.1f}%"
        
        # Add PCA information
        if pca_applied_for_clustering:
            output_df['pca_components_used'] = selected_n_components
            output_df['pca_applied'] = True
        else:
            output_df['pca_applied'] = False
        
        # Add clustering metrics
        output_df['dbscan_eps'] = dbscan_eps
        output_df['dbscan_min_samples'] = dbscan_min_samples
        output_df['total_clusters_found'] = n_clusters
        output_df['anomaly_detection_rate'] = f"{percentage_anomalies:.2f}%"
        
        output_file = "results_anomaly_detection.csv"
        try:
            output_df.to_csv(output_file, index=True)
            console.print(f"\n[green]✓ Complete results saved to [bold]{output_file}[/bold][/green]")
            
            # Show sample of detected anomalies
            anomalous_data = output_df[output_df['is_anomaly'] == True]
            if not anomalous_data.empty:
                console.print(f"\n[cyan]Sample of detected anomalies (first 5 rows):[/cyan]")
                display_cols = ['cluster_label', 'is_anomaly']
                if 'num_features_selected' in output_df.columns:
                    display_cols.extend(['num_features_selected', 'feature_reduction_ratio'])
                rprint(anomalous_data[display_cols].head())
            else:
                console.print("[cyan]No anomalies detected with current parameters.[/cyan]")

        except Exception as e:
            console.print(f"[yellow]⚠️ Could not save results: {e}[/yellow]")
    else:
        console.print(f"[yellow]⚠️ Index mismatch: Could not align original data with cluster labels.[/yellow]")

    # 9. Generate Pipeline Summary Report
    console.print(Panel("[bold green]--- Pipeline Summary Report ---[/bold green]", border_style="green"))
    
    summary_report = {
        "Pipeline Configuration": {
            "Data Source": specific_file if specific_file else f"All CSVs in {data_folder}",
            "Original Data Shape": f"{dataframe.shape[0]} rows × {dataframe.shape[1]} columns",
            "Preprocessing Scaler": scaler_type,
            "Bayesian Feature Selection": "Enabled (Optimized)" if use_bayesian_fs else "Disabled",
            "PCA Optimization": "Genetic Algorithm" if run_ga_for_pca else "Manual/None"
        },
        "Feature Selection Results": {},
        "PCA Results": {},
        "Anomaly Detection Results": {
            "DBSCAN Parameters": f"eps={dbscan_eps:.3f}, min_samples={dbscan_min_samples}",
            "Parameter Optimization": "Enabled" if auto_optimize else "Manual",
            "Normal Clusters Found": n_clusters,
            "Total Anomalies": f"{num_anomalies} ({percentage_anomalies:.2f}%)",
            "Silhouette Score": f"{clustering_metrics.get('silhouette_score', 'N/A'):.4f}" if clustering_metrics.get('silhouette_score') is not None else 'N/A',
            "Davies-Bouldin Score": f"{clustering_metrics.get('davies_bouldin_score', 'N/A'):.4f}" if clustering_metrics.get('davies_bouldin_score') is not None else 'N/A'
        }
    }
    
    # Add feature selection details
    if use_bayesian_fs and selected_features_list:
        reduction_ratio = (1 - len(selected_features_list)/X_scaled.shape[1])*100
        summary_report["Feature Selection Results"] = {
            "Method": "Optimized Bayesian Network with Markov Blanket Analysis",
            "Original Features": X_scaled.shape[1],
            "Selected Features": len(selected_features_list),
            "Reduction Ratio": f"{reduction_ratio:.1f}%",
            "Pre-filter Method": optimization_config.get('prefilter_method', 'correlation'),
            "Max Features for BN": optimization_config.get('max_features_for_bn', 50),
            "Max Parents": max_parents,
            "Processing Time": f"{elapsed_time:.1f} seconds"
        }
    else:
        summary_report["Feature Selection Results"] = {"Method": "None - All features used"}
    
    # Add PCA details
    if pca_applied_for_clustering:
        summary_report["PCA Results"] = {
            "Method": "Genetic Algorithm Optimized" if run_ga_for_pca else "Manual Configuration",
            "Components Selected": selected_n_components,
            "Input Features": X_for_analysis.shape[1],
            "Dimensionality Reduction": f"{X_for_analysis.shape[1]} → {selected_n_components}"
        }
    else:
        summary_report["PCA Results"] = {"Method": "None - Original dimensionality preserved"}
    
    # Print summary report
    console.print("\n[bold]📊 PIPELINE EXECUTION SUMMARY[/bold]")
    console.print("=" * 60)
    
    for section, details in summary_report.items():
        console.print(f"\n[bold cyan]{section}:[/bold cyan]")
        if isinstance(details, dict):
            for key, value in details.items():
                console.print(f"  {key}: [green]{value}[/green]")
        else:
            console.print(f"  [green]{details}[/green]")
    
    # Save summary report
    try:
        import json
        with open("models/pipeline_summary_report.json", "w") as f:
            json.dump(summary_report, f, indent=2, default=str)
        console.print(f"\n[green]✓ Pipeline summary saved to models/pipeline_summary_report.json[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not save summary report: {e}[/yellow]")
    
    # Final recommendations and insights
    console.print("\n[bold blue]🎯 RECOMMENDATIONS & INSIGHTS:[/bold blue]")
    
    if use_bayesian_fs and selected_features_list:
        reduction_pct = (1 - len(selected_features_list)/X_scaled.shape[1])*100
        if reduction_pct > 50:
            console.print(f"✓ [green]Excellent dimensionality reduction: {reduction_pct:.1f}% of features removed[/green]")
        elif reduction_pct > 20:
            console.print(f"✓ [green]Good feature selection: {reduction_pct:.1f}% reduction achieved[/green]")
        else:
            console.print(f"⚠️ [yellow]Limited feature reduction: {reduction_pct:.1f}% - consider stricter selection[/yellow]")
    
    # Safe comparison for silhouette score (handle None values)
    silhouette_score = clustering_metrics.get('silhouette_score')
    if silhouette_score is not None:
        if silhouette_score > 0.5:
            console.print("✓ [green]Good clustering quality detected[/green]")
        elif silhouette_score < 0.3:
            console.print("⚠️ [yellow]Consider adjusting DBSCAN parameters for better clustering[/yellow]")
        else:
            console.print(f"📊 [cyan]Clustering quality (Silhouette): {silhouette_score:.3f}[/cyan]")
    else:
        console.print("ℹ️ [blue]Clustering quality metrics unavailable (insufficient clusters for evaluation)[/blue]")
    
    # Anomaly detection assessment
    if percentage_anomalies > 15:
        console.print("⚠️ [yellow]High anomaly rate detected. Consider increasing eps or min_samples[/yellow]")
    elif percentage_anomalies < 1:
        console.print("⚠️ [red]Very few anomalies detected. DBSCAN parameters may be too strict![/red]")
        console.print("   💡 [cyan]Try: decreasing eps (e.g., eps=10-15) or min_samples (e.g., 3-4)[/cyan]")
    else:
        console.print("✓ [green]Reasonable anomaly detection rate[/green]")
    
    # Cluster structure assessment
    if n_clusters == 0:
        console.print("❌ [red]No clusters found. DBSCAN parameters too strict![/red]")
        console.print("   💡 [cyan]Try: significantly decrease eps (e.g., eps=5-10)[/cyan]")
    elif n_clusters == 1:
        console.print("⚠️ [yellow]Only one cluster found. DBSCAN treating most data as one big cluster![/yellow]")
        console.print("   💡 [cyan]Try: decrease eps (e.g., eps=10-15) for more granular clustering[/cyan]")
    elif n_clusters > 20:
        console.print("⚠️ [yellow]Many small clusters found. Consider increasing eps or min_samples[/yellow]")
    else:
        console.print(f"✓ [green]Good cluster structure: {n_clusters} clusters identified[/green]")
    
    # Specific recommendations for current results
    console.print("\n[bold blue]🔧 SPECIFIC PARAMETER RECOMMENDATIONS:[/bold blue]")
    if percentage_anomalies == 0.0 and n_clusters <= 1:
        console.print("📋 [yellow]Current issue: DBSCAN parameters too conservative[/yellow]")
        console.print("   🎯 [green]Recommended fixes:[/green]")
        console.print("      • Decrease eps from 20 to 8-12 (for tighter clustering)")
        console.print("      • Decrease min_samples from 5 to 3-4 (for smaller clusters)")
        console.print("      • This should reveal more cluster structure and identify anomalies")
        
        # Suggest automatic parameter tuning
        console.print("\n   🤖 [cyan]Consider enabling automatic parameter tuning in future versions[/cyan]")
    
    console.print(Panel("[bold blue]🎉 Optimized anomaly detection pipeline completed successfully![/bold blue]", border_style="blue"))
    
    # Show execution time summary
    end_time = time.time()
    total_time = end_time - start_time if 'start_time' in locals() else 0
    console.print(f"\n[bold]⏱️ Total Pipeline Execution Time: {total_time:.1f} seconds[/bold]")


if __name__ == "__main__":
    # Configure warnings and environment
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module='pgmpy')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Create necessary directories
    for directory in ['models', 'results']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            console.print(f"[cyan]Created '{directory}' directory.[/cyan]")
    
    # Validate configuration file exists
    if not os.path.exists('config.yaml'):
        console.print("[bold red]ERROR:[/bold red] Configuration file 'config.yaml' does not exist.", style="red")
        console.print("[yellow]Please ensure config.yaml is in the current directory.[/yellow]")
        exit(1)
    
    # Record start time for total execution tracking
    start_time = time.time()
    
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error in pipeline: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        exit(1)