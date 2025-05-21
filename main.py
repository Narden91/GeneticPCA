import time
import pandas as pd
import yaml
import os
import warnings
from rich.console import Console
from rich.panel import Panel
from rich import print

# Import modules
from data_loader import load_all_csvs_from_folder, load_specific_csv_from_folder
from preprocessor import preprocess_data
from classifier import train_and_evaluate_model
from calibration import calibrate_model, get_calibrated_probabilities
from explainer import explain_model_with_shap
from genetic_algorithm import GeneticFeatureSelector


# Create a console instance
console = Console()


def main():
    console.print(Panel.fit("[bold blue]Starting multiclass classification pipeline...[/bold blue]", 
                           border_style="blue"))

    # Load configuration from YAML file
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        console.print(f"[green]✓ Configuration loaded from [bold]{config_path}[/bold][/green]")
    except FileNotFoundError:
        console.print(f"[bold red]ERROR:[/bold red] Configuration file '[bold]{config_path}[/bold]' not found.", style="red")
        exit(1)
    except yaml.YAMLError as e:
        console.print(f"[bold red]ERROR:[/bold red] YAML parsing error in '[bold]{config_path}[/bold]': {e}", style="red")
        exit(1)
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] Unknown error loading '[bold]{config_path}[/bold]': {e}", style="red")
        exit(1)
    
    # 1. Load Data
    data_folder = config['data_loader']['data_folder_path']
    specific_file = config['data_loader'].get('specific_file')

    console.print(Panel(f"[yellow]📂 Loading data from: [bold]{data_folder}[/bold][/yellow]", 
                       border_style="yellow"))
    
    if specific_file:
        console.print(f"[yellow]Specific file: [bold]{specific_file}[/bold][/yellow]")
        dataframe = load_specific_csv_from_folder(data_folder, specific_file)
    else:
        console.print(f"[yellow]Loading all CSV files from folder[/yellow]")
        dataframe = load_all_csvs_from_folder(data_folder)

    if dataframe is not None:
        console.print(f"[green]✓ Data loaded: [/green][cyan]{len(dataframe)} rows, {len(dataframe.columns)} columns[/cyan]")
        
        # 2. Preprocess Data
        console.print(Panel("[bold magenta]--- Starting Preprocessing ---[/bold magenta]", border_style="magenta"))
        target_column = config['data_loader']['target_column'] 
        console.print(f"[magenta]Target column: [bold]{target_column}[/bold][/magenta]")
        
        with console.status("[bold magenta]Preprocessing data...[/bold magenta]", spinner="dots"):
            X_train, X_test, y_train, y_test = preprocess_data(dataframe, target_column, verbose=config["settings"].get("verbose", 0))
            
            # Check for data leakage - keep this important check
            if target_column in X_train.columns:
                console.print(f"[bold red]LEAKAGE ALERT: Target column '{target_column}' found in X_train![/bold red]")
            if target_column in X_test.columns:
                console.print(f"[bold red]LEAKAGE ALERT: Target column '{target_column}' found in X_test![/bold red]")
        
        # Show only important correlations if verbose
        if config["settings"].get("verbose", 0) > 0:
            correlations = X_train.corrwith(y_train.astype(float))
            console.print("[yellow]Top correlations with target:[/yellow]")
            console.print(correlations.abs().sort_values(ascending=False).head(5))
        
        console.print("[green]✓ Preprocessing completed[/green]")

        # Convert target to int if possible
        try:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        except Exception as e:
            console.print(f"[yellow]⚠️ Warning: Unable to convert target to integer format[/yellow]")

        # 3. Feature Selection
        if config["settings"].get("feature_selection", True):
            console.print(Panel("[bold yellow]--- Feature Selection with GA ---[/bold yellow]", border_style="yellow"))
            
            # Get GA parameters from config
            ga_params = config.get('genetic_algorithm', {}).get('params', {})
            
            # Initialize and run GA
            with console.status("[bold cyan]Running genetic algorithm...[/bold cyan]", spinner="dots"):
                ga_selector = GeneticFeatureSelector(
                    X_train=X_train, 
                    y_train=y_train,
                    population_size=ga_params.get('popolazione', 50),
                    generations=ga_params.get('generazioni', 30),
                    crossover_prob=ga_params.get('crossover_prob', 0.7),
                    mutation_prob=ga_params.get('mutation_prob', 0.2),
                    accuracy_weight=ga_params.get('accuracy_weight', 0.6),
                    feature_count_weight=ga_params.get('feature_count_weight', 0.2),
                    correlation_weight=ga_params.get('correlation_weight', 0.2)
                )
                
                selected_features = ga_selector.run()
                console.print(f"[green]✓ GA completed: [cyan]{len(selected_features)} features selected[/cyan][/green]")
            
            if config["settings"].get("verbose", 0) > 0:
                ga_selector.plot_fitness_history()
                
            # Filter features
            X_train = X_train[selected_features]
            X_test = X_test[selected_features] 
        
        # 4. Train and evaluate models
        console.print(Panel("[bold blue]--- Training & Evaluating Model ---[/bold blue]", border_style="blue"))
        
        model_config_yaml = config.get('model', {})
        cv_config_yaml = config.get('cross_validation', {})

        if not model_config_yaml.get('type'):
            console.print("[bold red]ERROR: Model type not specified in configuration (model.type).[/bold red]", style="red")
            exit(1)

        trained_model, conf_matrix, test_accuracy, class_report_dict = train_and_evaluate_model(
            X_train, X_test, y_train, y_test,
            model_config_yaml,
            cv_config_yaml,
            console
        )
        
        # Show feature importances if available
        if trained_model and hasattr(trained_model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': trained_model.feature_importances_
            }).sort_values('importance', ascending=False)
            console.print("[yellow]Top feature importances:[/yellow]")
            console.print(importances.head(5))
        
    else:
        console.print(Panel(f"[bold red]Pipeline aborted due to error loading data from '{data_folder}'.[/bold red]", 
                            border_style="red"))

    console.print(Panel("[bold blue]Multiclass classification pipeline completed.[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # Check if the config file exists
    if not os.path.exists('config.yaml'):
        console.print("[bold red]ERROR:[/bold red] Configuration file 'config.yaml' does not exist.", style="red")
        exit(1)
    
    main()