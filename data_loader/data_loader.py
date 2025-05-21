import pandas as pd
import os
import glob
from rich.console import Console


# Create console instance for Rich formatting
console = Console()


def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        console.print(f"[green]✓ Data successfully loaded from[/green] [bold]{file_path}[/bold]")
        return df
    except FileNotFoundError:
        console.print(f"[bold red]ERROR:[/bold red] File not found at [bold]{file_path}[/bold]", style="red")
        return None
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] Failed to load CSV file [bold]{file_path}[/bold]: {e}", style="red")
        return None


def load_specific_csv_from_folder(folder_path, file_name):
    """
    Load a specific CSV file from a folder.
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files
    file_name (str): Name of the specific file to load
    
    Returns:
    DataFrame or None: The loaded pandas DataFrame or None if error occurs
    """
    if not os.path.isdir(folder_path):
        console.print(f"[bold red]ERROR:[/bold red] Specified folder '[bold]{folder_path}[/bold]' does not exist or is not a directory.", style="red")
        return None
    
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        console.print(f"[bold red]ERROR:[/bold red] File '[bold]{file_name}[/bold]' does not exist in folder '[bold]{folder_path}[/bold]'.", style="red")
        return None
    
    try:
        df = pd.read_csv(file_path)
        console.print(f"[green]✓ File loaded successfully:[/green] [bold]{file_name}[/bold] ([cyan]{len(df)}[/cyan] rows)")
        return df
    except pd.errors.EmptyDataError:
        console.print(f"[yellow]⚠️ Warning: CSV file '[bold]{file_name}[/bold]' is empty.[/yellow]")
        return None
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] Failed to load CSV file '[bold]{file_name}[/bold]': {e}", style="red")
        return None


def load_all_csvs_from_folder(folder_path):
    """Load all CSV files from a specified folder and concatenate them."""
    if not os.path.isdir(folder_path):
        console.print(f"[bold red]ERROR:[/bold red] Specified folder '[bold]{folder_path}[/bold]' does not exist or is not a directory.", style="red")
        return None

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        console.print(f"[yellow]No CSV files found in folder '[bold]{folder_path}[/bold]'.[/yellow]")
        return None
    
    all_dfs = []
    console.print(f"[blue]Found {len(csv_files)} CSV files in folder '[bold]{folder_path}[/bold]'. Starting to load...[/blue]")
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            console.print(f"  [green]✓ Loaded:[/green] [bold]{os.path.basename(file_path)}[/bold] ([cyan]{len(df)}[/cyan] rows)")
        except FileNotFoundError:
            console.print(f"  [red]ERROR: File not found {file_path} (this shouldn't happen if glob found it).[/red]")
        except pd.errors.EmptyDataError:
            console.print(f"  [yellow]⚠️ Warning: CSV file '[bold]{os.path.basename(file_path)}[/bold]' is empty.[/yellow]")
        except Exception as e:
            console.print(f"  [red]ERROR: Failed to load CSV file '[bold]{os.path.basename(file_path)}[/bold]': {e}[/red]")

    if not all_dfs:
        console.print("[bold red]No DataFrames were successfully loaded.[/bold red]")
        return None
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    console.print(f"[green]✓ All CSV files have been loaded and concatenated. Combined DataFrame has [cyan]{len(combined_df)}[/cyan] rows.[/green]")
    return combined_df