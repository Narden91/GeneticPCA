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
    except pd.errors.EmptyDataError:
        console.print(f"[yellow]⚠️ Warning: CSV file '[bold]{os.path.basename(file_path)}[/bold]' is empty.[/yellow]")
        return pd.DataFrame() # Return empty DataFrame to avoid downstream errors
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] Failed to load CSV file [bold]{file_path}[/bold]: {e}", style="red")
        return None

def load_specific_csv_from_folder(folder_path, file_name):
    if not os.path.isdir(folder_path):
        console.print(f"[bold red]ERROR:[/bold red] Specified folder '[bold]{folder_path}[/bold]' does not exist or is not a directory.", style="red")
        return None
    
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        console.print(f"[bold red]ERROR:[/bold red] File '[bold]{file_name}[/bold]' does not exist in folder '[bold]{folder_path}[/bold]'.", style="red")
        return None
    
    return load_csv(file_path)


def load_all_csvs_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        console.print(f"[bold red]ERROR:[/bold red] Specified folder '[bold]{folder_path}[/bold]' does not exist or is not a directory.", style="red")
        return None

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        console.print(f"[yellow]No CSV files found in folder '[bold]{folder_path}[/bold]'.[/yellow]")
        return pd.DataFrame() # Return empty DataFrame
    
    all_dfs = []
    console.print(f"[blue]Found {len(csv_files)} CSV files in folder '[bold]{folder_path}[/bold]'. Starting to load...[/blue]")
    
    for file_path in csv_files:
        df = load_csv(file_path)
        if df is not None and not df.empty:
            all_dfs.append(df)
            console.print(f"  [green]✓ Loaded:[/green] [bold]{os.path.basename(file_path)}[/bold] ([cyan]{len(df)}[/cyan] rows)")
        elif df is not None and df.empty:
             console.print(f"  [yellow]✓ Loaded (empty):[/yellow] [bold]{os.path.basename(file_path)}[/bold]")


    if not all_dfs:
        console.print("[bold red]No DataFrames with content were successfully loaded.[/bold red]")
        return pd.DataFrame() # Return empty DataFrame
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if combined_df.empty:
        console.print("[yellow]Combined DataFrame is empty after loading all files.[/yellow]")
        return pd.DataFrame()

    console.print(f"[green]✓ All CSV files have been loaded and concatenated. Combined DataFrame has [cyan]{len(combined_df)}[/cyan] rows.[/green]")
    return combined_df