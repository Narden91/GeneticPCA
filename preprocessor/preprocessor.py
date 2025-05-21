import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console


# Create a console instance for consistent printing
console = Console()

def preprocess_data(df: pd.DataFrame, target_column: str, verbose: int = 0) -> tuple:
    """Performs basic preprocessing and data cleaning."""
    console.print("[magenta]Starting data preprocessing...[/magenta]")
    
    # Step 1: Handle missing values
    initial_rows = len(df)
    df_cleaned = df.dropna()  # Remove rows with NaN values
    rows_dropped = initial_rows - len(df_cleaned)
    
    if rows_dropped > 0:
        console.print(f"[yellow]Rows removed due to NaN values: {rows_dropped}[/yellow]")
    else:
        console.print("[green]No rows removed due to NaN values.[/green]")

    # Step 2: Check if target column exists
    if target_column not in df_cleaned.columns:
        console.print(f"[bold red]ERROR: Target column '{target_column}' not found in DataFrame after NaN removal.[/bold red]")
        return None, None, None, None 

    # Step 3: Split features and target
    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]

    # Step 4: Verify data shapes and distribution
    if verbose > 0:
        console.print(f"[cyan]X shape before split: {X.shape}[/cyan]")
        console.print(f"[cyan]y shape before split: {y.shape}[/cyan]")
        console.print(f"[cyan]Unique values in y before split: {y.nunique()}, Distribution:\n{y.value_counts(normalize=True)}[/cyan]")

    # Step 5: Check if stratification is possible
    min_class_count = y.value_counts().min()
    n_splits_for_stratify_check = 2 
    
    stratify_param = None
    if y.nunique() > 1 and min_class_count >= n_splits_for_stratify_check:
        stratify_param = y
        console.print("[green]Stratification enabled for train_test_split.[/green]")
    elif y.nunique() <= 1:
        console.print(f"[bold red]ERROR: Target column y has {y.nunique()} unique value(s). Cannot stratify or perform classification.[/bold red]")
        return None, None, None, None  # Cannot proceed with classification
    else:  # y.nunique() > 1 but min_class_count < n_splits_for_stratify_check
        console.print(f"[yellow]WARNING: Stratification disabled. The least frequent class has only {min_class_count} samples, which is less than required for splitting (minimum 2).[/yellow]")

    # Step 6: Perform train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=stratify_param
        )
        console.print("[green]Data successfully split into training and test sets.[/green]")
        
        # Step 7: Log split information if verbose
        if verbose > 0:
            console.print(f"[cyan]X_train shape: {X_train.shape}, y_train shape: {y_train.shape}[/cyan]")
            console.print(f"[cyan]X_test shape: {X_test.shape}, y_test shape: {y_test.shape}[/cyan]")
            console.print(f"[cyan]Class distribution in y_train:\n{y_train.value_counts(normalize=True)}[/cyan]")
            console.print(f"[cyan]Class distribution in y_test:\n{y_test.value_counts(normalize=True)}[/cyan]")

        return X_train, X_test, y_train, y_test
        
    except ValueError as e:
        console.print(f"[bold red]ERROR during train_test_split: {e}[/bold red]")
        console.print("[red]   This can happen if stratification fails due to insufficient samples in a class relative to test_size.[/red]")
        return None, None, None, None