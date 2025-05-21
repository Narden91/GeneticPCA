import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rich.console import Console

console = Console()

def preprocess_data_unsupervised(df: pd.DataFrame, scaler_type: str = "StandardScaler", verbose: int = 0) -> pd.DataFrame:
    """
    Performs preprocessing for unsupervised learning:
    1. Handles missing values (drops rows).
    2. Selects only numeric columns.
    3. Scales the data.
    """
    if df is None or df.empty:
        console.print("[yellow]⚠️ Preprocessing skipped: Input DataFrame is None or empty.[/yellow]")
        return pd.DataFrame()

    console.print("[magenta]Starting data preprocessing for unsupervised learning...[/magenta]")
    
    # Step 1: Handle missing values
    initial_rows = len(df)
    df_cleaned = df.dropna()
    rows_dropped = initial_rows - len(df_cleaned)
    
    if rows_dropped > 0:
        console.print(f"[yellow]Rows removed due to NaN values: {rows_dropped}[/yellow]")
    else:
        console.print("[green]No rows removed due to NaN values.[/green]")

    if df_cleaned.empty:
        console.print("[bold red]ERROR: DataFrame is empty after NaN removal.[/bold red]")
        return pd.DataFrame()

    # Step 2: Select only numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        console.print("[bold red]ERROR: No numeric columns found in the DataFrame for scaling and clustering.[/bold red]")
        return pd.DataFrame()
    
    X_numeric = df_cleaned[numeric_cols]
    if verbose > 1:
        console.print(f"[cyan]Selected {len(numeric_cols)} numeric columns: {list(numeric_cols)}[/cyan]")
        console.print(f"[cyan]Shape of numeric data: {X_numeric.shape}[/cyan]")

    # Step 3: Scale the data
    if scaler_type:
        if scaler_type.lower() == "standardscaler":
            scaler = StandardScaler()
        elif scaler_type.lower() == "minmaxscaler":
            scaler = MinMaxScaler()
        else:
            console.print(f"[yellow]⚠️ Unknown scaler type '{scaler_type}'. No scaling will be applied.[/yellow]")
            scaler = None
        
        if scaler:
            console.print(f"[magenta]Applying {scaler_type}...[/magenta]")
            X_scaled_array = scaler.fit_transform(X_numeric)
            X_scaled_df = pd.DataFrame(X_scaled_array, columns=X_numeric.columns, index=X_numeric.index)
            console.print("[green]✓ Data scaling completed.[/green]")
            if verbose > 1:
                console.print("[cyan]Scaled data head:[/cyan]")
                console.print(X_scaled_df.head(3))
            return X_scaled_df
    else:
        console.print("[yellow]No scaler specified. Using unscaled numeric data.[/yellow]")
        return X_numeric

    return X_numeric