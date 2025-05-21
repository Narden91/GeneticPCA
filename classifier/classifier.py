import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import numpy as np # For checking y_train unique values


def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series, 
                             model_config: dict, cv_config: dict, console: Console):
    """
    Trains and evaluates a specified classifier.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target.
        y_test: Testing target.
        model_config: Dictionary with model type and parameters.
                      Example: {'type': 'RandomForest', 
                                'RandomForest_params': {'n_estimators': 100},
                                'global_random_state': 42}
        cv_config: Dictionary with cross-validation parameters.
                   Example: {'n_splits': 5, 'shuffle': True}
        console: Rich Console object for printing.

    Returns:
        tuple: (trained_model, confusion_matrix, test_accuracy, classification_report_dict)
               Returns (None, None, None, None) if model type is unsupported or error occurs.
    """
    model_type = model_config.get('type', 'RandomForest').lower()
    # Use model-specific params if available, else empty dict
    model_params = model_config.get(f"{model_type}_params", {}) 
    
    global_random_state = model_config.get('global_random_state', 42)

    console.print(Panel(f"[bold cyan]Initializing Model: {model_type.capitalize()} [/bold cyan]", border_style="cyan"))

    # Ensure y_train and y_test are 1D
    if hasattr(y_train, 'ndim') and y_train.ndim > 1:
        y_train = y_train.squeeze()
    if hasattr(y_test, 'ndim') and y_test.ndim > 1:
        y_test = y_test.squeeze()

    # Check number of unique classes in y_train
    try:
        # Convert to numpy array first to handle potential mixed types if y_train is object dtype
        y_train_unique = np.unique(y_train.astype(str)) if y_train.dtype == 'object' else np.unique(y_train)
        num_class = len(y_train_unique)
        console.print(f"[cyan]Number of unique classes in y_train: {num_class}[/cyan]")
        if num_class <= 1:
            console.print(f"[bold red]ERROR: Target y_train contains {num_class} unique class(es). At least 2 classes are required for classification.[/bold red]", style="red")
            return None, None, None, None
    except Exception as e:
        console.print(f"[bold red]ERROR: Unable to determine number of classes from y_train: {e}[/bold red]", style="red")
        return None, None, None, None

    if model_type == 'randomforest':
        if 'random_state' not in model_params:
            model_params['random_state'] = global_random_state
        model = RandomForestClassifier(**model_params)
    elif model_type == 'logisticregression':
        if 'random_state' not in model_params:
            model_params['random_state'] = global_random_state
        
        # For multiclass, ensure we're using appropriate settings
        if num_class > 2:
            if 'multi_class' not in model_params:
                model_params['multi_class'] = 'multinomial'
            
            # Check if solver is compatible with multinomial
            if 'solver' not in model_params:
                model_params['solver'] = 'lbfgs'  # Default solver for multinomial
            elif model_params['solver'] in ['liblinear']:
                console.print(f"[yellow]⚠️ Warning: '{model_params['solver']}' solver does not support multinomial multi_class. Changing to 'lbfgs'.[/yellow]")
                model_params['solver'] = 'lbfgs'
                
            # Ensure max_iter is reasonably high for convergence in multiclass problems
            if 'max_iter' not in model_params:
                model_params['max_iter'] = 1000
        
        # Create the model
        model = LogisticRegression(**model_params)
    elif model_type == 'xgboost':
        if 'random_state' not in model_params:
            model_params['random_state'] = global_random_state
        
        # XGBoost specific handling for multiclass
        if num_class > 2:
            if 'objective' not in model_params:
                model_params['objective'] = 'multi:softprob'
            if 'num_class' not in model_params: # Required for multi:softprob if not inferred
                 model_params['num_class'] = num_class
            if 'eval_metric' not in model_params:
                model_params['eval_metric'] = 'mlogloss'
             # Check if y_train labels are in [0, num_class-1]
            min_label, max_label = y_train.min(), y_train.max()
            if not (min_label == 0 and max_label == num_class - 1 and y_train.nunique() == num_class):
                 console.print(f"[yellow]⚠️ Warning (XGBoost): y_train does not appear to be correctly encoded in the range [0, num_class-1].[/yellow]")
                 console.print(f"[yellow]   Min label: {min_label}, Max label: {max_label}, Unique labels: {y_train.nunique()}, Expected num_class: {num_class}.[/yellow]")
                 console.print(f"[yellow]   It is recommended to use LabelEncoder on the target column in the preprocessor.[/yellow]")

        else: # Binary case
            if 'objective' not in model_params:
                model_params['objective'] = 'binary:logistic'
            if 'eval_metric' not in model_params:
                model_params['eval_metric'] = 'logloss'
        
        model_params['use_label_encoder'] = False # Recommended for modern XGBoost
        model = xgb.XGBClassifier(**model_params)
    elif model_type == 'catboost':
        if 'random_seed' not in model_params:
            model_params['random_seed'] = global_random_state
        if 'verbose' not in model_params:
            model_params['verbose'] = 0 
        if num_class > 2 and 'loss_function' not in model_params:
            model_params['loss_function'] = 'MultiClass'
        elif num_class == 2 and 'loss_function' not in model_params: # Binary
            model_params['loss_function'] = 'Logloss'
        model = cb.CatBoostClassifier(**model_params)
    else:
        console.print(f"[bold red]ERROR: Model type '{model_type}' is not supported. Supported types: randomforest, logisticregression, xgboost, catboost.[/bold red]", style="red")
        return None, None, None, None

    # Cross-validation
    k_folds = cv_config.get('n_splits', 5)
    shuffle_cv = cv_config.get('shuffle', True)
    random_state_cv = cv_config.get('random_state', global_random_state)
    
    console.print(f"[cyan]Performing Cross-Validation ({k_folds}-fold) on training data...[/cyan]")
    try:
        # Ensure y_train is integer type for StratifiedKFold and some models
        y_train_cv = y_train.astype(int)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=shuffle_cv, random_state=random_state_cv)
        cv_scores = cross_val_score(model, X_train, y_train_cv, cv=skf, scoring='accuracy')
        console.print(f"[green]Cross-Validation mean accuracy: {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})[/green]")
    except ValueError as ve:
        console.print(f"[yellow]⚠️ Warning during Cross-Validation: {ve}[/yellow]")
        console.print(f"[yellow]   Possible cause: y_train is not in numeric/integer format or there are issues with class labels.[/yellow]")
        console.print(f"[yellow]   Proceeding with training on the complete set, but CV was skipped.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Unexpected error during Cross-Validation: {e}[/yellow]")
        console.print(f"[yellow]   Proceeding with training on the complete set, but CV was skipped.[/yellow]")

    # Model Training
    console.print(f"[cyan]Training {model_type.capitalize()} model on the complete training set...[/cyan]")
    try:
        model.fit(X_train, y_train)
        console.print(f"[green]✓ {model_type.capitalize()} model trained.[/green]")
    except Exception as e:
        console.print(f"[bold red]ERROR during model training: {e}[/bold red]", style="red")
        return None, None, None, None

    # Evaluation on Test Set
    console.print(f"[cyan]Evaluating {model_type.capitalize()} model on Test Set...[/cyan]")
    y_pred = model.predict(X_test)
    
    # Ensure y_test and y_pred are of compatible types for metrics
    try:
        y_test_eval = y_test.astype(int)
        y_pred_eval = y_pred.astype(int)
    except ValueError:
        console.print(f"[yellow]⚠️ Warning: Unable to convert y_test/y_pred to integers for evaluation. Attempting with original type.[/yellow]")
        y_test_eval = y_test
        y_pred_eval = y_pred
        # Fallback for CatBoost if it predicts string labels and y_test is numeric or vice-versa
        if isinstance(y_pred_eval, np.ndarray) and y_pred_eval.dtype == 'object':
            try:
                y_pred_eval = y_pred_eval.astype(y_test_eval.dtype)
            except ValueError:
                # Try converting y_test_eval to string if y_pred_eval seems to be string class labels
                if y_test_eval.dtype != 'object':
                    y_test_eval = y_test_eval.astype(str)

    accuracy = accuracy_score(y_test_eval, y_pred_eval)
    conf_matrix = confusion_matrix(y_test_eval, y_pred_eval)
    # Ensure labels for classification report are consistent if y_test has few unique values
    unique_labels_test = np.unique(y_test_eval)
    unique_labels_pred = np.unique(y_pred_eval)
    report_labels = sorted(list(set(unique_labels_test) | set(unique_labels_pred)))

    class_report_str = classification_report(y_test_eval, y_pred_eval, zero_division=0, labels=report_labels if len(report_labels) > 1 else None)
    class_report_dict = classification_report(y_test_eval, y_pred_eval, output_dict=True, zero_division=0, labels=report_labels if len(report_labels) > 1 else None)

    console.print(Panel(f"[bold green]Test Set Evaluation Results ({model_type.capitalize()})[/bold green]", border_style="green"))
    console.print(f"Accuracy: [bold cyan]{accuracy:.4f}[/bold cyan]")
    
    console.print("\n[bold]Classification Report:[/bold]")
    console.print(Text(class_report_str))
    
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(conf_matrix)
    
    return model, conf_matrix, accuracy, class_report_dict