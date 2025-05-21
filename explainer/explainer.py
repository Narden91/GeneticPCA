import shap
import pandas as pd

def explain_model_with_shap(model, X_train, X_test, model_type='tree'):
    """Genera e visualizza i valori SHAP per spiegare il modello."""
    print("Avvio della generazione dei valori SHAP...")
    
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model, X_train) 
    elif model_type == 'kernel': 
        X_train_summary = shap.kmeans(X_train, 10) 
        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)
    else: 
        try:
            explainer = shap.Explainer(model, X_train)
        except Exception as e:
            print(f"Errore nella creazione dell'Explainer SHAP generico: {e}. Provare a specificare model_type.")
            print("Restituzione di None per i valori SHAP.")
            return None, None

    shap_values = explainer.shap_values(X_test)
    print("Valori SHAP generati.")

    return explainer, shap_values

