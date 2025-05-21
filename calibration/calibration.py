from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd 

def calibrate_model(model, X_train, y_train, X_calib, method='isotonic'):
    """Calibra un modello addestrato."""
    print(f"Avvio della calibrazione del modello con il metodo: {method}...")
    
    try:

        if hasattr(model, 'is_calibrated_') and model.is_calibrated_():
            print("Il modello è già calibrato.")
            return model
        
        calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit' if hasattr(model, 'predict_proba') else 5) # cv=5 se il modello non è pre-fittato o per cross-validation
        
        if hasattr(model, 'predict_proba'):
            if X_calib is not None:
                print(f"Calibrazione del modello pre-addestrato su un set di calibrazione separato.")
                calibrated_model.fit(X_calib, y_train.loc[X_calib.index] if isinstance(X_calib, pd.DataFrame) and isinstance(y_train, pd.Series) else y_train[:len(X_calib)])
            else:
                print("Set di calibrazione non fornito. La calibrazione potrebbe essere sub-ottimale o fallire.")
                return model 
        else:
            print(f"Addestramento e calibrazione del modello con cross-validation (cv={calibrated_model.cv}).")
            calibrated_model.fit(X_train, y_train) 

        print("Calibrazione completata.")
        return calibrated_model
    except Exception as e:
        print(f"Errore durante la calibrazione: {e}")
        print("Restituzione del modello originale non calibrato.")
        return model

def get_calibrated_probabilities(calibrated_model, X_test):
    """Ottiene le probabilità calibrate per il set di test."""
    print("Ottenimento delle probabilità calibrate...")
    calibrated_probs = calibrated_model.predict_proba(X_test)
    print("Probabilità calibrate ottenute.")
    return calibrated_probs

