import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# --- CORRECTION: Removed trailing comma from import. ---
from sklearn.metrics import mean_squared_error, mean_absolute_error
from . import config
# This import is needed if you load a saved model for evaluation
# from .model_builder import LSTMModel

def save_prediction_plot(y_true, y_pred, filepath):
    """Saves a plot comparing predictions to actual values."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Predictions vs Actuals")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

# --- CORRECTION: Corrected function name and signature. ---
def evaluate_lstm(model, loader, device):
    """Evaluates a trained PyTorch LSTM model."""
    if model is None or loader is None:
        return {}
        
    preds, truth = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            batch_preds = model(xb).cpu().numpy().ravel()
            batch_truth = yb.cpu().numpy().ravel()
            
            # --- CORRECTION: Corrected variable names and method calls. ---
            preds.extend(np.expm1(batch_preds))
            truth.extend(np.expm1(batch_truth))
            
    preds = np.array(preds)
    truth = np.array(truth)
    
    results = {
        "lstm_mae": float(mean_absolute_error(truth, preds)),
        "lstm_rmse": float(np.sqrt(mean_squared_error(truth, preds)))
    }
    return results, truth, preds

def evaluate_rf(model, X_test, y_test):
    """Evaluates a trained scikit-learn RandomForest model."""
    if model is None or X_test is None or y_test is None:
        return {}

    # Get predictions in log space
    preds_log = model.predict(X_test)
    
    # Convert predictions and true values back to original scale
    preds_orig_scale = np.expm1(preds_log)
    y_test_orig_scale = np.expm1(y_test)
    
    # Calculate metrics
    results = {
        "rf_mae": float(mean_absolute_error(y_test_orig_scale, preds_orig_scale)),
        "rf_rmse": float(np.sqrt(mean_squared_error(y_test_orig_scale, preds_orig_scale)))
    }
    
    # Return the original scale values for optional plotting
    return results, y_test_orig_scale, preds_orig_scale