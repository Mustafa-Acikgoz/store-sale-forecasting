# src/evaluate.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score # Changed import
from . import config

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

def evaluate_lstm(model, loader, device, log_transformed):
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
            
            if log_transformed:
                batch_preds = np.expm1(batch_preds)
                batch_truth = np.expm1(batch_truth)
            
            preds.extend(batch_preds)
            truth.extend(batch_truth)
            
    preds = np.array(preds)
    truth = np.array(truth)
    
    # Updated metrics dictionary
    results = {
        "lstm_rmse": float(np.sqrt(mean_squared_error(truth, preds))),
        "lstm_r2_score": float(r2_score(truth, preds))
    }
    return results, truth, preds

def evaluate_rf(model, X_test, y_test, log_transformed):
    """Evaluates a trained scikit-learn RandomForest model."""
    if model is None or X_test is None or y_test is None:
        return {}

    preds_log = model.predict(X_test)
    
    if log_transformed:
        preds_orig_scale = np.expm1(preds_log)
        y_test_orig_scale = np.expm1(y_test)
    else:
        preds_orig_scale = preds_log
        y_test_orig_scale = y_test
    
    # Updated metrics dictionary
    results = {
        "rf_rmse": float(np.sqrt(mean_squared_error(y_test_orig_scale, preds_orig_scale))),
        "rf_r2_score": float(r2_score(y_test_orig_scale, preds_orig_scale))
    }
    
    return results, y_test_orig_scale, preds_orig_scale