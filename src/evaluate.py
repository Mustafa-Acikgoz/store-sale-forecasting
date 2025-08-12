import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Save predictions vs actual values as a plot
def save_prediction_plot(y_true, y_pred, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Predictions vs Actuals")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

# Evaluate trained LSTM model on a dataset
def evaluate_lstm(model, loader, device, log_transformed):
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for xb_cat, xb_num, yb in loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            yb = yb.to(device)
            yp = model(xb_cat, xb_num).detach().cpu().numpy().ravel()
            yt = yb.detach().cpu().numpy().ravel()
            if log_transformed:
                yp = np.expm1(yp)
                yt = np.expm1(yt)
            preds.append(yp)
            truth.append(yt)

    preds = np.concatenate(preds) if preds else np.array([])
    truth = np.concatenate(truth) if truth else np.array([])
    if preds.size == 0 or truth.size == 0:
        return {"lstm_rmse": None, "lstm_r2_score": None}, truth, preds

    results = {
        "lstm_rmse": float(np.sqrt(mean_squared_error(truth, preds))),
        "lstm_r2_score": float(r2_score(truth, preds)),
    }
    return results, truth, preds

# Save model weights to a file
def save_model_weights(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")
