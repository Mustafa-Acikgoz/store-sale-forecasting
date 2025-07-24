# src/evaluate.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os
from . import config
from .model_builder import LSTMModel # Import the model class

def evaluate_models(lstm_model, rf_model, test_loader, X_test_rf, y_test_rf):
    print("\n--- Evaluating Models ---")
    
    # --- LSTM Evaluation ---
    lstm_model.eval()
    predictions_lstm_log = []
    with torch.no_grad():
        for sequences, _ in test_loader:
            sequences = sequences.to(config.DEVICE)
            outputs = lstm_model(sequences)
            predictions_lstm_log.extend(outputs.cpu().numpy().flatten())
    
    # --- Random Forest Evaluation ---
    predictions_rf_log = rf_model.predict(X_test_rf)
    
    # --- Inverse Transform ---
    final_predictions_lstm = np.expm1(predictions_lstm_log)
    final_predictions_rf = np.expm1(predictions_rf_log)
    final_actuals = np.expm1(y_test_rf.values)
    
    min_len = min(len(final_predictions_lstm), len(final_predictions_rf))
    final_actuals = final_actuals[:min_len]
    final_predictions_lstm = final_predictions_lstm[:min_len]
    final_predictions_rf = final_predictions_rf[:min_len]

    # --- Calculate Metrics ---
    rmse_lstm = np.sqrt(mean_squared_error(final_actuals, final_predictions_lstm))
    mae_lstm = mean_absolute_error(final_actuals, final_predictions_lstm)
    rmse_rf = np.sqrt(mean_squared_error(final_actuals, final_predictions_rf))
    mae_rf = mean_absolute_error(final_actuals, final_predictions_rf)
    
    print("\n--- LSTM Model Performance ---")
    print(f"Root Mean Squared Error (RMSE): {rmse_lstm:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae_lstm:.4f}")
    print("\n--- Random Forest Model Performance ---")
    print(f"Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae_rf:.4f}\n")
    
    # --- Visualize ---
    plt.figure(figsize=(18, 8))
    plot_range = 300
    plt.plot(final_actuals[:plot_range], label='Actual Sales', color='black', linewidth=2)
    plt.plot(final_predictions_lstm[:plot_range], label='LSTM Predictions', color='blue', linestyle='--')
    plt.plot(final_predictions_rf[:plot_range], label='Random Forest Predictions', color='red', linestyle=':')
    plt.title('Sales Forecast vs. Actual Values', fontsize=16)
    plt.xlabel('Time Step (in test set)', fontsize=12)
    plt.ylabel('Number of Sales', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    if not os.path.exists(config.FIGURE_DIR):
        os.makedirs(config.FIGURE_DIR)
    plt.savefig(config.EVALUATION_FIGURE_PATH)
    print(f"Evaluation plot saved to {config.EVALUATION_FIGURE_PATH}")
    plt.show()

def load_and_evaluate():
    """Example function to load saved models and run evaluation."""
    print("Loading saved models for evaluation...")
    # Load RF model
    rf_model = joblib.load(config.RF_MODEL_PATH)
    
    # Load LSTM model
    # Note: We need to know the input size to initialize the model before loading weights
    # This is a simplification; a more robust solution would save model architecture info.
    # For now, we assume we can reconstruct the test data to get the shape.
    # This part would be called from main.py where test_loader and X_test_rf are available.
    print("This function is a placeholder. Evaluation should be called from a main script with live data.")