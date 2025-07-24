# main.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import all the necessary modules from the src package
from src import config
from src import data_preprocessing
from src import model_builder
from src import evaluate

def main():
    """
    Main function to run the entire data processing, model training, and evaluation pipeline.
    """
    # --- Phase 1 & 2: Data Loading and Preprocessing ---
    print("--- Starting Data Preprocessing Pipeline ---")
    raw_df, holidays_df = data_preprocessing.load_and_merge_data()
    
    if raw_df is None:
        print("Halting execution due to data loading failure.")
        return

    featured_df = data_preprocessing.create_features(raw_df, holidays_df)
    
    # This function now returns all necessary data splits and the scaler
    train_lstm_df, test_lstm_df, train_rf_df, test_rf_df, scaler = data_preprocessing.preprocess_for_modeling(featured_df)

    # --- Prepare Data for LSTM ---
    print("\n--- Preparing Data for LSTM Model ---")
    X_train_lstm, y_train_lstm = data_preprocessing.create_sequences(train_lstm_df, config.TARGET_COLUMN)
    X_test_lstm, y_test_lstm = data_preprocessing.create_sequences(test_lstm_df, config.TARGET_COLUMN)
    
    # Create PyTorch DataLoaders
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test_lstm.astype(np.float32)), 
            torch.from_numpy(y_test_lstm.astype(np.float32).reshape(-1,1))
        ), 
        batch_size=config.LSTM_BATCH_SIZE, 
        shuffle=False
    )

    # --- Prepare Data for Random Forest ---
    print("\n--- Preparing Data for Random Forest Model ---")
    X_train_rf = train_rf_df.drop(config.TARGET_COLUMN, axis=1)
    y_train_rf = train_rf_df[config.TARGET_COLUMN]
    X_test_rf = test_rf_df.drop(config.TARGET_COLUMN, axis=1)
    y_test_rf = test_rf_df[config.TARGET_COLUMN]

    # --- Phase 3: Train LSTM Model ---
    lstm_model = model_builder.train_lstm_model(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)

    # --- Phase 4: Train Random Forest Model ---
    rf_model = model_builder.train_rf_model(X_train_rf, y_train_rf)

    # --- Phase 5: Evaluate Models ---
    print("\n--- Starting Model Evaluation ---")
    evaluate.evaluate_models(lstm_model, rf_model, test_loader, X_test_rf, y_test_rf)

    print("\n--- Pipeline execution complete! ---")

if __name__ == '__main__':
    # This ensures the main function runs only when the script is executed directly
    main()