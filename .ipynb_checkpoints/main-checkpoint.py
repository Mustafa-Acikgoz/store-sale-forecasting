# main.py

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Import project modules
from src import config
from src import data_preprocessing as dp
from src import model_builder as mb
from src import evaluate as ev

def run_lstm_pipeline(train_df, test_df, config, model_builder, evaluator, log_transformed):
    """Prepares data, trains, and evaluates the LSTM model."""
    print("\n--- Running LSTM Pipeline ---")
    
    print("Creating sequences...")
    X_train_seq, y_train_seq = model_builder.make_sequences(train_df, config.SEQUENCE_LENGTH, config.TARGET_COLUMN)
    X_test_seq, y_test_seq = model_builder.make_sequences(test_df, config.SEQUENCE_LENGTH, config.TARGET_COLUMN)
    
    print("Training LSTM...")
    lstm_model = model_builder.train_lstm_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, config)
    
    if lstm_model is None:
        print("LSTM model training failed or was skipped. Skipping evaluation.")
        return None, {}
        
    print("Evaluating LSTM...")
    test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), batch_size=config.LSTM_BATCH_SIZE)
    lstm_results, _, _ = evaluator.evaluate_lstm(lstm_model, test_loader, config.DEVICE, log_transformed)
    
    print(f"LSTM Results: {lstm_results}")
    return lstm_model, lstm_results

def run_rf_pipeline(train_rf, test_rf, config, model_builder, evaluator, log_transformed):
    """Prepares data, trains, and evaluates the Random Forest model."""
    print("\n--- Running Random Forest Pipeline ---")
    
    train_features = [c for c in train_rf.columns if c not in ["date", config.TARGET_COLUMN]]
    X_train_rf = train_rf[train_features].values
    y_train_rf = train_rf[config.TARGET_COLUMN].values
    
    test_features = [c for c in test_rf.columns if c not in ["date", config.TARGET_COLUMN]]
    X_test_rf = test_rf[test_features].values
    y_test_rf = test_rf[config.TARGET_COLUMN].values
    
    print("Training Random Forest...")
    rf_model = model_builder.train_rf_model(X_train_rf, y_train_rf)
    
    print("Evaluating Random Forest...")
    rf_results, _, _ = evaluator.evaluate_rf(rf_model, X_test_rf, y_test_rf, log_transformed)
    
    print(f"Random Forest Results: {rf_results}")
    return rf_model, rf_results

def main():
    """Main function to run the entire ML pipeline."""
    print("--- Loading Full Dataset ---")
    data, holidays = dp.load_and_merge_data()

    # --- 1. Prepare data and run Random Forest on 2% of the data ---
    print("\n--- Preparing Data for Random Forest (2% sample) ---")
    rf_data_sample = data.sample(frac=0.02, random_state=42)
    rf_features = dp.create_features(rf_data_sample, holidays)
    _, _, rf_train, rf_test, _, rf_log_transformed = dp.preprocess_for_modeling(rf_features)
    _, rf_results = run_rf_pipeline(rf_train, rf_test, config, mb, ev, rf_log_transformed)

    # --- 2. Prepare data and run LSTM on 20% of the data ---
    print("\n--- Preparing Data for LSTM (20% sample) ---")
    lstm_data_sample = data.sample(frac=0.20, random_state=42)
    lstm_features = dp.create_features(lstm_data_sample, holidays)
    lstm_train, lstm_test, _, _, _, lstm_log_transformed = dp.preprocess_for_modeling(lstm_features)
    _, lstm_results = run_lstm_pipeline(lstm_train, lstm_test, config, mb, ev, lstm_log_transformed)

    # --- 3. Final Summary ---
    print("\n--- All Pipelines Complete ---")
    print("Final Random Forest Results:", rf_results)
    print("Final LSTM Results:", lstm_results)

if __name__ == "__main__":
    main()