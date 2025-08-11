import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Import project modules
from src import config
from src import data_preprocessing as dp
from src import model_builder as mb
from src import evaluate as ev

def run_lstm_pipeline(train_df, test_df, config, model_builder, evaluator):
    """Prepares data, trains, and evaluates the LSTM model."""
    print("\n--- Running LSTM Pipeline ---")
    
    # 1. Create sequences for LSTM
    print("Creating sequences...")
    X_train_seq, y_train_seq = model_builder.make_sequences(train_df, config.SEQUENCE_LENGTH, config.TARGET_COLUMN)
    X_test_seq, y_test_seq = model_builder.make_sequences(test_df, config.SEQUENCE_LENGTH, config.TARGET_COLUMN)
    
    # 2. Train the model
    print("Training LSTM...")
    lstm_model = model_builder.train_lstm_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, config)
    
    # 3. Evaluate the model
    print("Evaluating LSTM...")
    test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), batch_size=config.LSTM_BATCH_SIZE)
    lstm_results, lstm_true, lstm_pred = evaluator.evaluate_lstm(lstm_model, test_loader, config.DEVICE)
    
    print(f"LSTM Results: {lstm_results}")
    return lstm_model, lstm_results

def run_rf_pipeline(train_rf, test_rf, config, model_builder, evaluator):
    """Prepares data, trains, and evaluates the Random Forest model."""
    print("\n--- Running Random Forest Pipeline ---")
    
    # 1. Prepare data into X (features) and y (target) arrays
    train_features = [c for c in train_rf.columns if c not in ["date", config.TARGET_COLUMN]]
    X_train_rf = train_rf[train_features].values
    y_train_rf = train_rf[config.TARGET_COLUMN].values
    
    test_features = [c for c in test_rf.columns if c not in ["date", config.TARGET_COLUMN]]
    X_test_rf = test_rf[test_features].values
    y_test_rf = test_rf[config.TARGET_COLUMN].values
    
    # 2. Train the model
    print("Training Random Forest...")
    rf_model = model_builder.train_rf_model(X_train_rf, y_train_rf)
    
    # 3. Evaluate the model
    print("Evaluating Random Forest...")
    rf_results, _, _ = evaluator.evaluate_rf(rf_model, X_test_rf, y_test_rf)
    
    print(f"Random Forest Results: {rf_results}")
    return rf_model, rf_results

def main():
    """Main function to run the entire ML pipeline."""
    # --- Phase 1: Shared Data Preparation ---
    print("--- Starting Data Preprocessing Pipeline ---")
    data, holidays = dp.load_and_merge_data()
    features = dp.create_features(data, holidays)
    train_df, test_df, train_rf, test_rf, scaler = dp.preprocess_for_modeling(features)

    # --- Phase 2 & 3: Run Model-Specific Pipelines (Train & Evaluate) ---
    rf_model, rf_results = run_rf_pipeline(train_rf, test_rf, config, mb, ev)
    lstm_model, lstm_results = run_lstm_pipeline(train_df, test_df, config, mb, ev)

    # --- Final Summary ---
    print("\n--- All Pipelines Complete ---")
    print("Final Random Forest Results:", rf_results)
    print("Final LSTM Results:", lstm_results)

if __name__ == "__main__":
    main()