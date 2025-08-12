# config.py

import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directory Paths ---
RAW_DATA_DIR = "data"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
FIGURE_DIR = "reports/figures"

# --- File Paths ---
EVALUATION_FIGURE_PATH = f"{FIGURE_DIR}/prediction_vs_actuals.png"
TRAIN_FILE = "train.csv"
STORES_FILE = "stores.csv"
HOLIDAYS_FILE = "holidays_events.csv"
OIL_FILE = "oil.csv"
TRANSACTIONS_FILE = "transactions.csv"

# --- Model & Data Parameters ---
TARGET_COLUMN = "sales"
SEQUENCE_LENGTH = 30
TEST_DURATION_DAYS = 16
SKEW_THRESHOLD = 0.75  # Threshold for applying log transformation

# --- Random Forest GridSearchCV Hyperparameters ---
RF_GRID_SEARCH = True  # Enable/disable grid search
RF_CV_FOLDS = 3        # Number of cross-validation folds

# MODIFIED: Reduced grid to have 4 combinations (4 * 3 folds = 12 fits)
RF_PARAM_GRID = {
    'n_estimators': [150],            # 1 value
    'max_depth': [10, 20],            # 2 values
    'min_samples_split': [10],          # 1 value
    'min_samples_leaf': [2, 4]          # 2 values
}
RF_MODEL_PATH = f"{MODEL_DIR}/random_forest_model.joblib"

# --- LSTM Hyperparameters ---
LSTM_INPUT_SIZE = None   # Set at runtime from feature count
LSTM_HIDDEN_SIZE = 50
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT_PROB = 0.2
LSTM_OUTPUT_SIZE = 1
LSTM_LEARNING_RATE = 1e-3
LSTM_BATCH_SIZE = 64
LSTM_NUM_EPOCHS = 100
LSTM_PATIENCE = 10
LSTM_MODEL_PATH = f"{MODEL_DIR}/lstm_model.pt"