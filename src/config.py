# src/config.py

import torch

# --- Directory Paths ---
# Define paths relative to the project's root directory.
RAW_DATA_DIR = 'data'
PROCESSED_DATA_DIR = 'data/processed/'
MODEL_DIR = 'models/'
FIGURE_DIR = 'reports/figures/'

# --- Data Files ---
TRAIN_FILE = 'train.txt'
STORES_FILE = 'stores.csv'
OIL_FILE = 'oil.csv'
HOLIDAYS_FILE = 'holidays_events.csv'
TRANSACTIONS_FILE = 'transactions.csv'

# --- Data Processing Parameters ---
TEST_DURATION_DAYS = 16
SEQUENCE_LENGTH = 30  # For LSTM: use the last 30 days to predict the next day

# --- Model Parameters ---
TARGET_COLUMN = 'sales'

# LSTM Model Hyperparameters
LSTM_HIDDEN_SIZE = 50
LSTM_NUM_LAYERS = 2
LSTM_OUTPUT_SIZE = 1
LSTM_DROPOUT_PROB = 0.2
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 64
LSTM_NUM_EPOCHS = 100
LSTM_PATIENCE = 10
LSTM_MODEL_PATH = f"{MODEL_DIR}lstm_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Random Forest Model Hyperparameters
# These are example parameters; ideally, they would be the best found from RandomizedSearchCV.
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH = 30
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 4
RF_MAX_FEATURES = 'sqrt'
RF_MODEL_PATH = f"{MODEL_DIR}random_forest_model.joblib"

# --- Evaluation ---
EVALUATION_FIGURE_PATH = f"{FIGURE_DIR}predictions_vs_actuals.png"
