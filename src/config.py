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

# --- Random Forest Hyperparameters ---
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH = 30
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 4
RF_MAX_FEATURES = "sqrt"
RF_MODEL_PATH = f"{MODEL_DIR}/random_forest_model.joblib"

# --- LSTM Hyperparameters ---
LSTM_INPUT_SIZE = None  # Set at runtime from feature count
LSTM_HIDDEN_SIZE = 50
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT_PROB = 0.2
LSTM_OUTPUT_SIZE = 1
LSTM_LEARNING_RATE = 1e-3
LSTM_BATCH_SIZE = 64
LSTM_NUM_EPOCHS = 100
LSTM_PATIENCE = 10
LSTM_MODEL_PATH = f"{MODEL_DIR}/lstm_model.pt"