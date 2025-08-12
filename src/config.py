import torch

# Device for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data file paths
RAW_DATA_DIR = "data"
TRAIN_FILE = "train.csv"
STORES_FILE = "stores.csv"
HOLIDAYS_FILE = "holidays_events.csv"
OIL_FILE = "oil.csv"
TRANSACTIONS_FILE = "transactions.csv"

# Target and data split settings
TARGET_COLUMN = "sales"
TEST_DURATION_DAYS = 32
SKEW_THRESHOLD = 0.75

# LSTM hyperparameters
SEQ_LEN = 28
LSTM_HIDDEN_SIZE = 40
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT_PROB = 0.4
LSTM_OUTPUT_SIZE = 1
LSTM_LR = 3e-4
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 1024

# Feature columns
CATEGORICAL_COLS = ["store_nbr", "family", "city", "state", "type", "cluster"]

NUMERICAL_COLS = [
    "onpromotion", "transactions", "dcoilwtico",
    "dayofweek", "weekofyear", "month", "year", "is_weekend", "is_holiday",
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_mean_28",
]
