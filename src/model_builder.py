# src/model_builder.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from . import config

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(config.LSTM_NUM_LAYERS, x.size(0), config.LSTM_HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(config.LSTM_NUM_LAYERS, x.size(0), config.LSTM_HIDDEN_SIZE).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score, self.counter = score, 0
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_lstm_model(X_train, y_train, X_test, y_test):
    print("\n--- Training LSTM Model ---")
    input_size = X_train.shape[2]
    model = LSTMModel(input_size, config.LSTM_HIDDEN_SIZE, config.LSTM_NUM_LAYERS, config.LSTM_OUTPUT_SIZE, config.LSTM_DROPOUT_PROB).to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LEARNING_RATE)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32).reshape(-1,1))), batch_size=config.LSTM_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32).reshape(-1,1))), batch_size=config.LSTM_BATCH_SIZE, shuffle=False)
    
    early_stopper = EarlyStopping(patience=config.LSTM_PATIENCE, verbose=True, path=config.LSTM_MODEL_PATH)

    for epoch in range(config.LSTM_NUM_EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item() * sequences.size(0)
        
        val_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{config.LSTM_NUM_EPOCHS}, Validation Loss: {val_loss:.6f}')
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
            
    model.load_state_dict(torch.load(config.LSTM_MODEL_PATH))
    print("LSTM model trained and saved.")
    return model

# --- Random Forest Model ---
def train_rf_model(X_train, y_train):
    print("\n--- Training Random Forest Model ---")
    rf_model = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        max_features=config.RF_MAX_FEATURES,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    joblib.dump(rf_model, config.RF_MODEL_PATH)
    print("Random Forest model trained and saved.")
    return rf_model