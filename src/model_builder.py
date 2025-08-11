# src/model_builder.py
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from . import config

class EarlyStopping:
    """Stops training when validation loss doesn't improve."""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None

    def step(self, val_loss, model):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True
        return False

def make_sequences(df, seq_len, target_col):
    """Creates sequences for LSTM model."""
    feat_cols = [c for c in df.columns if c not in ["date", target_col]]
    X = df[feat_cols].values.astype("float32")
    y = df[target_col].values.astype("float32").reshape(-1, 1)
    
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])

    return torch.tensor(np.stack(X_seq)), torch.tensor(np.stack(y_seq))

class LSTMModel(nn.Module):
    """Defines the LSTM model architecture."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        out, _ = self.lstm(X)
        last_timestep_out = out[:, -1, :]
        pred = self.fc(last_timestep_out)
        return pred

def train_lstm_model(train_seq, train_y, val_seq, val_y, config):
    """Trains the LSTM model."""
    model = LSTMModel(
        input_size=train_seq.shape[2],
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        output_size=config.LSTM_OUTPUT_SIZE,
        dropout=config.LSTM_DROPOUT_PROB
    ).to(config.DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=config.LSTM_LEARNING_RATE)
    crit = nn.MSELoss()
    stopper = EarlyStopping(patience=config.LSTM_PATIENCE)
    
    train_loader = DataLoader(TensorDataset(train_seq, train_y), batch_size=config.LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_seq, val_y), batch_size=config.LSTM_BATCH_SIZE, shuffle=False)

    for epoch in range(config.LSTM_NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                pred = model(xb)
                val_loss += crit(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{config.LSTM_NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if stopper.step(val_loss, model):
            break

    if stopper.best_state:
        model.load_state_dict(stopper.best_state)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), config.LSTM_MODEL_PATH)
        print(f"Best model saved to {config.LSTM_MODEL_PATH}")

    return model

def train_rf_model(X_train, y_train):
    """Initializes and trains the Random Forest model using GridSearchCV."""
    if not config.RF_GRID_SEARCH:
        print("Skipping Grid Search. Using default Random Forest.")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        print("Starting GridSearchCV for Random Forest...")
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=config.RF_PARAM_GRID,
            cv=config.RF_CV_FOLDS,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters found: {grid_search.best_params_}")
        model = grid_search.best_estimator_

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(model, config.RF_MODEL_PATH)
    print(f"Best Random Forest model saved to {config.RF_MODEL_PATH}")
    return model