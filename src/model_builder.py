import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from . import config


class LSTMModel(nn.Module):
    def __init__(self, vocab_sizes, emb_dims, num_numerical_features):
        super().__init__()
        self.cat_cols = list(vocab_sizes.keys())

        # one embedding per categorical column
        self.embeddings = nn.ModuleDict({
            f"cat_{col}": nn.Embedding(vocab_sizes[col], emb_dims[col])
            for col in self.cat_cols
        })

        total_embed_dim = sum(emb_dims.values())
        lstm_input_size = total_embed_dim + num_numerical_features

        self.dropout = nn.Dropout(config.LSTM_DROPOUT_PROB)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.LSTM_DROPOUT_PROB if config.LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(config.LSTM_HIDDEN_SIZE, config.LSTM_OUTPUT_SIZE)

    def forward(self, x_cat, x_num):
        # x_cat: [B, T, Ccat], x_num: [B, T, Cnum]
        embeds = [self.embeddings[f"cat_{col}"](x_cat[:, :, i]) for i, col in enumerate(self.cat_cols)]
        x_emb = torch.cat(embeds, dim=-1)                       # [B, T, sum(emb_dims)]
        x_in  = torch.cat([x_emb, x_num], dim=-1) if x_num.shape[-1] > 0 else x_emb
        x_in  = self.dropout(x_in)

        out, _ = self.lstm(x_in)                                # [B, T, H]
        last  = out[:, -1, :]                                   # last timestep
        return self.fc(last).squeeze(-1)                        # [B]


def train_lstm_model(Xc_tr, Xn_tr, y_tr, Xc_te, Xn_te, y_te, vocab_sizes, emb_dims, log_transformed):
    device = config.DEVICE

    num_num_features = Xn_tr.shape[2] if Xn_tr.ndim == 3 else 0
    model = LSTMModel(vocab_sizes, emb_dims, num_num_features).to(device)

    # datasets & loaders
    train_ds = TensorDataset(
        torch.tensor(Xc_tr, dtype=torch.long),
        torch.tensor(Xn_tr, dtype=torch.float32),
        torch.tensor(y_tr,  dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(Xc_te, dtype=torch.long),
        torch.tensor(Xn_te, dtype=torch.float32),
        torch.tensor(y_te,  dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.LSTM_BATCH_SIZE, shuffle=False)

    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR)

    for epoch in range(config.LSTM_EPOCHS):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for xc, xn, yb in train_loader:
            xc, xn, yb = xc.to(device), xn.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xc, xn)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xc.size(0)
        tr_loss /= max(1, len(train_loader.dataset))

        # ---- validate ----
        model.eval()
        va_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for xc, xn, yb in val_loader:
                xc, xn, yb = xc.to(device), xn.to(device), yb.to(device)
                pr = model(xc, xn)
                loss = crit(pr, yb)
                va_loss += loss.item() * xc.size(0)
                y_true.extend(yb.detach().cpu().numpy())
                y_pred.extend(pr.detach().cpu().numpy())
        va_loss /= max(1, len(val_loader.dataset))

        # metrics (both spaces)
        r2_log = r2_score(y_true, y_pred)
        if log_transformed:
            y_true_o = np.expm1(y_true)
            y_pred_o = np.expm1(y_pred)
            r2_o     = r2_score(y_true_o, y_pred_o)
        else:
            r2_o = r2_log

        print(f"Epoch {epoch+1}/{config.LSTM_EPOCHS} | "
              f"Train {tr_loss:.4f} | Val {va_loss:.4f} | "
              f"R2_log {r2_log:.4f} | R2_orig {r2_o:.4f}")

    return model
