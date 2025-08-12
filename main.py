from src import data_preprocessing as dp
from src import model_builder as mb
from src import config
from torch.utils.data import TensorDataset, DataLoader
import torch
from src.evaluate import evaluate_lstm, save_prediction_plot, save_model_weights

def main():
    # basic leak guard
    assert config.TARGET_COLUMN not in config.NUMERICAL_COLS, "Leak: raw target in NUMERICAL_COLS."

    # load + features
    df, holidays = dp.load_and_merge_data()
    df = dp.create_features(df, holidays)

    # sequences
    print(f"Target skew: {df[config.TARGET_COLUMN].skew():.2f}")
    Xc_tr, Xn_tr, y_tr, Xc_te, Xn_te, y_te, vocab_sizes, emb_dims, log_t = dp.make_sequences_for_embedding(df)
    print(f"Train seqs: {len(y_tr):,} | Test seqs: {len(y_te):,} | SEQ_LEN={config.SEQ_LEN}, TEST_DAYS={config.TEST_DURATION_DAYS}")
    if len(y_te) == 0:
        raise RuntimeError("No test sequences. Adjust SEQ_LEN/TEST_DURATION_DAYS.")

    # train
    model = mb.train_lstm_model(Xc_tr, Xn_tr, y_tr, Xc_te, Xn_te, y_te, vocab_sizes, emb_dims, log_t)

    # evaluate
    val_ds = TensorDataset(
        torch.tensor(Xc_te, dtype=torch.long),
        torch.tensor(Xn_te, dtype=torch.float32),
        torch.tensor(y_te,  dtype=torch.float32),
    )
    val_loader = DataLoader(val_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=False)
    metrics, y_true, y_pred = evaluate_lstm(model, val_loader, config.DEVICE, log_t)
    print(metrics)
    save_prediction_plot(y_true, y_pred, "artifacts/lstm_pred_vs_actual.png")

    # save
    save_model_weights(model, "artifacts/lstm_weights.pth")

if __name__ == "__main__":
    main()
