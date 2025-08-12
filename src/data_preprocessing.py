import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and merge all CSV files
def load_and_merge_data():
    train        = pd.read_csv("data/train.csv", parse_dates=["date"])
    stores       = pd.read_csv("data/stores.csv")
    oil          = pd.read_csv("data/oil.csv", parse_dates=["date"])
    holidays     = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv("data/transactions.csv", parse_dates=["date"])

    df = (
        train
        .merge(stores, on="store_nbr", how="left")
        .merge(transactions, on=["date", "store_nbr"], how="left")
        .merge(oil, on="date", how="left")
    )

    if "dcoilwtico" in df.columns:
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    return df, holidays

# Columns that define a time series group
def _group_keys(df):
    if {"store_nbr", "family"}.issubset(df.columns):
        return ["store_nbr", "family"]
    elif "store_nbr" in df.columns:
        return ["store_nbr"]
    else:
        return []

# Add calendar, holiday, and time-series features
def create_features(df, holidays):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["dayofweek"]  = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    h = holidays[["date"]].drop_duplicates().copy()
    h["is_holiday"] = 1
    df = df.merge(h, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    if "dcoilwtico" in df.columns:
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    keys = _group_keys(df)
    tgt  = "sales"
    sort_cols = ([*keys, "date"] if keys else ["date"])
    df = df.sort_values(sort_cols)

    if keys:
        grp = df.groupby(keys)[tgt]
        for lag in (7, 14, 28):
            df[f"sales_lag_{lag}"] = grp.shift(lag)
    else:
        for lag in (7, 14, 28):
            df[f"sales_lag_{lag}"] = df[tgt].shift(lag)

    if keys:
        base = df.groupby(keys)[tgt].shift(1)
        gid = df[keys].astype(str).agg("|".join, axis=1)
        for window in (7, 28):
            df[f"sales_rolling_mean_{window}"] = (
                base.groupby(gid)
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
            )
    else:
        base = df[tgt].shift(1)
        for window in (7, 28):
            df[f"sales_rolling_mean_{window}"] = base.rolling(window, min_periods=1).mean()

    return df

# Prepare sequences for LSTM with embeddings
def make_sequences_for_embedding(df, seq_len=28):
    cat_cols = ["store_nbr", "family", "city", "state", "type", "cluster"]
    num_cols = [
        "onpromotion", "transactions", "dcoilwtico",
        "dayofweek", "weekofyear", "month", "year", "is_weekend", "is_holiday",
        "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_rolling_mean_7", "sales_rolling_mean_28",
    ]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    split_date = df["date"].max() - pd.Timedelta(days=32)
    train_df = df[df["date"] <= split_date].copy()
    test_df  = df[df["date"] >  split_date].copy()

    log_transformed = False
    if train_df["sales"].skew() > 0.75:
        log_transformed = True
        for _df in (train_df, test_df):
            _df["sales"] = np.log1p(_df["sales"])

    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols].fillna(0))
    test_df[num_cols]  = scaler.transform(test_df[num_cols].fillna(0))

    def build_sequences(data):
        Xc_list, Xn_list, y_list = [], [], []
        for (store_id, fam), block in data.groupby(["store_nbr", "family"], sort=False):
            block = block.sort_values("date")
            cat_data = block[cat_cols].values
            num_data = block[num_cols].values
            target   = block["sales"].values
            n = len(block)
            if n <= seq_len:
                continue
            for i in range(n - seq_len):
                Xc_list.append(cat_data[i:i + seq_len])
                Xn_list.append(num_data[i:i + seq_len])
                y_list.append(target[i + seq_len])
        return np.array(Xc_list), np.array(Xn_list), np.array(y_list)

    Xc_tr, Xn_tr, y_tr = build_sequences(train_df)
    Xc_te, Xn_te, y_te = build_sequences(test_df)

    vocab_sizes = {col: int(df[col].nunique()) for col in cat_cols}
    emb_dims    = {col: int(min(50, (n // 2) + 1)) for col, n in vocab_sizes.items()}

    return Xc_tr, Xn_tr, y_tr, Xc_te, Xn_te, y_te, vocab_sizes, emb_dims, log_transformed
