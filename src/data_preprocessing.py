import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from . import config

def load_and_merge_data():
    """Loads and merges all raw data files into a single DataFrame."""
    train_df = pd.read_csv(f"{config.RAW_DATA_DIR}/{config.TRAIN_FILE}", parse_dates=["date"])
    stores_df = pd.read_csv(f"{config.RAW_DATA_DIR}/{config.STORES_FILE}")
    oil_df = pd.read_csv(f"{config.RAW_DATA_DIR}/{config.OIL_FILE}", parse_dates=["date"])
    holidays_df = pd.read_csv(f"{config.RAW_DATA_DIR}/{config.HOLIDAYS_FILE}", parse_dates=["date"])
    transactions_df = pd.read_csv(f"{config.RAW_DATA_DIR}/{config.TRANSACTIONS_FILE}", parse_dates=["date"])

    # --- CORRECTION: Initial merge was incorrect. It now correctly merges train_df with stores_df. ---
    df = pd.merge(train_df, stores_df, on="store_nbr", how="left")
    df = pd.merge(df, oil_df, on="date", how="left")
    df = pd.merge(df, transactions_df, on=["date", "store_nbr"], how="left")

    # Fill missing oil prices
    if "dcoilwtico" in df:
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
        
    return df, holidays_df

def create_features(df, holidays_df):
    """Creates time-series and rolling features."""
    df["date"] = pd.to_datetime(df["date"])
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- CORRECTION: Check for the correct target column name. ---
    if {"store_nbr", "family", config.TARGET_COLUMN}.issubset(df.columns):
        df = df.sort_values(["store_nbr", "family", "date"])
        
        # --- CORRECTION: Loop was incorrectly indented and had f-string errors. ---
        for week in (7, 14, 28):
            col_name = f"sales_rolling_mean_{week}"
            df[col_name] = (df.groupby(["store_nbr", "family"])[config.TARGET_COLUMN]
                             .transform(lambda s: s.rolling(week, min_periods=1).mean()))
    return df

def preprocess_for_modeling(df):
    """Prepares the final data for modeling: encoding, scaling, and splitting."""
    categorical_cols = ["store_nbr", "family", "city", "state", "type", "cluster"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Log transform the target variable
    df[config.TARGET_COLUMN] = np.log1p(df[config.TARGET_COLUMN])

    # --- CORRECTION: Data splitting syntax was invalid. ---
    split_date = df["date"].max() - pd.Timedelta(days=config.TEST_DURATION_DAYS)
    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()
    
    # --- CORRECTION: Removed redundant features list creation. ---
    features = [c for c in train_df.columns if c not in ["date", config.TARGET_COLUMN]]
    
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    # Convenience for Random Forest (drop rows that became NaN after feature engineering)
    train_rf = train_df.dropna()
    test_rf = test_df.dropna()

    print("Data preprocessing complete.")
    return train_df, test_df, train_rf, test_rf, scaler