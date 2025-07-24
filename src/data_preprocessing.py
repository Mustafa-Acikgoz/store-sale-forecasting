# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from . import config

def load_and_merge_data():
    """Loads all raw data files and merges them into a single DataFrame."""
    print("Loading and merging data...")
    try:
        train_df = pd.read_csv(f"{config.RAW_DATA_DIR}{config.TRAIN_FILE}", sep='\t', parse_dates=['date'], dayfirst=True)
        stores_df = pd.read_csv(f"{config.RAW_DATA_DIR}{config.STORES_FILE}")
        oil_df = pd.read_csv(f"{config.RAW_DATA_DIR}{config.OIL_FILE}", parse_dates=['date'])
        holidays_df = pd.read_csv(f"{config.RAW_DATA_DIR}{config.HOLIDAYS_FILE}", parse_dates=['date'])
        transactions_df = pd.read_csv(f"{config.RAW_DATA_DIR}{config.TRANSACTIONS_FILE}", parse_dates=['date'])
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your data files are in the '{config.RAW_DATA_DIR}' directory.")
        return None, None

    data = pd.merge(train_df, stores_df, on='store_nbr', how='left')
    data = pd.merge(data, oil_df, on='date', how='left')
    data = pd.merge(data, transactions_df, on=['date', 'store_nbr'], how='left')
    
    data['dcoilwtico'] = data['dcoilwtico'].ffill().bfill()
    print("Data loading and merging complete.")
    return data, holidays_df

def create_features(df, holidays_df):
    """Engineers new features from the existing data."""
    print("Creating features...")
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    holidays_national = holidays_df[holidays_df['locale'] == 'National']
    df['is_national_holiday'] = df['date'].isin(holidays_national['date']).astype(int)

    df.sort_values(by=['store_nbr', 'family', 'date'], inplace=True)
    for lag in [7, 14, 28]:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
    for window in [7, 14, 28]:
        df[f'sales_rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(window).mean()
    
    df.fillna(0, inplace=True)
    print("Feature creation complete.")
    return df

def preprocess_for_modeling(df):
    """Handles final preprocessing steps for modeling."""
    print("Preprocessing data for modeling...")
    categorical_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df[config.TARGET_COLUMN] = np.log1p(df[config.TARGET_COLUMN])

    split_date = df['date'].max() - pd.to_timedelta(days=config.TEST_DURATION_DAYS)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()
    train_df.drop('date', axis=1, inplace=True)
    test_df.drop('date', axis=1, inplace=True)

    features = [col for col in train_df.columns if col != config.TARGET_COLUMN]
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    train_rf = train_df.dropna()
    test_rf = test_df.dropna()
    
    print("Data preprocessing complete.")
    return train_df, test_df, train_rf, test_rf, scaler

def create_sequences(df, target_col):
    """Creates 3D sequences for the LSTM model."""
    X, y = [], []
    feature_cols = [col for col in df.columns if col != target_col]
    X_data = df[feature_cols].values
    y_data = df[target_col].values

    for i in range(len(df) - config.SEQUENCE_LENGTH):
        X.append(X_data[i:(i + config.SEQUENCE_LENGTH)])
        y.append(y_data[i + config.SEQUENCE_LENGTH])
    
    return np.array(X), np.array(y)