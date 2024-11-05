import pandas as pd
import numpy as np
import os


def read_file(filepath):
    try:
        return pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filepath} does not contain any columns.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty

def gather_data():    
    path = "./IT3212-assignments/Assignment 1/Stocks/"
    # file = "evtc.us.txt"
    # filepaths = [path+f for f in os.listdir(path) if f.endswith('aaap.us.txt')]
    filepaths = [path+f for f in os.listdir(path)]
    # filepaths = [path+file]
    df = pd.concat(map(read_file, filepaths))

    return df

def seasonal_features(df):
    print("Creating seasonal features...")
    # Extract seasonal features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['Quarter'] = df['Date'].dt.quarter

    return df

def lag_features(df, lags):
    # Create lag features of open, high, low, close, and volume

    print("Creating lag features for...")

    feature_dict = {}

    for lag in lags:
        print(f"Lag {lag}...")

        feature_dict[f'Open_Lag_{lag}'] = df['Open'].shift(lag).fillna(method='bfill')
        feature_dict[f'High_Lag_{lag}'] = df['High'].shift(lag).fillna(method='bfill')
        feature_dict[f'Low_Lag_{lag}'] = df['Low'].shift(lag).fillna(method='bfill')
        feature_dict[f'Close_Lag_{lag}'] = df['Close'].shift(lag).fillna(method='bfill')
        feature_dict[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag).fillna(method='bfill')

    # Convert dictionary to DataFrame and concatenate with original
    features_df = pd.DataFrame(feature_dict, index=df.index)
    df = pd.concat([df, features_df], axis=1)

    return df

def statistical_features(df):
    """
    Statistics for each column of the original dataset; open, high, low, close, and volume. 
    Statistics will be for the last “trading week” (5 days of open stock market), 
    month (25 days of stock market), quartiles (one for every of the last 4 quartiles), 
    and year (stock market days in a year).
    - Mean
    - Absolute deviation
    - Volatility
    - Standard deviation
    """
    print("Creating statistical features for...")

    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_dict = {}

    for column in columns:
        print(f"Column '{column}'...")

        # Calculate the mean
        print("Calculating mean...")
        feature_dict[f'{column}_Mean_Week'] = df[column].rolling(window=5, min_periods=0).mean()
        feature_dict[f'{column}_Mean_Month'] = df[column].rolling(window=25, min_periods=0).mean()
        feature_dict[f'{column}_Mean_Quarter'] = df[column].rolling(window=63, min_periods=0).mean()
        feature_dict[f'{column}_Mean_Year'] = df[column].rolling(window=252, min_periods=0).mean()

        # Calculate the absolute deviation
        print("Calculating absolute deviation...")
        for window, suffix in zip([5, 25, 63, 252], ['Week', 'Month', 'Quarter', 'Year']):
            rolling_mean = df[column].rolling(window=window, min_periods=0).mean()
            abs_deviation = (df[column] - rolling_mean).abs().rolling(window=window, min_periods=0).mean()
            feature_dict[f'{column}_AbsDev_{suffix}'] = abs_deviation
        # feature_dict[f'{column}_AbsDev_Week'] = df[column].rolling(window=5).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        # feature_dict[f'{column}_AbsDev_Month'] = df[column].rolling(window=25).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        # feature_dict[f'{column}_AbsDev_Quarter'] = df[column].rolling(window=63).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        # feature_dict[f'{column}_AbsDev_Year'] = df[column].rolling(window=252).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        # Calculate the standard deviation
        print("Calculating standard deviation...")
        feature_dict[f'{column}_Std_Week'] = df[column].rolling(window=5, min_periods=0).std().fillna(method='bfill')
        feature_dict[f'{column}_Std_Month'] = df[column].rolling(window=25, min_periods=0).std().fillna(method='bfill')
        feature_dict[f'{column}_Std_Quarter'] = df[column].rolling(window=63, min_periods=0).std().fillna(method='bfill')
        feature_dict[f'{column}_Std_Year'] = df[column].rolling(window=252, min_periods=0).std().fillna(method='bfill')

    # Convert dictionary to DataFrame and concatenate with original
    features_df = pd.DataFrame(feature_dict, index=df.index)
    df = pd.concat([df, features_df], axis=1)

    return df

def extract_features(df):
    # Extract seasonal features
    df = seasonal_features(df)

    # Create lag features of the last 25 stock days (one month)
    lags = range(1, 26)
    df = lag_features(df, lags)

    # Extract statistical features
    df = statistical_features(df)

    return df

df = gather_data()
df = extract_features(df)
print(df.head())

# Save the preprocessed data to a CSV file
df.to_csv('./preprocessed_stock_data.csv', index=False)