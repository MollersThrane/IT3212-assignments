import pandas as pd
import os


def read_file(filepath):
    try:
        return pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filepath} does not contain any columns.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty

def gather_data():    
    path = "./IT3212-assignments/Assignment 1/Stocks/"
    filepaths = [path+f for f in os.listdir(path)]
    df = pd.concat(map(read_file, filepaths))

    return df

def seasonal_features(df):
    """
    Extract the following seasonal features from the date column of the dataset:
    - Day of year
    - Day of month
    - Day of week
    - Week-number
    - Month
    - Year
    - Quartile
    """

    print("Creating seasonal features...")
    # Extract seasonal features
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekNumber'] = df['Date'].dt.week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter

    return df

def lag_features(df, lags):
    # Create lag features of open, high, low, close, and volume

    print("Creating lag features for...")

    feature_dict = {}

    for lag in lags:
        print(f"Lag {lag}...")

        feature_dict[f'Open_Lag_{lag}'] = df['Open'].shift(lag).bfill()
        feature_dict[f'High_Lag_{lag}'] = df['High'].shift(lag).bfill()
        feature_dict[f'Low_Lag_{lag}'] = df['Low'].shift(lag).bfill()
        feature_dict[f'Close_Lag_{lag}'] = df['Close'].shift(lag).bfill()
        feature_dict[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag).bfill()

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
    - Standard deviation
    """
    print("Creating statistical features for...")

    columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    for column in columns:
        feature_dict = {}
        print(f"Column '{column}'...")

        # Calculate the mean
        print("Calculating mean...")
        feature_dict[f'{column}_Mean_Week'] = df[column].shift(1).rolling(window=5, min_periods=0).mean()
        feature_dict[f'{column}_Mean_Month'] = df[column].shift(1).rolling(window=25, min_periods=0).mean()
        feature_dict[f'{column}_Mean_Quarter'] = df[column].shift(1).rolling(window=63, min_periods=0).mean()
        feature_dict[f'{column}_Mean_Year'] = df[column].shift(1).rolling(window=252, min_periods=0).mean()

        # Calculate the absolute deviation
        print("Calculating absolute deviation...")
        for window, suffix in zip([5, 25, 63, 252], ['Week', 'Month', 'Quarter', 'Year']):
            rolling_mean = df[column].shift(1).rolling(window=window, min_periods=0).mean()
            abs_deviation = (df[column] - rolling_mean).abs().shift(1).rolling(window=window, min_periods=0).mean()
            feature_dict[f'{column}_AbsDev_{suffix}'] = abs_deviation

        # Calculate the standard deviation
        print("Calculating standard deviation...")
        # Uses backfilling to fill NaN values, as the first values will be NaN
        feature_dict[f'{column}_Std_Week'] = df[column].shift(1).rolling(window=5, min_periods=0).std().bfill()
        feature_dict[f'{column}_Std_Month'] = df[column].shift(1).rolling(window=25, min_periods=0).std().bfill()
        feature_dict[f'{column}_Std_Quarter'] = df[column].shift(1).rolling(window=63, min_periods=0).std().bfill()
        feature_dict[f'{column}_Std_Year'] = df[column].shift(1).rolling(window=252, min_periods=0).std().bfill()
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
# print(df.head())

# Save the preprocessed data to a CSV file
print("Saving preprocessed data to 'preprocessed_stock_data.csv'...")
df.to_csv('./preprocessed_stock_data.csv', index=False)
