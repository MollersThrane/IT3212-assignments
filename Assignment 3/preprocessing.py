import pandas as pd
import os

def read_file(filepath):
    try:
        return pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filepath} does not contain any columns.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty

def gather_data():    
    path = "./Stocks/ge.us.txt"
    # filepaths = [path+f for f in os.listdir(path)]
    # df = pd.concat(map(read_file, filepaths))
    df = read_file(path)

    return df

def detect_and_remove_outliers(df):
    """
    Detect outliers using Z-score and IQR methods and remove data points where both methods agree.
    """
    print("Detecting and removing outliers...")
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    indices_to_remove = set()

    for column in columns:
        # Compute the differences between consecutive values
        diff = df[column].diff()

        # Compute Z-score of differences
        mean_diff = diff.mean()
        std_diff = diff.std()
        z_scores = (diff - mean_diff) / std_diff

        # Identify outliers with Z-score > threshold (e.g., 3)
        threshold_z = 3
        outliers_z = z_scores.abs() > threshold_z

        # Compute IQR of differences
        Q1 = diff.quantile(0.25)
        Q3 = diff.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_iqr = (diff < lower_bound) | (diff > upper_bound)

        # Identify outliers where both methods agree
        outliers = outliers_z & outliers_iqr

        # Collect indices of outliers to remove
        indices = df.index[outliers].tolist()
        indices_to_remove.update(indices)

    # Remove the outliers from the DataFrame
    df = df.drop(index=indices_to_remove).reset_index(drop=True)

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
    df['WeekNumber'] = df['Date'].dt.isocalendar().week
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

        feature_dict[f'Open_Lag_{lag}'] = df['Open'].shift(lag).ffill()
        feature_dict[f'High_Lag_{lag}'] = df['High'].shift(lag).ffill()
        feature_dict[f'Low_Lag_{lag}'] = df['Low'].shift(lag).ffill()
        feature_dict[f'Close_Lag_{lag}'] = df['Close'].shift(lag).ffill()
        feature_dict[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag).ffill()

    for f in feature_dict.values():
        f.fillna(0, inplace=True)

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
            abs_deviation = (df[column].shift(1) - rolling_mean).abs().rolling(window=window, min_periods=0).mean()
            feature_dict[f'{column}_AbsDev_{suffix}'] = abs_deviation

        # Calculate the standard deviation
        print("Calculating standard deviation...")
        feature_dict[f'{column}_Std_Week'] = df[column].shift(1).rolling(window=5, min_periods=0).std().ffill()
        feature_dict[f'{column}_Std_Month'] = df[column].shift(1).rolling(window=25, min_periods=0).std().ffill()
        feature_dict[f'{column}_Std_Quarter'] = df[column].shift(1).rolling(window=63, min_periods=0).std().ffill()
        feature_dict[f'{column}_Std_Year'] = df[column].shift(1).rolling(window=252, min_periods=0).std().ffill()
        
        for f in feature_dict.values():
            f.fillna(0, inplace=True)

        # Convert dictionary to DataFrame and concatenate with original
        features_df = pd.DataFrame(feature_dict, index=df.index)
        df = pd.concat([df, features_df], axis=1)

    return df

def extract_features(df):

    df.drop(columns=['OpenInt'], inplace=True)

    # Extract seasonal features
    df = seasonal_features(df)

    # Create lag features of the last 25 stock days (one month)
    lags = range(1, 26)
    df = lag_features(df, lags)

    # Extract statistical features
    df = statistical_features(df)

    return df

df = gather_data()
df = detect_and_remove_outliers(df)
df = extract_features(df)
df.drop(columns=["Open", "High", "Low", "Volume"], inplace=True)
# print(df.head())

# Save the preprocessed data to a CSV file
print("Saving preprocessed data to 'preprocessed_stock_data.csv'...")
df.to_csv('./preprocessed_stock_data.csv', index=False)
