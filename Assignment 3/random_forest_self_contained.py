import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(file_path='./Stocks/ge.us.txt'):
    """Load dataset from the specified file path."""
    df = pd.read_csv(file_path)
    return df


def seasonal_features(df):
    """Add seasonal features to the dataset."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekNumber'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    return df


def lag_features(df, lags=25):
    """Add lag features to the dataset."""
    for lag in range(1, lags + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    return df


def extract_features(df, lags=25):
    """Combine seasonal and lag features into the dataset."""
    df = seasonal_features(df)
    df = lag_features(df, lags)
    return df


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2, y_pred


def save_window_plot(window_num, train_data, test_actual, test_predicted, mae, mse, r2, elapsed_time, output_dir):
    """Save plot for a specific window."""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_data)), train_data, label='Training Data', color='blue')
    plt.plot(range(len(train_data), len(train_data) + len(test_actual)), test_actual, label='Actual Test Data', color='green')
    plt.plot(range(len(train_data), len(train_data) + len(test_predicted)), test_predicted, label='Predicted Test Data', color='orange')

    plt.title(f"Random Forest - Window {window_num}")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Add metrics summary to the plot
    plt.text(0.05, 0.95, f"MAE: {mae:.3f}\nMSE: {mse:.3f}\nR²: {r2:.3f}\nTime: {elapsed_time:.2f}s",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"window_{window_num}.png")
    plt.savefig(plot_path)
    plt.close()


def main():
    start_time = time.time()

    # Load and preprocess data
    df = load_data()
    df = extract_features(df)
    df = df.dropna().sort_values(by='Date')

    # Define training and testing windows
    initial_train_size = int(len(df) * 0.8)
    test_window_size = 252  # Approx. 1 year of trading days

    # Metrics across all windows
    all_mae, all_mse, all_r2, all_times = [], [], [], []
    all_predictions, all_actuals = [], []
    output_dir = './Plots/random_forest_self_contained'

    # Expanding window loop
    for window_num, start in enumerate(range(initial_train_size, len(df) - test_window_size, test_window_size), start=1):
        train_df = df.iloc[:start]
        test_df = df.iloc[start:start + test_window_size]

        X_train = train_df.drop(columns=['Close', 'Date'])
        y_train = train_df['Close']
        X_test = test_df.drop(columns=['Close', 'Date'])
        y_test = test_df['Close']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Measure time for training and evaluation
        start_window_time = time.time()
        mae, mse, r2, y_pred = train_and_evaluate_rf(X_train_scaled, y_train, X_test_scaled, y_test)
        elapsed_time = time.time() - start_window_time

        # Collect metrics and predictions
        all_mae.append(mae)
        all_mse.append(mse)
        all_r2.append(r2)
        all_times.append(elapsed_time)
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)

        # Save plot for the current window
        save_window_plot(window_num, train_df['Close'], y_test, y_pred, mae, mse, r2, elapsed_time, output_dir)

        # Display metrics for the window
        print(f"Window {window_num}:")
        print(f"MAE: {mae}, MSE: {mse}, R2: {r2}, Time: {elapsed_time:.2f}s")
        print("-" * 40)

    # Final combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df[:initial_train_size]['Close'])), df[:initial_train_size]['Close'], label='Training Data', color='blue')
    plt.plot(range(initial_train_size, initial_train_size + len(all_actuals)), all_actuals, label='Actual Values', color='green')
    plt.plot(range(initial_train_size, initial_train_size + len(all_predictions)), all_predictions, label='Predicted Values', color='orange')

    plt.title("Random Forest - Combined Actual vs Predicted Close Prices")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Add overall metrics summary to the plot
    total_time = sum(all_times)
    avg_mae = np.mean(all_mae)
    avg_mse = np.mean(all_mse)
    avg_r2 = np.mean(all_r2)
    plt.text(0.05, 0.95, f"Avg MAE: {avg_mae:.3f}\nAvg MSE: {avg_mse:.3f}\nAvg R²: {avg_r2:.3f}\nTotal Time: {total_time:.2f}s",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))

    # Save final plot
    plt.savefig(os.path.join(output_dir, "final_combined_plot.png"))
    plt.close()

    # Metric trends plot
    metric_names = ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R² Score']
    metric_values = [all_mae, all_mse, all_r2]
    for metric, values in zip(metric_names, metric_values):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(values) + 1), values, marker='o', label=metric, color='red')
        plt.title(f"Metric Trend - {metric}")
        plt.xlabel("Expanding Window Number")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('²', '2')}_trend.png"))
        plt.close()

    # Display overall performance
    print("Overall Performance Across Expanding Windows")
    print(f"Average MAE: {avg_mae:.3f}")
    print(f"Average MSE: {avg_mse:.3f}")
    print(f"Average R2: {avg_r2:.3f}")
    print(f"Total Time: {total_time:.2f}s")

    elapsed_time = time.time() - start_time
    print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
