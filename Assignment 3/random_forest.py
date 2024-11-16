import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from expanding_window import ExpandingWindowByYear
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time


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
    # Load preprocessed data
    data_df = pd.read_csv('./Datasets/preprocessed_stock_data.csv')

    # Initialize the expanding window
    ew = ExpandingWindowByYear(
        data_df, initial_train_years=1, test_years=1, result_columns=["Close"]
    )

    all_mae, all_mse, all_r2, all_times = [], [], [], []
    all_actuals, all_predictions = [], []
    plot_dir = "./Plots/random_forest"
    os.makedirs(plot_dir, exist_ok=True)

    window_num = 1

    while True:
        try:
            # Get training and testing windows
            train_X, train_y, train_dates = ew.train_window()
            test_X, test_y, test_dates = ew.test_window()

            # Record time for training and evaluation
            start_time = time.time()
            mae, mse, r2, y_pred = train_and_evaluate_rf(
                train_X.values, train_y.values.ravel(), test_X.values, test_y.values.ravel()
            )
            elapsed_time = time.time() - start_time

            # Store metrics
            all_mae.append(mae)
            all_mse.append(mse)
            all_r2.append(r2)
            all_times.append(elapsed_time)

            # Accumulate predictions and actuals
            all_actuals.extend(test_y["Close"].values)
            all_predictions.extend(y_pred)

            # Save window-specific plot
            save_window_plot(window_num, train_y["Close"], test_y["Close"], y_pred, mae, mse, r2, elapsed_time, plot_dir)

            # Extend the window for the next iteration
            ew.extend_train_window()
            window_num += 1

        except IndexError:
            print("No more windows to process.")
            break
        except ValueError as e:
            print(f"Error encountered: {e}")
            break

    # Final combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        data_df['Date'], data_df['Close'],
        label="Actual Data (Training + Test)", color="green", alpha=0.6
    )
    plt.plot(
        data_df.iloc[-len(all_predictions):]['Date'], all_predictions,
        label="Predicted Data", color="orange", alpha=0.8
    )

    plt.title("Random Forest - Combined Actual vs Predicted Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Add overall metrics
    total_time = sum(all_times)
    avg_mae = np.mean(all_mae)
    avg_mse = np.mean(all_mse)
    avg_r2 = np.mean(all_r2)
    plt.text(
        0.05,
        0.95,
        f"Avg MAE: {avg_mae:.3f}\nAvg MSE: {avg_mse:.3f}\nAvg R²: {avg_r2:.3f}\nTotal Time: {total_time:.2f}s",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor='white', alpha=0.5),
    )

    plt.savefig(f"{plot_dir}/final_combined_plot.png")
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

        # Remove individual metric annotations for clarity
        plt.savefig(os.path.join(plot_dir, f"{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('²', '2')}_trend.png"))
        plt.close()

    # Print overall metrics
    print("Overall Performance Across Expanding Windows:")
    print(f"Avg MAE: {avg_mae:.3f}")
    print(f"Avg MSE: {avg_mse:.3f}")
    print(f"Avg R²: {avg_r2:.3f}")
    print(f"Total Time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
