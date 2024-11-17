import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from expanding_window import ExpandingWindowByYear
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def train_and_evaluate_boosting(X_train, y_train, X_test, y_test, base_model, n_estimators=50, learning_rate=1.0):
    """
    Train and evaluate a boosting model with a specified base algorithm.
    """
    model = AdaBoostRegressor(estimator=base_model, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, mae, mse, r2


def save_window_plot(window_num, train_data, test_actual, test_predicted, mae, mse, r2, elapsed_time, output_dir, model_name, model_params):
    """Save plot for a specific window."""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_data)), train_data, label='Training Data', color='blue')
    plt.plot(range(len(train_data), len(train_data) + len(test_actual)), test_actual, label='Actual Test Data', color='green')
    plt.plot(range(len(train_data), len(train_data) + len(test_predicted)), test_predicted, label='Boosted Predictions', color='orange')

    plt.title(f"Boosting - {model_name} (Window {window_num})\n{model_params}")
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


def boosting_pipeline(data_df, base_model, model_name, output_dir, n_estimators=50, learning_rate=1.0):
    """
    Perform boosting using an expanding window on the given dataset and base model.
    """
    os.makedirs(output_dir, exist_ok=True)
    expanding_window = ExpandingWindowByYear(data_df, initial_train_years=1, test_years=1, result_columns=["Close"])
    window_num = 1
    all_mae, all_mse, all_r2, all_times = [], [], [], []
    all_actuals, all_predictions = [], []
    model_params = str(base_model)

    # Ensure 'Date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data_df['Date']):
        data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')

    while True:
        try:
            # Get training and testing windows
            X_train, y_train, train_dates = expanding_window.train_window()
            X_test, y_test, test_dates = expanding_window.test_window()

            # Skip if test data is empty
            if X_test.empty or y_test.empty:
                print("Skipping empty test window.")
                expanding_window.extend_train_window()
                continue

            # Record time for training and evaluation
            start_time = time.time()
            _, y_pred, mae, mse, r2 = train_and_evaluate_boosting(
                X_train.values, y_train.values.ravel(), X_test.values, y_test.values,
                base_model, n_estimators=n_estimators, learning_rate=learning_rate
            )
            elapsed_time = time.time() - start_time

            # Store metrics
            all_mae.append(mae)
            all_mse.append(mse)
            all_r2.append(r2)
            all_times.append(elapsed_time)

            # Accumulate predictions and actuals
            all_actuals.extend(y_test.values.ravel())
            all_predictions.extend(y_pred)

            # Save window-specific plot
            save_window_plot(
                window_num, y_train.values.ravel(), y_test.values.ravel(), y_pred, 
                mae, mse, r2, elapsed_time, output_dir, model_name, model_params
            )

            # Extend training window
            expanding_window.extend_train_window()
            window_num += 1

        except IndexError:
            print("No more windows to process.")
            break
        except ValueError as e:
            print(f"Error encountered: {e}")
            break

    # Return empty results if no windows were processed
    if not all_mae:
        print(f"No valid windows processed for {model_name}.")
        return [], [], [], 0

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

    # Format the x-axis to show only selected years
    years = data_df['Date'].dt.year
    start_year, end_year = years.min(), years.max()
    year_range = end_year - start_year + 1
    max_labels = 10
    interval = max(1, year_range // max_labels)
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(interval))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

    plt.title(f"{model_name} - Combined Actual vs Predicted Close Prices\n{model_params}")
    plt.xlabel("Year")
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

    plt.savefig(f"{output_dir}/final_combined_plot.png")
    plt.close()

    return all_mae, all_mse, all_r2, total_time






if __name__ == "__main__":
    # Load preprocessed data
    data_df = pd.read_csv('./Datasets/preprocessed_stock_data.csv')

    # Define output directory
    base_output_dir = "./Plots/boosting"

    # Define base models for boosting
    base_models = [
        ("DecisionTree", DecisionTreeRegressor(max_depth=3)),
        ("LinearRegression", LinearRegression()),
        ("SVR", SVR(kernel='linear')),
        ("NeuralNetwork", MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500)),
        ("AdditiveModel", make_pipeline(PolynomialFeatures(degree=2), LinearRegression())),
    ]

    models_metrics = {}
    for model_name, model in base_models:
        print(f"Running Boosting with {model_name}...")
        model_output_dir = os.path.join(base_output_dir, model_name)
        mae, mse, r2, total_time = boosting_pipeline(data_df, model, model_name, model_output_dir)
        models_metrics[model_name] = (mae, mse, r2)
        print(f"{model_name}: Avg MAE = {np.mean(mae):.3f}, Avg R² = {np.mean(r2):.3f}, Total Time = {total_time:.2f}s")

    # Compare all models
    compare_models(models_metrics, base_output_dir)
