import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from expanding_window import ExpandingWindowByYear
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from pygam import LinearGAM, s


# --------------- Custom Base Models for Boosting --------------- #

def create_nn_base_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_additive_base_model():
    return LinearGAM(s(0))


# --------------- Boosting Pipeline --------------- #

def train_and_evaluate_boosting(X_train, y_train, X_test, y_test, base_model, n_estimators=50, learning_rate=1.0):
    model = AdaBoostRegressor(estimator=base_model, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, mae, mse, r2


def save_window_plot(window_num, train_data, test_actual, test_predicted, mae, mse, r2, elapsed_time, output_dir, model_name):
    """
    Save plot for a specific window.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_data)), train_data, label='Training Data', color='blue')
    plt.plot(range(len(train_data), len(train_data) + len(test_actual)), test_actual, label='Actual Test Data', color='green')
    plt.plot(range(len(train_data), len(train_data) + len(test_predicted)), test_predicted, label='Predicted Test Data', color='orange')

    plt.title(f"{model_name} - Window {window_num}")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Annotate with metrics
    plt.text(0.05, 0.95, f"MAE: {mae:.3f}\nMSE: {mse:.3f}\nR²: {r2:.3f}\nTime: {elapsed_time:.2f}s",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"window_{window_num}.png"))
    plt.close()


def boosting_pipeline(data_df, model_name, output_dir, base_model, n_estimators=50, learning_rate=1.0):
    os.makedirs(output_dir, exist_ok=True)
    expanding_window = ExpandingWindowByYear(data_df, initial_train_years=1, test_years=1, result_columns=["Close"])
    window_num = 1
    all_mae, all_mse, all_r2, all_times = [], [], [], []
    all_actuals, all_predictions = [], []

    while True:
        try:
            X_train, y_train, _ = expanding_window.train_window()
            X_test, y_test, _ = expanding_window.test_window()

            if X_test.empty or y_test.empty:
                print(f"Window {window_num}: Empty test window. Skipping.")
                expanding_window.extend_train_window()
                continue

            # Configure model dynamically
            if model_name == "NeuralNetwork":
                input_dim = X_train.shape[1]
                base_model = create_nn_base_model(input_dim)
            elif model_name == "AdditiveModel":
                base_model = create_additive_base_model()

            # Train and evaluate
            start_time = time.time()
            _, y_pred, mae, mse, r2 = train_and_evaluate_boosting(
                X_train.values, y_train.values.ravel(), X_test.values, y_test.values, base_model, n_estimators, learning_rate
            )
            elapsed_time = time.time() - start_time

            # Save window-specific plot
            save_window_plot(window_num, y_train.values.ravel(), y_test.values.ravel(), y_pred,
                             mae, mse, r2, elapsed_time, output_dir, model_name)

            # Store metrics
            all_mae.append(mae)
            all_mse.append(mse)
            all_r2.append(r2)
            all_times.append(elapsed_time)
            all_actuals.extend(y_test.values.ravel())
            all_predictions.extend(y_pred)

            expanding_window.extend_train_window()
            window_num += 1

        except IndexError:
            print("No more windows to process.")
            break

    return all_mae, all_mse, all_r2, sum(all_times), all_predictions, all_actuals


# --------------- Comparison Function --------------- #

def compare_models(models_metrics, all_predictions, all_actuals, all_dates, output_dir):
    """
    Generate comparison graphs for all models.
    """
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Final combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(all_dates, all_actuals, label="Actual Data", color="green", alpha=0.6)
    for model_name, predictions in all_predictions.items():
        plt.plot(all_dates[-len(predictions):], predictions, label=f"Predicted ({model_name})", alpha=0.8)
    plt.title("Actual vs Predicted Close Prices (All Models)")
    plt.xlabel("Year")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(comparison_dir, "predicted_vs_actual_all_models.png"))
    plt.close()

    # Metric Comparison
    for metric_name, idx in zip(["MAE", "MSE", "R²"], range(3)):
        metric_values = [np.mean(metrics[idx]) for metrics in models_metrics.values()]
        model_names = list(models_metrics.keys())
        plt.figure(figsize=(8, 6))
        plt.bar(model_names, metric_values)
        plt.title(f"{metric_name} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric_name)
        plt.grid(axis="y")
        plt.savefig(os.path.join(comparison_dir, f"{metric_name.lower()}_comparison.png"))
        plt.close()


# --------------- Main Script --------------- #

if __name__ == "__main__":
    data_df = pd.read_csv('./Datasets/preprocessed_stock_data.csv')
    base_output_dir = "./Plots/boosting"

    models = [
        ("RandomForest", (RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42))),
        ("LinearRegression", LinearRegression()),
        ("SVR", SVR(kernel='linear')),
        ("NeuralNetwork", None),
        ("AdditiveModel", None),
    ]

    models_metrics = {}
    all_predictions = {}
    all_actuals = None
    all_dates = data_df["Date"]

    for model_name, base_model in models:
        print(f"Running BOOSTING with {model_name}...")
        model_output_dir = os.path.join(base_output_dir, model_name)

        mae, mse, r2, total_time, predictions, actuals = boosting_pipeline(
            data_df, model_name, model_output_dir, base_model=base_model
        )

        models_metrics[model_name] = (mae, mse, r2)
        all_predictions[model_name] = predictions
        if all_actuals is None:
            all_actuals = actuals

    # Generate comparison plots
    compare_models(models_metrics, all_predictions, all_actuals, all_dates, base_output_dir)
