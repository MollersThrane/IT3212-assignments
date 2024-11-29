import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import time

from sklearn.base import BaseEstimator, RegressorMixin
from expanding_window import ExpandingWindowByYear
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.pipeline import make_pipeline
from pygam import LinearGAM, s
from statsmodels.gam.api import GLMGam, BSplines
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


# --------------- Custom Base Models for Boosting --------------- #

# def create_nn_base_model(input_dim):
#     model = Sequential([
#         Input(shape=(input_dim,)),
#         Dense(64, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=None, learning_rate=0.001):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None  # Placeholder for the Keras model

    def build_model(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def fit(self, X, y):
        if self.model is None:
            self.model = self.build_model()
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

    def get_params(self, deep=True):
        return {'input_dim': self.input_dim, 'learning_rate': self.learning_rate}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    # if early stopping is needed:
    # def fit(self, X, y):
    #     if self.model is None:
    #         self.model = self.build_model()
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #     self.model.fit(X, y, epochs=100, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])
    #     return self



from sklearn.base import BaseEstimator, RegressorMixin
from pygam import LinearGAM, s

from pygam import LinearGAM, s
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

# class LinearGAMWrapper(BaseEstimator, RegressorMixin):
#     def __init__(self, n_features, lam=1.0):
#         self.n_features = n_features
#         self.lam = lam
#         self.model = None

#     def fit(self, X, y):
#         X = self._ensure_2d_array(X)
#         terms = sum([s(i) for i in range(self.n_features)])  # Additive terms for all features
#         self.model = LinearGAM(terms).fit(X, y)
#         self.model.lam = self.lam
#         return self

#     def predict(self, X):
#         X = self._ensure_2d_array(X)
#         return self.model.predict(X)

#     def _ensure_2d_array(self, X):
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)
#         return np.array(X)

#     def get_params(self, deep=True):
#         return {'n_features': self.n_features, 'lam': self.lam}

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self


# def create_additive_base_model():
#     return LinearGAMWrapper(terms=s(0), lam=1.0)

# Below was an attempt to use the Statsmodels GAM API, but it is not fully supported for this purpose.
# class StatsmodelsGAMWrapper(BaseEstimator, RegressorMixin):
#     def __init__(self, n_features, df=10):
#         self.n_features = n_features
#         self.df = df
#         self.model = None

#     def fit(self, X, y):
#         X = np.asarray(X)
#         y = np.asarray(y)
#         # Generate spline bases for all features
#         spline_basis = BSplines(X, df=[self.df] * self.n_features, degree=[3] * self.n_features)
#         self.model = GLMGam(y, smoother=spline_basis).fit()
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def get_params(self, deep=True):
#         return {'n_features': self.n_features, 'df': self.df}

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self

#below implements additive model without using the Statsmodels GAM:
#implementaion:
# class additive_model(BaseEstimator, RegressorMixin):
#     def __init__(self, n_features, df=10):
#         self.n_features = n_features
#         self.df = df
#         self.model = None

#     def fit(self, x, y):
#         X = np.asarray(x)
#         y = np.asarray(y)
#         # Generate spline bases for all features
#         spline_basis = BSplines(x, df=[self.df] * self.n_features, degree=[3] * self.n_features)
#         self.model = GLMGam(y, smoother=spline_basis).fit()
#         return self

#     def predict(self, x):
#         return self.model.predict(x)

#     def get_params(self, deep=True):
#         return {'n_features': self.n_features, 'df': self.df}

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self


# additive_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
# class additive_model(BaseEstimator, RegressorMixin):
#     def __init__(self):
#         self.model = None

#     def fit(self, X, y):
#         self.model = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6)).fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def get_params(self, deep=True):
#         return {}

#     def set_params(self, **params):
#         return self
        
# --------------- Custom Base Models for Boosting --------------- #







# --------------- Boosting Pipeline --------------- #

def train_and_evaluate_boosting(X_train, y_train, X_test, y_test, base_model, n_estimators=50, learning_rate=1.0):
    model = AdaBoostRegressor(estimator=base_model, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    # Calculate MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), np.finfo(float).eps)))
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    return model, y_pred, mae, mape, rmse, r2



# Below window plot function includes training period, test period, and metrics.
# def save_window_plot(window_num, train_data, test_actual, test_predicted, mae, rmse, r2, elapsed_time, output_dir, model_name):
#     """
#     Save plot for a specific window.
#     """
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(len(train_data)), train_data, label='Training Data', color='blue')
#     plt.plot(range(len(train_data), len(train_data) + len(test_actual)), test_actual, label='Actual Test Data', color='green')
#     plt.plot(range(len(train_data), len(train_data) + len(test_predicted)), test_predicted, label='Predicted Test Data', color='orange')

#     plt.title(f"{model_name} - Window {window_num}")
#     plt.xlabel("Days")
#     plt.ylabel("Close Price")
#     plt.legend()
#     plt.grid()

#     # Annotate with metrics
#     plt.text(0.05, 0.95, f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}\nTime: {elapsed_time:.2f}s",
#          transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
#          bbox=dict(facecolor='white', alpha=0.5))


#     # Save the plot
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f"window_{window_num}.png"))
#     plt.close()

# Below window plot function includes only the test period and metrics.
def save_window_plot(window_num, test_actual, test_predicted, mae, rmse, mape, r2, elapsed_time, output_dir, model_name):
    """
    Save plot for a specific window showing only the test period.
    """
    plt.figure(figsize=(12, 6))

    # Plot actual test data
    plt.plot(range(len(test_actual)), test_actual, label='Actual Test Data', color='green')
    
    # Plot predicted test data
    plt.plot(range(len(test_predicted)), test_predicted, label='Predicted Test Data', color='orange')

    # Add title and labels
    plt.title(f"{model_name} - Window {window_num} (Test Period Only)")
    plt.xlabel("Test Period (Days)")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Annotate with metrics
    plt.text(0.05, 0.95, f"MAE: {mae:.3f}\nMAPE: {mape:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}\nTime: {elapsed_time:.2f}s",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"window_{window_num}_test_only.png"))
    plt.close()


def analyze_weak_learner_weights(boosting_model, output_dir, model_name, window_num):
    """
    Analyze and visualize weak learner weights and feature importance in the AdaBoostRegressor model for each window.
    """
    if not isinstance(boosting_model, AdaBoostRegressor):
        print("This function is only applicable to AdaBoostRegressor.")
        return

    # Extract weights and weak learners
    learner_weights = boosting_model.estimator_weights_
    weak_learners = boosting_model.estimators_
    print(f"Weights of Weak Learners for {model_name} (Window {window_num}): {learner_weights}")

    # Initialize aggregated feature importance (if supported by the base model)
    aggregated_importance = None
    feature_support = False

    # Calculate feature importance for tree-based weak learners
    for idx, learner in enumerate(weak_learners):
        if hasattr(learner, "feature_importances_"):  # Check if the learner supports feature importances
            feature_support = True
            importance = learner.feature_importances_ * learner_weights[idx]  # Weighted importance
            if aggregated_importance is None:
                aggregated_importance = importance
            else:
                aggregated_importance += importance

    # Save weak learner weights for this window
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(learner_weights) + 1), learner_weights, color='skyblue')
    plt.title(f"Weak Learner Weights for {model_name} (Window {window_num})")
    plt.xlabel("Weak Learner Index")
    plt.ylabel("Weight")
    plt.grid(axis="y")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_weak_learner_weights_window_{window_num}.png"))
    plt.close()

    # Save feature importance for this window (if supported)
    if feature_support and aggregated_importance is not None:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(aggregated_importance)), aggregated_importance, color='salmon')
        plt.title(f"Aggregated Feature Importance (Weighted) for {model_name} (Window {window_num})")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.grid(axis="y")
        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance_window_{window_num}.png"))
        plt.close()
    elif not feature_support:
        print(f"Feature importance analysis is not supported for the base model used in AdaBoost (Window {window_num}).")

def generate_final_aggregate_graphs(output_dir, model_name, all_weights, all_importances):
    """
    Generate final aggregate graphs for weak learner weights and feature importance across all windows.
    """
    # Aggregate weak learner weights
    avg_weights = np.mean(all_weights, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(avg_weights) + 1), avg_weights, color='skyblue')
    plt.title(f"Final Aggregated Weak Learner Weights for {model_name}")
    plt.xlabel("Weak Learner Index")
    plt.ylabel("Average Weight")
    plt.grid(axis="y")
    plt.savefig(os.path.join(output_dir, f"{model_name}_final_weak_learner_weights.png"))
    plt.close()

    # Aggregate feature importance (if applicable)
    if all_importances:
        avg_importance = np.mean(np.array(all_importances), axis=0)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_importance)), avg_importance, color='salmon')
        plt.title(f"Final Aggregated Feature Importance for {model_name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Average Importance")
        plt.grid(axis="y")
        plt.savefig(os.path.join(output_dir, f"{model_name}_final_feature_importance.png"))
        plt.close()
        
def generate_final_combined_plot(data_df, all_predictions, all_actuals, all_times, all_mae, all_mape, all_rmse, all_r2, output_dir):
    """
    Generate a final combined plot for actual vs predicted data across all windows,
    with improved readability (e.g., formatted x-axis for years).
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual data
    plt.plot(
        data_df['Date'], data_df['Close'],
        label="Actual Data (Training + Test)", color="green", alpha=0.6
    )
    
    # Plot predictions (only for test data range)
    plt.plot(
        data_df.iloc[-len(all_predictions):]['Date'], all_predictions,
        label="Predicted Data", color="orange", alpha=0.8
    )

    # Format the x-axis to show selected years
    years = data_df['Date'].dt.year
    start_year, end_year = years.min(), years.max()
    year_range = end_year - start_year + 1
    max_labels = 10
    interval = max(1, year_range // max_labels)
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(interval))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

    plt.title("Boosting - Combined Actual vs Predicted Close Prices")
    plt.xlabel("Year")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Add overall metrics
    total_time = sum(all_times)
    avg_mae = np.mean(all_mae)
    avg_rmse = np.mean(all_rmse)
    avg_mape = np.mean(all_mape)
    avg_r2 = np.mean(all_r2)
    plt.text(
        0.05,
        0.95,
        f"Avg MAE: {avg_mae:.3f}\nAvg MAPE: {avg_mape:.3f}\nAvg RMSE: {avg_rmse:.3f}\nAvg R²: {avg_r2:.3f}\nTotal Time: {total_time:.2f}s",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor='white', alpha=0.5),
    )

    # Save the final combined plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "final_combined_plot.png"))
    plt.close()

def generate_test_period_plot(test_dates, all_actuals, all_predictions, output_dir, model_name=None, mae=None, rmse=None, r2=None, mape=None, elapsed_time=None):
    """
    Generate a plot for only the test period actual vs. predicted data, including metrics.
    """
    plt.figure(figsize=(12, 6))

    # Plot actual test data
    plt.plot(test_dates, all_actuals, label="Actual Data", color="green", alpha=0.6)

    # Plot predicted test data
    plt.plot(test_dates, all_predictions, label="Predicted Data", color="orange", alpha=0.8)

    # Add title and labels
    title = f"Test-Period: Actual vs Predicted Close Prices"
    if model_name:
        title += f" - {model_name}"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Annotate metrics on the plot
    if mae is not None and mape is not None and rmse is not None and r2 is not None and elapsed_time is not None:
        plt.text(
            0.05,
            0.95,
            f"MAE: {mae:.3f}\nMAPE: {mape:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}\nTime: {elapsed_time:.2f}s",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"test_period_plot.png" if not model_name else f"test_period_plot_{model_name}.png"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()










# Below uses a window-based x-axis for metric trend plots rather than year-based x-axis.
# def generate_metric_trend_plots(all_mae, all_rmse, all_r2, output_dir):
#     """
#     Generate metric trend plots for MAE, rmse, and R² across all windows.
#     """
#     metric_names = ['Mean Absolute Error (MAE)', 'Mean Squared Error (rmse)', 'R² Score']
#     metric_values = [all_mae, all_rmse, all_r2]

#     for metric, values in zip(metric_names, metric_values):
#         plt.figure(figsize=(12, 6))
#         plt.plot(range(1, len(values) + 1), values, marker='o', label=metric, color='red')
#         plt.title(f"Metric Trend - {metric}")
#         plt.xlabel("Expanding Window Number")
#         plt.ylabel(metric)
#         plt.legend()
#         plt.grid()
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, f"{metric.replace(' ', '_')}_trend.png"))
#         plt.close()

# below uses year-based x-axis for metric trend plots rather than window-based x-axis.
def generate_combined_metric_trend_plot(all_metrics, output_dir, years):
    """
    Generate a single combined metric trend plot for MAE, MAPE, RMSE, and R² across all windows with test-year-based x-axis.
    """
    # Unpack metrics
    all_mae, all_mape, all_rmse, all_r2 = all_metrics

    # Ensure `years` is a list of integers
    years = list(map(int, years))  # Convert all year values to integers if not already

    # Create a combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(years, all_mae, label='MAE', marker='o')
    plt.plot(years, all_mape, label='MAPE', marker='o')
    plt.plot(years, all_rmse, label='RMSE', marker='o')
    plt.plot(years, all_r2, label='R²', marker='o')

    plt.title("Combined Metric Trend (Test Periods)")
    plt.xlabel("Year")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid()

    # Set x-axis ticks as integers
    plt.xticks(ticks=years, labels=[str(year) for year in years])

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "combined_metric_trend.png"))
    plt.close()

def generate_final_aggregated_test_plot(test_dates, all_actuals, all_predictions, output_dir, model_name, all_mae, all_mape, all_rmse, all_r2, total_time):
    """
    Generate a final aggregated plot for actual vs predicted data across all test periods,
    including metrics like MAE, MAPE, RMSE, R², and Total Time.
    """
    plt.figure(figsize=(12, 6))

    # Plot actual test data
    plt.plot(test_dates, all_actuals, label="Actual Data", color="green", alpha=0.6)

    # Plot predicted test data
    plt.plot(test_dates, all_predictions, label="Predicted Data", color="orange", alpha=0.8)

    # Calculate average metrics
    avg_mae = np.mean(all_mae)
    avg_mape = np.mean(all_mape)
    avg_rmse = np.mean(all_rmse)
    avg_r2 = np.mean(all_r2)

    # Add title and labels
    plt.title(f"Final Aggregated Test Period: Actual vs Predicted - {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Annotate with metrics
    plt.text(
        0.05,
        0.95,
        f"Avg MAE: {avg_mae:.3f}\nAvg MAPE: {avg_mape:.3f}\nAvg RMSE: {avg_rmse:.3f}\nAvg R²: {avg_r2:.3f}\nTotal Time: {total_time:.2f}s",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Save the aggregated plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"final_aggregated_test_period_{model_name}.png"))
    plt.close()





def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    """
    Train and evaluate a standalone model on the given dataset.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), np.finfo(float).eps)))
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    r2 = r2_score(y_test, y_pred)

    return model, y_pred, mae, mape, rmse, r2


def standalone_pipeline(data_df, model_name, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)

    # Calculate total years present in the dataset
    total_years = data_df['Date'].dt.year.max() - data_df['Date'].dt.year.min() + 1
    initial_train_years = int(total_years * 0.8)

    expanding_window = ExpandingWindowByYear(data_df, initial_train_years=initial_train_years, test_years=1, result_columns=["Close"])
    window_num = 1
    all_mae, all_mape, all_rmse, all_r2, all_times = [], [], [], [], []
    all_actuals, all_predictions, all_test_dates = [], [], []

    while True:
        try:
            X_train, y_train, _ = expanding_window.train_window()
            X_test, y_test, test_metadata = expanding_window.test_window()

            if X_test.empty or y_test.empty:
                print(f"Window {window_num}: Empty test window. Skipping remaining windows.")
                break

            # Dynamically configure Neural Network
            if model_name == "NeuralNetwork":
                input_dim = X_train.shape[1]
                model = KerasRegressorWrapper(input_dim=input_dim)

            # Debug shapes
            print(f"Window {window_num}: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}")

            # Train and evaluate
            start_time = time.time()
            _, y_pred, mae, mape, rmse, r2 = train_and_evaluate_model(
                X_train.values, y_train.values.ravel(), X_test.values, y_test.values, model
            )
            elapsed_time = time.time() - start_time

            # Store metrics
            all_mae.append(mae)
            all_mape.append(mape)
            all_rmse.append(rmse)
            all_r2.append(r2)
            all_times.append(elapsed_time)

            # Store actuals, predictions, and test dates
            all_actuals.extend(y_test.values.ravel())
            all_predictions.extend(y_pred)
            all_test_dates.extend(data_df.iloc[y_test.index]["Date"])

            # Generate window-specific test period plot
            test_dates = data_df.iloc[y_test.index]["Date"]
            generate_test_period_plot(
                test_dates=test_dates,               # Dates for the test period
                all_actuals=y_test.values.ravel(),   # Actual values for the test period
                all_predictions=y_pred,             # Predicted values for the test period
                output_dir=output_dir,              # Output directory
                model_name=f"{model_name}_Window_{window_num}",  # Model name and window for the plot
                mae=mae, mape=mape, rmse=rmse, r2=r2, elapsed_time=elapsed_time
            )

            expanding_window.extend_train_window()
            window_num += 1

        except ValueError as ve:
            print(f"ValueError encountered in window {window_num}: {ve}")
            break
        except IndexError:
            print("No more windows to process.")
            break

    # Generate final aggregated test period plot
    if all_predictions and all_actuals and all_test_dates:
        generate_final_aggregated_test_plot(
            pd.to_datetime(all_test_dates),  # Test dates
            all_actuals,                    # Actual values
            all_predictions,                # Predicted values
            output_dir,                     # Output directory
            model_name,                     # Model name
            all_mae, all_mape, all_rmse,    # Metric lists
            all_r2,                         # R² list
            sum(all_times)                  # Total time
        )
    else:
        print("Skipping final aggregated test period plot due to empty predictions or actuals.")

    # Generate metric trend plots
    generate_combined_metric_trend_plot((all_mae, all_mape, all_rmse, all_r2), output_dir, pd.to_datetime(all_test_dates).year.unique())

    return all_mae, all_mape, all_rmse, all_r2, sum(all_times), all_predictions, all_actuals







# --------------- Comparison Function --------------- #

def compare_models(models_metrics, all_predictions, all_actuals, all_dates, output_dir, total_times):
    """
    Generate comparison graphs for all models, including time spent.
    """
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Extract the test range of dates based on the length of all_actuals
    test_dates = all_dates[-len(all_actuals):]

    # Final combined plot for actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, all_actuals, label="Actual Data", color="green", alpha=0.6)
    for model_name, predictions in all_predictions.items():
        plt.plot(test_dates, predictions, label=f"Predicted ({model_name})", alpha=0.8)

    # Format x-axis to display years as integers
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

    plt.title("Actual vs Predicted Close Prices (All Models)")
    plt.xlabel("Year")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Save the combined plot
    plt.savefig(os.path.join(comparison_dir, "predicted_vs_actual_all_models.png"))
    plt.close()

    # Metric Comparison
    for metric_name, idx in zip(["MAE", "MAPE", "RMSE", "R²"], range(3)):
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

    # Time Comparison
    plt.figure(figsize=(8, 6))
    plt.bar(list(total_times.keys()), list(total_times.values()), color='orange')
    plt.title("Time Spent Comparison")
    plt.xlabel("Model")
    plt.ylabel("Total Time (seconds)")
    plt.grid(axis="y")
    plt.savefig(os.path.join(comparison_dir, "time_comparison.png"))
    plt.close()





# --------------- Main Script --------------- #
if __name__ == "__main__":
    # Load dataset
    data_df = pd.read_csv('./Datasets/preprocessed_stock_data.csv')
    data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
    data_df = data_df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    base_output_dir = "./Plots/standalone"

    models = [
        ("LinearRegression", LinearRegression()),
        ("AdditiveModel", make_pipeline(PolynomialFeatures(degree=2), LinearRegression())),
        ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)),
        ("NeuralNetwork", None),  # Dynamic base model
        ("SVR", SVR(kernel='linear')),
    ]

    models_metrics = {}
    all_predictions = {}
    all_actuals = None
    all_dates = data_df["Date"]
    total_times = {}

    for model_name, model in models:
        print(f"Running STANDALONE with {model_name}...")
        model_output_dir = os.path.join(base_output_dir, model_name)

        # Dynamically configure Neural Network
        if model_name == "NeuralNetwork":
            input_dim = data_df.drop(columns=["Date", "Close"]).shape[1]
            model = KerasRegressorWrapper(input_dim=input_dim)

        # Run standalone pipeline
        mae, mape, rmse, r2, total_time, predictions, actuals = standalone_pipeline(
            data_df, model_name, model_output_dir, model
        )

        models_metrics[model_name] = (mae, mape, rmse, r2)
        all_predictions[model_name] = predictions
        total_times[model_name] = total_time

        if all_actuals is None:
            all_actuals = actuals

    # Generate comparison plots
    compare_models(models_metrics, all_predictions, all_actuals, all_dates, base_output_dir, total_times)