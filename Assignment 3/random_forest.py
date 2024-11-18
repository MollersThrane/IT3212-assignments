import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import time
from model_base import AbstractModel
from expanding_window import ExpandingWindowByYear


class RandomForestStockModel(AbstractModel):
    def __init__(self, dataset: pd.DataFrame, initial_train_years=43, num_test_years=1):
        self.dataset = dataset.copy(deep=True)
        self.initial_train_years = initial_train_years
        self.num_test_years = num_test_years

        # Ensure 'Date' column is in datetime format
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])

        # Drop rows with missing values
        self.dataset.dropna(inplace=True)

        # Initialize ExpandingWindowByYear instance
        self.expanding_window = ExpandingWindowByYear(
            data=self.dataset,
            initial_train_years=self.initial_train_years,
            test_years=self.num_test_years,
            result_columns=["Close"]
        )

        # Initialize storage for results
        self.actuals = []
        self.predictions = []
        self.mae_scores = []
        self.rmse_scores = []
        self.r2_scores = []
        self.mape_scores = []
        self.times = []
        self.test_years = []

    def train_and_test(self):
        while True:
            try:
                # Retrieve train/test sets
                train_X, train_y, _ = self.expanding_window.train_window()
                test_X, test_y, _ = self.expanding_window.test_window()

                # Record the start time
                start_time = time.time()

                # Train and evaluate the Random Forest model
                model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
                model.fit(train_X, train_y.values.ravel())
                self.model = model
                y_pred = model.predict(test_X)

                # Calculate metrics
                mae = mean_absolute_error(test_y, y_pred)
                rmse = root_mean_squared_error(test_y, y_pred)
                r2 = r2_score(test_y, y_pred)
                mape = mean_absolute_percentage_error(test_y, y_pred)
                elapsed_time = time.time() - start_time

                # Store results
                self.actuals.extend(test_y["Close"].values)
                self.predictions.extend(y_pred)
                self.mae_scores.append(mae)
                self.rmse_scores.append(rmse)
                self.r2_scores.append(r2)
                self.mape_scores.append(mape)
                self.times.append(elapsed_time)
                self.test_years.append(self.expanding_window.current_test_start_year)

                # Extend to the next expanding window
                self.expanding_window.extend_train_window()

            except (IndexError, ValueError):
                # Stop if no more windows are available
                break

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        input_scaled = self.expanding_window.scaler.transform(input_df)
        input_scaled = self.expanding_window.pca.transform(input_scaled)
        df_pca = pd.DataFrame(input_scaled, columns=[f'PC{i+1}' for i in range(self.expanding_window.num_pca_components)])
        predictions = self.model.predict(df_pca).flatten()
        return pd.DataFrame({'Predictions': predictions}, index=input_df.index)

    def get_r2_rmse_mae_mape_per_year(self) -> pd.DataFrame:
        metrics = pd.DataFrame({
            "Year": self.test_years,
            "MAE": self.mae_scores,
            "RMSE": self.rmse_scores,
            "MAPE": self.mape_scores,
            "R2": self.r2_scores
        })
        metrics.set_index("Year", inplace=True)
        return metrics

    def plot_results(self):
        # Plot original vs predicted stock prices
        test_indices = self.dataset.iloc[-len(self.predictions):].index
        test_dates = self.dataset.loc[test_indices, 'Date']

        plt.figure(figsize=(14, 6))
        plt.plot(self.dataset['Date'], self.dataset['Close'], label="Actual Data (Training + Test)", color="green", alpha=0.6)
        plt.plot(test_dates, self.predictions, label="Predicted Data", color="orange", alpha=0.8)
        plt.title("Random Forest - Combined Actual vs Predicted Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot all metrics trends in one plot
        plt.figure(figsize=(14, 6))
        plt.plot(self.test_years, self.mae_scores, label="MAE", marker='o', color="blue")
        plt.plot(self.test_years, self.rmse_scores, label="RMSE", marker='s', color="red")
        plt.plot(self.test_years, self.mape_scores, label="MAPE", marker='^', color="orange")
        plt.plot(self.test_years, self.r2_scores, label="R²", marker='x', color="purple")
        plt.axhline(y=0, color='black', linestyle='--', label="Baseline")
        plt.title("Metrics Trends Over Test Years")
        plt.xlabel("Year")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid()
        plt.show()


# Usage Example
if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('./preprocessed_stock_data.csv')
    df.dropna(inplace=True)

    # Instantiate the RandomForestStockModel
    model = RandomForestStockModel(dataset=df, initial_train_years=43, num_test_years=1)

    # Train and test the model
    model.train_and_test()

    # Get metrics and display
    metrics = model.get_r2_rmse_mae_mape_per_year()
    print(metrics)

    # Plot the results
    model.plot_results()
