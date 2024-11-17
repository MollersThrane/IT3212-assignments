import numpy as np
import pandas as pd
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from model_base import AbstractModel  # Import AbstractModel
from expanding_window import ExpandingWindowByYear


class AdditiveStockModel(AbstractModel):
    def __init__(self, dataset: pd.DataFrame, initial_train_years=43, num_test_years=1):
        self.dataset = dataset.copy(deep=True)
        self.initial_train_years = initial_train_years
        self.num_test_years = num_test_years

        # Ensure 'Date' column is in datetime format
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])

        # Drop rows with missing values
        self.dataset.dropna(inplace=True)

        # Initialize the ExpandingWindowByYear instance
        self.expanding_window = ExpandingWindowByYear(
            data=self.dataset,
            initial_train_years=self.initial_train_years,
            test_years=self.num_test_years,
            result_columns=['Close']
        )

        # Initialize storage for results
        self.results = []

    def train_and_test(self):
        while True:
            try:
                # Retrieve train/test sets
                train_X, train_y = self.expanding_window.train_window()
                test_X, test_y = self.expanding_window.test_window()

                # Train the GAM model
                gam = LinearGAM(s(0))  # Using a single smooth term for simplicity
                gam.fit(train_X, train_y)

                self.model = gam

                # Predict on the test window
                predictions = gam.predict(test_X)

                # Store results
                self.results.append({
                    'train_end_year': self.expanding_window.current_train_end_year,
                    'test_start_year': self.expanding_window.current_test_start_year,
                    'predictions': predictions,
                    'actuals': test_y.values.flatten()
                })

                # Extend the training window
                self.expanding_window.extend_train_window()

            except (IndexError, ValueError):
                # Stop if there are no more test windows
                break


    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        input_scaled = self.expanding_window.scaler.transform(input_df)
        input_scaled = self.expanding_window.pca.transform(input_scaled)
        predictions = self.model.predict(input_scaled).flatten()
        return pd.DataFrame({'Predictions': predictions}, index=input_df.index)

    def get_r2_rmse_mae_mape_per_year(self) -> pd.DataFrame:
        metrics = []
        for result in self.results:
            predictions = result['predictions']
            actuals = result['actuals']
            year = result['test_start_year']

            mae = mean_absolute_error(actuals, predictions)
            mape = mean_absolute_percentage_error(actuals, predictions)
            rmse = root_mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)

            metrics.append({
                'Year': year,
                'MAE': mae,
                'MAPE': mape,
                'RMSE': rmse,
                'R2': r2
            })

        metrics_df = pd.DataFrame(metrics)
        metrics_df.set_index('Year', inplace=True)
        return metrics_df

    def plot_results(self):
        # Plot predictions vs actuals for each test window
        for result in self.results:
            plt.figure(figsize=(8, 6))
            plt.plot(result['actuals'], label='Actuals')
            plt.plot(result['predictions'], label='Predictions')
            plt.title(f"Predictions for Test Year {result['test_start_year']}")
            plt.legend()
            plt.show()

        # Plot metrics over years
        metrics_df = self.get_r2_rmse_mae_mape_per_year()
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_df.index, metrics_df['MAE'], label='MAE', marker='o')
        plt.plot(metrics_df.index, metrics_df['MAPE'], label='MAPE', marker='x')
        plt.plot(metrics_df.index, metrics_df['RMSE'], label='RMSE', marker='s')
        plt.plot(metrics_df.index, metrics_df['R2'], label='R2', marker='d')
        plt.title("Metrics Over Years")
        plt.xlabel("Year")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('./preprocessed_stock_data.csv')
    df.dropna(inplace=True)

    # Instantiate the AdditiveStockModel
    model = AdditiveStockModel(dataset=df, initial_train_years=43, num_test_years=1)

    # Train and test the model
    model.train_and_test()

    # Get metrics
    metrics = model.get_r2_rmse_mae_mape_per_year()
    print(metrics)

    # Plot results
    model.plot_results()
