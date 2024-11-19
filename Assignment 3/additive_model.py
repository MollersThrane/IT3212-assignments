import numpy as np
import pandas as pd
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from model_base import AbstractModel  # Import AbstractModel
from expanding_window import ExpandingWindowByYear


class AdditiveStockModel(AbstractModel):
    def __init__(self, dataset: pd.DataFrame, initial_train_years=43, num_test_years=1):
        self.dataset = dataset.copy(deep=True)
        self.initial_train_years = initial_train_years
        self.num_test_years = num_test_years

        # Ensure 'Date' column is in datetime format
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])

        # Sort dataset by Date to ensure chronological order
        self.dataset.sort_values('Date', inplace=True)

        # Reset index after sorting
        self.dataset.reset_index(drop=True, inplace=True)

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
        self.all_predictions = pd.DataFrame(columns=['Date', 'Prediction'])
        self.all_actuals = pd.DataFrame(columns=['Date', 'Actual'])
        self.model = None  # Initialize the model attribute

    def train_and_test(self):
        while True:
            try:
                # Retrieve train/test sets
                train_X, train_y, train_dates = self.expanding_window.train_window()
                test_X, test_y, test_dates = self.expanding_window.test_window()

                # Train the GAM model
                self.model = LinearGAM(s(0, n_splines=4, lam=0.1))  # Using a single smooth term for simplicity
                self.model.fit(train_X, train_y)

                # Predict on the test window
                predictions = self.model.predict(test_X)

                # Store results
                self.results.append({
                    'train_end_year': self.expanding_window.current_train_end_year,
                    'test_start_year': self.expanding_window.current_test_start_year,
                    'predictions': predictions,
                    'actuals': test_y.values.flatten(),
                    'test_dates': test_dates  # Assuming test_dates is a pandas Series or array-like of dates
                })

                # Aggregate all predictions and actuals
                temp_predictions = pd.DataFrame({
                    'Date': test_dates,
                    'Prediction': predictions
                })
                temp_actuals = pd.DataFrame({
                    'Date': test_dates,
                    'Actual': test_y.values.flatten()
                })
                self.all_predictions = pd.concat([self.all_predictions, temp_predictions], ignore_index=True)
                self.all_actuals = pd.concat([self.all_actuals, temp_actuals], ignore_index=True)

                # Extend the training window
                self.expanding_window.extend_train_window()

            except (IndexError, ValueError) as e:
                # Stop if there are no more test windows
                print(f"Stopped training because {e}")
                break

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Assuming AbstractModel handles scaling and PCA
        input_scaled = self.expanding_window.scaler.transform(input_df)
        input_scaled = self.expanding_window.pca.transform(input_scaled)
        df_pca = pd.DataFrame(input_scaled, columns=[f'PC{i+1}' for i in range(self.expanding_window.num_pca_components)])
        predictions = self.model.predict(df_pca).flatten()
        return pd.DataFrame({'Predictions': predictions}, index=input_df.index)

    def get_r2_rmse_mae_mape_per_year(self) -> pd.DataFrame:
        metrics = []
        for result in self.results:
            predictions = result['predictions']
            actuals = result['actuals']
            year = result['test_start_year']

            mae = mean_absolute_error(actuals, predictions)
            mape = mean_absolute_percentage_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
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
        # Combine all predictions and actuals into a single DataFrame
        merged_df = pd.merge(
            self.all_actuals,
            self.all_predictions,
            on='Date',
            how='inner'
        )

        # Sort by Date to ensure chronological order
        merged_df.sort_values('Date', inplace=True)

        # Set Date as the index for plotting
        merged_df.set_index('Date', inplace=True)

        # Plot Actual vs Predicted values
        plt.figure(figsize=(14, 8))
        plt.plot(merged_df.index, merged_df['Actual'], label='Actual', color='orange', linewidth=2)
        plt.plot(merged_df.index, merged_df['Prediction'], label='Predicted', color='blue', linewidth=1)
        plt.title("Predicted vs Actual Close Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot metrics over years
        metrics_df = self.get_r2_rmse_mae_mape_per_year()
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_df.index, metrics_df['MAE'], label='MAE', marker='o')
        plt.plot(metrics_df.index, metrics_df['MAPE'], label='MAPE', marker='x')
        plt.plot(metrics_df.index, metrics_df['RMSE'], label='RMSE', marker='s')
        plt.plot(metrics_df.index, metrics_df['R2'], label='R2', marker='d')
        plt.title("Performance Metrics Over Years")
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
