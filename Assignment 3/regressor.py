from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from expanding_window import ExpandingWindowByYear
from model_base import AbstractModel


class LinearRegressionStockModel(AbstractModel):
    def __init__(self, dataset: pd.DataFrame, initial_train_years=43, num_test_years=1):
        self.dataset = dataset.copy(deep=True)
        self.initial_train_years = initial_train_years
        self.num_test_years = num_test_years
        
        # Ensure 'Date' column is in datetime format
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        
        # Drop rows with missing values
        self.dataset.dropna(inplace=True)
        
        # Initialize the LinearRegression model
        self.model = LinearRegression()
        
        # Initialize storage for results
        self.metrics_results = pd.DataFrame(columns=['Year', 'MAE', 'MAPE', 'RMSE', 'R2'])
        self.actual_values = pd.DataFrame(columns=['Date', 'Actual'])
        self.model_predictions = pd.DataFrame(columns=['Date', 'Prediction'])

    def train_and_test(self):
        self.expanding_window = ExpandingWindowByYear(
            data=self.dataset, 
            initial_train_years=self.initial_train_years, 
            test_years=self.num_test_years, 
            result_columns=['Close']
        )
        
        keep_testing = True
        while keep_testing:
            # Retrieve train/test sets
            train_input, train_results, train_dates = self.expanding_window.train_window()
            test_input, test_results, test_dates = self.expanding_window.test_window()
            
            # Train the model and predict
            self.model.fit(train_input, train_results)
            predictions = self.model.predict(test_input)
            
            # Calculate metrics
            mae = mean_absolute_error(test_results, predictions)
            mape = mean_absolute_percentage_error(test_results, predictions)
            rmse = root_mean_squared_error(test_results, predictions)
            r2 = r2_score(test_results, predictions)
            year = test_dates.dt.year.max()
            
            # Append metrics
            temp_df = pd.DataFrame({
                'Year': [year],
                'MAE': [mae],
                'MAPE': [mape],
                'RMSE': [rmse],
                'R2': [r2]
            })
            self.metrics_results = pd.concat([self.metrics_results, temp_df], ignore_index=True)
            
            # Store actual and predicted values
            self.actual_values = pd.concat([
                self.actual_values,
                pd.DataFrame({
                    'Date': test_dates,
                    'Actual': test_results.values.flatten()
                })
            ], ignore_index=True)
            
            self.model_predictions = pd.concat([
                self.model_predictions,
                pd.DataFrame({
                    'Date': test_dates,
                    'Prediction': predictions.flatten()
                })
            ], ignore_index=True)
            
            try:
                self.expanding_window.extend_train_window()
            except Exception:
                keep_testing = False

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        input_scaled = self.expanding_window.scaler.transform(input_df)
        input_scaled = self.expanding_window.pca.transform(input_scaled)
        predictions = self.model.predict(input_scaled).flatten()
        return pd.DataFrame({'Predictions': predictions}, index=input_df.index)

    def get_r2_rmse_mae_mape_per_year(self) -> pd.DataFrame:
        self.metrics_results.set_index('Year', inplace=True)
        return self.metrics_results

    def plot_results(self):
        # Plot metrics over years
        plt.figure(figsize=(12, 8))
        plt.plot(self.metrics_results.index, self.metrics_results['MAE'], label="MAE", marker="o")
        plt.plot(self.metrics_results.index, self.metrics_results['MAPE'], label="MAPE", marker="x")
        plt.plot(self.metrics_results.index, self.metrics_results['RMSE'], label="RMSE", marker="s")
        plt.plot(self.metrics_results.index, self.metrics_results['R2'], label="R2", marker="d")
        plt.title("Metrics Over Years")
        plt.xlabel("Year")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Plot actual vs predicted values
        self.actual_values.set_index('Date', inplace=True)
        self.model_predictions.set_index('Date', inplace=True)
        
        merged_df = pd.merge(
            self.actual_values, 
            self.model_predictions, 
            left_index=True, 
            right_index=True
        )
        merged_df['Year'] = merged_df.index.year
        
        plt.figure(figsize=(14, 8))
        plt.plot(merged_df.index, merged_df['Actual'], color='orange', label="Actual", linewidth=2, zorder=1)
        plt.plot(merged_df.index, merged_df['Prediction'], color='blue', label="Predicted", linewidth=1, zorder=2)
        
        plt.title("Predicted vs Actual Values")
        plt.xlabel("Date")
        plt.ylabel("Stock Close Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("./preprocessed_stock_data.csv")
    df.dropna(inplace=True)
    model = LinearRegressionStockModel(dataset=df, initial_train_years=43, num_test_years=1)
    model.train_and_test()
    metrics = model.get_r2_rmse_mae_mape_per_year()
    print(metrics)
    model.plot_results()
