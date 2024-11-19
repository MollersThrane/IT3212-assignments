import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from model_base import AbstractModel
from expanding_window import ExpandingWindowByYear


class NeuralNetworkStockModel(AbstractModel):
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
        self.predictions = []
        self.actuals = []
        self.r2_scores = []
        self.test_years = []

    def train_and_test(self):
        while True:
            try:
                # Retrieve train/test sets
                X_train, y_train, _ = self.expanding_window.train_window()
                X_test, y_test, _ = self.expanding_window.test_window()

                # Ensure non-empty test sets
                if X_test.empty or y_test.empty:
                    print("No more data available for the next test window. Stopping.")
                    break

                # Scale the data using MinMaxScaler
                scaler = MinMaxScaler()
                X_train_scaled = X_train #scaler.fit_transform(X_train)
                X_test_scaled = X_test #scaler.transform(X_test)

                # Define and compile the neural network
                model = Sequential([
                    Input(shape=(X_train_scaled.shape[1],)),
                    Dense(64, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Early stopping
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                # Train the model
                model.fit(
                    X_train_scaled, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_data=(X_test_scaled, y_test),
                    verbose=0,
                    callbacks=[early_stopping]
                )

                self.model = model

                # Predict on the test data
                y_test_pred = model.predict(X_test_scaled).flatten()

                # Store predictions and actual values
                self.predictions.extend(y_test_pred)
                self.actuals.extend(y_test.values.flatten())

                # Calculate and store metrics
                r2 = r2_score(y_test, y_test_pred)
                self.r2_scores.append(r2)
                self.test_years.append(self.expanding_window.current_test_start_year)
                print(f"R2 Score for Year {self.expanding_window.current_test_start_year}: {r2}")

                # Move to the next expanding window
                self.expanding_window.extend_train_window()

            except (IndexError, ValueError):
                # Stop if no more test data is available
                print("Reached the end of available data.")
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
        metrics = []
        for i, year in enumerate(self.test_years):
            y_actual = self.actuals[i:i + len(self.test_years)]
            y_pred = self.predictions[i:i + len(self.test_years)]

            mae = mean_absolute_error(y_actual, y_pred)
            mape = mean_absolute_percentage_error(y_actual, y_pred)
            rmse = root_mean_squared_error(y_actual, y_pred)
            r2 = self.r2_scores[i]

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
        # Ensure alignment of test dates with predictions
        test_indices = self.dataset.iloc[-len(self.predictions):].index
        test_dates = self.dataset.loc[test_indices, 'Date']

        # Plot original vs predicted stock prices
        plt.figure(figsize=(14, 6))
        plt.plot(self.dataset['Date'], self.dataset['Close'], label="Original Close Price", color="blue")
        plt.plot(test_dates, self.predictions, label="Predicted Close Price", color="orange")
        plt.title("Original vs. Predicted Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

        self.metrics_results = self.get_r2_rmse_mae_mape_per_year()
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


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("./preprocessed_stock_data.csv")
    df.dropna(inplace=True)

    # Instantiate and run the Neural Network Stock Model
    model = NeuralNetworkStockModel(dataset=df, initial_train_years=43, num_test_years=1)

    # Train and test the model
    model.train_and_test()

    # Get metrics
    metrics = model.get_r2_rmse_mae_mape_per_year()
    print(metrics)

    # Plot results
    model.plot_results()
