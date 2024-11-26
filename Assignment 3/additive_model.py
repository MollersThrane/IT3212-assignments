import numpy as np
import pandas as pd
from pygam import LinearGAM, s, intercept
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from expanding_window import ExpandingWindowByYear, _perform_pca

def additive_model_basic(df):
    """
    Fit an additive model to the dataset using a Generalized Additive Model (GAM).

    Parameters:
    - df (DataFrame): The dataset to fit the model to.
    """

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf with NaN
    df.dropna(inplace=True)  # Drop rows with NaN values

    # Split the data into train and test sets 80/20 based on year
    train_df = df[df["Year"] <= 2015]
    test_df = df[df["Year"] > 2015]

    # Split the data into X and y
    X_train = train_df.drop(columns=["Close"])
    y_train = train_df["Close"]

    X_test = test_df.drop(columns=["Close"])
    y_test = test_df["Close"]

    # Perform pca on the training data
    # pca_result, components, variance_ratios, pca, _ = _perform_pca(X_train)

    # Initialize model with spline term
    gam = LinearGAM(s(0))

    # Fit the model
    gam.fit(X_train, y_train)

    # Perform pca on the test data
    # test_pca = pca.transform(X_test)

    # Make predictions
    predictions = gam.predict(X_test)

    r2 = r2_score(y_test, predictions)
    print(f"R² Score: {r2:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual", color="blue")
    plt.plot(y_test.index, predictions, label="Predicted", color="orange")
    plt.title("Predictions vs Actuals")
    plt.xlabel("Date")
    plt.ylabel("Stock Close Price")
    plt.legend()
    plt.show()

def additive_model_with_expanding_window(df, gam_params=s(0)):
    """
    Fit an additive model to the dataset using a Generalized Additive Model (GAM).

    Parameters:
    - df (DataFrame): The dataset to fit the model to.

    Returns:
    - results (list): List of dictionaries containing the results for each test window.
    """
    expanding_window = ExpandingWindowByYear(
        data=df,  # Replace with your DataFrame
        initial_train_years=1,  # Starting with 5 years of training data
        test_years=1,  # Expanding by 1 year at a time
        result_columns=["Close"],  # Replace with the name of your target column
        variance_to_keep=0.80,
    )

    # Initialize a list to store results
    results = []

    while True:
        try:
            # Get the current train and test windows
            train_X, train_y, train_dates = expanding_window.train_window()
            test_X, test_y, test_dates = expanding_window.test_window()

            # Ensure train_X and train_y are NumPy arrays
            train_X = np.array(train_X)
            train_y = np.array(train_y)
            test_X = np.array(test_X)
            test_y = np.array(test_y)

            # Initialize and fit the LinearGAM model on the current training window
            # gam = LinearGAM(lam=0.1, n_splines=40)  # Initialize the model with the specified parameters
            # gam = LinearGAM(s(0) + s(1))  # Can't use this with expanding window, since ew only has 1 feature
            gam = LinearGAM(s(0, n_splines=4, lam=0.1))
            # gam = LinearGAM(s(0))

            # gam = LinearGAM(s(0, n_splines=5, lam=0.1) + s(1, n_splines=5, lam=0.1) +
            #     s(2, n_splines=5, lam=0.1) + s(3, n_splines=5, lam=0.1))

            
            # gridsearch_results = gam.gridsearch(train_X, train_y, return_scores=True)
            gam.fit(train_X, train_y)

            # Make predictions on the test window
            predictions = gam.predict(test_X)

            # Calculate metrics (MAE, MAPE, RMSE, R²)
            mae = mean_absolute_error(test_y, predictions)
            mape = mean_absolute_percentage_error(test_y, predictions)
            r2 = r2_score(test_y, predictions)
            rmse = np.sqrt(np.mean((test_y - predictions) ** 2))

            # Store results (you can calculate and store metrics here)
            results.append(
                {
                    "train_end_year": expanding_window.current_train_end_year,
                    "test_start_year": expanding_window.current_test_start_year,
                    "test_dates": test_dates,
                    "predictions": predictions,
                    "actuals": test_y,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
                    "rmse": rmse,
                }
            )

            # Extend the training window
            expanding_window.extend_train_window()

        except IndexError as e:
            # Stop if we've exhausted all available windows

            break
        except ValueError as e:
            # Stop if there's no more test data to predict on
            break

    # Results are stored in `results`, which includes predictions for each test window

    return results




# Example usage:
if __name__ == "__main__":
    df = pd.read_csv("./Datasets/preprocessed_stock_data.csv")

    # additive_model_basic(df)

    results = additive_model_with_expanding_window(df, gam_params=s(0))

    # Plot the metrics
    mae = [result["mae"] for result in results]
    mape = [result["mape"] for result in results]
    r2 = [result["r2"] for result in results]
    rmse = [result["rmse"] for result in results]

    years = [result["test_start_year"] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(years, mae, label="MAE", marker="o")
    plt.plot(years, mape, label="MAPE", marker="x")
    plt.plot(years, r2, label="R²", marker="d")
    plt.plot(years, rmse, label="RMSE", marker="s")
    plt.title("Model Metrics for Additive Model")
    plt.xlabel("Test Year")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # # Plot the results
    # for result in results:
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(result['actuals'], label='Actuals')
    #     plt.plot(result['predictions'], label='Predictions')
    #     plt.title(f"Predictions for Test Year {result['test_start_year']}")
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig(f"./Datasets/additive_results/predictions_{result['test_start_year']}.png")
    #     plt.close()  # Close the figure to avoid memory issues

    # Aggregate predictions and actuals for all test windows
    all_predictions = []
    all_actuals = []
    all_test_dates = []

    for result in results:
        all_predictions.extend(result['predictions'])
        all_actuals.extend(result['actuals'])
        all_test_dates.extend(result['test_dates'])

    # Convert test dates to a Pandas DataFrame for better handling
    all_test_dates = pd.to_datetime(all_test_dates)

    # Plot the aggregated results
    plt.figure(figsize=(12, 8))
    plt.plot(all_test_dates, all_actuals, label="Actuals", color="blue", alpha=0.7)
    plt.plot(all_test_dates, all_predictions, label="Predictions", color="orange", alpha=0.7)
    plt.title("Predictions vs Actuals for Additive Model")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Datasets/additive_results/overall_predictions.png")  # Save the figure
    plt.show()