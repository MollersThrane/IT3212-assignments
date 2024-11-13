import numpy as np
import pandas as pd
from pygam import LinearGAM, s
import matplotlib.pyplot as plt    

import os
from expanding_window import ExpandingWindowByYear

def additive_model(df):
    """
    Fit an additive model to the dataset using a Generalized Additive Model (GAM).

    Parameters:
    - df (DataFrame): The dataset to fit the model to.

    Returns:
    - results (list): List of dictionaries containing the results for each test window.
    """
    expanding_window = ExpandingWindowByYear(
        data=df,                # Replace with your DataFrame
        initial_train_years=1,              # Starting with 5 years of training data
        test_years=1,                       # Expanding by 1 year at a time
        result_columns=['Close'],   # Replace with the name of your target column
        variance_to_keep=0.80
    )

    # Initialize a list to store results
    results = []

    while True:
        try:
            # Get the current train and test windows
            train_X, train_y = expanding_window.train_window()
            test_X, test_y = expanding_window.test_window()

            # Initialize and fit the LinearGAM model on the current training window
            gam = LinearGAM(s(0))  # Assume we're using a single smooth term for now
            # gam.gridsearch(train_X, train_y)
            gam.fit(train_X, train_y)

            # Make predictions on the test window
            predictions = gam.predict(test_X)
            # print(test_y.values)

            # Store results (you can calculate and store metrics here)
            results.append({
                'train_end_year': expanding_window.current_train_end_year,
                'test_start_year': expanding_window.current_test_start_year,
                'predictions': predictions,
                'actuals': test_y.values.flatten()  # Flattening if test_y is a DataFrame/Series
            })

            # Extend the training window
            expanding_window.extend_train_window()

        except IndexError:
            # Stop if we've exhausted all available windows
            break
        except ValueError:
            # Stop if there's no more test data to predict on
            break

    # Results are stored in `results`, which includes predictions for each test window

    return results

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv('./preprocessed_stock_data.csv')

    # Assuming you have loaded your data into a DataFrame `df`
    results = additive_model(df)

    # Plot the results and save the figure to a file in ./additive_results/
    for result in results:

        plt.figure(figsize=(8, 6))
        plt.plot(result['actuals'].flatten(), label='Actuals')
        plt.plot(result['predictions'], label='Predictions')
        plt.title(f"Predictions for Test Year {result['test_start_year']}")
        plt.legend()
        # plt.show()
        plt.savefig(f"./additive_results/predictions_{result['test_start_year']}.png")
        plt.close()  # Close the figure to avoid memory issues