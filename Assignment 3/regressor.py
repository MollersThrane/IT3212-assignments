from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from expanding_window import ExpandingWindowByYear

df = pd.read_csv("./preprocessed_stock_data.csv")
df.dropna(inplace=True)
print(df)

# Create models with unique names for PolynomialFeatures
models = [("LinearRegression", LinearRegression())]

# # Add Polynomial models with different degrees
# for degree in range(2, 5):
#     model_name = f"PolynomialRegression_Degree_{degree}"
#     models.append((model_name, make_pipeline(PolynomialFeatures(degree), LinearRegression())))

# DataFrame to store metrics after each test window
metrics_results = pd.DataFrame(columns=['Model', 'Iteration', 'MAE', 'MAPE', 'RMSE', 'R2'])
actual_values = pd.DataFrame(columns=['Actual', 'Model', 'Iteration'])
model_predictions = pd.DataFrame(columns=['Prediction', 'Model', 'Iteration'])


# Train and evaluate each model
for model_name, model in models:
    data = df.copy(deep=True)
    e_w = ExpandingWindowByYear(data=data, initial_train_years=1, test_years=1, result_columns=['Close'])
    
    keep_testing = True
    iteration = 1
    while keep_testing:
        # Get the train and test sets for the current window
        train_input, train_results = e_w.train_window()
        test_input, test_results = e_w.test_window()

        # Fit the model and make predictions
        model.fit(train_input, train_results)
        predictions = model.predict(test_input)

        # Calculate metrics for the current test window
        mae = mean_absolute_error(test_results, predictions)
        mape = mean_absolute_percentage_error(test_results, predictions)
        rmse = root_mean_squared_error(test_results, predictions)
        r2 = r2_score(test_results, predictions)

        # Create a temporary DataFrame with the current metrics to concatenate
        temp_df = pd.DataFrame({
            'Model': [model_name],
            'Iteration': [iteration],
            'MAE': [mae],
            'MAPE': [mape],
            'RMSE': [rmse],
            'R2': [r2]
        })

        # Concatenate the temporary DataFrame to metrics_results
        metrics_results = pd.concat([metrics_results, temp_df], ignore_index=True)

        # Store actual and predicted values for visualization
        temp_actual_df = pd.DataFrame({
            'Actual': test_results.values.flatten(),
            'Model': model_name,
            'Iteration': iteration
        })
        actual_values = pd.concat([actual_values, temp_actual_df], ignore_index=True)
        
        temp_pred_df = pd.DataFrame({
            'Prediction': predictions.flatten(),
            'Model': model_name,
            'Iteration': iteration
        })
        model_predictions = pd.concat([model_predictions, temp_pred_df], ignore_index=True)

        # Try to extend the train window for the next iteration
        try:
            e_w.extend_train_window()
            iteration += 1
        except Exception:
            keep_testing = False

# Display metrics DataFrame
print(metrics_results)

# Plot the metrics over iterations for each model
for model_name in metrics_results['Model'].unique():
    model_data = metrics_results[metrics_results['Model'] == model_name]
    
    plt.figure(figsize=(12, 8))
    plt.plot(model_data['Iteration'], model_data['MAE'], label="MAE", marker="o")
    plt.plot(model_data['Iteration'], model_data['MAPE'], label="MAPE", marker="x")
    plt.plot(model_data['Iteration'], model_data['RMSE'], label="RMSE", marker="s")
    plt.plot(model_data['Iteration'], model_data['R2'], label="R2", marker="d")
    plt.title(f"Metrics Over Time for {model_name}")
    plt.xlabel("Iteration (One iteration = one year)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.show()

# Reset the index of actual_values and model_predictions to use as RowIndex
actual_values = actual_values.reset_index().rename(columns={'index': 'RowIndex'})
model_predictions = model_predictions.reset_index().rename(columns={'index': 'RowIndex'})

# Define colors for plotting
color_actual = 'orange'  # Color for actual values
color_pred1 = 'blue'     # First color for predicted lines
color_pred2 = 'green'    # Second color for predicted lines

# Plot predicted vs actual values for each model with two alternating colors for iterations
for model_name in actual_values['Model'].unique():
    model_actuals = actual_values[actual_values['Model'] == model_name]
    model_preds = model_predictions[model_predictions['Model'] == model_name]
    
    # Ensure both DataFrames have the same RowIndex
    model_preds = model_preds.sort_values('RowIndex').reset_index(drop=True)
    model_actuals = model_actuals.sort_values('RowIndex').reset_index(drop=True)
    
    # Create a scatter plot
    plt.figure(figsize=(14, 8))
    
    # Plot actual values first (behind)
    plt.plot(model_actuals['RowIndex'], model_actuals['Actual'], 
             color=color_actual, label="Actual", linewidth=2, zorder=1)
    
    # Plot predicted values as lines without markers, alternating colors
    unique_iterations = model_preds['Iteration'].unique()
    for iter_num in unique_iterations:
        iter_preds = model_preds[model_preds['Iteration'] == iter_num]
        # Choose color based on iteration parity (even or odd)
        if iter_num % 2 == 0:
            color = color_pred1
        else:
            color = color_pred2
        plt.plot(iter_preds['RowIndex'], iter_preds['Prediction'], 
                 color=color, alpha=0.75, linewidth=1, zorder=2)
    
    plt.title(f"Predicted vs Actual Values for {model_name}")
    plt.xlabel("Row Index")
    plt.ylabel("Stock Close Value")
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_actual, lw=2, label='Actual'),
        Line2D([0], [0], color=color_pred1, lw=1, label='Predicted (Even Iterations (one iteration = 1 year)'),
        Line2D([0], [0], color=color_pred2, lw=1, label='Predicted (Odd Iterations (one iteration = 1 year))')
    ]
    plt.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
