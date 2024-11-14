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

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

print(df)


# Create models with unique names for PolynomialFeatures
models = [("LinearRegression", LinearRegression())]

# # Add Polynomial models with different degrees
# for degree in range(2, 5):
#     model_name = f"PolynomialRegression_Degree_{degree}"
#     models.append((model_name, make_pipeline(PolynomialFeatures(degree), LinearRegression())))

# Initialize DataFrames to store metrics, actual values, and predictions
metrics_results = pd.DataFrame(columns=['Model', 'Year', 'MAE', 'MAPE', 'RMSE', 'R2'])
actual_values = pd.DataFrame(columns=['Date', 'Actual', 'Model'])
model_predictions = pd.DataFrame(columns=['Date', 'Prediction', 'Model'])

# Train and evaluate each model
for model_name, model in models:
    data = df.copy(deep=True)
    e_w = ExpandingWindowByYear(data=data, initial_train_years=1, test_years=1, result_columns=['Close'])
    
    keep_testing = True
    iteration = 1
    while keep_testing:
        # Get the train and test sets for the current window
        train_input, train_results, train_dates = e_w.train_window()
        test_input, test_results, test_dates = e_w.test_window()

        # Fit the model and make predictions
        model.fit(train_input, train_results)
        predictions = model.predict(test_input)

        # Calculate metrics for the current test window
        mae = mean_absolute_error(test_results, predictions)
        mape = mean_absolute_percentage_error(test_results, predictions)
        rmse = root_mean_squared_error(test_results, predictions)
        r2 = r2_score(test_results, predictions)

        # Extract the year from the representative_date
        # Assuming 'Date' is in the test_dates
        representative_year = test_dates.dt.year.max()

        # Create a temporary DataFrame with the current metrics to concatenate
        temp_df = pd.DataFrame({
            'Model': [model_name],
            'Year': [representative_year],
            'MAE': [mae],
            'MAPE': [mape],
            'RMSE': [rmse],
            'R2': [r2]
        })

        # Concatenate the temporary DataFrame to metrics_results
        metrics_results = pd.concat([metrics_results, temp_df], ignore_index=True)

        # Store actual values with their corresponding dates
        temp_actual_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': test_results.values.flatten(),
            'Model': model_name
        })
        actual_values = pd.concat([actual_values, temp_actual_df], ignore_index=True)
        
        # Store predicted values with their corresponding dates
        temp_pred_df = pd.DataFrame({
            'Date': test_dates,
            'Prediction': predictions.flatten(),
            'Model': model_name
        })
        model_predictions = pd.concat([model_predictions, temp_pred_df], ignore_index=True)

        # Try to extend the train window for the next iteration
        try:
            e_w.extend_train_window()
            iteration += 1
        except Exception:
            keep_testing = False

# Set 'Year' as the index for metrics_results
metrics_results.set_index('Year', inplace=True)
print("Metrics Results:")
print(metrics_results)

# Set 'Date' as the index for actual_values and model_predictions
actual_values.set_index('Date', inplace=True)
model_predictions.set_index('Date', inplace=True)

# Display the head of the DataFrames to verify
print("\nActual Values:")
print(actual_values.head())

print("\nModel Predictions:")
print(model_predictions.head())

# Plot the metrics over years for each model
for model_name in metrics_results['Model'].unique():
    model_data = metrics_results[metrics_results['Model'] == model_name]
    
    plt.figure(figsize=(12, 8))
    plt.plot(model_data.index, model_data['MAE'], label="MAE", marker="o")
    plt.plot(model_data.index, model_data['MAPE'], label="MAPE", marker="x")
    plt.plot(model_data.index, model_data['RMSE'], label="RMSE", marker="s")
    plt.plot(model_data.index, model_data['R2'], label="R2", marker="d")
    plt.title(f"Metrics Over Years for {model_name}")
    plt.xlabel("Year")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Define colors for plotting
color_actual = 'orange'  # Color for actual values
color_pred1 = 'blue'     # First color for predicted lines
color_pred2 = 'green'    # Second color for predicted lines

# Plot predicted vs actual values for each model with two alternating colors for years
for model_name in actual_values['Model'].unique():
    model_actuals = actual_values[actual_values['Model'] == model_name].sort_index()
    model_preds = model_predictions[model_predictions['Model'] == model_name].sort_index()
    
    # Merge actual and predicted values on Date to ensure alignment
    merged_df = pd.merge(model_actuals, model_preds, left_index=True, right_index=True)
    
    # Extract Year from Date for coloring
    merged_df['Year'] = merged_df.index.year
    
    # Create a scatter plot
    plt.figure(figsize=(14, 8))
    
    # Plot actual values
    plt.plot(merged_df.index, merged_df['Actual'], 
             color=color_actual, label="Actual", linewidth=2, zorder=1)
    
    # Plot predicted values as lines without markers, alternating colors based on Year
    unique_years = sorted(merged_df['Year'].unique())
    for year in unique_years:
        iter_preds = merged_df[merged_df['Year'] == year]
        # Choose color based on year parity (even or odd)
        if year % 2 == 0:
            color = color_pred1
            label = 'Predicted (Even Years)'
        else:
            color = color_pred2
            label = 'Predicted (Odd Years)'
        # To prevent duplicate labels in legend
        if year == unique_years[0]:
            plt.plot(iter_preds.index, iter_preds['Prediction'], 
                     color=color, alpha=0.75, linewidth=1, label=label, zorder=2)
        else:
            plt.plot(iter_preds.index, iter_preds['Prediction'], 
                     color=color, alpha=0.75, linewidth=1, zorder=2)
    
    plt.title(f"Predicted vs Actual Values for {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Stock Close Value")
    
    # Create custom legend to avoid duplicate labels
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_actual, lw=2, label='Actual'),
        Line2D([0], [0], color=color_pred1, lw=1, label='Predicted (Even Years)'),
        Line2D([0], [0], color=color_pred2, lw=1, label='Predicted (Odd Years)')
    ]
    plt.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
