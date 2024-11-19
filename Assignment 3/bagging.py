import pandas as pd
import numpy as np
from regressor import LinearRegressionStockModel
from additive_model import AdditiveStockModel
from neural_network import NeuralNetworkStockModel
from random_forest import RandomForestStockModel
from svm_kernel import SupportVectorRegressorStockModel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score

# Load and preprocess data
df = pd.read_csv("./preprocessed_stock_data.csv")
df.dropna(inplace=True)

# Calculate the number of years in the dataset
num_years = df['Year'].max() - df['Year'].min()
print(num_years)

model_classes = [LinearRegressionStockModel, AdditiveStockModel, NeuralNetworkStockModel, RandomForestStockModel, SupportVectorRegressorStockModel]

# Split DataFrame into N parts
df_split = np.array_split(df, len(model_classes) + 1)

# Get test data
test_data = df_split[-1]
test_actuals = test_data[['Close']].copy(deep=True)
test_years = test_data['Year'].copy(deep=True)
test_data.drop(columns=['Close', 'Date'], inplace=True)

# Generate predictions from each model
predictions = {}
model_names = []

# Train and test models on their respective subsets, and predict
for i, part in enumerate(df_split[:-1]):
    model = model_classes[i](dataset=part, initial_train_years=int((num_years / (len(model_classes)+1))*0.8), num_test_years=1)
    model.train_and_test()
    model_name = model.__class__.__name__
    model_names.append(model_name)
    pred = model.predict(test_data)
    predictions[model_name] = pred.values.flatten()  # Ensure it's a 1D array

# Create DataFrame of predictions
predictions_df = pd.DataFrame(predictions, index=test_data.index)

# Calculate the average prediction
predictions_df['Average_Predictions'] = predictions_df.mean(axis=1)

# Calculate metrics per year using average predictions
metrics_per_year = []
for year in test_years.unique():
    idx = test_years == year
    actual = test_actuals.loc[idx, 'Close']
    pred = predictions_df.loc[idx, 'Average_Predictions']
    
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    rmse = root_mean_squared_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    metrics_per_year.append({
        'Year': year,
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'R2': r2
    })

metrics_df = pd.DataFrame(metrics_per_year).set_index('Year')

# Plot metrics over years
plt.figure(figsize=(12, 8))
plt.plot(metrics_df.index, metrics_df['MAE'], label="MAE", marker="o")
plt.plot(metrics_df.index, metrics_df['MAPE'], label="MAPE", marker="x")
plt.plot(metrics_df.index, metrics_df['RMSE'], label="RMSE", marker="s")
plt.plot(metrics_df.index, metrics_df['R2'], label="R²", marker="d")
plt.title("Metrics Over Years")
plt.xlabel("Year")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot actual vs predicted values, including individual model predictions
plt.figure(figsize=(14, 8))
plt.plot(test_actuals.index, test_actuals['Close'], color='orange', label="Actual", linewidth=2, zorder=1)
for model_name in model_names:
    plt.plot(predictions_df.index, predictions_df[model_name], label=model_name, linewidth=1, zorder=2, alpha=0.45)
plt.plot(predictions_df.index, predictions_df['Average_Predictions'], color='blue', label="Predicted (Bagging Average)", linewidth=2, zorder=3)
plt.title("Predicted vs Actual Values")
plt.xlabel("Index")
plt.ylabel("Stock Close Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()