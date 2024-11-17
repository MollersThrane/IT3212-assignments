import pandas as pd
import numpy as np
from regressor import LinearStockRegressor
from additive_model import AdditiveStockModel
from neural_network import NeuralNetworkStockModel
from neural_network import RandomForestStockModel
import matplotlib.pyplot as plt

df = pd.read_csv("./preprocessed_stock_data.csv")
df.dropna(inplace=True)

num_years = df['Year'].max() - df['Year'].min()

model_classes = [LinearStockRegressor, AdditiveStockModel, NeuralNetworkStockModel, RandomForestStockModel]

# Split DataFrame into N parts
df_split = np.array_split(df, len(model_classes) + 1)

# Accessing each part
for i, part in enumerate(df_split):
    if i == len(df_split) - 1:
        break
    model_classes[i] = model_classes[i](dataset = part, initial_train_years=int(num_years/len(model_classes)), num_test_years=1)
    model_classes[i].train_and_test()

test_data = df_split[-1]
test_actuals = test_data['Close'].copy(deep=True)
test_data.drop(columns=['Close'], inplace=True)
predictions = []

for model in model_classes:
    predictions.append(model.predict(test_data))

predictions_list = [df['Predictions'] for df in predictions]

combined_predictions = pd.concat(predictions_list, axis=1)

combined_predictions['Average_Predictions'] = combined_predictions.mean(axis=1)

plt.figure(figsize=(14, 8))
plt.plot(test_data.index, test_data['Close'], color='orange', label="Actual", linewidth=2, zorder=1)
plt.plot(combined_predictions.index, combined_predictions['Average_Predictions'], color='blue', label="Bagging prediction (average)", linewidth=1, zorder=2)

plt.title("Predicted (bagging - average) vs Actual Values")
plt.xlabel("Date")
plt.ylabel("Stock Close Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
