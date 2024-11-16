import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from expanding_window import ExpandingWindowByYear  # Ensure to import the class correctly

# Load your data
data_df = pd.read_csv("./preprocessed_stock_data.csv")
data_df['Year'] = pd.to_datetime(data_df['Date']).dt.year

# Initialize the expanding window with 1 year initial training and 1 year testing
expanding_window = ExpandingWindowByYear(data_df, initial_train_years=1, test_years=1, result_columns=["Close"])

# Store the predictions, actuals, years, and R2 scores for visualization
all_predictions = []
all_actuals = []
r2_scores = []
test_years = []

# Iterate through each expanding window
while True:
    try:
        # Get the current train and test window
        X_train, y_train, _ = expanding_window.train_window()
        X_test, y_test, _ = expanding_window.test_window()
        
        # Ensure the test window is not empty
        if X_test.empty or y_test.empty:
            print("No more data available for the next test window. Stopping.")
            break
        
        # Scale the data using MinMaxScaler
        # scaler = MinMaxScaler()
        X_train_scaled = X_train #scaler.fit_transform(X_train)
        X_test_scaled = X_test #scaler.transform(X_test)
        
        # Get the input dimension for the current window
        input_dim = X_train_scaled.shape[1]

        # Define and compile the neural network model with the current input dimension
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Set early stopping criteria
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model on the current window
        model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_data=(X_test_scaled, y_test), 
                  verbose=0, callbacks=[early_stopping])

        # Predict only on the test data
        y_test_pred = model.predict(X_test_scaled)

        # Store predictions and actual values for visualization
        all_predictions.extend(y_test_pred.flatten().tolist())  # Predict values from X_test only
        all_actuals.extend(y_test.values.flatten().tolist())    # True values for X_test only

        
        # Calculate and store R2 score
        r2 = r2_score(y_test, y_test_pred)
        r2_scores.append(r2)
        test_years.append(expanding_window.current_test_start_year)
        print(f"R2 Score on Test Set for Year {expanding_window.current_test_start_year}: {r2}")

        # Move to the next window
        expanding_window.extend_train_window()

        
    except IndexError:
        # Stop the loop if no more test data is available
        print("Reached the end of available data.")
        break
    except ValueError as ve:
        # Break the loop if test data is empty
        print(f"Error: {ve}")
        break

# Ensure alignment of test dates with predictions
test_indices = data_df.iloc[-len(all_predictions):].index
test_dates = data_df.loc[test_indices, 'Date']

# Convert to datetime if not already
data_df['Date'] = pd.to_datetime(data_df['Date'])
test_dates = pd.to_datetime(test_dates)

# Plot the original vs predicted stock prices
plt.figure(figsize=(14, 6))
plt.plot(data_df['Date'], data_df['Close'], label="Original Close Price", color="blue")
plt.plot(test_dates, all_predictions, label="Predicted Close Price", color="orange")
plt.title("Original vs. Predicted Stock Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


# Plot the R² score for each year
plt.figure(figsize=(14, 6))
plt.plot(test_years, r2_scores, marker='o', linestyle='-', color='purple', label="R² Score by Year")
plt.axhline(y=0, color='red', linestyle='--')
plt.title("R² Score for Each Test Year")
plt.xlabel("Year")
plt.ylabel("R² Score")
plt.legend()
plt.show()