# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the training dataset
print("Loading training dataset...")
df = pd.read_csv("train.csv")
df.drop(['date'], axis=1, inplace=True)
df = df.replace(',', '.', regex=True)  # Replace commas with dots for numeric columns
df = df.astype(float)  # Convert all columns to float

# Splitting data into features (X) and target variable (y)
X = df.drop(['% Silica Concentrate'], axis=1)
y = df['% Silica Concentrate']

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ANN Model
print("Training Artificial Neural Network...")
model_ann = Sequential([
    Dense(22, input_shape=(22,), activation='relu'),  # Input layer with 22 features
    Dense(14, activation='relu'),  # Hidden layer
    Dense(1, activation='linear')  # Output layer (linear for regression)
])

model_ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model_ann.fit(X_train, y_train, epochs=25, batch_size=16, verbose=1)

# Evaluating ANN on test data
y_preds_ann = model_ann.predict(X_test)
mse_ann = mean_squared_error(y_test, y_preds_ann)
r2_ann = r2_score(y_test, y_preds_ann)
print(f"\nANN Model - Mean Squared Error: {mse_ann:.4f}, R² Score: {r2_ann:.4f}")

# Decision Tree Regressor Model
print("\nTraining Decision Tree Regressor...")
model_dtr = DecisionTreeRegressor(random_state=42)
model_dtr.fit(X_train, y_train)

# Evaluating DTR on test data
y_preds_dtr = model_dtr.predict(X_test)
mse_dtr = mean_squared_error(y_test, y_preds_dtr)
r2_dtr = r2_score(y_test, y_preds_dtr)
print(f"DTR Model - Mean Squared Error: {mse_dtr:.4f}, R² Score: {r2_dtr:.4f}")

# Load the test dataset for predictions
print("\nLoading test dataset for predictions...")
df_test = pd.read_csv("test.csv")
df_test.drop(['date'], axis=1, inplace=True)
df_test = df_test.replace(',', '.', regex=True)  # Replace commas with dots for numeric columns
df_test = df_test.astype(float)  # Convert all columns to float

# Predicting % Silica Concentrate on unseen data
print("\nGenerating predictions on test dataset...")
y_preds_ann_test = model_ann.predict(df_test)
y_preds_dtr_test = model_dtr.predict(df_test)

# Saving predictions to CSV files
print("\nSaving predictions to CSV files...")
df_preds_ann = pd.DataFrame(y_preds_ann_test, columns=['% Silica Concentrate'])
df_preds_ann.index = df_preds_ann.index + 1  # Start index from 1
df_preds_ann.index.name = 'ID'
df_preds_ann.to_csv('ann_predictions.csv', index=True)

df_preds_dtr = pd.DataFrame(y_preds_dtr_test, columns=['% Silica Concentrate'])
df_preds_dtr.index = df_preds_dtr.index + 1  # Start index from 1
df_preds_dtr.index.name = 'ID'
df_preds_dtr.to_csv('dtr_predictions.csv', index=True)

print("\nPredictions saved successfully as 'ann_predictions.csv' and 'dtr_predictions.csv'.")
print("Program completed.")
