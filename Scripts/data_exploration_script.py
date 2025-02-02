# data_exploration_script.py

import pandas as pd
import numpy as np

# Load the datasets
train = pd.read_csv('/content/drive/MyDrive/air_quality_forecasting/train.csv')
test = pd.read_csv('/content/drive/MyDrive/air_quality_forecasting/test.csv')

# Display the first few rows of the training dataset
print("Training Data Overview:")
print(train.head())

# Display the first few rows of the test dataset
print("\nTest Data Overview:")
print(test.head())

# shape of the datasets
print("\nTraining Data Shape:", train.shape)
print("Test Data Shape:", test.shape)

# showing the column names and data types
print("\nTraining Data Columns and Data Types:")
print(train.info())

print("\nTest Data Columns and Data Types:")
print(test.info())

# Convert 'datetime' column to datetime format and set as index
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
train.set_index('datetime', inplace=True)
test.set_index('datetime', inplace=True)

# Handle missing values using forward and backward fill
train['pm2.5'].fillna(method='ffill', inplace=True)
train['pm2.5'].fillna(method='bfill', inplace=True)

# Ensuring no missing values remain
print("\nMissing Values in Training Data After Handling:")
print(train.isnull().sum())

# Time-based feature engineering
def add_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

# Adding the time features to the datasets
train = add_time_features(train)
test = add_time_features(test)

print("\nTraining data with time-based features:")
print(train.head())
