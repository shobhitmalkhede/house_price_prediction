import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
# You can replace this with any open-source housing dataset
data = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")

# Selecting relevant features
data = data.dropna()  # Remove missing values
X = data[["median_income"]]  # Using median income as the only feature for simplicity
y = data["median_house_value"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model training complete and saved as 'house_price_model.pkl'.")
