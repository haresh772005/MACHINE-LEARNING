import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data  # Features
y = data.target  # Target variable (house prices)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R² Score: {r2:.3f}")

new_house = np.array([[2.5, 3, 1500, 0.2, 15, 1, 2.0, 37]])  # Example features: (avg rooms, avg bedrooms, area, etc.)
predicted_price = rf_model.predict(new_house)
print(f"Predicted House Price: ${predicted_price[0]:.2f}")
