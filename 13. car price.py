import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Generate synthetic car dataset
np.random.seed(42)
n_samples = 500

data = {
    'car_year': np.random.randint(2000, 2023, n_samples),
    'mileage': np.random.randint(5000, 200000, n_samples),
    'engine_size': np.random.randint(1, 6, n_samples),
    'car_brand': np.random.choice(['Toyota', 'Ford', 'BMW', 'Audi', 'Mercedes'], n_samples),
    'price': np.random.randint(10000, 50000, n_samples)  # Synthetic target variable
}

df = pd.DataFrame(data)

# Preprocess the data
df['car_brand'] = LabelEncoder().fit_transform(df['car_brand'])
X = df[['car_year', 'mileage', 'engine_size', 'car_brand']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Example prediction
new_car = pd.DataFrame({'car_year': [2020], 'mileage': [30000], 'engine_size': [2.5], 'car_brand': [0]})  # Toyota (encoded)
print(f"Predicted Car Price: ${model.predict(new_car)[0]:.2f}")
