import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Generate synthetic dataset
np.random.seed(42)
n_samples = 200
df = pd.DataFrame({
    'battery': np.random.randint(2500, 6000, size=n_samples),
    'ram': np.random.randint(3, 16, size=n_samples),
    'screen_size': np.round(np.random.uniform(5.0, 7.0, size=n_samples), 1),
    'camera': np.random.randint(8, 108, size=n_samples),
    'processor': np.random.choice(['Qualcomm', 'MediaTek', 'Exynos', 'Apple'], size=n_samples),
    'price': np.random.randint(1000, 3000, size=n_samples)  # Simplified price generation
})

# Encode processor and split data
df['processor'] = LabelEncoder().fit_transform(df['processor'])
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict
model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}, R²: {r2_score(y_test, y_pred):.3f}")

# Prediction for a new mobile
new_mobile = pd.DataFrame({'battery': [4000], 'ram': [6], 'screen_size': [6.5], 'camera': [48], 'processor': [0]})
print(f"Predicted Mobile Price: ${model.predict(new_mobile)[0]:.2f}")
