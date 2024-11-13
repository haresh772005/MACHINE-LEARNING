import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Simulate a credit score dataset
np.random.seed(42)
n_samples = 500

# Generating random data for features
age = np.random.randint(18, 70, n_samples)  # Age
income = np.random.randint(30000, 120000, n_samples)  # Annual income in dollars
loan_amount = np.random.randint(5000, 50000, n_samples)  # Loan amount requested
credit_score = np.random.randint(300, 850, n_samples)  # Credit score (300-850 range)

# Generating target variable (Good/Bad Credit)
# Assume credit score above 650 is good, below 650 is bad
good_credit = (credit_score >= 650).astype(int)

# Creating DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'good_credit': good_credit  # Target variable
})

# Features (X) and target (y)
X = df[['age', 'income', 'loan_amount', 'credit_score']]  # Features
y = df['good_credit']  # Target

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{cm}")

# Example prediction: New customer data
new_customer = pd.DataFrame({
    'age': [35],
    'income': [55000],
    'loan_amount': [15000],
    'credit_score': [700]  # Example of good credit
})

# Predict if the new customer has good credit
new_prediction = model.predict(new_customer)
print(f"Prediction for new customer (Good Credit = 1, Bad Credit = 0): {new_prediction[0]}")
