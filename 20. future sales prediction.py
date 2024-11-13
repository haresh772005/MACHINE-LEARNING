import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data: Historical sales (Month vs. Sales)
# Assuming sales data for 12 months
data = {
    'Month': np.arange(1, 13),  # Months from 1 to 12
    'Sales': [200, 220, 250, 275, 300, 330, 360, 400, 430, 470, 500, 520]  # Example sales data
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target variable
X = df[['Month']]  # Feature (month)
y = df['Sales']    # Target (sales)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plotting the results
plt.scatter(X, y, color="blue", label="Actual Sales Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.title("Monthly Sales Prediction")
plt.show()

# Predict future sales (e.g., for the next 6 months)
future_months = np.arange(13, 19).reshape(-1, 1)  # Months 13 to 18
future_sales = model.predict(future_months)

# Display predictions
for month, sale in zip(future_months.flatten(), future_sales):
    print(f"Predicted sales for month {month}: {sale:.2f}")
