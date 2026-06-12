# ==========================================
# PROJECT: HOUSE PRICE PREDICTION AI
# DESCRIPTION: A Simple Linear Regression model to predict house prices.
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("🤖 Initializing House Price Prediction Model...")

# 1. Load the California Housing Dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Display basic structure of the data
print("\n📋 Dataset Preview (First 5 rows):")
print(df.head())

# 2. Separate Features (X) and Target Price (y)
X = df.drop(columns=['MedHouseVal'])  # Features like rooms, income, etc.
y = df['MedHouseVal']                 # Target: Median house value

# 3. Split data into Training set (80%) and Testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Linear Regression Model
model = LinearRegression()
print("\n🏋️ Training the AI model...")
model.fit(X_train, y_train)

# 5. Make Predictions on the test data
y_pred = model.predict(X_test)

# 6. Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance Evaluation:")
print(f"👉 Mean Squared Error (MSE): {mse:.4f}")
print(f"👉 R-squared Score (Accuracy metric): {r2:.4f}")

# 7. Create a Visual Plot (Required for Output Images/Screenshots)
# Create an output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Prices ($100k)')
plt.ylabel('Predicted Prices ($100k)')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()

# Save the plot automatically to the folder
plt.savefig('output/actual_vs_predicted.png')
print("\n📈 Plot successfully saved to 'output/actual_vs_predicted.png'!")