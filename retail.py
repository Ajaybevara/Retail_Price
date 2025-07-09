# retail_price_optimization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Step 1: Load dataset
file_path = r"C:\Users\ajayb\OneDrive\Desktop\AIML_projects\Retail_Price\retail_price.csv"
df = pd.read_csv(file_path)

# Step 2: Preview data
print("Dataset Preview:\n", df.head())

# Step 3: Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Drop rows with missing values
df.dropna(inplace=True)

# Step 5: Drop non-numeric/irrelevant columns for modeling
df = df.drop(['product_id', 'month_year', 'product_category_name'], axis=1)

# Step 6: Show remaining columns
print("\nRemaining Columns:\n", df.columns)

# Step 7: Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ➕ Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(df['unit_price'], kde=True, bins=30, color='green')
plt.title("Distribution of Unit Price")
plt.xlabel("Unit Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# ➕ Pairplot of top correlated features
top_corr = df.corr()['unit_price'].abs().sort_values(ascending=False).index[1:5]
sns.pairplot(df[top_corr.to_list() + ['unit_price']])
plt.suptitle("Pairplot of Top Correlated Features", y=1.02)
plt.show()

# Step 8: Define features (X) and target (y)
X = df.drop('unit_price', axis=1)
y = df['unit_price']

# Step 9: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Step 11: Train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 12: Evaluation Function
def evaluate_model(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Step 13: Evaluate Models
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest Regressor", y_test, y_pred_rf)

# Step 14: Plot Actual vs Predicted for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Random Forest)")
plt.grid(True)
plt.show()

# ➕ Actual vs Predicted Line Plot (First 50 samples)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:50], label="Actual Price", marker='o')
plt.plot(y_pred_rf[:50], label="Predicted Price", marker='x')
plt.title("Actual vs Predicted Price (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Unit Price")
plt.legend()
plt.grid(True)
plt.show()

# ➕ Residuals Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred_rf, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted (Random Forest)")
plt.xlabel("Predicted Unit Price")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# Step 15: Feature Importance (Random Forest)
feature_importance = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Step 16: Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("\n✅ Model saved as 'rf_model.pkl'")

# Step 17: Predict price from user input
print("\n🔍 Predict Unit Price for a New Product")

input_data = []
for feature in X.columns:
    val = float(input(f"Enter value for {feature}: "))
    input_data.append(val)

input_df = pd.DataFrame([input_data], columns=X.columns)
predicted_price = rf.predict(input_df)[0]

print(f"\n💰 Predicted Unit Price: ₹{predicted_price:.2f}")

# Step 18: Show user input and predicted price as bar chart (fixed warning)
plot_data = input_df.iloc[0].to_dict()
plot_data['Predicted Unit Price'] = predicted_price

plt.figure(figsize=(10, 6))
sns.barplot(
    x=list(plot_data.keys()),
    y=list(plot_data.values()),
    hue=list(plot_data.keys()),
    dodge=False,
    palette="Set2",
    legend=False
)
plt.xticks(rotation=45)
plt.ylabel("Value")
plt.title("User Input Feature Values and Predicted Unit Price")
plt.tight_layout()
plt.grid(True)
plt.show()
