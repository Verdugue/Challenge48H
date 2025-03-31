import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display basic information about the training data
print("Training Data Shape:", train_data.shape)
print("\nFirst few rows of training data:")
print(train_data.head())
print("\nBasic information about the training data:")
print(train_data.info())
print("\nMissing values in training data:")
print(train_data.isnull().sum())

# Separate features and target
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Basic preprocessing
# 1. Handle missing values
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Impute missing values
X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
X_val[numeric_features] = numeric_imputer.transform(X_val[numeric_features])
X_val[categorical_features] = categorical_imputer.transform(X_val[categorical_features])

# 2. Encode categorical variables
X_train = pd.get_dummies(X_train, columns=categorical_features)
X_val = pd.get_dummies(X_val, columns=categorical_features)

# 3. Scale numeric features
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val[numeric_features] = scaler.transform(X_val[numeric_features])

# Train a simple Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on validation set
y_pred = rf_model.predict(X_val)

# Calculate metrics
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print("\nModel Performance:")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close() 