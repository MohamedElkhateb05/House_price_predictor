This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data = pd.read_csv("/kaggle/input/housing-prices-dataset/Housing.csv")

data.head()
data.info()
data.describe()

def find_outliers_iqr(data, feature):
    """Identify outliers in a feature using the IQR method."""
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    return outliers

# Find outliers for 'price'
price_outliers = find_outliers_iqr(data, 'price')
print(f"Number of outliers in price: {len(price_outliers)}")
print(price_outliers[['price', 'area']].sort_values(by='price', ascending=False).head())

print("\n---\n")

# Find outliers for 'area'
area_outliers = find_outliers_iqr(data, 'area')
print(f"Number of outliers in area: {len(area_outliers)}")
print(area_outliers[['area', 'price']].sort_values(by='area', ascending=False).head())

# Get the indices of the outlier rows for 'price' and 'area'
price_outlier_indices = find_outliers_iqr(data, 'price').index
area_outlier_indices = find_outliers_iqr(data, 'area').index

print("="*60)
print("PRICE OUTLIERS (Full Records)")
print("="*60)
display(data.loc[price_outlier_indices].sort_values(by='price', ascending=False))

print("\n" + "="*60)
print("AREA OUTLIERS (Full Records)")
print("="*60)
display(data.loc[area_outlier_indices].sort_values(by='area', ascending=False))


# Create a list of indices to drop
indices_to_drop = [403, 125, 211]

# Drop these rows from the DataFrame
data_cleaned = data.drop(index=indices_to_drop)

# Verify the new shape of the DataFrame (should have 3 fewer rows)
print(f"Original shape: {data.shape}")
print(f"Cleaned shape: {data_cleaned.shape}")

data_cleaned.describe()

# Apply log transformation to the cleaned DataFrame
data_cleaned['log_price'] = np.log(data_cleaned['price'])
data_cleaned['log_area'] = np.log(data_cleaned['area'])

# Check the new distribution of the log-transformed features
print(data_cleaned[['log_price', 'log_area']].describe())

One-hot encode 'furnishingstatus' and drop the first category to avoid multicollinearity
furnishing_dummies = pd.get_dummies(data_cleaned['furnishingstatus'], prefix='furnishing', drop_first=True)

# Concatenate the new dummy columns with the main DataFrame
data_cleaned = pd.concat([data_cleaned, furnishing_dummies], axis=1)

# Now we can drop the original 'furnishingstatus' column
data_cleaned.drop('furnishingstatus', axis=1, inplace=True)

data_cleaned.info()

# List of binary categorical features that are still strings
binary_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Map 'yes' to 1 and 'no' to 0 for each column
for col in binary_vars:
    data_cleaned[col] = data_cleaned[col].map({'yes': 1, 'no': 0})

# Verify the conversion
print(data_cleaned[binary_vars].head())

# Check the data types of all columns in the cleaned DataFrame
print(data_cleaned.dtypes)


# Use the log-transformed target
y = data_cleaned['log_price']

# Create feature set - drop original price/area and keep log_area plus all other features
X = data_cleaned.drop(['price', 'area', 'log_price'], axis=1)  # Keep 'log_area' as a feature

# Verify the shapes
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test set - X: {X_test.shape}, y: {y_test.shape}")

# Initialize scaler
scaler = StandardScaler()

# Fit on training data and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for better readability (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("First few rows of scaled training data:")
print(X_train_scaled.head())


# Initialize lists to store errors
train_errors, test_errors = [], []
degrees = range(1, 6)  # Start from degree=1 (linear) to degree=5

for d in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train_scaled) # Use scaled features!
    X_test_poly = poly.transform(X_test_scaled)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predict and calculate errors
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    # Print the score for each degree for clarity
    print(f"Degree {d}: Train MSE = {train_errors[-1]:.4f}, Test MSE = {test_errors[-1]:.4f}")

# Find the best degree (the one with the lowest TEST error)
best_degree = degrees[np.argmin(test_errors)]
print(f"\nThe best degree is: {best_degree}")


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
plt.plot(degrees, test_errors, 's-', label='Testing Error', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.title('Bias-Variance Tradeoff: Model Complexity vs. Error', fontsize=14, fontweight='bold')
plt.axvline(x=best_degree, color='red', linestyle='--', alpha=0.7, label=f'Best Degree ({best_degree})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Results:")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

