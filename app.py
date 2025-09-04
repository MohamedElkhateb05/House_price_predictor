# =====================
# PHASE 0: IMPORTS & SETUP
# =====================
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =====================
# PHASE 1: DATA LOADING FROM KAGGLE
# =====================
print("=== PHASE 1: DOWNLOADING DATA FROM KAGGLE ===")

try:
    # Download the dataset
    print("Downloading housing dataset from Kaggle...")
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Find and load the CSV file
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        data_file = os.path.join(path, csv_files[0])
        data = pd.read_csv(data_file)
        print(f"Successfully loaded {csv_files[0]} with shape {data.shape}")
    else:
        raise FileNotFoundError("No CSV files found in the dataset directory")
        
except Exception as e:
    print(f"Error downloading from Kaggle: {e}")
    print("Falling back to local file...")
    # Fallback: try to load from local path
    try:
        data = pd.read_csv("Housing.csv")
        print("Loaded from local file 'Housing.csv'")
    except:
        print("Could not load data. Please check your Kaggle credentials or provide the data manually.")
        exit()

print("\nFirst 5 rows of the dataset:")
print(data.head())
print(f"\nData types:\n{data.dtypes}")

# =====================
# PHASE 2: DATA CLEANING
# =====================
print("\n=== PHASE 2: DATA CLEANING ===")

# Create a clean copy
data_cleaned = data.copy()

# Display initial summary to identify outliers
print("Initial summary statistics:")
print(data_cleaned[['price', 'area']].describe())

# Remove clear outlier errors (based on typical housing data)
# These would be properties with extremely large area but very low price
price_q1 = data_cleaned['price'].quantile(0.25)
price_q3 = data_cleaned['price'].quantile(0.75)
price_iqr = price_q3 - price_q1

area_q1 = data_cleaned['area'].quantile(0.25)
area_q3 = data_cleaned['area'].quantile(0.75)
area_iqr = area_q3 - area_q1

# Identify outliers (you can adjust these thresholds)
price_outliers = data_cleaned[
    (data_cleaned['price'] < price_q1 - 1.5 * price_iqr) | 
    (data_cleaned['price'] > price_q3 + 1.5 * price_iqr)
]

area_outliers = data_cleaned[
    (data_cleaned['area'] < area_q1 - 1.5 * area_iqr) | 
    (data_cleaned['area'] > area_q3 + 1.5 * area_iqr)
]

print(f"Found {len(price_outliers)} price outliers and {len(area_outliers)} area outliers")

# Remove only the most extreme outliers (manual inspection would be better)
extreme_outliers = data_cleaned[
    (data_cleaned['area'] > 10000) & (data_cleaned['price'] < 5000000)
]
print(f"Removing {len(extreme_outliers)} extreme outliers")
data_cleaned = data_cleaned.drop(extreme_outliers.index)

print(f"New data shape after cleaning: {data_cleaned.shape}")

# =====================
# PHASE 3: FEATURE ENGINEERING
# =====================
print("\n=== PHASE 3: FEATURE ENGINEERING ===")

# Apply log transformation to price and area (crucial for housing data)
data_cleaned['log_price'] = np.log(data_cleaned['price'])
data_cleaned['log_area'] = np.log(data_cleaned['area'])

# Encode binary features (yes/no to 1/0)
binary_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_vars:
    data_cleaned[col] = data_cleaned[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishingstatus
furnishing_dummies = pd.get_dummies(data_cleaned['furnishingstatus'], prefix='furnishing', drop_first=True)
data_cleaned = pd.concat([data_cleaned, furnishing_dummies], axis=1)
data_cleaned.drop('furnishingstatus', axis=1, inplace=True)

print("Feature engineering completed successfully")
print(f"New columns: {data_cleaned.columns.tolist()}")

# =====================
# PHASE 4: TRAIN-TEST SPLIT & SCALING
# =====================
print("\n=== PHASE 4: TRAIN-TEST SPLIT & SCALING ===")

# Define features and target
X = data_cleaned.drop(['price', 'area', 'log_price'], axis=1)
y = data_cleaned['log_price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for better handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("Feature scaling completed")

# =====================
# PHASE 5: MODEL SELECTION
# =====================
print("\n=== PHASE 5: MODEL SELECTION ===")

# Test different polynomial degrees
print("Testing polynomial degrees...")
train_errors, test_errors = [], []
degrees = range(1, 6)

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    print(f"Degree {d}: Train MSE = {train_errors[-1]:.4f}, Test MSE = {test_errors[-1]:.4f}")

# Find best degree
best_degree = degrees[np.argmin(test_errors)]
print(f"\nBest degree: {best_degree}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
plt.plot(degrees, test_errors, 's-', label='Testing Error', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Model Complexity vs. Error')
plt.axvline(x=best_degree, color='r', linestyle='--', label=f'Best Degree ({best_degree})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =====================
# PHASE 6: FINAL MODEL TRAINING
# =====================
print("\n=== PHASE 6: FINAL MODEL TRAINING ===")

# Train final model with best degree
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluate final model
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"FINAL MODEL RESULTS (Degree {best_degree}):")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Convert error to USD interpretation
actual_prices = np.exp(y_test)
predicted_prices = np.exp(y_pred)
mape = np.mean(np.abs(actual_prices - predicted_prices) / actual_prices) * 100
print(f"Mean Absolute Percentage Error: {mape:.1f}%")

# =====================
# PHASE 7: INTERACTIVE PREDICTION SYSTEM
# =====================
print("\n=== PHASE 7: INTERACTIVE PREDICTION SYSTEM ===")

def predict_house_price():
    """Interactive house price prediction in USD"""
    print("🏠 HOUSE PRICE PREDICTION SYSTEM (USD)")
    print("Please enter the following property details:\n")
    
    try:
        # Get user input
        bedrooms = int(input("Number of bedrooms: "))
        bathrooms = int(input("Number of bathrooms: "))
        stories = int(input("Number of stories: "))
        area = float(input("Total area (sq.ft): "))
        parking = int(input("Number of parking spaces: "))
        
        print("\nPlease answer yes/no for the following:")
        mainroad = 1 if input("On main road? (yes/no): ").lower() == 'yes' else 0
        guestroom = 1 if input("Has guest room? (yes/no): ").lower() == 'yes' else 0
        basement = 1 if input("Has basement? (yes/no): ").lower() == 'yes' else 0
        hotwaterheating = 1 if input("Has hot water heating? (yes/no): ").lower() == 'yes' else 0
        airconditioning = 1 if input("Has air conditioning? (yes/no): ").lower() == 'yes' else 0
        prefarea = 1 if input("In preferred area? (yes/no): ").lower() == 'yes' else 0
        
        print("\nFurnishing status:")
        print("1. Furnished")
        print("2. Semi-Furnished") 
        print("3. Unfurnished")
        furnishing_choice = int(input("Enter choice (1-3): "))
        
        furnishing_semi = 1 if furnishing_choice == 2 else 0
        furnishing_unfurnished = 1 if furnishing_choice == 3 else 0
        
        # Create input array with correct feature order
        input_features = np.array([[
            bedrooms, bathrooms, stories, mainroad, guestroom,
            basement, hotwaterheating, airconditioning, parking,
            prefarea, np.log(area), furnishing_semi, furnishing_unfurnished
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_features)
        
        # Apply polynomial transformation
        input_poly = poly.transform(input_scaled)
        
        # Predict
        log_prediction = model.predict(input_poly)[0]
        predicted_price = np.exp(log_prediction)
        
        # Confidence interval based on model performance
        lower_bound = predicted_price * (1 - mape/100)
        upper_bound = predicted_price * (1 + mape/100)
        
        # Display results
        print("\n" + "="*50)
        print("📊 PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Price: ${predicted_price:,.2f}")
        print(f"Confidence Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
        print(f"Model Accuracy: ±{mape:.1f}%")
        print("="*50)
        
        return predicted_price
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to enter valid numbers.")
        return None

# =====================
# PHASE 8: SAVE MODEL COMPONENTS
# =====================
print("\n=== PHASE 8: SAVE MODEL COMPONENTS ===")

# Save all components for future use
model_components = {
    'model': model,
    'poly': poly,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'best_degree': best_degree,
    'performance': {'mse': mse, 'r2': r2, 'mape': mape}
}

joblib.dump(model_components, 'house_price_predictor_usd.pkl')
print("Model saved to 'house_price_predictor_usd.pkl'")

# =====================
# PHASE 9: RUN THE PREDICTION SYSTEM
# =====================
print("\n=== PHASE 9: PREDICTION SYSTEM READY ===")
print("The model is trained and ready for predictions!")
print("Run predict_house_price() to start predicting house prices in USD.")

# Start the prediction system
predict_house_price()
