import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=== PHASE 1: DOWNLOADING DATA FROM KAGGLE ===")

try:
    print("Downloading housing dataset from Kaggle...")
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    print(f"Dataset downloaded to: {path}")

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
    try:
        data = pd.read_csv("Housing.csv")
        print("Loaded from local file 'Housing.csv'")
    except:
        print("Could not load data. Please check your Kaggle credentials or provide the data manually.")
        exit()

print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\n=== PHASE 2: DATA CLEANING ===")

data_cleaned = data.copy()

print("Initial summary statistics:")
print(data_cleaned[['price', 'area']].describe())

price_q1 = data_cleaned['price'].quantile(0.25)
price_q3 = data_cleaned['price'].quantile(0.75)
price_iqr = price_q3 - price_q1

area_q1 = data_cleaned['area'].quantile(0.25)
area_q3 = data_cleaned['area'].quantile(0.75)
area_iqr = area_q3 - area_q1

price_outliers = data_cleaned[
    (data_cleaned['price'] < price_q1 - 1.5 * price_iqr) |
    (data_cleaned['price'] > price_q3 + 1.5 * price_iqr)
]

area_outliers = data_cleaned[
    (data_cleaned['area'] < area_q1 - 1.5 * area_iqr) |
    (data_cleaned['area'] > area_q3 + 1.5 * area_iqr)
]

print(f"Found {len(price_outliers)} price outliers and {len(area_outliers)} area outliers")

extreme_outliers = data_cleaned[
    (data_cleaned['area'] > 10000) & (data_cleaned['price'] < 5000000)
]
print(f"Removing {len(extreme_outliers)} extreme outliers")
data_cleaned = data_cleaned.drop(extreme_outliers.index)

print(f"New data shape after cleaning: {data_cleaned.shape}")

print("\n=== PHASE 3: FEATURE ENGINEERING ===")

data_cleaned['log_price'] = np.log(data_cleaned['price'])
data_cleaned['log_area'] = np.log(data_cleaned['area'])

binary_vars = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_vars:
    data_cleaned[col] = data_cleaned[col].map({'yes': 1, 'no': 0})

furnishing_dummies = pd.get_dummies(
    data_cleaned['furnishingstatus'], prefix='furnishing', drop_first=True)
data_cleaned = pd.concat([data_cleaned, furnishing_dummies], axis=1)
data_cleaned.drop('furnishingstatus', axis=1, inplace=True)

print("Feature engineering completed successfully")
print(f"New columns: {data_cleaned.columns.tolist()}")

print("\n=== PHASE 4: TRAIN-TEST SPLIT & SCALING ===")

X = data_cleaned.drop(['price', 'area', 'log_price'], axis=1)
y = data_cleaned['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(
    X_test_scaled, columns=X.columns, index=X_test.index)

print("Feature scaling completed")

print("\n=== PHASE 5: MODEL SELECTION ===")

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

    print(
        f"Degree {d}: Train MSE = {train_errors[-1]:.4f}, Test MSE = {test_errors[-1]:.4f}")

best_degree = degrees[np.argmin(test_errors)]
print(f"\nBest degree: {best_degree}")

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'o-',
         label='Training Error', linewidth=2, markersize=8)
plt.plot(degrees, test_errors, 's-',
         label='Testing Error', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Model Complexity vs. Error')
plt.axvline(x=best_degree, color='r', linestyle='--',
            label=f'Best Degree ({best_degree})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n=== PHASE 6: FINAL MODEL TRAINING ===")

poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"FINAL MODEL RESULTS (Degree {best_degree}):")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

actual_prices = np.exp(y_test)
predicted_prices = np.exp(y_pred)
mape = np.mean(np.abs(actual_prices - predicted_prices) / actual_prices) * 100
print(f"Mean Absolute Percentage Error: {mape:.1f}%")

print("\n=== PHASE 6.5: VISUALIZATIONS ===")

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8F71']
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Housing Price Analysis - Key Insights', fontsize=16, fontweight='bold', y=0.98)

axes[0, 0].scatter(actual_prices, predicted_prices, alpha=0.6, color=colors[0])
axes[0, 0].plot([actual_prices.min(), actual_prices.max()], 
                [actual_prices.min(), actual_prices.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title('Actual vs Predicted Prices\n(Model Accuracy)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(actual_prices, bins=30, color=colors[1], alpha=0.7, edgecolor='black')
axes[0, 1].axvline(actual_prices.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: ${actual_prices.mean():,.0f}')
axes[0, 1].set_xlabel('Price ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Housing Prices', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_[1:len(X.columns)+1])
}).sort_values('importance', ascending=True)

axes[0, 2].barh(feature_importance['feature'], feature_importance['importance'], 
                color=colors[2])
axes[0, 2].set_xlabel('Importance (Absolute Coefficient Value)')
axes[0, 2].set_title('Top Factors Influencing House Prices', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='x')

error_percentage = ((actual_prices - predicted_prices) / actual_prices) * 100
axes[1, 0].hist(error_percentage, bins=30, color=colors[3], alpha=0.7, edgecolor='black')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 0].set_xlabel('Prediction Error (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Prediction Errors', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(data_cleaned['area'], data_cleaned['price'], 
                  alpha=0.6, color=colors[4], s=30)
axes[1, 1].set_xlabel('Area (sq.ft)')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].set_title('Price vs Area\n(Strongest Correlation)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

metrics = ['R² Score', 'MAPE', 'MSE']
values = [r2, mape, mse]
color_bars = ['#2E86AB', '#A23B72', '#F18F01']

bars = axes[1, 2].bar(metrics, values, color=color_bars, alpha=0.8)
axes[1, 2].set_ylabel('Value')
axes[1, 2].set_title('Model Performance Metrics', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('housing_price_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = data_cleaned[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Heatmap', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations created and saved as PNG files")

print("\n=== PHASE 7: INTERACTIVE PREDICTION SYSTEM ===")

def predict_house_price():
    print("🏠 HOUSE PRICE PREDICTION SYSTEM (USD)")
    print("Please enter the following property details:\n")

    try:
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

        input_features = np.array([[
            bedrooms, bathrooms, stories, mainroad, guestroom,
            basement, hotwaterheating, airconditioning, parking,
            prefarea, np.log(area), furnishing_semi, furnishing_unfurnished
        ]])

        input_scaled = scaler.transform(input_features)
        input_poly = poly.transform(input_scaled)
        log_prediction = model.predict(input_poly)[0]
        predicted_price = np.exp(log_prediction)

        lower_bound = predicted_price * (1 - mape/100)
        upper_bound = predicted_price * (1 + mape/100)

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

print("\n=== PHASE 8: SAVE MODEL COMPONENTS ===")

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

print("\n=== PHASE 9: PREDICTION SYSTEM READY ===")
print("The model is trained and ready for predictions!")
print("Run predict_house_price() to start predicting house prices in USD.")

predict_house_price()
