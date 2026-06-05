import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 80)
print("HOUSE PRICE REGRESSION ANALYSIS")
print("=" * 80)

df = pd.read_csv('house_price_regression_dataset.csv')
print("\nDataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")

# Separate features and target
X = df.drop('House_Price', axis=1)
y = df['House_Price']

print("\n" + "=" * 80)
print("FEATURE ANALYSIS AND PREPROCESSING")
print("=" * 80)

# Analyze each feature for skewness
from scipy import stats
feature_analysis = {}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(X.columns):
    skewness = stats.skew(X[col])
    feature_analysis[col] = {'skewness': skewness}
    
    print(f"\n{col}:")
    print(f"  Skewness: {skewness:.4f}")
    
    # Plot histogram
    axes[idx].hist(X[col], bins=10, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col}\n(Skewness: {skewness:.4f})')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')

# Remove empty subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=100, bbox_inches='tight')
print("\n✓ Feature distributions saved to 'feature_distributions.png'")

# Preprocess features based on skewness
X_preprocessed = X.copy()
preprocessors = {}

for col in X.columns:
    skewness = abs(feature_analysis[col]['skewness'])
    if skewness > 0.5:  # Consider as skewed
        print(f"\n  ✓ {col} is skewed (|skewness| = {skewness:.4f}), applying PowerTransformer")
        pt = PowerTransformer(method='yeo-johnson')
        X_preprocessed[col] = pt.fit_transform(X[[col]])
        preprocessors[col] = ('power', pt)
    else:
        print(f"\n  ✓ {col} is not skewed (|skewness| = {skewness:.4f}), applying StandardScaler")
        ss = StandardScaler()
        X_preprocessed[col] = ss.fit_transform(X[[col]])
        preprocessors[col] = ('standard', ss)

# Preprocess target variable
target_scaler = StandardScaler()
y_preprocessed = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Calculate correlations with original values
print("\nCorrelation between features and target (original values):")
correlations = {}
for col in X.columns:
    corr = X[col].corr(y)
    correlations[col] = corr
    print(f"  {col}: {corr:.4f}")

# Visualize correlations
fig, ax = plt.subplots(figsize=(10, 6))
corr_data = pd.DataFrame({
    'Feature': list(correlations.keys()),
    'Correlation': list(correlations.values())
})
sns.barplot(data=corr_data, x='Feature', y='Correlation', ax=ax)
ax.set_title('Feature Correlation with House Price')
ax.set_ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('correlations.png', dpi=100, bbox_inches='tight')
print("✓ Correlation plot saved to 'correlations.png'")

print("\n" + "=" * 80)
print("DATA SPLITTING")
print("=" * 80)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y_preprocessed, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\n" + "=" * 80)
print("MODEL TRAINING AND EVALUATION")
print("=" * 80)

# Dictionary to store models and results
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'SVM': SVR(kernel='rbf'),
}

results = []

print("\nTraining 10 different regression models...\n")

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    results.append({
        'Model': name,
        'RMSE (Train)': rmse_train,
        'RMSE (Test)': rmse_test,
        'R² Score': r2
    })
    
    print(f"{name:25} | Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f} | R²: {r2:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE (Test)').reset_index(drop=True)

print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE (Sorted by Test RMSE)")
print("=" * 80)
print("\n" + results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n✓ Best Model: {best_model_name}")
print(f"  Test RMSE: {results_df.iloc[0]['RMSE (Test)']:.4f}")
print(f"  R² Score: {results_df.iloc[0]['R² Score']:.4f}")

# Save results table
results_df.to_csv('model_comparison.csv', index=False)
print("\n✓ Model comparison table saved to 'model_comparison.csv'")

# Visualize results
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(results_df))
ax.bar(x_pos - 0.2, results_df['RMSE (Train)'], width=0.4, label='Train RMSE', alpha=0.8)
ax.bar(x_pos + 0.2, results_df['RMSE (Test)'], width=0.4, label='Test RMSE', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('RMSE')
ax.set_title('Model Comparison - RMSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
print("✓ Model comparison plot saved to 'model_comparison.png'")

# Save best model and preprocessing objects
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessors, 'feature_preprocessors.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl')
joblib.dump(correlations, 'correlations.pkl')

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("✓ best_model.pkl - Best trained model")
print("✓ feature_preprocessors.pkl - Feature preprocessing objects")
print("✓ target_scaler.pkl - Target variable scaler")
print("✓ model_comparison.csv - Comparison table")
print("✓ feature_distributions.png - Feature distribution plots")
print("✓ correlations.png - Correlation plot")
print("✓ model_comparison.png - Model comparison plot")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! Run 'streamlit run app.py' to start the prediction app.")
print("=" * 80)
