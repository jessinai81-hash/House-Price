# House Price Regression Project

## 📊 Project Overview
This project performs comprehensive house price prediction using machine learning regression models with data preprocessing and analysis.

## 🎯 Key Components

### 1. Data Analysis & Preprocessing (`analysis_and_modeling.py`)
- **Feature Analysis**: Analyzes skewness of each feature
  - **Skewed Features**: Preprocessed with PowerTransformer
  - **Non-skewed Features**: Preprocessed with StandardScaler
  
- **Features Analyzed**:
  - Area (sq ft): Skewness = 0.3038 → StandardScaler
  - Bedrooms: Skewness = 0.0220 → StandardScaler
  - Bathrooms: Skewness = 0.0805 → StandardScaler
  - House Age: Skewness = 0.4996 → StandardScaler
  - Distance (km): Skewness = 0.3353 → StandardScaler

- **Target Variable**: House Price (preprocessed with StandardScaler)

### 2. Correlation Analysis
Feature correlations with House Price:
- **Area_sqft**: 0.9947 (very strong positive)
- **Bedrooms**: 0.9517 (very strong positive)
- **Bathrooms**: 0.9431 (very strong positive)
- **House_Age**: -0.9086 (very strong negative)
- **Distance_KM**: -0.9212 (very strong negative)

### 3. Model Comparison (10 Algorithms)

| Model | Train RMSE | Test RMSE | R² Score |
|-------|-----------|-----------|----------|
| Ridge Regression | 0.1401 | 0.0595 | 0.9961 ⭐ |
| XGBoost | 0.0007 | 0.2264 | 0.9435 |
| Gradient Boosting | 0.0000 | 0.2473 | 0.9326 |
| KNN (k=5) | 0.2515 | 0.2544 | 0.9286 |
| Random Forest | 0.0665 | 0.2606 | 0.9251 |
| Linear Regression | 0.0711 | 0.2925 | 0.9056 |
| Decision Tree | 0.0000 | 0.2942 | 0.9046 |
| SVM | 0.1292 | 0.3202 | 0.8870 |
| AdaBoost | 0.0440 | 0.3243 | 0.8840 |
| Lasso Regression | 1.0116 | 0.9524 | -0.0001 |

**Best Model**: Ridge Regression (Test RMSE: 0.0595, R²: 0.9961)

### 4. Data Split
- Training Set: 16 samples (80%)
- Test Set: 4 samples (20%)

## 📁 Generated Files

### Models & Data
- `best_model.pkl` - Ridge Regression model (best performing)
- `feature_preprocessors.pkl` - Feature preprocessing objects
- `target_scaler.pkl` - Target variable scaler
- `model_comparison.csv` - Complete comparison table

### Visualizations
- `feature_distributions.png` - Distribution plots of all features
- `correlations.png` - Feature correlation bar chart
- `model_comparison.png` - Model performance comparison

## 🚀 How to Use

### 1. Run Analysis & Training
```bash
python analysis_and_modeling.py
```
This generates all models, preprocessors, and visualizations.

### 2. Start the Prediction App
```bash
streamlit run app.py
```

This launches an interactive web app with:
- **Predict Price Tab**: Enter house details to get price predictions
- **Model Comparison Tab**: View all 10 models' performance
- **Feature Correlations Tab**: Analyze feature relationships
- **Analysis Tab**: View visualizations and dataset statistics

## 📋 Features of the Streamlit App

### Prediction Interface
Input fields:
- Area (sq ft)
- Number of Bedrooms
- Number of Bathrooms
- House Age (years)
- Distance from City Center (km)

Output:
- Predicted house price
- Input summary table

### Model Comparison
- Interactive bar chart comparing all 10 models
- RMSE scores for training and test sets
- R² scores for model evaluation

### Correlation Analysis
- Sorted correlation values
- Interactive visualization
- Interpretation guide

### Analysis Dashboard
- Feature distribution plots
- Model performance comparison charts
- Dataset statistics

## 🔧 Requirements

All required packages are installed in the virtual environment:
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- streamlit
- plotly
- xgboost
- scipy
- Pillow

## 💡 Model Performance Insights

**Ridge Regression** was selected as the best model because:
1. **Lowest Test RMSE**: 0.0595 (best generalization)
2. **Highest R² Score**: 0.9961 (explains 99.61% of variance)
3. **Best Balance**: Good performance without overfitting
4. **Robustness**: Ridge regularization prevents overfitting

## 📝 Notes

- The dataset contains 20 house samples
- All features show strong correlation with price
- Negative correlations indicate features that decrease price (age, distance)
- Positive correlations indicate features that increase price (area, bedrooms, bathrooms)
- The model is ready for deployment and real-world predictions
