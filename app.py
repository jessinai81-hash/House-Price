import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load saved models and preprocessors
@st.cache_resource
def load_models():
    model = joblib.load('best_model.pkl')
    preprocessors = joblib.load('feature_preprocessors.pkl')
    target_scaler = joblib.load('target_scaler.pkl')
    correlations = joblib.load('correlations.pkl')
    return model, preprocessors, target_scaler, correlations

model, preprocessors, target_scaler, correlations = load_models()

# Load comparison data
@st.cache_data
def load_comparison():
    return pd.read_csv('model_comparison.csv')

comparison_df = load_comparison()

# App title
st.title("🏠 House Price Prediction System")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Predict Price", "Model Comparison", "Feature Correlations", "Analysis"])

# Tab 1: Prediction
with tab1:
    st.header("Enter House Details for Price Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input(
            "Area (sq ft)",
            min_value=500,
            max_value=5000,
            value=2000,
            step=100
        )
        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=5,
            value=2,
            step=1
        )
    
    with col2:
        house_age = st.number_input(
            "House Age (years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )
        distance = st.number_input(
            "Distance from City Center (km)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.5
        )
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Area_sqft': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'House_Age': [house_age],
        'Distance_KM': [distance]
    })
    
    # Preprocess input
    input_preprocessed = input_data.copy()
    for col in input_data.columns:
        if col in preprocessors:
            processor_type, processor = preprocessors[col]
            input_preprocessed[col] = processor.transform(input_data[[col]])
    
    # Make prediction
    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        # Predict on preprocessed data
        y_pred_preprocessed = model.predict(input_preprocessed)
        
        # Inverse transform to get original scale
        y_pred = target_scaler.inverse_transform(y_pred_preprocessed.reshape(-1, 1))[0][0]
        
        st.success("✓ Prediction Complete!")
        
        # Display prediction with large font
        st.metric(
            label="Predicted House Price",
            value=f"${y_pred:,.2f}",
            delta=None
        )
        
        # Display input summary
        st.subheader("Input Summary")
        summary_data = {
            'Feature': ['Area', 'Bedrooms', 'Bathrooms', 'House Age', 'Distance from City'],
            'Value': [
                f'{area:,} sq ft',
                f'{bedrooms} bedroom(s)',
                f'{bathrooms} bathroom(s)',
                f'{house_age} years',
                f'{distance:.1f} km'
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# Tab 2: Model Comparison
with tab2:
    st.header("Model Performance Comparison")
    
    st.subheader("All Regression Models - RMSE Scores")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Create bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['RMSE (Train)'],
        name='Train RMSE',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df['RMSE (Test)'],
        name='Test RMSE',
        marker_color='darkblue'
    ))
    fig.update_layout(
        title='Model Comparison - RMSE Scores',
        xaxis_title='Model',
        yaxis_title='RMSE',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    best_model_info = comparison_df.iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", best_model_info['Model'])
    with col2:
        st.metric("Test RMSE", f"{best_model_info['RMSE (Test)']:.4f}")
    with col3:
        st.metric("R² Score", f"{best_model_info['R² Score']:.4f}")

# Tab 3: Feature Correlations
with tab3:
    st.header("Feature Correlation Analysis")
    
    corr_df = pd.DataFrame({
        'Feature': list(correlations.keys()),
        'Correlation with Price': list(correlations.values())
    })
    corr_df = corr_df.sort_values('Correlation with Price', ascending=False)
    
    st.subheader("Correlation Coefficients")
    st.dataframe(corr_df, use_container_width=True, hide_index=True)
    
    # Visualization
    fig = px.bar(
        corr_df,
        x='Feature',
        y='Correlation with Price',
        color='Correlation with Price',
        color_continuous_scale='RdBu',
        title='Feature Correlation with House Price'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(
        "💡 **Interpretation:** Features with higher absolute correlation values have stronger relationships with house prices. "
        "Positive values mean the feature increases with price, while negative values mean the opposite."
    )

# Tab 4: Analysis & Visualization
with tab4:
    st.header("Data Analysis & Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Distributions")
        try:
            from PIL import Image
            img = Image.open('feature_distributions.png')
            st.image(img, use_container_width=True)
        except:
            st.warning("Feature distribution plot not found")
    
    with col2:
        st.subheader("Model Performance Comparison")
        try:
            from PIL import Image
            img = Image.open('model_comparison.png')
            st.image(img, use_container_width=True)
        except:
            st.warning("Model comparison plot not found")
    
    st.subheader("Dataset Statistics")
    try:
        df = pd.read_csv('house_price_regression_dataset.csv')
        st.dataframe(df.describe(), use_container_width=True)
    except:
        st.warning("Dataset not found")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>House Price Regression Model | Built with Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
