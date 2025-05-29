import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model
model = joblib.load("accident_model.pkl")

# Configure page
st.set_page_config(page_title="Accident Risk Predictor", layout="wide")
st.title("🚦 Road Accident Risk Prediction System")

# Sidebar info
with st.sidebar:
    st.markdown("""
    **About This System**
    
    Predicts accident risk (0-120 scale) based on:
    - Traffic conditions
    - Weather
    - Road type
    - Historical accident data
    
    **Model Performance**
    - R2 Score: 0.89
    - MAE: 4.23
    """)

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        traffic = st.slider("Traffic Density (vehicles/km)", 0, 200, 100)
        speed = st.slider("Average Speed (km/h)", 0, 120, 60)
        weather = st.selectbox(
            "Weather Conditions",
            ["Clear", "Rainy", "Foggy", "Snowy"]
        )
        
    with col2:
        time = st.selectbox(
            "Time of Day",
            ["Morning", "Afternoon", "Night"]
        )
        road = st.selectbox(
            "Road Type",
            ["Highway", "City Road", "Rural Road"]
        )
        prev_accidents = st.slider("Previous Accidents (last year)", 0, 10, 2)
    
    submitted = st.form_submit_button("Predict Risk")

# Prediction logic
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'traffic_density': traffic,
        'avg_speed': speed,
        'weather': weather,
        'time_of_day': time,
        'road_type': road,
        'previous_accidents': prev_accidents
    }])
    
    # One-hot encode
    input_processed = pd.get_dummies(input_data)
    
    # Ensure all expected columns exist
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    
    input_processed = input_processed[expected_columns]
    
    # Predict
    risk = model.predict(input_processed)[0]
    
    # Display results
    st.subheader("Prediction Result")
    
    if risk < 40:
        color = "green"
        status = "Low Risk"
    elif risk < 70:
        color = "orange"
        status = "Moderate Risk"
    else:
        color = "red"
        status = "High Risk"
    
    st.markdown(f"""
    <div style="
        background-color: {color}20;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid {color};
        margin: 1rem 0;
    ">
        <h3 style="margin:0;color:{color}">
            Predicted Risk: <strong>{risk:.1f}</strong> ({status})
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        st.subheader("Key Risk Factors")
        importance_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(5)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# Run locally: streamlit run app.py