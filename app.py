import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualization", "Predict", "Model Performance"])

# Home page
if page == "Home":
    st.title("🍷 Wine Quality Prediction")
    st.write("This app predicts wine quality using a Random Forest model trained on the Wine Quality Dataset from Kaggle.")

# Data Exploration
elif page == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.subheader("Filter by Alcohol and pH")
    min_a, max_a = st.slider("Select alcohol range", 8.0, 14.0, (8.5, 12.5))
    min_ph, max_ph = st.slider("Select pH range", 2.5, 4.0, (3.0, 3.5))
    filtered_df = df[(df['alcohol'] >= min_a) & (df['alcohol'] <= max_a) & (df['pH'] >= min_ph) & (df['pH'] <= max_ph)]
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        st.write(filtered_df)

# Visualization
elif page == "Visualization":
    st.subheader("Quality Distribution")
    fig1 = px.histogram(df, x='quality', title="Wine Quality Distribution")
    st.plotly_chart(fig1)

    st.subheader("Alcohol vs Quality")
    fig2 = px.box(df, x='label', y='alcohol', title="Alcohol Content by Quality")
    st.plotly_chart(fig2)

    st.subheader("Correlation Heatmap")
    fig3 = px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlations")
    st.plotly_chart(fig3)

    st.subheader("Alcohol vs Sulphates")
    fig4 = px.scatter(df, x='alcohol', y='sulphates', color='label', title="Alcohol vs Sulphates")
    st.plotly_chart(fig4)

# Predict
elif page == "Predict":
    st.subheader("Enter Wine Features")
    try:
        with st.form("prediction_form"):
            fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, help="Acidity from non-volatile acids (4.0-16.0)")
            volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, help="Acidity from volatile acids (0.1-1.5)")
            citric_acid = st.number_input("Citric Acid", 0.0, 1.0, help="Citric acid content (0.0-1.0)")
            residual_sugar = st.number_input("Residual Sugar", 0.9, 15.0, help="Sugar content (0.9-15.0)")
            chlorides = st.number_input("Chlorides", 0.01, 0.2, help="Salt content (0.01-0.2)")
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1.0, 70.0, help="Free SO2 (1.0-70.0)")
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 6.0, 300.0, help="Total SO2 (6.0-300.0)")
            density = st.number_input("Density", 0.99000, 1.00500, help="Density of wine (0.99000-1.00500)")
            pH = st.number_input("pH", 2.5, 4.0, help="pH level (2.5-4.0)")
            sulphates = st.number_input("Sulphates", 0.3, 2.0, help="Sulphate content (0.3-2.0)")
            alcohol = st.number_input("Alcohol", 8.0, 15.0, help="Alcohol content in % (8.0-15.0)")
            submitted = st.form_submit_button("Predict")

        if submitted:
            with st.spinner("Predicting..."):
                input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                        density, pH, sulphates, alcohol]])
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][pred]
                result = "🍷 Good Quality" if pred == 1 else "❌ Not Good"
                st.success(f"Prediction: {result}")
                st.info(f"Confidence: {prob * 100:.2f}%")
    except Exception as e:
        st.error(f"Input Error: {e}")

# Model Performance
elif page == "Model Performance":
    st.subheader("Evaluate Model on Full Dataset")
    X = df.drop(columns=['quality', 'label'])
    y = df['label']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.text("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Model Comparison")
    fig5 = px.bar(x=['Random Forest', 'Logistic Regression'], y=[0.934, 0.899],
                  labels={'x': 'Model', 'y': 'Accuracy'}, title="Model Accuracy Comparison")
    st.plotly_chart(fig5)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig6 = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig6)