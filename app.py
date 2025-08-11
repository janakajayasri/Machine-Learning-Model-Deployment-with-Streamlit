# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load Data and Model
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("winequality.csv")  # Change to your dataset filename if different
    df['label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    return df

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

df = load_data()
df_model = df.copy()

model, scaler = load_model()

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Data Exploration",
    "Visualization",
    "Predict",
    "Model Performance"
])

# =========================
# Page: Home
# =========================
if page == "Home":
    st.title("Wine Quality Prediction App")
    st.write("""
        This app predicts whether a wine is of good quality (label = 1) or not (label = 0) 
        based on its physicochemical properties.
    """)

# =========================
# Page: Data Exploration
# =========================
elif page == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Class Distribution")
    st.write(df['label'].value_counts())

# =========================
# Page: Visualization
# =========================
elif page == "Visualization":
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select a feature to visualize", df.columns[:-2])  # exclude quality & label
    fig = px.histogram(df, x=feature, color="label", barmode="overlay", title=f"{feature} Distribution")
    st.plotly_chart(fig)

# =========================
# Page: Predict
# =========================
elif page == "Predict":
    st.subheader("Make a Prediction")

    # Create input fields for features
    input_data = []
    for col in df_model.columns[:-2]:  # exclude quality & label
        value = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data.append(value)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.write("Prediction:", "Good Quality" if prediction == 1 else "Not Good Quality")

# =========================
# Page: Model Performance
# =========================
elif page == "Model Performance":
    st.subheader("Model Evaluation")

    X = df_model.drop(columns=['label', 'quality'])
    y = df_model['label']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y, y_pred)
    st.write(f"Accuracy: {acc:.2f}")

    st.write("Classification Report")
    st.text(classification_report(y, y_pred))

    st.write("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)
