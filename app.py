import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load data and model
df = pd.read_csv("/content/WineQT.csv")
model = joblib.load("model.joblib")

# Preprocess data for model performance page
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
X = df.drop('quality', axis=1)
y = df['quality']

# Corrected train_test_split (now capturing all 4 values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_pred = model.predict(X_test)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance"])

# Home Page
if page == "Home":
    st.title("🍷 Wine Quality Predictor")
    st.image("https://images.unsplash.com/photo-1514933651103-005eec06c04b", width=600)
    st.write("""
    This app predicts wine quality (0: Average, 1: Excellent) using machine learning.
    - **Dataset:** Red Wine Quality from Kaggle
    - **Model:** Random Forest (Accuracy: 89%)
    """)

# Data Exploration Page
elif page == "Data Exploration":
    st.title("🔍 Data Exploration")
    st.subheader("Dataset Overview")
    st.write(f"**Shape:** {df.shape} | **Columns:** {df.columns.tolist()}")
    
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    st.subheader("Filter Data")
    col_filter = st.selectbox("Select column to filter", df.columns)
    min_val, max_val = float(df[col_filter].min()), float(df[col_filter].max())
    user_val = st.slider(f"Filter {col_filter}", min_val, max_val, (min_val, max_val))
    filtered_df = df[(df[col_filter] >= user_val[0]) & (df[col_filter] <= user_val[1])]
    st.dataframe(filtered_df)

# Visualizations Page
elif page == "Visualizations":
    st.title("📊 Data Visualizations")
    
    # Histogram
    st.subheader("Feature Distribution")
    hist_col = st.selectbox("Select feature for histogram", df.columns)
    fig1 = px.histogram(df, x=hist_col, color='quality', nbins=30)
    st.plotly_chart(fig1)
    
    # Scatter plot
    st.subheader("Correlation Analysis")
    col_x = st.selectbox("X-axis", df.columns)
    col_y = st.selectbox("Y-axis", df.columns)
    fig2 = px.scatter(df, x=col_x, y=col_y, color='quality')
    st.plotly_chart(fig2)
    
    # Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig3 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig3)

# Prediction Page
elif page == "Prediction":
    st.title("🔮 Quality Prediction")
    st.write("Adjust feature values to predict wine quality:")
    
    # Input widgets
    inputs = {}
    cols = st.columns(3)
    for i, col in enumerate(X.columns):  # Use X.columns instead of df.columns[:-1]
        with cols[i % 3]:
            inputs[col] = st.slider(
                f"{col}",
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].mean())
            )
    
    # Predict
    if st.button("Predict Quality"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.success(f"Prediction: {'Excellent 🎉' if prediction == 1 else 'Average 👍'}")
        st.metric("Probability of Excellent Quality", f"{proba:.0%}")

# Model Performance Page
elif page == "Model Performance":
    st.title("📈 Model Evaluation")
    accuracy = accuracy_score(y_test, rf_pred)
    f1 = f1_score(y_test, rf_pred)
    st.subheader("Random Forest Performance")
    st.write(f"**Accuracy:** {accuracy:.2%} | **F1-Score:** {f1:.2f}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, rf_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Average', 'Excellent'],
                yticklabels=['Average', 'Excellent'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    feat_importance = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(feat_importance)

