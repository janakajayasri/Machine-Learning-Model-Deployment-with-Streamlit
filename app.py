import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('winequality-red.csv', sep=';')  # Assume the CSV is in the same directory
    df['label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    df.drop('quality', axis=1, inplace=True)
    return df

df = load_data()

# Features list
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# Load saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Function to compute model performance metrics
@st.cache_data
def compute_metrics():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_scaler = StandardScaler()  # Fit a new scaler for consistency in computation
    X_train_scaled = train_scaler.fit_transform(X_train)
    X_test_scaled = train_scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred)
    lr_cm = confusion_matrix(y_test, lr_pred)

    return rf_acc, rf_report, rf_cm, lr_acc, lr_report, lr_cm

# App title and description
st.title("Wine Quality Prediction App")
st.write("""
This application allows you to explore the Wine Quality dataset, visualize features, 
predict wine quality using a trained Random Forest model, and view model performance metrics.
The model classifies wine as 'Good' (1) if quality >=7 or 'Bad' (0) otherwise.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sections", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])

if page == "Data Exploration":
    st.header("Data Exploration")
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {df.columns.tolist()}")
    st.write("Data Types:")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    st.subheader("Interactive Data Filtering")
    st.write("Filter data based on alcohol content:")
    min_alcohol, max_alcohol = st.slider("Alcohol range", float(df['alcohol'].min()), float(df['alcohol'].max()), 
                                         (float(df['alcohol'].min()), float(df['alcohol'].max())))
    filtered_df = df[(df['alcohol'] >= min_alcohol) & (df['alcohol'] <= max_alcohol)]
    st.dataframe(filtered_df)
    st.info("Use the slider to filter the dataset by alcohol values.")

elif page == "Visualizations":
    st.header("Visualizations")
    st.write("Explore interactive visualizations of the dataset.")

    # Interactive feature selection
    selected_feature = st.selectbox("Select feature for Histogram and Boxplot", features)

    # Chart 1: Histogram
    st.subheader(f"Histogram of {selected_feature.capitalize()}")
    fig1, ax1 = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax1)
    ax1.set_title(f"Distribution of {selected_feature.capitalize()}")
    st.pyplot(fig1)

    # Chart 2: Boxplot vs Label
    st.subheader(f"Boxplot of {selected_feature.capitalize()} vs Quality Label")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='label', y=selected_feature, data=df, ax=ax2)
    ax2.set_title(f"{selected_feature.capitalize()} by Quality")
    ax2.set_xticklabels(['Bad', 'Good'])
    st.pyplot(fig2)

    # Chart 3: Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)

    st.info("Select a feature from the dropdown to update the histogram and boxplot.")

elif page == "Model Prediction":
    st.header("Model Prediction")
    st.write("Enter feature values to predict wine quality using the Random Forest model.")

    # User inputs with min/max from data
    input_values = {}
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    for i, feature in enumerate(features):
        min_v = float(df[feature].min())
        max_v = float(df[feature].max())
        default = float(df[feature].median())
        with columns[i % 3]:
            input_values[feature] = st.number_input(feature.replace('_', ' ').capitalize(), 
                                                    min_value=min_v, max_value=max_v, value=default)

    if st.button("Predict"):
        features_list = [input_values[feature] for feature in features]
        features_array = np.array([features_list])
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("Good Wine")
        else:
            st.error("Bad Wine")
        st.write(f"Probability of being Good: {probability:.2f}")

    st.info("Adjust the input values and click 'Predict' for real-time results. Inputs are constrained to realistic ranges based on the dataset.")

elif page == "Model Performance":
    st.header("Model Performance")
    st.write("Evaluation metrics for the trained models on the test set.")

    rf_acc, rf_report, rf_cm, lr_acc, lr_report, lr_cm = compute_metrics()

    st.subheader("Model Comparison")
    st.write(f"Random Forest Accuracy: {rf_acc:.4f}")
    st.write(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    st.write("The Random Forest model performs better and is used for predictions.")

    st.subheader("Random Forest Classification Report")
    st.text(rf_report)

    st.subheader("Random Forest Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_xticklabels(['Bad', 'Good'])
    ax_cm.set_yticklabels(['Bad', 'Good'])
    st.pyplot(fig_cm)

    st.subheader("Logistic Regression Classification Report")
    st.text(lr_report)

    st.info("Metrics are computed on a 20% test split with random_state=42 for reproducibility.")
