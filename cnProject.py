import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def preprocess_data(df):
    df = df.drop(columns=['Time'], errors='ignore')
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    df = df.fillna(df.median())
    return df

def train_model(df):
    X = df.drop(columns=['bad_packet'], errors='ignore')
    y = df['bad_packet']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return model, cm, accuracy, X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def plot_feature_importance(model, feature_names):
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    fig, ax = plt.subplots()
    feature_importances.sort_values(ascending=False).plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

st.set_page_config(page_title="ML Model Dashboard", layout="wide")
st.title("🚀 Technical ML Dashboard: Bad Packet Classification")

df = load_data()
if df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    df = preprocess_data(df)
    
    model, cm, accuracy, X_train, X_test, y_train, y_test = train_model(df)
    
    st.subheader("Confusion Matrix & Accuracy")
    plot_confusion_matrix(cm)
    st.write(f"**Model Accuracy:** {accuracy:.2%}")
    
    st.subheader("Feature Importance")
    plot_feature_importance(model, df.drop(columns=['bad_packet'], errors='ignore').columns)