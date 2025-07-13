import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🔬 Diabetes Prediction App")
st.markdown("Enter patient details to predict the likelihood of diabetes.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 72)
        insulin = st.number_input("Insulin", 0, 900, 80)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47)
    with col2:
        glucose = st.number_input("Glucose", 0, 200, 120)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
        bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
        age = st.number_input("Age", 10, 100, 33)

    submitted = st.form_submit_button(" Predict")

if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                            bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader(" Prediction Result")
    if prediction == 1:
        st.error(f" The patient is **likely diabetic** (Probability: {prob:.2f})")
    else:
        st.success(f" The patient is **not likely diabetic** (Probability: {prob:.2f})")

if hasattr(model, 'feature_importances_'):
    st.subheader(" Feature Importance")
    importance = model.feature_importances_
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=feature_names, ax=ax)
    ax.set_title("Feature Importance (Tree-based model)")
    st.pyplot(fig)

if st.checkbox(" Show Dataset Visual Insights"):
    data = pd.read_csv("diabetes.csv")
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
    data.fillna(data.median(), inplace=True)

    st.markdown("###  Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    st.pyplot(fig1)

    st.markdown("###  Glucose Distribution by Outcome")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=data, x='Glucose', hue='Outcome', kde=True, ax=ax2)
    st.pyplot(fig2)

    st.markdown("###  BMI vs Age by Outcome")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=data, x='Age', y='BMI', hue='Outcome', ax=ax3)
    st.pyplot(fig3)
    

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
model_name = type(model).__name__

st.sidebar.markdown(f" Model Used: **{model_name}**")

if model_name == "XGBClassifier":
    st.info(" You're using a XGradient Boosting model, which is an ensemble of decision trees.")

st.markdown("---")
st.markdown(" Built with Streamlit |  Model: XGBoost |  Sakir's Portfolio Project")
