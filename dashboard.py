# app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 1. Load Dataset (Pima Indians Diabetes Dataset)
# -------------------------------------------------
@st.cache_data
def load_data():
    
    data = pd.read_csv("Diabetes.csv")
    return data

data = load_data()

# -------------------------------------------------
# 2. Train Logistic Regression Model
# -------------------------------------------------
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------------------
# 3. Streamlit App UI
# -------------------------------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Predict whether a patient is likely to have diabetes using health data. "
         "This model is trained on the **Pima Indians Diabetes Dataset**.")

st.sidebar.header("Enter Patient Details:")

# Sidebar Inputs
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
bp = st.sidebar.number_input("Blood Pressure (Diastolic)", min_value=0, max_value=122, value=70)
skin = st.sidebar.number_input("Triceps Skin Fold Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

# Collect input features
input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

# Prediction Button
if st.sidebar.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The model predicts this patient is **Diabetic** with {probability*100:.2f}% probability.")
    else:
        st.success(f"✅ The model predicts this patient is **Not Diabetic** with {probability*100:.2f}% probability.")

# -------------------------------------------------
# 4. Show Dataset & Model Info
# -------------------------------------------------
with st.expander("📊 View Dataset Sample"):
    st.dataframe(data.head())

with st.expander("ℹ️ About the Model"):
    st.write("""
    - Algorithm: Logistic Regression  
    - Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DPF, Age  
    - Target: Outcome (1 = Diabetic, 0 = Not Diabetic)  
    - Train/Test Split: 80/20  
    """)
