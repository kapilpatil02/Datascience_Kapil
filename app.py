import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

# Encoding
sex_male = 1 if sex == "Male" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, age, sibsp, parch, fare,
                            sex_male, embarked_C, embarked_Q]])
    
    input_data[:, [1, 4]] = scaler.transform(input_data[:, [1, 4]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
