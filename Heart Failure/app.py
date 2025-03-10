import streamlit as st
import pickle
import numpy as np
import xgboost as xgb

# Load the model
model = pickle.load(open('hf1.pkl', 'rb'))

def predict_xgboost(inputs):
    input_array = np.array([inputs]).astype(np.float64)
    prediction = model.predict(input_array)
    return float('{0:.{1}f}'.format(prediction[0], 2))

def main():
    st.title("Heart Failure Prediction App (XGBoost)")

    # Input fields with validation
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    anaemia = st.selectbox("Anaemia", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0.0, value=10.0)
    diabetes = st.selectbox("Diabetes", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100, value=50)
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
    platelets = st.number_input("Platelets", min_value=0.0, value=250000.0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0.0, value=135.0)
    sex = st.selectbox("Sex", [0, 1])  # Assuming 0 for female, 1 for male
    smoking = st.selectbox("Smoking", [0, 1])
    time = st.number_input("Time", min_value=0, value=10)

    if st.button("Predict"):
        inputs = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                  high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]
        output = predict_xgboost(inputs)
        st.success(f'The probability of heart failure is {output}')

        if output > 0.5:
            st.markdown("<div style='background-color:#F08080;padding:10px'><h2 style='color:black;text-align:center;'>You are in danger</h2></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color:#F4D03F;padding:10px'><h2 style='color:white;text-align:center;'>You are safe</h2></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
