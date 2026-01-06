import streamlit as st
import joblib
import numpy as np

model = joblib.load("vgsales_logistic_model.pkl")

st.title("🎮 Video Game Sales Prediction App")
st.write("### Predict if a Game is a HIT or NOT")

na = st.number_input("NA Sales (millions)", min_value=0.0)
eu = st.number_input("EU Sales (millions)", min_value=0.0)
jp = st.number_input("JP Sales (millions)", min_value=0.0)
other = st.number_input("Other Sales (millions)", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[na, eu, jp, other]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"🔥 HIT GAME (Probability: {probability[0][1]:.2f})")
    else:
        st.error(f"❌ NOT A HIT (Probability: {probability[0][0]:.2f})")
