import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="💉", layout="wide")

# Custom background and styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1588776814546-b74b66f4838d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: None;
    border-radius: 10px;
    padding: 10px 24px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("💉 Diabetes Risk Prediction")
st.subheader("🔎 Check your health status and get personalized advice")

# Sidebar inputs
st.sidebar.header("📝 Enter Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 150, 80)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 85)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 28.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 1, 120, 35)

# Prepare input
input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
standardized_input = scaler.transform(input_data)

# Predict button
if st.button("🚀 Predict"):
    prediction = model.predict(standardized_input)[0]

    if prediction == 0:
        st.success("✅ You are likely **not diabetic**. Great job maintaining your health!")
        st.balloons()
        advice = "🎯 Keep up your healthy lifestyle. Maintain a balanced diet and regular exercise."
    else:
        st.error("⚠️ The model predicts a high risk of **Diabetes**.")
        st.snow()
        advice = "💡 Please consult a doctor immediately. Monitor blood sugar and follow a controlled diet."

    st.markdown(f"**💬 Health Advice:** {advice}")

    # Chart (optional - static risk slice)

    # Generate PDF
    def generate_pdf(data, prediction):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Diabetes Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)

        labels = ['Pregnancies', 'Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']
        for name, value in zip(labels, data[0]):
            pdf.cell(200, 10, f"{name}: {value:.2f}", ln=True)

        pdf.ln(5)
        if prediction == 1:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(200, 10, "Prediction: High risk of Diabetes", ln=True)
            advice_text = "Advice: Please consult a doctor immediately. Monitor blood sugar and follow a controlled diet."
        else:
            pdf.set_text_color(0, 150, 0)
            pdf.cell(200, 10, "Prediction: No risk of Diabetes", ln=True)
            advice_text = "Advice: Maintain a healthy lifestyle, balanced diet, and regular exercise."

        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 10, advice_text)
        pdf.output("diabetes_report.pdf")
        return "diabetes_report.pdf"

   

    pdf_path = generate_pdf(input_data, prediction)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="diabetes_report.pdf", mime="application/pdf")
    os.remove(pdf_path)

    # Show input data
    with st.expander("📋 View Entered Details"):
        st.write(pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']))
with st.expander("🗂️ View Input Summary"):
    st.write(pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']))

# Footer
st.markdown("""
---
<center>
Made by Alisha Vashisht | 2025 👩‍⚕️🩺
</center>
""", unsafe_allow_html=True) 

