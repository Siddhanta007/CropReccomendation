import streamlit as st
import numpy as np
import joblib as jb

def main():
    st.title("Crop Recommendation")
    st.write("Enter the following details for crop recommendation:")

    # Input fields for crop recommendation
    nitrogen = st.number_input("Nitrogen (kg/ha)", value=100.0, step=1.0)
    phosphorous = st.number_input("Phosphorous (kg/ha)", value=50.0, step=1.0)
    potassium = st.number_input("Potassium (kg/ha)", value=50.0, step=1.0)
    pH = st.number_input("Soil pH", value=7.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", value=500.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", value=60.0, step=1.0)

    # Submit button
    submit_button = st.button("Recommend")

    # Display crop recommendation on submission
    if submit_button:
        crop_recommendation = recommend_crop(nitrogen, phosphorous, potassium, pH, rainfall, temperature, humidity)
        st.write("Based on the inputs provided, we recommend growing", crop_recommendation)

def recommend_crop(nitrogen, phosphorous, potassium, pH, rainfall, temperature, humidity):
    # Logic for crop recommendation based on the input parameters
    # Replace with your custom recommendation logic
    
    data = np.array([[nitrogen, phosphorous, potassium, pH, rainfall, temperature, humidity]])
    RF = jb.load('/Users/srikrishna/Coding/DataScience/final_year_project/Crop_Reccomendation/model/RandomForest.pkl')
    prediction = RF.predict(data)
    return prediction[0]


if __name__ == "__main__":
    main()
