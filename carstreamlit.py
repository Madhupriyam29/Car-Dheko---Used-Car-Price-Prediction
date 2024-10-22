import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
from PIL import Image
import os

# Load the trained model using joblib
model = joblib.load(r"C:\Users\Madhu\Downloads\model.pkl")

# Function to preprocess input data
def preprocess_data(data):
    features = [
        'modelYear',
        'Engine Displacement',
        'Power Steering',
        'transmission',
        'Engine Type',
        'Turbo Charger',
        'Mileage',
        'Max Power',
        'Torque',
        'No Of Airbags',
        'Leather Seats',
        'Air Conditioner',
        'Bluetooth',
        'Touch Screen',
        'Wheel Size',
        'Alloy Wheels',
        'Roof Rail',
        'Rear Camera',
        'Length',
        'Drive Type'
    ]
    
    df = pd.DataFrame([data], columns=features)
    
    categorical_features = [
        'transmission', 'Engine Type', 'Turbo Charger',
        'Leather Seats', 'Air Conditioner', 'Bluetooth', 
        'Touch Screen', 'Alloy Wheels', 'Roof Rail', 
        'Rear Camera', 'Drive Type'
    ]
    numeric_features = [
        'modelYear', 'Engine Displacement', 'Power Steering',
        'Mileage', 'Max Power', 'Torque', 'No Of Airbags', 
        'Wheel Size', 'Length'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    df_transformed = preprocessor.fit_transform(df)
    
    return df_transformed

# Streamlit app
st.title("Vehicle Price Prediction")

# Sidebar for input fields
st.sidebar.header("Input Fields")
model_year = st.sidebar.number_input("Model Year", min_value=2000, max_value=2024, value=2020)
engine_displacement = st.sidebar.number_input("Engine Displacement (L)", min_value=0, format="%d")
power_steering = st.sidebar.selectbox("Power Steering", ["Yes", "No"])
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "CVT", "DCT", "Direct Drive"])
engine_type = st.sidebar.selectbox("Engine Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
turbo_charger = st.sidebar.selectbox("Turbo Charger", ["Yes", "No"])
mileage = st.sidebar.number_input("Mileage (km/l)", min_value=0, format="%d")
max_power = st.sidebar.number_input("Max Power (hp)", min_value=0, format="%d")
torque = st.sidebar.number_input("Torque (Nm)", min_value=0, format="%d")
no_of_airbags = st.sidebar.number_input("No Of Airbags", min_value=0, max_value=10, format="%d")
leather_seats = st.sidebar.selectbox("Leather Seats", ["Yes", "No"])
air_conditioner = st.sidebar.selectbox("Air Conditioner", ["Yes", "No"])
bluetooth = st.sidebar.selectbox("Bluetooth", ["Yes", "No"])
touch_screen = st.sidebar.selectbox("Touch Screen", ["Yes", "No"])
wheel_size = st.sidebar.number_input("Wheel Size (inches)", min_value=0, format="%d")
alloy_wheels = st.sidebar.selectbox("Alloy Wheels", ["Yes", "No"])
roof_rail = st.sidebar.selectbox("Roof Rail", ["Yes", "No"])
rear_camera = st.sidebar.selectbox("Rear Camera", ["Yes", "No"])
length = st.sidebar.number_input("Length (mm)", min_value=0, format="%d")
drive_type = st.sidebar.selectbox("Drive Type", ["FWD", "RWD", "AWD"])

# Create a dictionary with the input data
input_data = {
    'modelYear': model_year,
    'Engine Displacement': engine_displacement,
    'Power Steering': 1 if power_steering == "Yes" else 0,
    'transmission': transmission,
    'Engine Type': engine_type,
    'Turbo Charger': 1 if turbo_charger == "Yes" else 0,
    'Mileage': mileage,
    'Max Power': max_power,
    'Torque': torque,
    'No Of Airbags': no_of_airbags,
    'Leather Seats': 1 if leather_seats == "Yes" else 0,
    'Air Conditioner': 1 if air_conditioner == "Yes" else 0,
    'Bluetooth': 1 if bluetooth == "Yes" else 0,
    'Touch Screen': 1 if touch_screen == "Yes" else 0,
    'Wheel Size': wheel_size,
    'Alloy Wheels': 1 if alloy_wheels == "Yes" else 0,
    'Roof Rail': 1 if roof_rail == "Yes" else 0,
    'Rear Camera': 1 if rear_camera == "Yes" else 0,
    'Length': length,
    'Drive Type': drive_type
}

# Preprocess the input data
processed_data = preprocess_data(input_data)

# Predict the price
if st.sidebar.button("Predict Price"):
    prediction = model.predict(processed_data)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")

    # Display car photo
    # car_model = st.selectbox("Select Car Model", ["ModelA", "ModelB", "ModelC"])  # Replace with actual car models
    # photo_path = os.path.join(CAR_PHOTOS_DIR, f"{car_model}.jpg")  # Path to car photos
    # if os.path.exists(photo_path):
    #     image = Image.open(photo_path)
    #     st.image(image, caption=f"{car_model} Photo", use_column_width=True)
    # else:
    #     st.write("Photo not available for the selected model.")