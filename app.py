import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset and train model
data = pd.read_csv("internet_usage_data.csv")
X = data[['streaming_hours', 'social_media_hours', 'online_classes_hours', 'gaming_hours', 'devices']]
y = data['monthly_usage_gb']
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Internet Usage Plan Advisor")

streaming = st.number_input("Streaming Hours/Week", min_value=0)
social = st.number_input("Social Media Hours/Week", min_value=0)
classes = st.number_input("Online Classes Hours/Week", min_value=0)
gaming = st.number_input("Gaming Hours/Week", min_value=0)
devices = st.number_input("Number of Devices", min_value=1)

if st.button("Predict Usage & Suggest Plan"):
    input_data = np.array([[streaming, social, classes, gaming, devices]])
    usage_pred = model.predict(input_data)[0]
    
    if usage_pred <= 50:
        plan = "50GB Plan"
    elif usage_pred <= 100:
        plan = "100GB Plan"
    else:
        plan = "Unlimited Plan"
    
    st.success(f"Predicted Monthly Usage: {usage_pred:.2f} GB")
    st.info(f"Suggested Plan: {plan}")
