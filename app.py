import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ==================================
# Load Dataset
# ==================================
data = pd.read_csv("internet_usage_data.csv")

X = data[['streaming_hours',
          'social_media_hours',
          'online_classes_hours',
          'gaming_hours',
          'devices']]
y = data['monthly_usage_gb']

# ==================================
# Train Multiple Regression Models
# ==================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("model", LinearRegression())
    ])
}

scores = {}
trained_models = {}

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    scores[name] = r2_score(y, y_pred)
    trained_models[name] = model

# Select best model
best_model_name = max(scores, key=scores.get)
best_model = trained_models[best_model_name]

# ==================================
# Streamlit UI
# ==================================
st.set_page_config(page_title="Internet Usage Plan Advisor", layout="centered")
st.title("📶 Internet Usage Plan Advisor")
st.write("Smart prediction using advanced regression techniques")

st.subheader("Enter Your Weekly Usage")

streaming = st.number_input("🎬 Streaming Hours / Week", min_value=0)
social = st.number_input("📱 Social Media Hours / Week", min_value=0)
classes = st.number_input("🎓 Online Classes / Work Hours / Week", min_value=0)
gaming = st.number_input("🎮 Gaming Hours / Week", min_value=0)
devices = st.number_input("📱 Number of Connected Devices", min_value=1)

# ==================================
# Prediction
# ==================================
if st.button("🔍 Predict Usage & Suggest Plan"):

    input_data = np.array([[streaming, social, classes, gaming, devices]])
    usage_pred = best_model.predict(input_data)[0]
    daily_usage = usage_pred / 30

    # Plan logic
    if usage_pred <= 50:
        plan = "50GB Plan"
        cost = "₹399 / month"
        category = "Light User"
    elif usage_pred <= 100:
        plan = "100GB Plan"
        cost = "₹699 / month"
        category = "Moderate User"
    else:
        plan = "Unlimited Plan"
        cost = "₹999 / month"
        category = "Heavy User"

    # ==================================
    # Output
    # ==================================
    st.success(f"📊 Predicted Monthly Usage: {usage_pred:.2f} GB")
    st.info(f"📅 Estimated Daily Usage: {daily_usage:.2f} GB/day")
    st.success(f"🏷️ Usage Category: {category}")

    st.subheader("📦 Recommended Plan")
    st.info(f"Plan: {plan}")
    st.warning(f"Estimated Cost: {cost}")

    st.subheader("🧠 Model Used (Automatically Selected)")
    st.write(f"✅ **{best_model_name}**")
    st.write(f"R² Score: {scores[best_model_name]:.3f}")

    # ==================================
    # Usage Breakdown
    # ==================================
    st.subheader("📊 Estimated Usage Breakdown")

    st.write(f"🎬 Streaming: {streaming * 3:.1f} GB")
    st.write(f"📱 Social Media: {social * 1.5:.1f} GB")
    st.write(f"🎓 Online Classes: {classes * 2:.1f} GB")
    st.write(f"🎮 Gaming: {gaming * 4:.1f} GB")

    # ==================================
    # Smart Tips
    # ==================================
    st.subheader("💡 Smart Usage Tips")

    if streaming > 20:
        st.write("🔹 Reduce video quality to save streaming data.")
    if gaming > 15:
        st.write("🔹 Avoid background downloads while gaming.")
    if social > 20:
        st.write("🔹 Disable auto-play videos on social media.")
    if devices > 3:
        st.write("🔹 Disconnect unused devices.")

# Footer
st.markdown("---")
st.caption("Mini Project | Internet Usage Plan Advisor using Advanced Regression")
